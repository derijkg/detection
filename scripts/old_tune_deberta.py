#!/usr/bin/env python3
# scripts/tune_deberta.py

import os
import sys

# ---------------------------------------------------------------------
# CRITICAL GPU & THREADING ISOLATION (Must be set BEFORE importing torch)
# ---------------------------------------------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# ---------------------------------------------------------------------

import argparse
import gc
import json
import shutil
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.multiprocessing as mp

# Set spawn start method safely to prevent CUDA worker deadlocks
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

# Calculate project root dynamically (~/detection)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"

from src.data.data_loader import DataFilter, DetectionDataManager

# Suppress Optuna verbose trial logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptunaPruningCallback(TrainerCallback):
    """
    Custom TrainerCallback that reports intermediate validation metrics to Optuna
    at exact step intervals and prunes unpromising trials early.
    """
    def __init__(self, trial: optuna.Trial, monitor: str = "eval_pauc_001"):
        self.trial = trial
        self.monitor = monitor

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and self.monitor in metrics:
            current_value = metrics[self.monitor]
            step = state.global_step
            self.trial.report(current_value, step=step)
            if self.trial.should_prune():
                raise optuna.TrialPruned(f"Trial pruned at step {step} with {self.monitor}={current_value:.6f}")


def ensure_length_column(dataset):
    """
    Ensures the dataset has an explicit integer 'length' column.
    Prevents LengthGroupedSampler from attempting to compare PyTorch 1D input_ids
    tensors directly during sorting.
    """
    if "length" not in dataset.column_names:
        lengths = [len(x) for x in dataset["input_ids"]]
        dataset = dataset.add_column("length", lengths)
    return dataset


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]

    # Calculate probabilities directly via Sigmoid (faster than NumPy Softmax)
    probs = 1.0 / (1.0 + np.exp(-(logits[:, 1] - logits[:, 0])))
    preds = np.argmax(logits, axis=-1)

    acc = float(accuracy_score(labels, preds))
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)

    try:
        roc_auc = float(roc_auc_score(labels, probs))
    except ValueError:
        roc_auc = 0.5

    # Safe pAUC Calculation
    num_negatives = np.sum(labels == 0)
    if num_negatives > 0:
        # Ensure max_fpr is at least 2 / num_negatives to prevent scikit-learn ValueError
        safe_max_fpr = max(0.01, 2.0 / float(num_negatives))
        try:
            pauc_001 = float(roc_auc_score(labels, probs, max_fpr=safe_max_fpr))
        except ValueError:
            pauc_001 = roc_auc
    else:
        pauc_001 = 0.5

    return {
        "pauc_001": pauc_001,
        "roc_auc": roc_auc,
        "accuracy": acc,
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
    }


def get_hardware_precision():
    """
    Dynamically determine optimal precision format (bf16, fp16, or fp32) based on
    exact CUDA hardware Compute Capability.
    """
    use_fp16 = False
    use_bf16 = False
    
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        # Hardware BF16 cuBLAS tensor cores require Ampere (Compute Capability >= 8.0) or newer
        if capability[0] >= 8 and torch.cuda.is_bf16_supported():
            use_bf16 = True
        elif capability[0] >= 7:  # Volta (7.0) or Turing (7.5)
            use_fp16 = True

    return use_fp16, use_bf16


def optuna_objective(
    trial,
    train_ds,
    dev_ds,
    tokenizer,
    model_name,
    scope_dir,
    eval_steps=100,  # Reduced from 200 for better pruning granularity
    early_stopping_patience=3,
    early_stopping_threshold=0.001,
    seed=42,
):
    print(f"\n" + "=" * 70)
    print(f" Starting Optuna Trial #{trial.number} ")
    print("=" * 70)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # --- Improved Search Space ---
    num_train_epochs = trial.suggest_int("num_train_epochs", 2, 4)
    learning_rate = trial.suggest_float("learning_rate", 5e-6, 4e-5, log=True)
    
    # 1. Real Effective Batch Size Tuning
    per_device_train_batch_size = 4  # Fixed physical batch size for speed/VRAM
    effective_batch_size = trial.suggest_categorical("effective_batch_size", [16, 32])
    gradient_accumulation_steps = effective_batch_size // per_device_train_batch_size

    weight_decay = trial.suggest_float("weight_decay", 1e-2, 1e-1, log=True)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.05, 0.2)
    label_smoothing_factor = trial.suggest_float("label_smoothing_factor", 0.0, 0.1)
    
    # 2. Learning Rate Scheduler Search (Cosine vs Linear)
    lr_scheduler_type = trial.suggest_categorical("lr_scheduler_type", ["linear", "cosine"])

    use_fp16, use_bf16 = get_hardware_precision()

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "Human", 1: "LLM"},
        label2id={"Human": 0, "LLM": 1},
        use_safetensors=True,
    )
    model.config.use_cache = False

    trial_output_dir = scope_dir / "optuna_trials" / f"trial_{trial.number}"
    optimizer_type = "adamw_torch_fused" if torch.cuda.is_available() and torch.__version__ >= "2.0" else "adamw_torch"

    training_args = TrainingArguments(
        output_dir=str(trial_output_dir),
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="no",
        save_steps=eval_steps,
        save_total_limit=1,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,  # Applied scheduler
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=8,  # Increased eval batch size for faster evaluation
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        label_smoothing_factor=label_smoothing_factor,
        load_best_model_at_end=False,
        metric_for_best_model="pauc_001",
        greater_is_better=True,
        fp16=use_fp16,
        bf16=use_bf16,
        group_by_length=True,
        length_column_name="length",
        optim=optimizer_type,
        gradient_checkpointing=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=0,
        eval_accumulation_steps=10,
        logging_steps=25,
        report_to="none",
        disable_tqdm=False,
        seed=seed,
        data_seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8 if (use_fp16 or use_bf16) else None),
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=early_stopping_threshold,
            ),
            OptunaPruningCallback(trial=trial, monitor="eval_pauc_001"),
        ],
    )

    target_metric = None

    try:
        trainer.train()
        eval_metrics = trainer.evaluate()
        target_metric = eval_metrics["eval_pauc_001"]

        try:
            prev_best_val = trial.study.best_value
            prev_best_num = trial.study.best_trial.number
            if target_metric > prev_best_val:
                best_str = f"🎉 NEW BEST! {target_metric:.6f} (Previous Best: {prev_best_val:.6f} from Trial #{prev_best_num})"
            else:
                best_str = f"Best So Far: {prev_best_val:.6f} (Trial #{prev_best_num})"
        except ValueError:
            best_str = f"🎉 NEW BEST! {target_metric:.6f} (First completed trial)"

        print("\n" + "-" * 70)
        print(f"--> [Trial #{trial.number} Finished]")
        print(f"    Current Trial pAUC@FPR<=0.01 : {target_metric:.6f} | ROC-AUC: {eval_metrics['eval_roc_auc']:.4f}")
        print(f"    {best_str}")
        print("-" * 70 + "\n")

    except optuna.TrialPruned:
        try:
            prev_best_val = trial.study.best_value
            prev_best_num = trial.study.best_trial.number
            best_info = f"Best So Far: {prev_best_val:.6f} (Trial #{prev_best_num})"
        except ValueError:
            best_info = "Best So Far: None"

        print(f"\n[PRUNED] Trial #{trial.number} was pruned by Optuna pruner. ({best_info})\n")
        raise

    except torch.OutOfMemoryError:
        print(f"\n[OOM WARNING] Trial #{trial.number} ran out of CUDA memory. Pruning trial...\n")
        target_metric = None

    finally:
        # Explicitly break internal references to prevent VRAM leakage
        if 'trainer' in locals():
            trainer.model = None
            trainer.optimizer = None
            trainer.lr_scheduler = None
            del trainer
        if 'model' in locals():
            del model
            
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if trial_output_dir.exists():
            shutil.rmtree(trial_output_dir, ignore_errors=True)

    return target_metric


def tune_scope(scope: str, args, manager: DetectionDataManager):
    base_outputs_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUTS_DIR
    scope_dir = base_outputs_dir / "mdeberta" / scope
    scope_dir.mkdir(parents=True, exist_ok=True)

    db_path = scope_dir / "optuna_study.db"
    storage_url = f"sqlite:///{db_path}"

    # --- 1. RESET STUDY CLEANUP ---
    if args.reset_study:
        print(f"\n[RESET STUDY] Cleaning up existing Optuna database and trial artifacts for scope '{scope}'...")
        
        for ext in ["", "-wal", "-shm", "-journal"]:
            f = Path(str(db_path) + ext)
            if f.exists():
                try:
                    f.unlink()
                except Exception as e:
                    print(f"Warning: Could not remove {f}: {e}")

        trials_dir = scope_dir / "optuna_trials"
        if trials_dir.exists():
            shutil.rmtree(trials_dir, ignore_errors=True)

        best_json = scope_dir / "best_hyperparameters.json"
        if best_json.exists():
            try:
                best_json.unlink()
            except Exception:
                pass

    # --- 2. INITIALIZE SQLITE STORAGE WITH WAL & TIMEOUT ---
    storage = optuna.storages.RDBStorage(
        url=storage_url,
        engine_kwargs={
            "connect_args": {"timeout": 120.0},
            "pool_pre_ping": True,
        }
    )

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        visible_env = os.environ.get("CUDA_VISIBLE_DEVICES", "Not set")
        gpu_status = f"{gpu_name} (CUDA_VISIBLE_DEVICES={visible_env})"
    else:
        gpu_status = "CPU Mode"

    # --- Scope-Specific Sample Size & Sequence Length Resolution ---
    if scope == "full":
        train_sample_size = args.full_train_sample_size if args.full_train_sample_size != -1 else args.train_sample_size
        dev_sample_size = args.full_dev_sample_size if args.full_dev_sample_size != -1 else args.dev_sample_size
        max_length = args.full_max_length if args.full_max_length != -1 else args.max_length
    elif scope == "sentence":
        train_sample_size = args.sentence_train_sample_size if args.sentence_train_sample_size != -1 else args.train_sample_size
        dev_sample_size = args.sentence_dev_sample_size if args.sentence_dev_sample_size != -1 else args.dev_sample_size
        max_length = args.sentence_max_length if args.sentence_max_length != -1 else args.max_length
    else:
        train_sample_size = args.train_sample_size
        dev_sample_size = args.dev_sample_size
        max_length = args.max_length

    print("\n" + "=" * 70)
    print(f" TUNING mDeBERTa-v3 HYPERPARAMETERS FOR SCOPE: '{scope.upper()}' ")
    print(f" Active Device          : {gpu_status}")
    print(f" Target Metric          : pAUC @ max FPR <= 0.01")
    print(f" Output Folder          : {scope_dir}")
    print(f" Configured Sample Size : Train={train_sample_size} | Dev={dev_sample_size}")
    print(f" Max Sequence Length    : {max_length}")
    print(f" Gradient Checkpointing : False")
    print(f" Eval Interval          : Every {args.eval_steps} steps")
    print(f" Early Stopping Patience: {args.early_stopping_patience} check(s)")
    print(f" Early Stopping Thresh  : {args.early_stopping_threshold}")
    print("=" * 70 + "\n")

    train_df = manager.filter_dataframe(DataFilter(splits=["train"], scopes=[scope]), sample_size=train_sample_size, seed=args.seed)
    dev_df = manager.filter_dataframe(DataFilter(splits=["dev"], scopes=[scope]), sample_size=dev_sample_size, seed=args.seed)

    print(f"Loaded DataFrames -> Train Samples: {len(train_df)} | Dev Samples: {len(dev_df)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)

    train_ds = manager.get_tokenized_dataset(
        scope=scope, split="train", tokenizer=tokenizer, max_length=max_length, return_format="torch"
    )
    dev_ds = manager.get_tokenized_dataset(
        scope=scope, split="dev", tokenizer=tokenizer, max_length=max_length, return_format="torch"
    )

    # --- VECTORIZED FAST DATASET FILTERING ---
    if train_sample_size > 0 and len(train_ds) > train_sample_size:
        if "_id" in train_df.columns and "_id" in train_ds.column_names:
            s_tr = set(train_df["_id"].astype(str))
            ds_ids = np.array(train_ds["_id"]).astype(str)
            tr_indices = np.where(np.isin(ds_ids, list(s_tr)))[0].tolist()
            train_ds = train_ds.select(tr_indices)
        else:
            train_ds = train_ds.select(range(train_sample_size))

    if dev_sample_size > 0 and len(dev_ds) > dev_sample_size:
        if "_id" in dev_df.columns and "_id" in dev_ds.column_names:
            s_dev = set(dev_df["_id"].astype(str))
            ds_ids = np.array(dev_ds["_id"]).astype(str)
            dev_indices = np.where(np.isin(ds_ids, list(s_dev)))[0].tolist()
            dev_ds = dev_ds.select(dev_indices)
        else:
            dev_ds = dev_ds.select(range(dev_sample_size))

    print(f"Final Tokenized Dataset -> Train: {len(train_ds)} | Dev: {len(dev_ds)}")

    # Precompute integer sequence lengths for batch bucketing
    train_ds = ensure_length_column(train_ds)
    dev_ds = ensure_length_column(dev_ds)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=args.eval_steps)
    sampler = optuna.samplers.TPESampler(
    multivariate=True, 
    group=True, 
    seed=args.seed
    )

    study = optuna.create_study(
        direction="maximize", 
        pruner=pruner,
        sampler=sampler,
        study_name=f"mdeberta_{scope}_pauc_optimization",
        storage=storage,
        load_if_exists=True,
    )

    print(f"\nStarting Optuna Search ({args.n_trials} Trials)...")
    study.optimize(
        lambda trial: optuna_objective(
            trial,
            train_ds=train_ds,
            dev_ds=dev_ds,
            tokenizer=tokenizer,
            model_name=args.model_name,
            scope_dir=scope_dir,
            eval_steps=args.eval_steps,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold,
            seed=args.seed,
        ),
        n_trials=args.n_trials,
        n_jobs=1,
    )

    best_trial = study.best_trial
    print("\n" + "=" * 60)
    print(f"OPTUNA SEARCH COMPLETE [{scope.upper()}]")
    print(f"Best Trial Number                : #{best_trial.number}")
    print(f"Best Validation pAUC @ FPR<=0.01 : {best_trial.value:.6f}")
    print("Best Hyperparameters Found       :")
    for k, v in best_trial.params.items():
        print(f"  - {k}: {v}")
    print("=" * 60 + "\n")

    best_params_save_path = scope_dir / "best_hyperparameters.json"
    hyperparams_json = {
        "scope": scope,
        "max_length": max_length,
        "best_trial_number": best_trial.number,
        "best_val_pauc_001": best_trial.value,
        "best_hyperparameters": best_trial.params,
        "train_sample_size": len(train_ds),
        "dev_sample_size": len(dev_ds),
        "eval_steps": args.eval_steps,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_threshold": args.early_stopping_threshold,
    }

    with open(best_params_save_path, "w") as f:
        json.dump(hyperparams_json, f, indent=4)

    print(f"[SAVED PARAMS] Best hyperparameters saved to: '{best_params_save_path}'")


def main():
    parser = argparse.ArgumentParser(description="Tune mDeBERTa-v3 hyperparameters for pAUC at max FPR 0.01.")

    parser.add_argument(
        "--scopes",
        nargs="+",
        default=["full", "sentence"],
        choices=["full", "sentence"],
        help="List of scopes to tune (default: full sentence)."
    )
    parser.add_argument("--model_name", type=str, default="microsoft/mdeberta-v3-base", help="Hugging Face model checkpoint.")
    
    # --- Global Fallbacks ---
    parser.add_argument("--sample_size", type=int, default=-1, help="Global sample size fallback (-1 for full dataset).")
    parser.add_argument("--train_sample_size", type=int, default=-1, help="Global train sample size fallback.")
    parser.add_argument("--dev_sample_size", type=int, default=-1, help="Global dev sample size fallback.")
    parser.add_argument("--max_length", type=int, default=256, help="Global fallback max sequence length.")

    # --- Scope-Specific Sample Sizes ---
    parser.add_argument("--full_train_sample_size", type=int, default=5000, help="Train sample size for 'full' abstract scope.")
    parser.add_argument("--full_dev_sample_size", type=int, default=-1, help="Dev sample size for 'full' abstract scope.")
    parser.add_argument("--sentence_train_sample_size", type=int, default=10000, help="Train sample size for 'sentence' scope.")
    parser.add_argument("--sentence_dev_sample_size", type=int, default=-1, help="Dev sample size for 'sentence' scope.")

    # --- Scope-Specific Max Sequence Lengths ---
    parser.add_argument("--full_max_length", type=int, default=256, help="Max sequence length for 'full' scope.")
    parser.add_argument("--sentence_max_length", type=int, default=128, help="Max sequence length for 'sentence' scope.")

    # --- Evaluation & Pruning Frequency ---
    parser.add_argument("--eval_steps", type=int, default=200, help="Interval in steps between evaluations and Optuna pruning checks.")

    # --- Early Stopping Configuration ---
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Number of eval checks with no metric improvement before stopping training early.")
    parser.add_argument("--early_stopping_threshold", type=float, default=0.001, help="Minimum metric improvement required to reset early stopping timer.")

    parser.add_argument("--n_trials", type=int, default=10, help="Number of Optuna search trials per scope.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output_dir", type=str, default=None, help="Base outputs directory.")
    parser.add_argument("--reset_study", action="store_true", help="Delete previous Optuna study database and trial artifacts to start fresh.")

    args = parser.parse_args()
    manager = DetectionDataManager()

    for scope in args.scopes:
        tune_scope(scope=scope, args=args, manager=manager)


if __name__ == "__main__":
    main()