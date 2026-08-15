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
import sqlite3
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.multiprocessing as mp

# Enable cuDNN benchmark for Turing (RTX 2080 Ti) hardware kernel auto-tuning
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# Set spawn start method safely to prevent CUDA worker deadlocks
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

# Attempt to import PEFT for LoRA tuning
try:
    from peft import LoraConfig, get_peft_model, TaskType
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

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


def debug_print(msg: str, enabled: bool = False):
    """Prints debug messages when --debug flag is set."""
    if enabled:
        print(f"[DEBUG] {msg}")


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling hard negative boundary examples
    and boosting performance on low-FPR metrics like pAUC@0.01.
    """
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = float(gamma)

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class CustomTrainer(Trainer):
    """
    Subclassed Hugging Face Trainer supporting custom Focal Loss
    and custom LLRD AdamW optimizer.
    """
    def __init__(self, *args, focal_gamma: float = 0.0, use_focal_loss: bool = False, custom_optimizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_gamma = focal_gamma
        self.use_focal_loss = use_focal_loss
        self.custom_optimizer = custom_optimizer
        if self.use_focal_loss:
            self.focal_loss_fn = FocalLoss(gamma=self.focal_gamma)

    def create_optimizer(self):
        if self.custom_optimizer is not None:
            self.optimizer = self.custom_optimizer
            return self.optimizer
        return super().create_optimizer()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits") if isinstance(outputs, dict) else outputs[1]

        if self.use_focal_loss and labels is not None:
            loss = self.focal_loss_fn(logits, labels)
        else:
            loss = outputs.get("loss") if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


class OptunaPruningCallback(TrainerCallback):
    """
    Custom TrainerCallback that reports intermediate validation metrics to Optuna
    at exact step intervals and prunes unpromising trials early.
    """
    def __init__(self, trial: optuna.Trial, monitor: str = "eval_pauc_001", fallback_monitor: str = "eval_roc_auc"):
        self.trial = trial
        self.monitor = monitor
        self.fallback_monitor = fallback_monitor

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            # Fall back to roc_auc if pauc_001 is missing or zero/baseline early on
            val = metrics.get(self.monitor)
            if val is None or val <= 0.5:
                val = metrics.get(self.fallback_monitor, 0.5)

            step = state.global_step
            self.trial.report(val, step=step)
            if self.trial.should_prune():
                raise optuna.TrialPruned(f"Trial pruned at step {step} with value={val:.6f}")


def ensure_length_column(dataset):
    if "length" not in dataset.column_names:
        lengths = [len(x) for x in dataset["input_ids"]]
        dataset = dataset.add_column("length", lengths)
    return dataset


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]

    # Numerically stable Sigmoid probabilities
    probs = 1.0 / (1.0 + np.exp(-(logits[:, 1] - logits[:, 0])))
    preds = np.argmax(logits, axis=-1)

    acc = float(accuracy_score(labels, preds))
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)

    try:
        roc_auc = float(roc_auc_score(labels, probs))
    except ValueError:
        roc_auc = 0.5

    # Safe pAUC Calculation (prevents ValueError when negative sample size is small)
    num_negatives = np.sum(labels == 0)
    if num_negatives > 0:
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
    Targeted Precision Check for NVIDIA Turing Architecture (RTX 2080 Ti):
    - FP16 Tensor Cores: Fully Supported
    - BF16: Not Supported natively on SM 7.5
    """
    use_fp16 = False
    use_bf16 = False
    
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        if capability[0] >= 8 and torch.cuda.is_bf16_supported():
            use_bf16 = True
        elif capability[0] >= 7:
            use_fp16 = True

    return use_fp16, use_bf16


def create_llrd_optimizer(model, base_lr: float, weight_decay: float, llrd_decay: float, is_debug: bool = False):
    """
    Constructs an AdamW optimizer with Layer-wise Learning Rate Decay (LLRD).
    Decays learning rate exponentially from top classifier layers down to bottom embeddings.
    """
    if hasattr(model, "deberta"):
        backbone = model.deberta
    elif hasattr(model, "base_model"):
        backbone = model.base_model
    else:
        backbone = getattr(model, model.base_model_prefix, model)

    encoder_layers = list(backbone.encoder.layer) if hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer") else []
    num_layers = len(encoder_layers)
    
    debug_print(f"Creating LLRD Optimizer | Base LR={base_lr:.2e} | LLRD Decay={llrd_decay:.4f} | Encoder Layers={num_layers}", is_debug)

    optimizer_grouped_parameters = []
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]

    # 1. Classifier Head (Gets full base_lr)
    head_decay, head_no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(h in name for h in ["classifier", "pooler", "score"]):
            if any(nd in name for nd in no_decay):
                head_no_decay.append(param)
            else:
                head_decay.append(param)

    optimizer_grouped_parameters.extend([
        {"params": head_decay, "weight_decay": weight_decay, "lr": base_lr, "name": "head_decay"},
        {"params": head_no_decay, "weight_decay": 0.0, "lr": base_lr, "name": "head_no_decay"},
    ])

    # 2. Encoder Layers (Decay geometrically downwards)
    for layer_idx in range(num_layers - 1, -1, -1):
        layer = encoder_layers[layer_idx]
        depth = (num_layers - 1 - layer_idx) + 1
        layer_lr = base_lr * (llrd_decay ** depth)
        
        p_decay, p_no_decay = [], []
        for name, param in layer.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in no_decay):
                p_no_decay.append(param)
            else:
                p_decay.append(param)

        optimizer_grouped_parameters.extend([
            {"params": p_decay, "weight_decay": weight_decay, "lr": layer_lr, "name": f"layer_{layer_idx}_decay"},
            {"params": p_no_decay, "weight_decay": 0.0, "lr": layer_lr, "name": f"layer_{layer_idx}_no_decay"},
        ])

    # 3. Embeddings Layer
    if hasattr(backbone, "embeddings"):
        emb_lr = base_lr * (llrd_decay ** (num_layers + 1))
        emb_decay, emb_no_decay = [], []
        for name, param in backbone.embeddings.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in no_decay):
                emb_no_decay.append(param)
            else:
                emb_decay.append(param)

        optimizer_grouped_parameters.extend([
            {"params": emb_decay, "weight_decay": weight_decay, "lr": emb_lr, "name": "embeddings_decay"},
            {"params": emb_no_decay, "weight_decay": 0.0, "lr": emb_lr, "name": "embeddings_no_decay"},
        ])

    filtered_groups = [g for g in optimizer_grouped_parameters if len(g["params"]) > 0]
    
    if is_debug:
        for g in filtered_groups:
            debug_print(f"  Group '{g.get('name')}': {len(g['params'])} tensors, lr={g['lr']:.3e}", True)

    return torch.optim.AdamW(filtered_groups)


def optuna_objective(
    trial,
    train_ds,
    dev_ds,
    tokenizer,
    model_name,
    scope_dir,
    max_length,
    args,
):
    print(f"\n" + "=" * 70)
    print(f" Starting Optuna Trial #{trial.number} ")
    print("=" * 70)

    # Define trial_output_dir at top to avoid unbound scope errors in finally block
    trial_output_dir = scope_dir / "optuna_trials" / f"trial_{trial.number}"

    debug_print("Clearing CUDA memory cache...", args.debug)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

    num_train_epochs = trial.suggest_int("num_train_epochs", 2, 4)
    
    # Dynamic LR range based on LoRA vs Full Fine-Tuning
    if args.enable_lora:
        learning_rate = trial.suggest_float("learning_rate", 5e-5, 1e-3, log=True)
    else:
        learning_rate = trial.suggest_float("learning_rate", 5e-6, 1e-4, log=True)
    
    # Optimized batch size for 11GB VRAM with gradient checkpointing
    per_device_train_batch_size = 16 if max_length <= 128 else 8
    effective_batch_size = trial.suggest_categorical("effective_batch_size", [16, 32])
    gradient_accumulation_steps = max(1, effective_batch_size // per_device_train_batch_size)

    weight_decay = trial.suggest_float("weight_decay", 1e-2, 1e-1, log=True)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.05, 0.2)
    label_smoothing_factor = trial.suggest_float("label_smoothing_factor", 0.0, 0.1)
    lr_scheduler_type = trial.suggest_categorical("lr_scheduler_type", ["linear", "cosine"])

    # --- LLRD Multiplicative Decay Tuning ---
    use_llrd = args.enable_llrd
    if use_llrd:
        one_minus_decay = trial.suggest_float("one_minus_llrd_decay", 0.05, 0.20, log=True)
        llrd_decay = 1.0 - one_minus_decay
        debug_print(f"Suggested LLRD Decay Rate: {llrd_decay:.4f} (1 - decay = {one_minus_decay:.4f})", args.debug)
    else:
        llrd_decay = 1.0

    # --- Focal Loss Tuning ---
    use_focal_loss = args.enable_focal_loss
    if use_focal_loss:
        focal_gamma = trial.suggest_float("focal_gamma", 0.5, 3.0)
        debug_print(f"Suggested Focal Loss Gamma: {focal_gamma:.3f}", args.debug)
    else:
        focal_gamma = 0.0

    use_fp16, use_bf16 = get_hardware_precision()

    debug_print(f"Loading Base Checkpoint: {model_name}", args.debug)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "Human", 1: "LLM"},
        label2id={"Human": 0, "LLM": 1},
        use_safetensors=True,
    )
    model.config.use_cache = False

    # --- LoRA PEFT Tuning ---
    if args.enable_lora:
        if not HAS_PEFT:
            raise RuntimeError("PEFT package not installed. Run 'pip install peft' to enable LoRA tuning.")
        
        lora_r = trial.suggest_categorical("lora_r", [8, 16, 32])
        lora_alpha = trial.suggest_categorical("lora_alpha", [16, 32, 64])
        lora_dropout = trial.suggest_float("lora_dropout", 0.05, 0.2)

        debug_print(f"Applying LoRA: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout:.3f}", args.debug)
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query_proj", "key_proj", "value_proj", "dense"],
        )
        model = get_peft_model(model, peft_config)
        if args.debug:
            trainable_params, all_params = model.get_nb_trainable_parameters()
            debug_print(f"LoRA Params: Trainable={trainable_params:,} | Total={all_params:,} ({100 * trainable_params / all_params:.2f}%)", True)

    # Build LLRD Optimizer
    custom_opt = create_llrd_optimizer(model, learning_rate, weight_decay, llrd_decay, is_debug=args.debug) if use_llrd else None

    # Scaled Eval Batch Size for 2080 Ti
    eval_batch_size = 32 if max_length <= 128 else 16
    debug_print(f"Configured Eval Batch Size: {eval_batch_size}", args.debug)

    training_args = TrainingArguments(
        output_dir=str(trial_output_dir),
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="no",              # FAST: Disable disk checkpointing during tuning
        load_best_model_at_end=False,    # FAST: Rely on Optuna for metric tracking
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        label_smoothing_factor=label_smoothing_factor if not use_focal_loss else 0.0,
        metric_for_best_model="pauc_001",
        greater_is_better=True,
        fp16=use_fp16,
        bf16=use_bf16,
        fp16_full_eval=False,             # Prevents FP16 eval overflow/underflow issues on DeBERTa
        gradient_checkpointing=True,     # Slashes activation VRAM usage by ~60%
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,
        length_column_name="length",
        dataloader_pin_memory=True,
        dataloader_num_workers=4,        # Boosted prefetch threads for Linux CPU->GPU pipeline
        dataloader_persistent_workers=True, # Keeps worker threads alive across evaluation steps
        eval_accumulation_steps=None,    # Keep predictions on GPU for maximum eval speed
        logging_steps=25,
        report_to="none",
        disable_tqdm=False,
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8 if (use_fp16 or use_bf16) else None),
        compute_metrics=compute_metrics,
        focal_gamma=focal_gamma,
        use_focal_loss=use_focal_loss,
        custom_optimizer=custom_opt,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            ),
            OptunaPruningCallback(trial=trial, monitor="eval_pauc_001"),
        ],
    )

    target_metric = None

    try:
        debug_print("Starting Trainer.train()...", args.debug)
        trainer.train()
        eval_metrics = trainer.evaluate()
        target_metric = eval_metrics["eval_pauc_001"]

        # Save model state_dict for Greedy Model Soup
        saved_weights_dir = scope_dir / "saved_trial_weights"
        saved_weights_dir.mkdir(parents=True, exist_ok=True)
        weight_file = saved_weights_dir / f"trial_{trial.number}.pt"
        
        debug_print(f"Saving completed trial weights to: '{weight_file}'", args.debug)
        
        # If LoRA was used, merge weights back to base model structure before saving
        if args.enable_lora and HAS_PEFT and hasattr(model, "merge_and_unload"):
            save_model = model.merge_and_unload()
            torch.save(save_model.state_dict(), weight_file)
        else:
            torch.save(model.state_dict(), weight_file)

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

    except Exception as e:
        print(f"\n[ERROR] Trial #{trial.number} failed with exception: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        target_metric = None

    finally:
        debug_print("Cleaning up PyTorch/CUDA objects for trial...", args.debug)
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
            torch.cuda.ipc_collect()

        if 'trial_output_dir' in locals() and trial_output_dir.exists():
            shutil.rmtree(trial_output_dir, ignore_errors=True)

    if target_metric is None:
        raise optuna.TrialPruned()

    return target_metric


def run_greedy_model_soup(study, scope_dir, dev_ds, tokenizer, model_name, scope: str, args):
    """
    Performs Greedy Model Soup across the top completed Optuna trials.
    Iteratively averages candidate trial state dicts to maximize validation pAUC@0.01.
    Reuses model and Trainer in memory to eliminate re-initialization overhead.
    """
    print("\n" + "=" * 60)
    print(f" RUNNING GREEDY MODEL SOUP EVALUATION [{scope.upper()}] ")
    print("=" * 60)

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed_trials) < 2:
        print("[SOUP WARNING] Not enough completed trials to perform Greedy Model Soup (need >= 2).")
        return

    completed_trials.sort(key=lambda t: t.value, reverse=True)
    weights_dir = scope_dir / "saved_trial_weights"
    
    debug_print(f"Found {len(completed_trials)} completed trials for soup evaluation.", args.debug)

    def load_base_model():
        m = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            id2label={0: "Human", 1: "LLM"},
            label2id={"Human": 0, "LLM": 1},
            use_safetensors=True,
        )
        m.config.use_cache = False
        return m

    eval_model = load_base_model()
    use_fp16, use_bf16 = get_hardware_precision()
    eval_args = TrainingArguments(
        output_dir=str(scope_dir / "soup_tmp"),
        per_device_eval_batch_size=32 if "sentence" in scope else 16,
        fp16=use_fp16,
        bf16=use_bf16,
        dataloader_num_workers=2,
        dataloader_persistent_workers=False,
        report_to="none",
    )
    eval_trainer = Trainer(
        model=eval_model,
        args=eval_args,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    def evaluate_state_dict(state_dict):
        eval_model.load_state_dict(state_dict, strict=False)
        metrics = eval_trainer.evaluate()
        return metrics["eval_pauc_001"]

    best_trial = completed_trials[0]
    best_weights_path = weights_dir / f"trial_{best_trial.number}.pt"
    if not best_weights_path.exists():
        print(f"[SOUP ERROR] Weights file for best Trial #{best_trial.number} not found.")
        return

    debug_print(f"Loading baseline weights from Trial #{best_trial.number}...", args.debug)
    soup_state_dict = torch.load(best_weights_path, map_location="cpu")
    best_soup_score = evaluate_state_dict(soup_state_dict)
    print(f"[SOUP BASELINE] Trial #{best_trial.number} Validation pAUC@0.01: {best_soup_score:.6f}")

    soup_trials_included = [best_trial.number]

    # Evaluate up to top 5 candidates
    for candidate_trial in completed_trials[1:5]:
        cand_weights_path = weights_dir / f"trial_{candidate_trial.number}.pt"
        if not cand_weights_path.exists():
            continue

        debug_print(f"Evaluating candidate Trial #{candidate_trial.number} (pAUC={candidate_trial.value:.6f})...", args.debug)
        cand_state_dict = torch.load(cand_weights_path, map_location="cpu")
        
        # Calculate averaged state dict
        num_models = len(soup_trials_included) + 1
        candidate_soup_dict = {}
        for k in soup_state_dict.keys():
            if k in cand_state_dict and soup_state_dict[k].dtype.is_floating_point:
                candidate_soup_dict[k] = (soup_state_dict[k] * (num_models - 1) + cand_state_dict[k]) / float(num_models)
            else:
                candidate_soup_dict[k] = soup_state_dict[k]

        cand_score = evaluate_state_dict(candidate_soup_dict)
        debug_print(f"  Candidate Soup Score: {cand_score:.6f} vs Current Best: {best_soup_score:.6f}", args.debug)

        if cand_score > best_soup_score:
            print(f"🎉 SOUP IMPROVEMENT! pAUC increased from {best_soup_score:.6f} to {cand_score:.6f} (Included Trial #{candidate_trial.number})")
            best_soup_score = cand_score
            soup_state_dict = candidate_soup_dict
            soup_trials_included.append(candidate_trial.number)
        else:
            debug_print(f"  Candidate Trial #{candidate_trial.number} rejected.", args.debug)

    soup_save_path = scope_dir / "greedy_model_soup.pt"
    torch.save(soup_state_dict, soup_save_path)
    
    print("\n" + "=" * 60)
    print(f"GREEDY MODEL SOUP COMPLETE [{scope.upper()}]")
    print(f"Final Soup Validation pAUC@0.01 : {best_soup_score:.6f}")
    print(f"Trials Included                 : {soup_trials_included}")
    print(f"Saved Soup Weights              : '{soup_save_path}'")
    print("=" * 60 + "\n")

    del eval_model, eval_trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    shutil.rmtree(scope_dir / "soup_tmp", ignore_errors=True)


def tune_scope(scope: str, args, manager: DetectionDataManager):
    base_outputs_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUTS_DIR
    scope_dir = base_outputs_dir / "mdeberta" / scope
    scope_dir.mkdir(parents=True, exist_ok=True)

    db_path = scope_dir / "optuna_study.db"
    storage_url = f"sqlite:///{db_path}"

    if args.reset_study:
        print(f"\n[RESET STUDY] Cleaning up existing Optuna database and trial artifacts for scope '{scope}'...")
        for ext in ["", "-wal", "-shm", "-journal"]:
            f = Path(str(db_path) + ext)
            if f.exists():
                try:
                    f.unlink()
                except Exception as e:
                    print(f"Warning: Could not remove {f}: {e}")

        for sub_dir in ["optuna_trials", "saved_trial_weights"]:
            d = scope_dir / sub_dir
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)

        for json_f in ["best_hyperparameters.json", "greedy_model_soup.pt"]:
            f = scope_dir / json_f
            if f.exists():
                try:
                    f.unlink()
                except Exception:
                    pass

    # Enable SQLite WAL mode to prevent database locks
    try:
        conn = sqlite3.connect(db_path, timeout=60.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.close()
    except Exception as e:
        debug_print(f"SQLite WAL mode initialization notice: {e}", args.debug)

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
    print(f" Enable LLRD Decay      : {args.enable_llrd}")
    print(f" Enable Focal Loss      : {args.enable_focal_loss}")
    print(f" Enable LoRA Search     : {args.enable_lora}")
    print(f" Enable Greedy Soup     : {args.enable_greedy_soup}")
    print(f" Debug Logging          : {args.debug}")
    print("=" * 70 + "\n")

    train_df = manager.filter_dataframe(DataFilter(splits=["train"], scopes=[scope]), sample_size=train_sample_size, seed=args.seed)
    dev_df = manager.filter_dataframe(DataFilter(splits=["dev"], scopes=[scope]), sample_size=dev_sample_size, seed=args.seed)

    debug_print(f"Loaded DataFrames -> Train: {len(train_df)} | Dev: {len(dev_df)}", args.debug)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)

    train_ds = manager.get_tokenized_dataset(
        scope=scope, split="train", tokenizer=tokenizer, max_length=max_length, return_format="torch"
    )
    dev_ds = manager.get_tokenized_dataset(
        scope=scope, split="dev", tokenizer=tokenizer, max_length=max_length, return_format="torch"
    )

    # Fast Dataset Selection
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

    train_ds = ensure_length_column(train_ds)
    dev_ds = ensure_length_column(dev_ds)
    dev_ds = dev_ds.sort("length")

    warmup_steps = args.eval_steps * getattr(args, "pruner_warmup_evals", 3)

    # Use PercentilePruner to prune only bottom 25% after 5 baseline trials
    pruner = optuna.pruners.PercentilePruner(
        percentile=25.0,
        n_startup_trials=getattr(args, "pruner_startup_trials", 5),
        n_warmup_steps=warmup_steps,
        interval_steps=args.eval_steps,
    )
    sampler = optuna.samplers.TPESampler(multivariate=True, group=True, seed=args.seed)

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
            max_length=max_length,
            args=args,
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

    # Run Greedy Model Soup if Enabled
    if args.enable_greedy_soup:
        run_greedy_model_soup(study, scope_dir, dev_ds, tokenizer, args.model_name, scope, args)


def main():
    parser = argparse.ArgumentParser(description="Tune mDeBERTa-v3 hyperparameters with LLRD, Focal Loss, LoRA, and Model Soup.")
    parser.add_argument("--pruner_warmup_evals", type=int, default=3, help="Number of initial evaluation steps to skip before pruning.")
    parser.add_argument("--pruner_startup_trials", type=int, default=5, help="Number of initial completed trials before pruning starts.")
    parser.add_argument(
        "--scopes",
        nargs="+",
        default=["full", "sentence"],
        choices=["full", "sentence"],
        help="List of scopes to tune (default: full sentence)."
    )
    parser.add_argument("--model_name", type=str, default="microsoft/mdeberta-v3-base", help="Hugging Face model checkpoint.")
    
    # --- Advanced ML Flags ---
    parser.add_argument("--enable_llrd", action="store_true", default=True, help="Enable Layer-wise Learning Rate Decay (LLRD).")
    parser.add_argument("--enable_focal_loss", action="store_true", default=True, help="Enable custom Focal Loss tuning.")
    parser.add_argument("--enable_lora", action="store_true", default=False, help="Enable PEFT / LoRA hyperparameter tuning.")
    parser.add_argument("--enable_greedy_soup", action="store_true", default=True, help="Enable Greedy Model Soup after Optuna search.")
    parser.add_argument("--debug", action="store_true", help="Print detailed debug information during trial execution.")

    # --- Global Fallbacks ---
    parser.add_argument("--sample_size", type=int, default=-1, help="Global sample size fallback (-1 for full dataset).")
    parser.add_argument("--train_sample_size", type=int, default=-1, help="Global train sample size fallback.")
    parser.add_argument("--dev_sample_size", type=int, default=-1, help="Global dev sample size fallback.")
    parser.add_argument("--max_length", type=int, default=256, help="Global fallback max sequence length.")

    # --- Scope-Specific Sample Sizes ---
    parser.add_argument("--full_train_sample_size", type=int, default=5000, help="Train sample size for 'full' abstract scope.")
    parser.add_argument("--full_dev_sample_size", type=int, default=8000, help="Dev sample size for 'full' abstract scope (-1 for full dev set).")
    parser.add_argument("--sentence_train_sample_size", type=int, default=10000, help="Train sample size for 'sentence' scope.")
    parser.add_argument("--sentence_dev_sample_size", type=int, default=11000, help="Dev sample size for 'sentence' scope (-1 for full dev set).")

    # --- Scope-Specific Max Sequence Lengths ---
    parser.add_argument("--full_max_length", type=int, default=256, help="Max sequence length for 'full' scope.")
    parser.add_argument("--sentence_max_length", type=int, default=128, help="Max sequence length for 'sentence' scope.")

    # --- Evaluation & Pruning Frequency ---
    parser.add_argument("--eval_steps", type=int, default=100, help="Interval in steps between evaluations.")

    # --- Early Stopping Configuration ---
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Number of eval checks with no metric improvement.")
    parser.add_argument("--early_stopping_threshold", type=float, default=0.001, help="Minimum metric improvement required.")

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