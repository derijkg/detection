#!/usr/bin/env python3
"""
mDeBERTa-v3 Fine-Tuning Pipeline Optimized for RTX 2080 Ti (11GB VRAM)

Features:
- Optuna Objective Target: Maximize TPR @ 5% FPR (Low False Positive Operating Point)
- Automatic Hyperparameter Extrapolation from 5k Subsample -> 400k+ Full Dataset
- FP16 Mixed Precision & CUDA Memory Allocator Optimization
- VRAM OOM Protection (Gradient Checkpointing & Managed Eval Batches)
"""

import os
import sys

# RTX 2080 Ti CUDA Allocator Memory Optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score
)

import optuna
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)

# Enable cuDNN autotuner for Turing Architecture
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Setup Project Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.data.data_loader import DetectionDataManager, DataFilter
except ImportError:
    from data_loader import DetectionDataManager, DataFilter


# ---------------------------------------------------------
# 1. METRICS AT OPERATING POINT (TPR @ 5% FPR)
# ---------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()

    # Calculate ROC Curve
    fpr, tpr, _ = roc_curve(labels, probs)

    # 1. Calculate TPR at exactly 1% FPR (Operating Point)
    idx_5fpr = np.where(fpr <= 0.01)[0]
    tpr_at_5fpr = float(tpr[idx_5fpr[-1]]) if len(idx_5fpr) > 0 else 0.0

    # 2. Calculate Partial AUC for FPR in [0.0, 0.05]
    try:
        pauc_1fpr = float(roc_auc_score(labels, probs, max_fpr=0.01))
    except ValueError:
        pauc_1fpr = 0.5

    # 3. Overall ROC-AUC
    try:
        overall_auc = float(roc_auc_score(labels, probs))
    except ValueError:
        overall_auc = 0.5

    acc = float(accuracy_score(labels, preds))
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )

    return {
        'tpr_at_5fpr': tpr_at_5fpr,  # Target metric for Optuna
        'pauc_1fpr': pauc_1fpr,
        'roc_auc': overall_auc,
        'accuracy': acc,
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall)
    }


# ---------------------------------------------------------
# 2. OPTUNA OBJECTIVE (MAXIMIZING TPR @ 1% FPR)
# ---------------------------------------------------------
def optuna_objective(trial, optuna_train_ds, optuna_val_ds, tokenizer, model_name, tune_dir, num_workers, scope):
    print(f"\n--- Starting Optuna Trial #{trial.number} ---")

    # Improved LR Ranges for mDeBERTa-v3
    if scope == "sentence":
        learning_rate = trial.suggest_float("learning_rate", 2.5e-5, 1e-4, log=True)
        num_train_epochs = trial.suggest_int("num_train_epochs", 4, 6)
        per_device_train_batch_size = 8
        gradient_accumulation_steps = trial.suggest_categorical("gradient_accumulation_steps", [2, 4])
    else:  # full abstract
        learning_rate = trial.suggest_float("learning_rate", 1.5e-5, 6e-5, log=True)
        num_train_epochs = trial.suggest_int("num_train_epochs", 3, 5)
        per_device_train_batch_size = 8
        gradient_accumulation_steps = trial.suggest_categorical("gradient_accumulation_steps", [2, 4])

    weight_decay = trial.suggest_float("weight_decay", 1e-2, 1e-1, log=True)
    label_smoothing_factor = trial.suggest_float("label_smoothing_factor", 0.0, 0.08)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.05, 0.15)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "Human", 1: "LLM"},
        label2id={"Human": 0, "LLM": 1},
        use_safetensors=True
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(tune_dir, f"trial_{trial.number}"),
        eval_strategy="epoch",
        save_strategy="no",
        save_total_limit=1,
        learning_rate=learning_rate,
        
        # --- VRAM OOM PREVENTION SAFEGUARDS ---
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=16,
        eval_accumulation_steps=10,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # --------------------------------------

        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        label_smoothing_factor=label_smoothing_factor,
        load_best_model_at_end=True,
        metric_for_best_model="tpr_at_5fpr",
        greater_is_better=True,
        fp16=True,
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=True,
        logging_steps=20,
        report_to="none",
        disable_tqdm=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=optuna_train_ds,
        eval_dataset=optuna_val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()
    eval_metrics = trainer.evaluate()
    target_score = eval_metrics["eval_tpr_at_5fpr"]

    print(f"--> [Trial #{trial.number} Finished] Validation TPR @ 1% FPR: {target_score:.4f}")

    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return target_score


# ---------------------------------------------------------
# 3. EXTRAPOLATE HYPERPARAMETERS (5k -> 400k)
# ---------------------------------------------------------
def extrapolate_params_for_full_dataset(best_params, full_train_size, scope):
    """
    Extrapolates hyperparameters tuned on a small subsample to the full 400k dataset.
    """
    full_params = best_params.copy()

    # Rule 1: Scale down epochs for large dataset (1 to 2 epochs max)
    if full_train_size > 50000:
        full_params["num_train_epochs"] = 2 if scope == "sentence" else 1
        print(f"\n[EXTRAPOLATION] Dataset size is large ({full_train_size:,} samples).")
        print(f"  - Scaled training epochs: {best_params.get('num_train_epochs')} (subsample) -> {full_params['num_train_epochs']} (full dataset)")

    # Rule 2: Ensure hardware-safe batch size for RTX 2080 Ti
    full_params["per_device_train_batch_size"] = 8
    full_params["gradient_accumulation_steps"] = best_params.get("gradient_accumulation_steps", 2)

    return full_params


# ---------------------------------------------------------
# 4. FAST INFERENCE & REPORTING
# ---------------------------------------------------------
def run_fast_inference(model, test_ds, tokenizer, batch_size=32, device="cuda"):
    model.to(device)
    model.eval()

    dataloader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=True
    )

    all_logits = []
    with torch.inference_mode():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
            outputs = model(**inputs)
            all_logits.append(outputs.logits.cpu())

    logits_tensor = torch.cat(all_logits, dim=0)
    probs_tensor = F.softmax(logits_tensor, dim=-1)

    return logits_tensor.numpy(), probs_tensor[:, 1].numpy()


def evaluate_paper_results(df_test, probs_llm, preds, save_dir="./outputs"):
    os.makedirs(save_dir, exist_ok=True)
    labels = df_test['label'].values

    fpr, tpr, _ = roc_curve(labels, probs_llm)
    roc_auc_val = auc(fpr, tpr)
    precision_curve, recall_curve, _ = precision_recall_curve(labels, probs_llm)
    pr_auc_val = average_precision_score(labels, probs_llm)

    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)

    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    tpr_at_1fpr = tpr[np.where(fpr <= 0.01)[0][-1]] if len(np.where(fpr <= 0.01)[0]) > 0 else 0.0
    tpr_at_5fpr = tpr[np.where(fpr <= 0.05)[0][-1]] if len(np.where(fpr <= 0.05)[0]) > 0 else 0.0

    overall_metrics = {
        "Total Test Samples": len(labels),
        "ROC-AUC": round(float(roc_auc_val), 4),
        "PR-AUC (AP)": round(float(pr_auc_val), 4),
        "Accuracy": round(float(acc), 4),
        "F1-Score": round(float(f1), 4),
        "Precision": round(float(prec), 4),
        "Recall (Sensitivity)": round(float(rec), 4),
        "Specificity": round(float(specificity), 4),
        "TPR @ 1% FPR": round(float(tpr_at_1fpr), 4),
        "TPR @ 5% FPR": round(float(tpr_at_5fpr), 4)
    }

    print("\n" + "=" * 70)
    print(" FINAL TEST SET EVALUATION REPORT ")
    print("=" * 70)
    for k, v in overall_metrics.items():
        print(f"  {k:<22}: {v}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    axes[0].plot(fpr, tpr, color='#2b5c8f', lw=2, label=f'mDeBERTa-v3 (TPR@5%FPR = {tpr_at_5fpr:.4f})')
    axes[0].axvline(x=0.05, color='red', linestyle=':', label='Operating Point (5% FPR)')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.5)

    axes[1].hist(probs_llm[labels == 0], bins=25, alpha=0.6, color='#1b9e77', label='Human')
    axes[1].hist(probs_llm[labels == 1], bins=25, alpha=0.6, color='#7570b3', label='LLM')
    axes[1].set_xlabel('Predicted Probability P(LLM)')
    axes[1].set_title('Output Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "paper_evaluation_plots.png"), dpi=300)
    plt.close()

    with open(os.path.join(save_dir, "paper_evaluation_summary.json"), "w") as f:
        json.dump(overall_metrics, f, indent=4)

    return overall_metrics


# ---------------------------------------------------------
# 5. MAIN PIPELINE
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="mDeBERTa-v3 Pipeline Maximizing TPR @ 5% FPR")

    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--scope", type=str, choices=["sentence", "full"], default="sentence")
    parser.add_argument("--train_sample_size", type=int, default=-1)
    parser.add_argument("--val_sample_size", type=int, default=-1)
    parser.add_argument("--test_sample_size", type=int, default=-1)
    parser.add_argument("--optuna_sample_size", type=int, default=100000)
    parser.add_argument("--max_length", type=int, default=256)

    parser.add_argument("--n_trials", type=int, default=8)
    parser.add_argument("--skip_tuning", action="store_true")
    parser.add_argument("--model_name", type=str, default="microsoft/mdeberta-v3-base")

    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--tune_dir", type=str, default="./optuna_trials")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.tune_dir, exist_ok=True)

    num_workers = min(4, os.cpu_count() or 1)

    print("\n" + "=" * 70)
    print(f" TARGET GPU: NVIDIA GeForce RTX 2080 Ti | SCOPE: {args.scope.upper()}")
    print(" METRIC TARGET: Maximize TPR @ 5% False Positive Rate (FPR)")
    print("=" * 70)

    # 1. Load Datasets
    manager = DetectionDataManager(data_path=args.data_path)

    train_df = manager.filter_dataframe(scopes=[args.scope], splits=['train'], sample_size=args.train_sample_size, seed=args.seed)
    val_df = manager.filter_dataframe(scopes=[args.scope], splits=['val'], sample_size=args.val_sample_size, seed=args.seed)
    test_df = manager.filter_dataframe(scopes=[args.scope], splits=['test'], sample_size=args.test_sample_size, seed=args.seed)

    print(f"\nLoaded Datasets:")
    print(f"  - Train : {len(train_df):,} rows")
    print(f"  - Val   : {len(val_df):,} rows")
    print(f"  - Test  : {len(test_df):,} rows")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)

    def tokenize_fn(examples):
        return tokenizer(examples['text'], truncation=True, max_length=args.max_length)

    train_ds = Dataset.from_pandas(train_df, preserve_index=False).map(tokenize_fn, batched=True)
    val_ds = Dataset.from_pandas(val_df, preserve_index=False).map(tokenize_fn, batched=True)
    test_ds = Dataset.from_pandas(test_df, preserve_index=False).map(tokenize_fn, batched=True)

    params_file = os.path.join(args.output_dir, "best_hyperparameters.json")

    # 2. Optuna Tuning
    if not args.skip_tuning and args.n_trials > 0:
        target_per_class = args.optuna_sample_size // 2

        # 50/50 Class-Balanced Sampling for Optuna
        opt_train_h = manager.filter_dataframe(scopes=[args.scope], splits=['train'], labels=[0], sample_size=int(target_per_class * 0.8), seed=args.seed)
        opt_train_l = manager.filter_dataframe(scopes=[args.scope], splits=['train'], labels=[1], sample_size=int(target_per_class * 0.8), seed=args.seed)
        opt_train_df = pd.concat([opt_train_h, opt_train_l]).sample(frac=1, random_state=args.seed).reset_index(drop=True)

        opt_val_h = manager.filter_dataframe(scopes=[args.scope], splits=['val'], labels=[0], sample_size=int(target_per_class * 0.2), seed=args.seed)
        opt_val_l = manager.filter_dataframe(scopes=[args.scope], splits=['val'], labels=[1], sample_size=int(target_per_class * 0.2), seed=args.seed)
        opt_val_df = pd.concat([opt_val_h, opt_val_l]).sample(frac=1, random_state=args.seed).reset_index(drop=True)

        opt_train_ds = Dataset.from_pandas(opt_train_df, preserve_index=False).map(tokenize_fn, batched=True)
        opt_val_ds = Dataset.from_pandas(opt_val_df, preserve_index=False).map(tokenize_fn, batched=True)

        print(f"\n[OPTUNA] Subsampled {len(opt_train_df)} train & {len(opt_val_df)} val balanced sentences for tuning.")

        study = optuna.create_study(direction="maximize", study_name="mdeberta_tpr5fpr_tuning")
        study.optimize(
            lambda trial: optuna_objective(trial, opt_train_ds, opt_val_ds, tokenizer, args.model_name, args.tune_dir, num_workers, args.scope),
            n_trials=args.n_trials
        )

        best_params = study.best_trial.params
        best_params["per_device_train_batch_size"] = 8

        with open(params_file, "w") as f:
            json.dump({"best_val_tpr_at_5fpr": float(study.best_value), "params": best_params}, f, indent=4)
        print(f"\n[PARAMS SAVED] Best Optuna parameters saved to '{params_file}'")

    else:
        if os.path.exists(params_file):
            print(f"\n[PARAMS LOADED] Loading saved hyperparameters from '{params_file}'")
            with open(params_file, "r") as f:
                data = json.load(f)
                best_params = data.get("params", data)
        else:
            best_params = {
                "num_train_epochs": 2,
                "learning_rate": 3.5e-5,
                "per_device_train_batch_size": 8,
                "gradient_accumulation_steps": 2,
                "weight_decay": 0.01,
                "label_smoothing_factor": 0.03,
                "warmup_ratio": 0.1
            }

    # 3. Extrapolate Parameters for Full Dataset Training
    final_params = extrapolate_params_for_full_dataset(best_params, len(train_df), args.scope)

    # 4. Final Production Training
    print("\n" + "=" * 70)
    print(" TRAINING PRODUCTION MODEL ON FULL DATASET ")
    print("=" * 70)

    model_save_path = os.path.join(args.output_dir, "best_mdeberta_detector")

    final_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "Human", 1: "LLM"},
        label2id={"Human": 0, "LLM": 1},
        use_safetensors=True
    )

    training_args = TrainingArguments(
        output_dir=model_save_path,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=final_params["learning_rate"],
        per_device_train_batch_size=final_params["per_device_train_batch_size"],
        gradient_accumulation_steps=final_params["gradient_accumulation_steps"],
        per_device_eval_batch_size=16,
        eval_accumulation_steps=10,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False},
        num_train_epochs=final_params["num_train_epochs"],
        weight_decay=final_params["weight_decay"],
        warmup_ratio=final_params["warmup_ratio"],
        label_smoothing_factor=final_params["label_smoothing_factor"],
        load_best_model_at_end=True,
        metric_for_best_model="tpr_at_5fpr",
        greater_is_better=True,
        fp16=True,
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=True,
        logging_steps=50,
        report_to="none"
    )

    trainer = Trainer(
        model=final_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    # 5. Inference & Results CSV Export
    print("\n[INFERENCE] Running evaluation on held-out test set...")
    logits, probs_llm = run_fast_inference(
        final_model, test_ds, tokenizer, batch_size=32
    )
    preds = np.argmax(logits, axis=-1)

    evaluate_paper_results(test_df, probs_llm, preds, save_dir=args.output_dir)

    output_df = test_df.copy()
    output_df['pred'] = preds
    output_df['prob_llm'] = probs_llm
    output_df['logit_human'] = logits[:, 0]
    output_df['logit_llm'] = logits[:, 1]
    output_df['logit_diff'] = logits[:, 1] - logits[:, 0]

    csv_out_path = os.path.join(args.output_dir, f"mdeberta_{args.scope}_predictions.csv")
    output_df.to_csv(csv_out_path, index=False)
    print(f"\n[COMPLETE] Prediction CSV exported to: '{csv_out_path}'")


if __name__ == "__main__":
    main()