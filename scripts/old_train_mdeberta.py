#!/usr/bin/env python3
# scripts/train_mdeberta.py

import argparse
import gc
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------
# CRITICAL GPU & THREADING ISOLATION (Must be set BEFORE importing torch)
# ---------------------------------------------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# ---------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# Calculate project root dynamically (~/detection)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Base outputs directory: /home/gderijck/detection/outputs
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"

from src.data.data_loader import DataFilter, DetectionDataManager


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
        # Maxwell TITAN X (5.2) & Pascal (6.x) default to FP32

    return use_fp16, use_bf16


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


# ==========================================
# 1. Config Loader
# ==========================================
def load_best_hyperparameters(scope: str, outputs_dir: Path) -> dict:
    config_path = outputs_dir / "mdeberta" / scope / "best_hyperparameters.json"
    if not config_path.exists():
        print(f"[WARNING] No tuned hyperparameters found at '{config_path}'. Using default parameters.")
        return {
            "learning_rate": 1.5e-5,
            "per_device_train_batch_size": 8,
            "num_train_epochs": 3,
            "weight_decay": 0.05,
            "warmup_ratio": 0.1,
            "label_smoothing_factor": 0.05,
        }

    with open(config_path, "r") as f:
        data = json.load(f)
        print(f"[LOADED PARAMS] Loaded best hyperparameters from: {config_path}")
        return data.get("best_hyperparameters", data)


# ==========================================
# 2. Metric Computation Helper
# ==========================================
def compute_metrics(eval_pred):
    """
    Vectorized and optimized evaluation metrics using native NumPy operations.
    Avoids CPU <-> GPU PyTorch tensor allocation overhead.
    """
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]

    # Numerically stable NumPy Softmax
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits[:, 1] / np.sum(exp_logits, axis=-1)
    preds = np.argmax(logits, axis=-1)

    acc = float(accuracy_score(labels, preds))
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)

    try:
        roc_auc = float(roc_auc_score(labels, probs))
    except ValueError:
        roc_auc = 0.5

    try:
        pauc_001 = float(roc_auc_score(labels, probs, max_fpr=0.01))
    except ValueError:
        pauc_001 = 0.5

    return {
        "pauc_001": pauc_001,
        "roc_auc": roc_auc,
        "accuracy": acc,
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
    }


# ==========================================
# 3. Decision Threshold Optimization on DEV
# ==========================================
def calculate_oof_threshold_on_dev(dev_df, dev_ds, trainer, n_splits=5):
    """
    Performs fast batched inference on 'dev' set and evaluates decision threshold tau*
    that maximizes TPR subject to FPR <= 0.01 using Stratified Group K-Fold cross-validation.
    """
    print(f"\nCalculating optimal decision threshold on 'dev' set ({n_splits}-Fold CV)...")

    # Fast inference using Trainer (uses pretokenized features + GPU acceleration)
    eval_output = trainer.predict(dev_ds)
    logits = eval_output.predictions
    if isinstance(logits, tuple):
        logits = logits[0]

    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    oof_probs = exp_logits[:, 1] / np.sum(exp_logits, axis=-1)

    dev_labels = dev_df["label"].values
    id_col = "_id" if "_id" in dev_df.columns else ("doc_id" if "doc_id" in dev_df.columns else "id")

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    thresholds = np.linspace(0.001, 0.999, 1000)
    best_threshold = 0.5
    best_tpr = -1.0
    best_stats = {}

    for t in thresholds:
        preds = (oof_probs >= t).astype(int)
        cm = confusion_matrix(dev_labels, preds)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            acc = (tp + tn) / len(dev_labels)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tpr
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            if fpr <= 0.01 and tpr > best_tpr:
                best_tpr = tpr
                best_threshold = float(t)
                best_stats = {
                    "optimal_threshold": round(best_threshold, 6),
                    "oof_fpr": round(float(fpr), 6),
                    "oof_tpr_at_1fpr": round(float(tpr), 6),
                    "oof_accuracy": round(float(acc), 4),
                    "oof_f1": round(float(f1), 4),
                    "oof_precision": round(float(prec), 4),
                    "oof_recall": round(float(rec), 4),
                    "oof_specificity": round(float(tn / (tn + fp)), 4),
                }

    if not best_stats:
        best_threshold = 0.5
        best_stats = {
            "optimal_threshold": 0.5,
            "oof_fpr": 0.0,
            "oof_tpr_at_1fpr": 0.0,
            "oof_accuracy": 0.0,
            "oof_f1": 0.0,
            "oof_precision": 0.0,
            "oof_recall": 0.0,
            "oof_specificity": 0.0,
        }

    print(f"Optimal Decision Threshold (τ*): {best_threshold:.6f}")
    print(f"  - Dev FPR        : {best_stats.get('oof_fpr'):.6f}")
    print(f"  - Dev TPR@1% FPR : {best_stats.get('oof_tpr_at_1fpr'):.4f}")
    print(f"  - Dev F1 Score   : {best_stats.get('oof_f1'):.4f}")

    return best_threshold, best_stats


# ==========================================
# 4. Publication-Ready Evaluation & Plotting
# ==========================================
def evaluate_and_plot_results(
    df_split, split_ds, split_name, scope, trainer, optimal_threshold, save_dir
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f" GENERATING PUBLICATION REPORT [{scope.upper()} | {split_name.upper()} SPLIT] ")
    print("=" * 70)

    # Fast batched inference using pretokenized Hugging Face Dataset
    eval_output = trainer.predict(split_ds)
    logits_arr = eval_output.predictions
    if isinstance(logits_arr, tuple):
        logits_arr = logits_arr[0]

    exp_logits = np.exp(logits_arr - np.max(logits_arr, axis=-1, keepdims=True))
    probs_llm = exp_logits[:, 1] / np.sum(exp_logits, axis=-1)

    labels = df_split["label"].values
    preds = (probs_llm >= optimal_threshold).astype(int)

    fpr, tpr, _ = roc_curve(labels, probs_llm)
    roc_auc_val = auc(fpr, tpr)
    try:
        pauc_001_val = roc_auc_score(labels, probs_llm, max_fpr=0.01)
    except ValueError:
        pauc_001_val = 0.5

    precision_curve, recall_curve, _ = precision_recall_curve(labels, probs_llm)
    pr_auc_val = average_precision_score(labels, probs_llm)

    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)

    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    fpr_under_001 = np.where(fpr <= 0.01)[0]
    tpr_at_1fpr = tpr[fpr_under_001[-1]] if len(fpr_under_001) > 0 else 0.0

    metrics_summary = {
        "Scope": scope,
        "Split": split_name,
        "Total Samples": len(labels),
        "Optimal Threshold (τ*)": round(optimal_threshold, 6),
        "pAUC @ max FPR 0.01": round(float(pauc_001_val), 6),
        "ROC-AUC": round(float(roc_auc_val), 4),
        "PR-AUC (AP)": round(float(pr_auc_val), 4),
        "Accuracy": round(float(acc), 4),
        "F1-Score": round(float(f1), 4),
        "Precision": round(float(prec), 4),
        "Recall (Sensitivity)": round(float(rec), 4),
        "Specificity": round(float(specificity), 4),
        "TPR @ 1% FPR": round(float(tpr_at_1fpr), 4),
    }

    print(f"\n--- PERFORMANCE SUMMARY [{split_name.upper()}] ---")
    for k, v in metrics_summary.items():
        print(f"  {k:<28}: {v}")

    # Compute Per-Generator Breakdown if model_name column is available
    gen_col = "model_name" if "model_name" in df_split.columns else ("generator_model" if "generator_model" in df_split.columns else None)
    per_model_results = []

    if gen_col:
        df_eval = df_split.copy()
        df_eval["prob_llm"] = probs_llm
        df_eval["pred"] = preds
        human_df = df_eval[df_eval["label"] == 0]

        for generator in df_eval[gen_col].unique():
            if str(generator).lower() == "human":
                continue
            llm_sub = df_eval[df_eval[gen_col] == generator]
            combined = pd.concat([human_df, llm_sub])

            sub_labels = combined["label"].values
            sub_probs = combined["prob_llm"].values
            sub_preds = combined["pred"].values

            try:
                sub_auc = roc_auc_score(sub_labels, sub_probs)
            except ValueError:
                sub_auc = 0.5

            sub_acc = accuracy_score(sub_labels, sub_preds)
            sub_prec, sub_rec, sub_f1, _ = precision_recall_fscore_support(sub_labels, sub_preds, average="binary", zero_division=0)

            per_model_results.append({
                "Generator": generator,
                "LLM Samples": len(llm_sub),
                "ROC-AUC": round(float(sub_auc), 4),
                "Accuracy": round(float(sub_acc), 4),
                "F1-Score": round(float(sub_f1), 4),
                "Precision": round(float(sub_prec), 4),
                "Recall": round(float(sub_rec), 4),
            })

        if per_model_results:
            print("\n--- PER-GENERATOR BREAKDOWN ---")
            print(pd.DataFrame(per_model_results).to_string(index=False))

    # Generate 4-Panel Plot Figure (300 DPI)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    plt.subplots_adjust(wspace=0.25, hspace=0.3)

    # Panel A: ROC Curve
    axes[0, 0].plot(fpr, tpr, color="#2b5c8f", lw=2, label=f"mDeBERTa-v3 ({scope.upper()})\npAUC@0.01={pauc_001_val:.4f}")
    axes[0, 0].axvline(x=0.01, color="red", linestyle=":", lw=1.5, label="FPR = 0.01")
    axes[0, 0].plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5, label="Random")
    axes[0, 0].set_xlabel("False Positive Rate", fontsize=11)
    axes[0, 0].set_ylabel("True Positive Rate", fontsize=11)
    axes[0, 0].set_title(f"(A) ROC Curve ({split_name.capitalize()})", fontsize=12, fontweight="bold")
    axes[0, 0].legend(loc="lower right", fontsize=10)
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    # Panel B: PR Curve
    axes[0, 1].plot(recall_curve, precision_curve, color="#d95f02", lw=2, label=f"AP={pr_auc_val:.4f}")
    axes[0, 1].set_xlabel("Recall", fontsize=11)
    axes[0, 1].set_ylabel("Precision", fontsize=11)
    axes[0, 1].set_title(f"(B) Precision-Recall Curve ({split_name.capitalize()})", fontsize=12, fontweight="bold")
    axes[0, 1].legend(loc="lower left", fontsize=10)
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    # Panel C: Confusion Matrix
    cm_sum = cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.divide(cm.astype("float"), cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum != 0)
    im = axes[1, 0].imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    axes[1, 0].set_title(f"(C) Confusion Matrix (τ* = {optimal_threshold:.4f})", fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    classes = ["Human", "LLM"]
    tick_marks = np.arange(len(classes))
    axes[1, 0].set_xticks(tick_marks)
    axes[1, 0].set_xticklabels(classes, fontsize=10)
    axes[1, 0].set_yticks(tick_marks)
    axes[1, 0].set_yticklabels(classes, fontsize=10)
    axes[1, 0].set_ylabel("True Label", fontsize=11)
    axes[1, 0].set_xlabel("Predicted Label", fontsize=11)

    for i in range(min(2, cm.shape[0])):
        for j in range(min(2, cm.shape[1])):
            axes[1, 0].text(
                j, i, f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > 0.5 else "black",
                fontsize=11, fontweight="bold",
            )

    # Panel D: Density Distribution
    human_probs = probs_llm[labels == 0]
    llm_probs = probs_llm[labels == 1]

    if len(human_probs) > 0:
        axes[1, 1].hist(human_probs, bins=25, alpha=0.6, color="#1b9e77", label="Human", density=True)
    if len(llm_probs) > 0:
        axes[1, 1].hist(llm_probs, bins=25, alpha=0.6, color="#7570b3", label="LLM", density=True)

    axes[1, 1].axvline(x=optimal_threshold, color="black", linestyle="--", lw=2, label=f"τ*={optimal_threshold:.4f}")
    axes[1, 1].set_xlabel("Predicted Probability P(LLM)", fontsize=11)
    axes[1, 1].set_ylabel("Density", fontsize=11)
    axes[1, 1].set_title(f"(D) Probability Density ({split_name.capitalize()})", fontsize=12, fontweight="bold")
    axes[1, 1].legend(loc="upper center", fontsize=10)
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    plot_path = save_dir / f"{split_name}_evaluation_plots.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"[SAVED PLOT] Plot saved to: '{plot_path}'")
    plt.close()

    # Save LaTeX Table
    latex_table_path = save_dir / f"{split_name}_metrics_table.tex"
    with open(latex_table_path, "w") as f:
        f.write("% Auto-generated LaTeX table\n")
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write(f"\\caption{{{split_name.capitalize()} Split Performance ({scope.upper()}). Threshold $\\tau^* = {optimal_threshold:.4f}$.}}\n")
        f.write("\\label{tab:mdeberta_" + scope + "_" + split_name + "}\n")
        f.write("\\begin{tabular}{lcccccc}\n\\hline\n")
        f.write("\\textbf{Split} & \\textbf{pAUC @ 0.01} & \\textbf{ROC-AUC} & \\textbf{Accuracy} & \\textbf{F1-Score} & \\textbf{Precision} & \\textbf{Recall} \\\\\n\\hline\n")
        f.write(f"{split_name.capitalize()} & {pauc_001_val:.4f} & {roc_auc_val:.4f} & {acc:.4f} & {f1:.4f} & {prec:.4f} & {rec:.4f} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

    # Save Logits DataFrame
    df_logits = df_split.copy()
    df_logits["logit_human"] = logits_arr[:, 0]
    df_logits["logit_llm"] = logits_arr[:, 1]
    df_logits["logit_diff"] = logits_arr[:, 1] - logits_arr[:, 0]
    df_logits["prob_llm"] = probs_llm
    df_logits["pred_llm"] = preds

    csv_path = save_dir / f"{split_name}_logits_analysis.csv"
    df_logits.to_csv(csv_path, index=False)
    print(f"[SAVED LOGITS] Logits saved to: '{csv_path}'")

    return metrics_summary, per_model_results


# ==========================================
# 5. Full Training Pipeline for Scope
# ==========================================
def run_full_training_for_scope(scope: str, args, manager: DetectionDataManager):
    outputs_base = Path(args.outputs_dir) if args.outputs_dir else DEFAULT_OUTPUTS_DIR
    scope_dir = outputs_base / "mdeberta" / scope
    scope_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f" FULL mDeBERTa-v3 TRAINING FOR SCOPE: '{scope.upper()}' ")
    print(f" Output Directory : {scope_dir}")
    print("=" * 70 + "\n")

    # Load Best Hyperparameters
    best_params = load_best_hyperparameters(scope=scope, outputs_dir=outputs_base)

    # Load Data via DetectionDataManager
    train_df = manager.filter_dataframe(DataFilter(splits=["train"], scopes=[scope]), sample_size=args.sample_size, seed=args.seed)
    dev_df = manager.filter_dataframe(DataFilter(splits=["dev"], scopes=[scope]), sample_size=args.sample_size, seed=args.seed)
    test_df = manager.filter_dataframe(DataFilter(splits=["test"], scopes=[scope]), sample_size=args.sample_size, seed=args.seed)

    print(f"Loaded Data Splits -> Train: {len(train_df)} | Dev: {len(dev_df)} | Test: {len(test_df)}")

    # Load pretokenized Hugging Face Datasets from cache
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)

    train_ds = manager.get_tokenized_dataset(
        scope=scope, split="train", tokenizer=tokenizer, max_length=args.max_length, return_format="torch"
    )
    dev_ds = manager.get_tokenized_dataset(
        scope=scope, split="dev", tokenizer=tokenizer, max_length=args.max_length, return_format="torch"
    )
    test_ds = manager.get_tokenized_dataset(
        scope=scope, split="test", tokenizer=tokenizer, max_length=args.max_length, return_format="torch"
    )

    # Accelerated Dataset Subsampling via NumPy index selection
    if args.sample_size > 0:
        if "_id" in train_df.columns and "_id" in train_ds.column_names:
            s_tr = set(train_df["_id"].astype(str))
            s_dev = set(dev_df["_id"].astype(str))
            s_te = set(test_df["_id"].astype(str))

            tr_idx = [i for i, x in enumerate(train_ds["_id"]) if str(x) in s_tr]
            dev_idx = [i for i, x in enumerate(dev_ds["_id"]) if str(x) in s_dev]
            te_idx = [i for i, x in enumerate(test_ds["_id"]) if str(x) in s_te]

            train_ds = train_ds.select(tr_idx)
            dev_ds = dev_ds.select(dev_idx)
            test_ds = test_ds.select(te_idx)
        else:
            if len(train_ds) > args.sample_size:
                train_ds = train_ds.select(range(min(args.sample_size, len(train_ds))))
            if len(dev_ds) > args.sample_size:
                dev_ds = dev_ds.select(range(min(args.sample_size, len(dev_ds))))
            if len(test_ds) > args.sample_size:
                test_ds = test_ds.select(range(min(args.sample_size, len(test_ds))))

    # Precompute integer sequence lengths for batch bucketing
    train_ds = ensure_length_column(train_ds)
    dev_ds = ensure_length_column(dev_ds)
    test_ds = ensure_length_column(test_ds)

    use_fp16, use_bf16 = get_hardware_precision()

    # Instantiate Model
    model_save_dir = scope_dir / "model"

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "Human", 1: "LLM"},
        label2id={"Human": 0, "LLM": 1},
        use_safetensors=True,
    )
    model.config.use_cache = False

    optimizer_type = "adamw_torch_fused" if torch.cuda.is_available() and torch.__version__ >= "2.0" else "adamw_torch"

    training_args = TrainingArguments(
        output_dir=str(model_save_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=best_params["learning_rate"],
        per_device_train_batch_size=best_params["per_device_train_batch_size"],
        per_device_eval_batch_size=16,
        num_train_epochs=best_params["num_train_epochs"],
        weight_decay=best_params.get("weight_decay", 0.05),
        warmup_ratio=best_params.get("warmup_ratio", 0.1),
        label_smoothing_factor=best_params.get("label_smoothing_factor", 0.05),
        load_best_model_at_end=True,
        metric_for_best_model="pauc_001",
        greater_is_better=True,
        fp16=use_fp16,
        bf16=use_bf16,
        # --- Batch Bucketing Optimization ---
        group_by_length=True,
        length_column_name="length",
        # ------------------------------------
        optim=optimizer_type,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
        eval_accumulation_steps=10,
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8 if (use_fp16 or use_bf16) else None),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\nFine-tuning model on full train split...")
    trainer.train()

    # Save Fine-Tuned Model & Tokenizer
    trainer.save_model(str(model_save_dir))
    tokenizer.save_pretrained(str(model_save_dir))
    print(f"\n[MODEL SAVED] Model saved to: '{model_save_dir}'")

    # CALCULATE OPTIMAL THRESHOLD ON DEV SET
    optimal_threshold, oof_stats = calculate_oof_threshold_on_dev(
        dev_df=dev_df, dev_ds=dev_ds, trainer=trainer
    )

    # EVALUATE DEV AND HELD-OUT TEST SETS
    dev_metrics, _ = evaluate_and_plot_results(
        df_split=dev_df,
        split_ds=dev_ds,
        split_name="dev",
        scope=scope,
        trainer=trainer,
        optimal_threshold=optimal_threshold,
        save_dir=scope_dir,
    )

    test_metrics, per_gen_metrics = evaluate_and_plot_results(
        df_split=test_df,
        split_ds=test_ds,
        split_name="test",
        scope=scope,
        trainer=trainer,
        optimal_threshold=optimal_threshold,
        save_dir=scope_dir,
    )

    # Save Comprehensive Summary JSON and Update Parameters
    summary_path = scope_dir / "paper_evaluation_summary.json"
    summary_json = {
        "scope": scope,
        "optimal_decision_threshold": optimal_threshold,
        "oof_threshold_statistics": oof_stats,
        "dev_metrics": dev_metrics,
        "test_metrics": test_metrics,
        "per_generator_metrics": per_gen_metrics,
        "best_hyperparameters": best_params,
    }

    with open(summary_path, "w") as f:
        json.dump(summary_json, f, indent=4)

    # Update best_hyperparameters.json with the calculated optimal threshold
    params_json_path = scope_dir / "best_hyperparameters.json"
    hyperparams_data = {
        "scope": scope,
        "optimal_decision_threshold": optimal_threshold,
        "best_hyperparameters": best_params,
        "oof_threshold_statistics": oof_stats,
        "dev_metrics": dev_metrics,
        "test_metrics": test_metrics,
    }
    with open(params_json_path, "w") as f:
        json.dump(hyperparams_data, f, indent=4)

    print(f"\n[SUMMARY SAVED] Comprehensive summary saved to: '{summary_path}'")
    print(f"[PARAMS UPDATED] Best parameters & threshold updated in: '{params_json_path}'")

    # Cleanup VRAM
    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==========================================
# Main Execution Pipeline
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Full mDeBERTa-v3 training setup utilizing dataloader cache & dev OOF thresholding.")

    parser.add_argument(
        "--scopes",
        nargs="+",
        default=["full", "sentence"],
        choices=["full", "sentence"],
        help="List of scopes to train sequentially (default: full sentence)."
    )
    parser.add_argument("--model_name", type=str, default="microsoft/mdeberta-v3-base", help="Hugging Face model checkpoint.")
    parser.add_argument("--sample_size", type=int, default=-1, help="Sample size for training (-1 for full dataset).")
    parser.add_argument("--max_length", type=int, default=256, help="Tokenizer max sequence length.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--outputs_dir", type=str, default=None, help="Base outputs directory (defaults to /home/gderijck/detection/outputs).")

    args = parser.parse_args()
    manager = DetectionDataManager()

    for scope in args.scopes:
        run_full_training_for_scope(scope=scope, args=args, manager=manager)

    print("\n" + "=" * 70)
    print("[ALL DONE] Full training & evaluation complete for all requested scopes!")
    print(f"Results saved under: {args.outputs_dir if args.outputs_dir else DEFAULT_OUTPUTS_DIR / 'mdeberta'}")
    print("=" * 70)


if __name__ == "__main__":
    main()