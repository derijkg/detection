#!/usr/bin/env python3
"""
Direct mDeBERTa-v3 Fine-Tuning & Evaluation Pipeline
Optimized for RTX 2080 (8GB VRAM Windows) & RTX 2080 Ti (11GB VRAM Linux)

Key Features & Scientific Enhancements:
- Effective Evaluation Metric: Optimized for Partial AUROC up to 0.01 FPR (pauc_1fpr).
- Dynamic Dev Set Resampling: Draws a fresh stratified random sample from full Dev Set on EVERY evaluation step.
- Pretokenized Cache Support: Automatically utilizes disk-cached tokenized datasets if present.
- Operating Point Calibration: Finds threshold tau on Dev Set guaranteeing FPR <= target_fpr (e.g., 1%).
- Numerical Precision Safeguards: Clamps calibration thresholds to prevent test set recall collapse.
- Leave-One-LLM-Out (LOMO) zero-shot evaluation support.
- `--test_only` Flag: Direct test set evaluation using calibrated operating point on pre-trained best model.
"""

import os
import sys
from pathlib import Path
# RTX CUDA Memory Allocator Optimization
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

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)

# Enable cuDNN autotuner
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


# Setup Project Root (~/detection)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Clean import from project root
from src.data.data_loader import DetectionDataManager, DataFilter


# ---------------------------------------------------------
# DYNAMIC DEV SET RESAMPLING & SAFE TRAINER SUBCLASS
# ---------------------------------------------------------
class DynamicSafeDebertaTrainer(Trainer):
    """
    Custom Trainer that:
    1. Overrides `evaluate()` to sample a FRESH stratified dynamic subset from the full Dev set on EVERY evaluation step.
    2. Ensures robust parameter and gradient precision hand-offs during PyTorch FP16 AMP scaling.
    """
    def __init__(self, *args, full_val_df=None, eval_sample_size=None, tokenizer=None, max_len=256, seed=42, **kwargs):
        super().__init__(*args, **kwargs)
        self.full_val_df = full_val_df
        self.eval_sample_size = eval_sample_size
        self.val_tokenizer = tokenizer
        self.max_len = max_len
        self.eval_seed = seed
        self.eval_call_count = 0

    def _get_dynamic_eval_dataset(self):
        if self.full_val_df is None or self.eval_sample_size is None or self.eval_sample_size <= 0:
            return None

        if len(self.full_val_df) <= self.eval_sample_size:
            eval_df = self.full_val_df
        else:
            self.eval_call_count += 1
            current_seed = self.eval_seed + self.eval_call_count
            
            target_per_class = self.eval_sample_size // 2
            h_df = self.full_val_df[self.full_val_df['label'] == 0]
            l_df = self.full_val_df[self.full_val_df['label'] == 1]

            v_h = h_df.sample(n=min(target_per_class, len(h_df)), random_state=current_seed)
            v_l = l_df.sample(n=min(target_per_class, len(l_df)), random_state=current_seed)
            
            eval_df = pd.concat([v_h, v_l]).sample(frac=1, random_state=current_seed).reset_index(drop=True)

        def tokenize_fn(examples):
            return self.val_tokenizer(examples['text'], truncation=True, max_length=self.max_len)

        dynamic_ds = Dataset.from_pandas(eval_df, preserve_index=False).map(
            tokenize_fn, 
            batched=True, 
            load_from_cache_file=False,
            new_fingerprint=f"eval_step_{self.eval_call_count}"
        )
        return dynamic_ds

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", **kwargs):
        # Draw a dynamic resampled dev subset if eval_dataset is not explicitly provided
        if eval_dataset is None and self.full_val_df is not None:
            eval_dataset = self._get_dynamic_eval_dataset()

        return super().evaluate(
            eval_dataset=eval_dataset, 
            ignore_keys=ignore_keys, 
            metric_key_prefix=metric_key_prefix, 
            **kwargs
        )

    def training_step(self, *args, **kwargs):
        loss = super().training_step(*args, **kwargs)
        # Prevent numerical NaN overflow in FP16 gradients
        for p in self.model.parameters():
            if p.requires_grad and p.grad is not None:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    p.grad.zero_()
        return loss


# ---------------------------------------------------------
# 1. EVALUATION METRICS (FOCUSED ON pAUC @ 1% FPR)
# ---------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()

    if len(np.unique(labels)) < 2:
        return {
            'pauc_1fpr': 0.5,
            'tpr_at_1fpr': 0.0,
            'tpr_at_5fpr': 0.0,
            'roc_auc': 0.5,
            'accuracy': 0.0,
            'f1': 0.0
        }

    try:
        fpr, tpr, _ = roc_curve(labels, probs)
        
        idx_1fpr = np.where(fpr <= 0.01)[0]
        tpr_at_1fpr = float(tpr[idx_1fpr[-1]]) if len(idx_1fpr) > 0 else 0.0

        idx_5fpr = np.where(fpr <= 0.05)[0]
        tpr_at_5fpr = float(tpr[idx_5fpr[-1]]) if len(idx_5fpr) > 0 else 0.0

        try:
            pauc_1fpr = float(roc_auc_score(labels, probs, max_fpr=0.01))
        except Exception:
            pauc_1fpr = 0.5

        overall_auc = float(roc_auc_score(labels, probs))
    except Exception:
        pauc_1fpr, tpr_at_1fpr, tpr_at_5fpr, overall_auc = 0.5, 0.0, 0.0, 0.5

    acc = float(accuracy_score(labels, preds))
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )

    return {
        'pauc_1fpr': pauc_1fpr,  # PRIMARY MODEL SELECTION METRIC
        'tpr_at_1fpr': tpr_at_1fpr,
        'tpr_at_5fpr': tpr_at_5fpr,
        'roc_auc': overall_auc,
        'accuracy': acc,
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall)
    }


# ---------------------------------------------------------
# 2. DEV SET THRESHOLD CALIBRATION
# ---------------------------------------------------------
def calibrate_threshold_at_fpr(val_labels, val_probs, max_fpr=0.01):
    """
    Finds the largest decision threshold tau on the validation set 
    that guarantees FPR <= max_fpr (e.g., 0.01).
    Applies strict numerical safeguards against scikit-learn threshold upper bounds.
    """
    fpr, tpr, thresholds = roc_curve(val_labels, val_probs)
    valid_indices = np.where(fpr <= max_fpr)[0]

    if len(valid_indices) == 0:
        print(f"[WARNING] Could not achieve FPR <= {max_fpr}. Defaulting threshold to 0.5.")
        return 0.5, 0.0, 0.0

    best_idx = valid_indices[-1]
    raw_thresh = thresholds[best_idx]
    
    max_prob_val = float(np.max(val_probs)) if len(val_probs) > 0 else 1.0
    calibrated_thresh = float(np.clip(raw_thresh, 0.0, max_prob_val))

    calibrated_fpr = float(fpr[best_idx])
    calibrated_tpr = float(tpr[best_idx])

    print("\n" + "=" * 70)
    print(f" DEV SET CALIBRATION (@ FPR <= {max_fpr*100:.1f}%) ")
    print("=" * 70)
    print(f"  Calibrated Threshold (tau) : {calibrated_thresh:.6f}")
    print(f"  Dev Set FPR achieved       : {calibrated_fpr*100:.2f}%")
    print(f"  Dev Set TPR (Sensitivity)  : {calibrated_tpr*100:.2f}%")
    print("=" * 70)

    return calibrated_thresh, calibrated_fpr, calibrated_tpr


# ---------------------------------------------------------
# 3. FAST INFERENCE
# ---------------------------------------------------------
def run_fast_inference(model, test_ds, tokenizer, batch_size=16, device="cuda"):
    model.to(device)
    model.eval()

    model_input_cols = ['input_ids', 'attention_mask', 'token_type_ids']
    keep_cols = [c for c in model_input_cols if c in test_ds.column_names]
    remove_cols = [c for c in test_ds.column_names if c not in keep_cols]
    
    test_ds_filtered = test_ds.remove_columns(remove_cols) if remove_cols else test_ds

    dataloader = torch.utils.data.DataLoader(
        test_ds_filtered,
        batch_size=batch_size,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=True
    )

    all_logits = []
    with torch.inference_mode():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            all_logits.append(outputs.logits.cpu())

    logits_tensor = torch.cat(all_logits, dim=0)
    probs_tensor = F.softmax(logits_tensor, dim=-1)

    return logits_tensor.numpy(), probs_tensor[:, 1].numpy()


# ---------------------------------------------------------
# 4. PUBLICATION REPORTING & PLOTS
# ---------------------------------------------------------
def evaluate_paper_results(df_test, probs_llm, preds, save_dir, eval_name="test", held_out_llm=None, calibrated_tau=0.5):
    os.makedirs(save_dir, exist_ok=True)
    labels = df_test['label'].values

    fpr, tpr, _ = roc_curve(labels, probs_llm)
    roc_auc_val = auc(fpr, tpr)
    pr_auc_val = average_precision_score(labels, probs_llm)

    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)

    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    empirical_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    tpr_at_1fpr = tpr[np.where(fpr <= 0.01)[0][-1]] if len(np.where(fpr <= 0.01)[0]) > 0 else 0.0
    tpr_at_5fpr = tpr[np.where(fpr <= 0.05)[0][-1]] if len(np.where(fpr <= 0.05)[0]) > 0 else 0.0

    try:
        pauc_1fpr = float(roc_auc_score(labels, probs_llm, max_fpr=0.01))
    except Exception:
        pauc_1fpr = 0.5

    overall_metrics = {
        "Evaluation Set": eval_name,
        "Calibrated Threshold (tau)": round(float(calibrated_tau), 6),
        "Held-Out Training LLM": held_out_llm if held_out_llm else "None (Trained on All LLMs)",
        "Total Test Samples": len(labels),
        "ROC-AUC": round(float(roc_auc_val), 4),
        "PR-AUC (AP)": round(float(pr_auc_val), 4),
        "Partial AUROC @ 1% FPR": round(float(pauc_1fpr), 4),
        "Empirical Test FPR": round(float(empirical_fpr), 4),
        "TPR @ 1% FPR": round(float(tpr_at_1fpr), 4),
        "TPR @ 5% FPR": round(float(tpr_at_5fpr), 4),
        "Accuracy": round(float(acc), 4),
        "F1-Score": round(float(f1), 4),
        "Precision": round(float(prec), 4),
        "Recall (Sensitivity)": round(float(rec), 4),
        "Specificity": round(float(specificity), 4)
    }

    print("\n" + "=" * 70)
    print(f" EVALUATION REPORT [{eval_name.upper()}] (Calibrated tau = {calibrated_tau:.6f}) ")
    print("=" * 70)
    for k, v in overall_metrics.items():
        print(f"  {k:<26}: {v}")

    # Per-Generator Model Breakdown
    df_test_eval = df_test.copy()
    df_test_eval['prob_llm'] = probs_llm
    df_test_eval['pred'] = preds

    per_model_results = []
    human_df = df_test_eval[df_test_eval['label'] == 0]
    model_col = 'model_name' if 'model_name' in df_test_eval.columns else 'generator_model'

    if model_col in df_test_eval.columns:
        for model_name in df_test_eval[model_col].unique():
            if str(model_name).lower() in ["human", "original", "none", "nan"]:
                continue

            llm_sub_df = df_test_eval[df_test_eval[model_col] == model_name]
            combined_sub = pd.concat([human_df, llm_sub_df])

            sub_labels = combined_sub['label'].values
            sub_probs = combined_sub['prob_llm'].values
            sub_preds = combined_sub['pred'].values

            try:
                sub_auc = roc_auc_score(sub_labels, sub_probs)
            except ValueError:
                sub_auc = 0.5

            sub_acc = accuracy_score(sub_labels, sub_preds)
            sub_prec, sub_rec, sub_f1, _ = precision_recall_fscore_support(
                sub_labels, sub_preds, average='binary', zero_division=0
            )

            is_held_out = (held_out_llm is not None) and (held_out_llm.lower() in str(model_name).lower())

            per_model_results.append({
                "Generator Model": str(model_name) + (" [HELD-OUT ZERO-SHOT]" if is_held_out else ""),
                "LLM Samples": len(llm_sub_df),
                "ROC-AUC": round(float(sub_auc), 4),
                "Accuracy": round(float(sub_acc), 4),
                "F1-Score": round(float(sub_f1), 4),
                "Precision": round(float(sub_prec), 4),
                "Recall": round(float(sub_rec), 4)
            })

    per_model_df = pd.DataFrame(per_model_results)
    if not per_model_df.empty:
        print(f"\n--- PER-LLM GENERATOR BREAKDOWN [{eval_name.upper()}] ---")
        print(per_model_df.to_string(index=False))

    # Publication Figures (300 DPI)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    
    axes[0].plot(fpr, tpr, color='#2b5c8f', lw=2, label=f'mDeBERTa-v3 (AUC={roc_auc_val:.4f})')
    axes[0].axvline(x=0.01, color='red', linestyle=':', label=f'1% FPR Operating Point (TPR={tpr_at_1fpr:.4f})')
    axes[0].axvline(x=0.05, color='orange', linestyle='--', label=f'5% FPR (TPR={tpr_at_5fpr:.4f})')
    axes[0].set_xlabel('False Positive Rate', fontsize=11)
    axes[0].set_ylabel('True Positive Rate', fontsize=11)
    axes[0].set_title(f'ROC Curve ({eval_name.upper()})', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, linestyle='--', alpha=0.5)

    axes[1].hist(probs_llm[labels == 0], bins=25, alpha=0.6, color='#1b9e77', label='Human Text', density=True)
    axes[1].hist(probs_llm[labels == 1], bins=25, alpha=0.6, color='#7570b3', label='LLM Text', density=True)
    axes[1].axvline(x=calibrated_tau, color='red', linestyle='--', lw=2, label=f'Calibrated tau ({calibrated_tau:.4f})')
    axes[1].set_xlabel('Predicted Probability P(LLM)', fontsize=11)
    axes[1].set_ylabel('Density', fontsize=11)
    axes[1].set_title(f'Output Distribution ({eval_name.upper()})', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"evaluation_plots_{eval_name}.png"), dpi=300)
    plt.close()

    summary_path = os.path.join(save_dir, f"evaluation_summary_{eval_name}.json")
    with open(summary_path, "w") as f:
        json.dump({"overall_metrics": overall_metrics, "per_model_metrics": per_model_results}, f, indent=4)

    return overall_metrics, per_model_df


# ---------------------------------------------------------
# 5. MAIN PIPELINE
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Direct mDeBERTa-v3 Fine-Tuning Pipeline")

    parser.add_argument("--scope", type=str, choices=["sentence", "full", "both"], default="sentence",
                        help="'sentence', 'full', or 'both' (Joint Multi-Granularity Training)")
    parser.add_argument("--leave_out_llm", type=str, default=None,
                        help="LLM model name to exclude from training/val sets for zero-shot testing (e.g. 'llama3')")
    
    parser.add_argument("--data_path", type=str, default=None, help="Path to preprocessed parquet dataset")
    parser.add_argument("--learning_rate", type=float, default=3.0e-5, help="Learning rate (default: 3.0e-5)")
    parser.add_argument("--epochs", type=int, default=3, help="Max training epochs (default: 3)")
    parser.add_argument("--model_name", type=str, default="microsoft/mdeberta-v3-base", help="Pretrained model identifier")

    parser.add_argument("--train_sample_size", type=int, default=-1, help="Rows for training (-1 for full dataset)")
    parser.add_argument("--val_sample_size", type=int, default=-1, help="Rows for val (-1 for full dataset)")
    parser.add_argument("--val_eval_sample_size", type=int, default=None, 
                        help="Subsample size for fast dynamic validation checks during training (Auto-scales by scope if None)")
    parser.add_argument("--test_sample_size", type=int, default=-1, help="Rows for test (-1 for full dataset)")

    parser.add_argument("--target_fpr", type=float, default=0.01, help="Target Dev FPR for threshold calibration (default: 0.01)")
    parser.add_argument("--output_dir", type=str, default="./outputs_fast", help="Base directory for outputs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    parser.add_argument("--test_only", action="store_true", 
                        help="Skip training and run threshold calibration + test evaluation using saved best model.")

    args = parser.parse_args()

    # Dynamic Output Directory: outputs/deberta_no_tune/deberta_{scope}
    scope_subfolder = f"deberta_{args.scope}"
    active_output_dir = os.path.join(args.output_dir, "deberta_no_tune", scope_subfolder)
    os.makedirs(active_output_dir, exist_ok=True)
    model_save_path = os.path.join(active_output_dir, f"best_mdeberta_{args.scope}")

    num_workers = min(4, os.cpu_count() or 1)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 70)
    print(f" DEVICE DETECTED   : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f" SCOPE MODE        : {args.scope.upper()}" + (" [JOINT MULTI-GRANULARITY DETECTOR]" if args.scope == "both" else ""))
    print(f" RUN MODE          : {'TEST ONLY (Evaluation Mode)' if args.test_only else 'TRAIN & EVALUATE'}")
    print(f" TARGET DEV FPR    : {args.target_fpr * 100:.1f}% Operating Point Threshold Calibration")
    print(f" MODEL SELECTION   : Partial AUROC @ 1% FPR (pauc_1fpr)")
    print(f" HELD-OUT LLM MODEL: {args.leave_out_llm if args.leave_out_llm else 'None (Include All)'}")
    print(f" OUTPUT DIRECTORY  : '{active_output_dir}'")
    print("=" * 70)

    manager = DetectionDataManager(data_path=args.data_path)
    max_len = 256 if args.scope == "sentence" else 512

    # ---------------------------------------------------------
    # 1. LOAD DATASETS (TRAIN / VAL / TEST)
    # ---------------------------------------------------------
    if args.scope in ["sentence", "full"]:
        if not args.test_only:
            train_df = manager.filter_dataframe(scopes=[args.scope], splits=['train'], sample_size=args.train_sample_size, seed=args.seed)
            train_df = train_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
        else:
            train_df = None
        
        val_df = manager.filter_dataframe(scopes=[args.scope], splits=['val'], sample_size=args.val_sample_size, seed=args.seed)
        test_df = manager.filter_dataframe(scopes=[args.scope], splits=['test'], sample_size=args.test_sample_size, seed=args.seed)
        test_dfs = {args.scope: test_df}

    else:  # args.scope == "both"
        print("\n[JOINT DATASET] Loading Sentence + Full Abstract datasets...")
        if not args.test_only:
            s_train = manager.filter_dataframe(scopes=['sentence'], splits=['train'], sample_size=args.train_sample_size, seed=args.seed)
            f_train = manager.filter_dataframe(scopes=['full'], splits=['train'], sample_size=args.train_sample_size, seed=args.seed)
            train_df = pd.concat([s_train, f_train]).sample(frac=1, random_state=args.seed).reset_index(drop=True)
        else:
            train_df = None

        s_val = manager.filter_dataframe(scopes=['sentence'], splits=['val'], sample_size=args.val_sample_size, seed=args.seed)
        f_val = manager.filter_dataframe(scopes=['full'], splits=['val'], sample_size=args.val_sample_size, seed=args.seed)
        val_df = pd.concat([s_val, f_val]).sample(frac=1, random_state=args.seed).reset_index(drop=True)

        s_test = manager.filter_dataframe(scopes=['sentence'], splits=['test'], sample_size=args.test_sample_size, seed=args.seed)
        f_test = manager.filter_dataframe(scopes=['full'], splits=['test'], sample_size=args.test_sample_size, seed=args.seed)
        c_test = pd.concat([s_test, f_test]).sample(frac=1, random_state=args.seed).reset_index(drop=True)

        test_dfs = {
            "sentence_test": s_test,
            "full_test": f_test,
            "combined_test": c_test
        }

    # 2. Apply Leave-One-LLM-Out (LOMO) Filtering to Train & Val sets
    model_col = 'model_name' if 'model_name' in val_df.columns else 'generator_model'

    if args.leave_out_llm and model_col in val_df.columns:
        llm_pattern = args.leave_out_llm.lower().strip()
        all_models = val_df[model_col].unique()
        matched_models = [m for m in all_models if llm_pattern in str(m).lower()]

        if matched_models:
            print(f"\n[LOMO FILTER] Excluding LLM model(s) matching '{args.leave_out_llm}': {matched_models}")
            if train_df is not None:
                train_mask = train_df[model_col].apply(lambda x: not any(m in str(x) for m in matched_models))
                train_df = train_df[train_mask].reset_index(drop=True)

            val_mask = val_df[model_col].apply(lambda x: not any(m in str(x) for m in matched_models))
            val_df = val_df[val_mask].reset_index(drop=True)
            print(f"  -> Remaining Val rows for calibration: {len(val_df):,}")
            print("  -> Test set retains ALL models to evaluate zero-shot detection.")

    # ---------------------------------------------------------
    # TEST ONLY MODE: LOAD SAVED MODEL
    # ---------------------------------------------------------
    if args.test_only:
        if not os.path.exists(model_save_path):
            raise FileNotFoundError(
                f"Cannot find saved best model at '{model_save_path}'. "
                f"Please run training first without --test_only."
            )

        print(f"\n[TEST-ONLY MODE] Loading fine-tuned model & tokenizer from: '{model_save_path}'")
        tokenizer = AutoTokenizer.from_pretrained(model_save_path, use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(model_save_path).to(device)

    # ---------------------------------------------------------
    # FULL TRAINING PIPELINE
    # ---------------------------------------------------------
    else:
        print(f"\nDataset Sizes for Training:")
        print(f"  - Train : {len(train_df):,} rows")
        print(f"  - Val   : {len(val_df):,} rows (Full validation set available)")

        # Scale intermediate dynamic validation subsample sizes
        if args.val_eval_sample_size is None or args.val_eval_sample_size <= 0:
            eval_sample_size = 10000 if args.scope == "sentence" else (3000 if args.scope == "full" else 5000)
        else:
            eval_sample_size = args.val_eval_sample_size

        eval_val_df = val_df.sample(n=min(eval_sample_size, len(val_df)), random_state=args.seed).reset_index(drop=True)
        print(f"  -> Dynamic Intermediate Val Size: {len(eval_val_df):,} rows (Resampled EVERY eval step)")

        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)

        def tokenize_fn_train(examples):
            return tokenizer(examples['text'], truncation=True, max_length=max_len)

        train_ds = Dataset.from_pandas(train_df, preserve_index=False).map(tokenize_fn_train, batched=True)
        eval_val_ds = Dataset.from_pandas(eval_val_df, preserve_index=False).map(tokenize_fn_train, batched=True)

        # Batching setup
        if max_len == 512:
            train_batch_size = 4
            grad_accum_steps = 4
            eval_batch_size = 8
        else:
            train_batch_size = 16
            grad_accum_steps = 2 
            eval_batch_size = 32

        effective_bs = train_batch_size * grad_accum_steps
        steps_per_epoch = len(train_ds) // effective_bs

        if steps_per_epoch > 1000:
            eval_strategy = "steps"
            eval_steps = max(200, min(500, steps_per_epoch // 20))
            save_strategy = "steps"
            save_steps = eval_steps
            patience = 4
            print(f"\n[EVAL STRATEGY] High-Frequency Step Evaluation Active ({steps_per_epoch:,} steps/epoch).")
            print(f"  -> Dynamically evaluating every {eval_steps:,} steps on fresh Dev subsets.")
        else:
            eval_strategy = "epoch"
            eval_steps = None
            save_strategy = "epoch"
            save_steps = None
            patience = 2
            print(f"\n[EVAL STRATEGY] Standard dataset size. Evaluating once per epoch dynamically.")

        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=2,
            id2label={0: "Human", 1: "LLM"},
            label2id={"Human": 0, "LLM": 1},
            use_safetensors=True
        )

        training_args = TrainingArguments(
            output_dir=model_save_path,
            eval_strategy=eval_strategy,
            eval_steps=eval_steps,
            save_strategy=save_strategy,
            save_steps=save_steps,
            save_total_limit=1,
            
            learning_rate=args.learning_rate,
            lr_scheduler_type="cosine",
            per_device_train_batch_size=train_batch_size,
            gradient_accumulation_steps=grad_accum_steps,
            per_device_eval_batch_size=eval_batch_size,
            eval_accumulation_steps=10,
            
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_grad_norm=1.0,
            fp16=True,
            
            num_train_epochs=args.epochs,
            weight_decay=0.01,
            warmup_steps=500,
            
            load_best_model_at_end=True,
            metric_for_best_model="pauc_1fpr",  # EFFECTIVE EVALUATION METRIC: pAUC @ 1% FPR
            greater_is_better=True,
            
            dataloader_num_workers=num_workers,
            dataloader_pin_memory=True,
            logging_steps=50,
            report_to="none"
        )

        trainer = DynamicSafeDebertaTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_val_ds,  # Initial seed dataset to pass HF Trainer __init__ checks
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
            full_val_df=val_df,
            eval_sample_size=eval_sample_size,
            tokenizer=tokenizer,
            max_len=max_len,
            seed=args.seed
        )

        print("\nStarting Fine-Tuning Training...")
        trainer.train()

        trainer.save_model(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        print(f"\n[MODEL SAVED] Best model (selected by highest pAUC @ 1% FPR) saved to '{model_save_path}'.")

    # ---------------------------------------------------------
    # 6. DEV SET OPERATING POINT THRESHOLD CALIBRATION
    # ---------------------------------------------------------
    eval_batch_size = 32 if args.scope == "sentence" else 8

    def tokenize_fn_eval(examples):
        return tokenizer(examples['text'], truncation=True, max_length=max_len)

    print(f"\n[CALIBRATION] Running inference on full Dev Set ({len(val_df):,} rows) to calibrate threshold at FPR <= {args.target_fpr * 100:.1f}%...")
    val_ds = Dataset.from_pandas(val_df, preserve_index=False).map(tokenize_fn_eval, batched=True)
    val_logits, val_probs = run_fast_inference(model, val_ds, tokenizer, batch_size=eval_batch_size, device=device)

    calibrated_tau, dev_fpr, dev_tpr = calibrate_threshold_at_fpr(
        val_labels=val_df['label'].values,
        val_probs=val_probs,
        max_fpr=args.target_fpr
    )

    # ---------------------------------------------------------
    # 7. TEST SET INFERENCE & EXPORTS USING CALIBRATED THRESHOLD
    # ---------------------------------------------------------
    print("\nRunning Evaluation & Prediction Exports on Test Sets using Calibrated Threshold...")
    for test_key, sub_test_df in test_dfs.items():
        sub_test_ds = Dataset.from_pandas(sub_test_df, preserve_index=False).map(tokenize_fn_eval, batched=True)
        
        logits, probs_llm = run_fast_inference(
            model, sub_test_ds, tokenizer, batch_size=eval_batch_size, device=device
        )
        
        # APPLY CALIBRATED DECISION THRESHOLD (tau)
        preds = (probs_llm >= calibrated_tau).astype(int)

        # Generate Reports & Plots
        evaluate_paper_results(
            sub_test_df, probs_llm, preds, 
            save_dir=active_output_dir, 
            eval_name=test_key, 
            held_out_llm=args.leave_out_llm,
            calibrated_tau=calibrated_tau
        )

        # Export Predictions CSV
        out_csv_df = sub_test_df.copy()
        out_csv_df['prob_llm'] = probs_llm
        out_csv_df['calibrated_threshold'] = calibrated_tau
        out_csv_df['pred'] = preds
        out_csv_df['logit_human'] = logits[:, 0]
        out_csv_df['logit_llm'] = logits[:, 1]
        out_csv_df['logit_diff'] = logits[:, 1] - logits[:, 0]

        csv_path = os.path.join(active_output_dir, f"mdeberta_predictions_{test_key}.csv")
        out_csv_df.to_csv(csv_path, index=False)
        print(f"[CSV EXPORTED] Saved: '{csv_path}' (Calibrated threshold tau = {calibrated_tau:.6f})")

    print("\n" + "=" * 70)
    print(" PIPELINE COMPLETED SUCCESSFULLY ")
    print(f" ALL OUTPUTS SAVED TO: '{active_output_dir}'")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()