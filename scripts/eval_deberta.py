#!/usr/bin/env python3
# scripts/eval_deberta.py

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
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

# Calculate project root dynamically (~/detection)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"

from src.data.data_loader import DataFilter, DetectionDataManager


# ==========================================
# 1. PyTorch Dataset Helper
# ==========================================
class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = [str(t) if t is not None else "" for t in texts]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


# ==========================================
# 2. Model & Config Loader
# ==========================================
def load_deberta_model_and_config(scope: str, outputs_dir: Path, device: torch.device):
    scope_dir = outputs_dir / "deberta" / scope
    config_path = scope_dir / "best_hyperparameters.json"

    if not scope_dir.exists():
        raise FileNotFoundError(
            f"No trained DeBERTa model directory found at: '{scope_dir}'.\n"
            f"Please run your DeBERTa training script for scope '{scope}' first."
        )

    print(f"[LOADING MODEL] Loading DeBERTa checkpoint from: {scope_dir}")
    tokenizer = AutoTokenizer.from_pretrained(scope_dir)
    model = AutoModelForSequenceClassification.from_pretrained(scope_dir)
    model.to(device)
    model.eval()

    config = {}
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
        print(f"[LOADED CONFIG] Loaded configuration from: {config_path}")
    else:
        print(f"[WARNING] No best_hyperparameters.json found at '{config_path}'. Defaulting threshold to 0.5.")

    return model, tokenizer, config, scope_dir


# ==========================================
# 3. Batched DeBERTa Inference Function
# ==========================================
def run_deberta_inference(
    model, tokenizer, texts, device: torch.device, batch_size: int = 32, max_length: int = 512
):
    dataset = TextDataset(texts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_probs = []
    print(f"[INFERENCE] Running batched inference on {len(texts)} samples (Batch Size: {batch_size}, Max Length: {max_length})...")

    with torch.no_grad():
        for batch_texts in dataloader:
            inputs = tokenizer(
                list(batch_texts),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            outputs = model(**inputs)
            logits = outputs.logits

            # Calculate probability of class 1 (LLM-generated text)
            if logits.shape[1] == 1:
                probs = torch.sigmoid(logits).squeeze(-1)
            else:
                probs = torch.softmax(logits, dim=-1)[:, 1]

            all_probs.extend(probs.cpu().numpy())

    return np.array(all_probs)


# ==========================================
# 4. Error Categorization & Analysis
# ==========================================
def analyze_and_save_errors(df_eval: pd.DataFrame, scope_dir: Path, scope: str):
    """Categorizes misclassifications, exports detailed CSV files, and prints breakdowns by generator/dataset."""
    conditions = [
        (df_eval["label"] == 1) & (df_eval["pred_llm"] == 1),
        (df_eval["label"] == 0) & (df_eval["pred_llm"] == 0),
        (df_eval["label"] == 0) & (df_eval["pred_llm"] == 1),
        (df_eval["label"] == 1) & (df_eval["pred_llm"] == 0),
    ]
    choices = ["TP", "TN", "FP", "FN"]
    df_eval["error_type"] = np.select(conditions, choices, default="UNKNOWN")
    df_eval["is_error"] = df_eval["error_type"].isin(["FP", "FN"])

    # 1. Export Full Predictions CSV (Binary Preds + Probabilities + Error Types)
    preds_csv = scope_dir / "test_predictions.csv"
    df_eval.to_csv(preds_csv, index=False)
    print(f"[CSV SAVED] Full test predictions saved to: '{preds_csv}'")

    # 2. Export Dedicated Error Files
    errors_df = df_eval[df_eval["is_error"]].copy()
    errors_csv = scope_dir / "test_errors.csv"
    errors_df.to_csv(errors_csv, index=False)

    fp_df = df_eval[df_eval["error_type"] == "FP"].copy()
    fn_df = df_eval[df_eval["error_type"] == "FN"].copy()
    fp_df.to_csv(scope_dir / "test_false_positives.csv", index=False)
    fn_df.to_csv(scope_dir / "test_false_negatives.csv", index=False)

    print(f"[CSV SAVED] Saved {len(errors_df)} errors total ({len(fp_df)} FPs, {len(fn_df)} FNs) to '{scope_dir}'")

    # 3. Print Error Summary
    print(f"\n--- ERROR BREAKDOWN ANALYSIS [{scope.upper()} SCOPE] ---")
    total_samples = len(df_eval)
    total_errors = len(errors_df)
    err_rate = (total_errors / total_samples) * 100 if total_samples > 0 else 0.0
    print(f"  Total Test Samples : {total_samples}")
    print(f"  Total Errors       : {total_errors} ({err_rate:.2f}%)")
    print(f"  False Positives    : {len(fp_df)} (Human text misclassified as LLM)")
    print(f"  False Negatives    : {len(fn_df)} (LLM text misclassified as Human)")

    # 4. Error Breakdown by Generator Model
    gen_col = None
    for col in ["model_name", "generator_model", "generator", "model"]:
        if col in df_eval.columns:
            gen_col = col
            break

    if gen_col:
        gen_breakdown = []
        for gen, group in df_eval.groupby(gen_col):
            g_total = len(group)
            g_errors = group["is_error"].sum()
            g_fp = (group["error_type"] == "FP").sum()
            g_fn = (group["error_type"] == "FN").sum()
            g_err_rate = (g_errors / g_total * 100) if g_total > 0 else 0.0
            mean_prob = group["prob_llm"].mean()

            gen_breakdown.append({
                "Generator / Source": gen,
                "Total Samples": g_total,
                "Errors": g_errors,
                "Error Rate (%)": round(g_err_rate, 2),
                "False Positives": g_fp,
                "False Negatives": g_fn,
                "Mean Prob P(LLM)": round(mean_prob, 4),
            })

        gen_df = pd.DataFrame(gen_breakdown).sort_values(by="Error Rate (%)", ascending=False)
        print("\n  [Error Breakdown by Generator / Source Model]")
        print(gen_df.to_string(index=False))

        gen_csv = scope_dir / "test_error_breakdown.csv"
        gen_df.to_csv(gen_csv, index=False)
        print(f"\n  [CSV SAVED] Generator error breakdown saved to: '{gen_csv}'")

    # 5. Error Breakdown by Synthetic Dataset Source (if available)
    dataset_col = None
    for col in ["dataset", "data_source", "domain", "dataset_name"]:
        if col in df_eval.columns and col != gen_col:
            dataset_col = col
            break

    if dataset_col:
        ds_breakdown = []
        for ds, group in df_eval.groupby(dataset_col):
            d_total = len(group)
            d_errors = group["is_error"].sum()
            d_fp = (group["error_type"] == "FP").sum()
            d_fn = (group["error_type"] == "FN").sum()
            ds_breakdown.append({
                "Dataset": ds,
                "Total Samples": d_total,
                "Errors": d_errors,
                "Error Rate (%)": round((d_errors / d_total) * 100, 2),
                "False Positives": d_fp,
                "False Negatives": d_fn,
            })
        ds_df = pd.DataFrame(ds_breakdown).sort_values(by="Error Rate (%)", ascending=False)
        print("\n  [Error Breakdown by Dataset]")
        print(ds_df.to_string(index=False))

        ds_csv = scope_dir / "test_error_breakdown_by_dataset.csv"
        ds_df.to_csv(ds_csv, index=False)
        print(f"  [CSV SAVED] Dataset error breakdown saved to: '{ds_csv}'")

    # 6. Print Top High-Confidence Misclassifications
    if len(fp_df) > 0:
        print("\n  [Top High-Confidence False Positives (Human text falsely predicted as LLM)]")
        top_fps = fp_df.sort_values(by="prob_llm", ascending=False).head(3)
        for _, row in top_fps.iterrows():
            text_snippet = str(row.get("text", ""))[:100].replace("\n", " ")
            source_info = row.get(gen_col, "Human") if gen_col else "Human"
            print(f"    - P(LLM): {row['prob_llm']:.4f} | Source: {source_info} | Text: {text_snippet}...")

    if len(fn_df) > 0:
        print("\n  [Top High-Confidence False Negatives (Synthetic LLM text falsely predicted as Human)]")
        top_fns = fn_df.sort_values(by="prob_llm", ascending=True).head(3)
        for _, row in top_fns.iterrows():
            text_snippet = str(row.get("text", ""))[:100].replace("\n", " ")
            gen_info = row.get(gen_col, "LLM") if gen_col else "LLM"
            print(f"    - P(LLM): {row['prob_llm']:.4f} | Generator: {gen_info} | Text: {text_snippet}...")

    return df_eval


# ==========================================
# 5. Main Test Evaluation Function
# ==========================================
def evaluate_deberta_test_set(scope: str, args, manager: DetectionDataManager, device: torch.device):
    outputs_base = Path(args.outputs_dir) if args.outputs_dir else DEFAULT_OUTPUTS_DIR
    model, tokenizer, config, scope_dir = load_deberta_model_and_config(scope=scope, outputs_dir=outputs_base, device=device)

    # Load test split raw DataFrame
    test_df = manager.filter_dataframe(DataFilter(splits=["test"], scopes=[scope])).copy()
    labels = test_df["label"].values
    optimal_threshold = config.get("optimal_decision_threshold", 0.5)

    print("\n" + "=" * 70)
    print(f" EVALUATING DEBERTA MODEL ON HELD-OUT TEST SET [{scope.upper()} SCOPE] ")
    print(f" Optimal Threshold (\u03c4*): {optimal_threshold:.6f}")
    print(f" Test Set Size           : {len(labels)} samples")
    print("=" * 70)

    # 1. Inference
    max_len = args.max_length if args.max_length else (128 if scope == "sentence" else 512)
    probs_llm = run_deberta_inference(
        model=model,
        tokenizer=tokenizer,
        texts=test_df["text"].tolist(),
        device=device,
        batch_size=args.batch_size,
        max_length=max_len,
    )
    preds = (probs_llm >= optimal_threshold).astype(int)

    # 2. Overall Metrics
    fpr, tpr, _ = roc_curve(labels, probs_llm)
    roc_auc_val = auc(fpr, tpr)
    pauc_001_val = roc_auc_score(labels, probs_llm, max_fpr=0.01)

    precision_curve, recall_curve, _ = precision_recall_curve(labels, probs_llm)
    pr_auc_val = average_precision_score(labels, probs_llm)

    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)

    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    tpr_at_1fpr = tpr[np.where(fpr <= 0.01)[0][-1]] if len(np.where(fpr <= 0.01)[0]) > 0 else 0.0
    tpr_at_5fpr = tpr[np.where(fpr <= 0.05)[0][-1]] if len(np.where(fpr <= 0.05)[0]) > 0 else 0.0

    overall_metrics = {
        "Scope": scope,
        "Total Test Samples": len(labels),
        "Optimal Decision Threshold (\u03c4*)": round(optimal_threshold, 6),
        "pAUC @ max FPR 0.01": round(pauc_001_val, 6),
        "ROC-AUC": round(roc_auc_val, 4),
        "PR-AUC (AP)": round(pr_auc_val, 4),
        "Accuracy": round(acc, 4),
        "F1-Score": round(f1, 4),
        "Precision": round(prec, 4),
        "Recall (Sensitivity)": round(rec, 4),
        "Specificity": round(specificity, 4),
        "TPR @ 1% FPR": round(tpr_at_1fpr, 4),
        "TPR @ 5% FPR": round(tpr_at_5fpr, 4),
    }

    print("\n--- OVERALL TEST SET PERFORMANCE ---")
    for k, v in overall_metrics.items():
        print(f"  {k:<32}: {v}")

    # 3. Per-LLM Generator Breakdown
    gen_col = "model_name" if "model_name" in test_df.columns else ("generator_model" if "generator_model" in test_df.columns else None)
    per_model_results = []

    if gen_col:
        df_eval = test_df.copy()
        df_eval["prob_llm"] = probs_llm
        df_eval["pred_llm"] = preds
        human_df = df_eval[df_eval["label"] == 0]

        for generator in df_eval[gen_col].unique():
            if str(generator).lower() == "human":
                continue

            llm_sub = df_eval[df_eval[gen_col] == generator]
            combined = pd.concat([human_df, llm_sub])

            sub_labels = combined["label"].values
            sub_probs = combined["prob_llm"].values
            sub_preds = combined["pred_llm"].values

            sub_auc = roc_auc_score(sub_labels, sub_probs)
            sub_acc = accuracy_score(sub_labels, sub_preds)
            sub_prec, sub_rec, sub_f1, _ = precision_recall_fscore_support(sub_labels, sub_preds, average="binary", zero_division=0)

            per_model_results.append({
                "Generator Model": generator,
                "LLM Samples": len(llm_sub),
                "ROC-AUC": round(sub_auc, 4),
                "Accuracy": round(sub_acc, 4),
                "F1-Score": round(sub_f1, 4),
                "Precision": round(sub_prec, 4),
                "Recall": round(sub_rec, 4),
            })

        if per_model_results:
            print("\n--- PER-LLM GENERATOR BREAKDOWN ---")
            print(pd.DataFrame(per_model_results).to_string(index=False))

    # 4. Error Analysis & Export CSVs
    df_analysis = test_df.copy()
    df_analysis["prob_llm"] = probs_llm
    df_analysis["pred_llm"] = preds
    analyze_and_save_errors(df_analysis, scope_dir=scope_dir, scope=scope)

    # 5. Generate 4-Panel Publication Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    plt.subplots_adjust(wspace=0.25, hspace=0.3)

    # Panel A: ROC Curve
    axes[0, 0].plot(fpr, tpr, color="#2b5c8f", lw=2, label=f"DeBERTa ({scope.upper()})\npAUC@0.01={pauc_001_val:.4f}")
    axes[0, 0].axvline(x=0.01, color="red", linestyle=":", lw=1.5, label="FPR = 0.01")
    axes[0, 0].plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5, label="Random Guess")
    axes[0, 0].set_xlabel("False Positive Rate", fontsize=11)
    axes[0, 0].set_ylabel("True Positive Rate", fontsize=11)
    axes[0, 0].set_title(f"(A) ROC Curve ({scope.upper()})", fontsize=12, fontweight="bold")
    axes[0, 0].legend(loc="lower right", fontsize=10)
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    # Panel B: PR Curve
    axes[0, 1].plot(recall_curve, precision_curve, color="#d95f02", lw=2, label=f"AP={pr_auc_val:.4f}")
    axes[0, 1].set_xlabel("Recall", fontsize=11)
    axes[0, 1].set_ylabel("Precision", fontsize=11)
    axes[0, 1].set_title(f"(B) Precision-Recall Curve ({scope.upper()})", fontsize=12, fontweight="bold")
    axes[0, 1].legend(loc="lower left", fontsize=10)
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    # Panel C: Confusion Matrix
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    im = axes[1, 0].imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    axes[1, 0].set_title(f"(C) Confusion Matrix (\u03c4* = {optimal_threshold:.4f})", fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    classes = ["Human", "LLM"]
    tick_marks = np.arange(len(classes))
    axes[1, 0].set_xticks(tick_marks)
    axes[1, 0].set_xticklabels(classes, fontsize=10)
    axes[1, 0].set_yticks(tick_marks)
    axes[1, 0].set_yticklabels(classes, fontsize=10)
    axes[1, 0].set_ylabel("True Label", fontsize=11)
    axes[1, 0].set_xlabel("Predicted Label", fontsize=11)

    for i in range(2):
        for j in range(2):
            axes[1, 0].text(
                j, i, f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > 0.5 else "black",
                fontsize=11, fontweight="bold",
            )

    # Panel D: Density Distribution
    human_probs = probs_llm[labels == 0]
    llm_probs = probs_llm[labels == 1]

    axes[1, 1].hist(human_probs, bins=25, alpha=0.6, color="#1b9e77", label="Human Text", density=True)
    axes[1, 1].hist(llm_probs, bins=25, alpha=0.6, color="#7570b3", label="LLM Text", density=True)
    axes[1, 1].axvline(x=optimal_threshold, color="black", linestyle="--", lw=2, label=f"Threshold (\u03c4*={optimal_threshold:.4f})")
    axes[1, 1].set_xlabel("Predicted Probability P(LLM)", fontsize=11)
    axes[1, 1].set_ylabel("Density", fontsize=11)
    axes[1, 1].set_title(f"(D) Probability Density ({scope.upper()})", fontsize=12, fontweight="bold")
    axes[1, 1].legend(loc="upper center", fontsize=10)
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    plot_path = scope_dir / "test_evaluation_plots.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"\n[FIGURE SAVED] Publication plot saved to: '{plot_path}'")
    plt.close()

    # 6. Export LaTeX Table
    latex_table_path = scope_dir / "test_metrics_table.tex"
    with open(latex_table_path, "w") as f:
        f.write("% Auto-generated LaTeX table for DeBERTa test evaluation\n")
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write(f"\\caption{{Performance Evaluation of DeBERTa Detector ({scope.upper()}) on Held-out Test Set. Threshold $\\tau^* = {optimal_threshold:.4f}$.}}\n")
        f.write("\\label{tab:deberta_" + scope + "_test_results}\n")
        f.write("\\begin{tabular}{lcccccc}\n\\hline\n")
        f.write("\\textbf{Evaluation Group} & \\textbf{pAUC @ 0.01} & \\textbf{ROC-AUC} & \\textbf{Accuracy} & \\textbf{F1-Score} & \\textbf{TPR @ 1\\% FPR} \\\\\n\\hline\n")
        f.write(f"Overall Test Set & {pauc_001_val:.4f} & {roc_auc_val:.4f} & {acc:.4f} & {f1:.4f} & {tpr_at_1fpr:.4f} \\\\\n")
        f.write("\\hline\n")
        for row in per_model_results:
            f.write(f"vs. {row['Generator Model']} & -- & {row['ROC-AUC']:.4f} & {row['Accuracy']:.4f} & {row['F1-Score']:.4f} & -- \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

    print(f"[LATEX SAVED] Copy-pasteable LaTeX table saved to: '{latex_table_path}'")

    # 7. Save Structured JSON Summary
    summary_path = scope_dir / "test_paper_evaluation_summary.json"
    json_data = {
        "overall_test_metrics": overall_metrics,
        "per_model_metrics": per_model_results,
        "best_hyperparameters": config.get("best_hyperparameters", {}),
    }
    with open(summary_path, "w") as f:
        json.dump(json_data, f, indent=4)

    print(f"[JSON SAVED] Test evaluation summary saved to: '{summary_path}'\n")


# ==========================================
# Main Execution Pipeline
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate DeBERTa model on held-out test split.")

    parser.add_argument(
        "--scopes",
        nargs="+",
        default=["sentence", "full"],
        choices=["sentence", "full"],
        help="List of scopes to evaluate sequentially (default: sentence full)."
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Inference batch size (default: 32).")
    parser.add_argument("--max_length", type=int, default=None, help="Tokenizer max sequence length override.")
    parser.add_argument("--outputs_dir", type=str, default=None, help="Base outputs directory.")

    args = parser.parse_args()
    manager = DetectionDataManager()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[HARDWARE] Using device: {device}")

    for scope in args.scopes:
        evaluate_deberta_test_set(scope=scope, args=args, manager=manager, device=device)

    print("=" * 70)
    print("[ALL DONE] DeBERTa test evaluation complete for both sentence and full scopes!")
    print("=" * 70)


if __name__ == "__main__":
    main()