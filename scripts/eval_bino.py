#!/usr/bin/env python3
# scripts/eval_binoculars.py

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
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
from tqdm import tqdm

# Calculate project root dynamically (~/detection)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Add mgteval submodule path
MGTEVAL_ROOT = PROJECT_ROOT / "evals" / "mgteval"
if str(MGTEVAL_ROOT) not in sys.path:
    sys.path.append(str(MGTEVAL_ROOT))

DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"

from src.data.data_loader import DataFilter, DetectionDataManager

# Attempt importing Binoculars from mgteval submodule or official binoculars library
try:
    from binoculars import Binoculars as OfficialBinoculars
    HAS_BINOCULARS_LIB = True
except ImportError:
    HAS_BINOCULARS_LIB = False


# ==========================================
# 0. Standalone Binoculars Fallback Inferencer
# ==========================================
def compute_binoculars_scores(texts: list, observer_model_name: str, performer_model_name: str, device: str, batch_size: int = 4) -> np.ndarray:
    """Computes Binoculars cross-perplexity vs perplexity ratio scores."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[BINOCULARS] Loading observer model '{observer_model_name}' and performer model '{performer_model_name}' on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(observer_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    obs_model = AutoModelForCausalLM.from_pretrained(observer_model_name, torch_dtype=torch.float16 if "cuda" in device else torch.float32).to(device)
    perf_model = AutoModelForCausalLM.from_pretrained(performer_model_name, torch_dtype=torch.float16 if "cuda" in device else torch.float32).to(device)

    obs_model.eval()
    perf_model.eval()

    scores = []
    print(f"[INFERENCE] Running Binoculars on {len(texts)} samples...")

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Binoculars Batch Score"):
            batch_texts = texts[i : i + batch_size]
            encodings = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            input_ids = encodings.input_ids
            attention_mask = encodings.attention_mask

            # Perplexity from Observer
            logits_obs = obs_model(input_ids).logits
            labels = input_ids[:, 1:].unsqueeze(-1)
            
            log_probs_obs = torch.log_softmax(logits_obs[:, :-1, :], dim=-1)
            lprobs_selected = log_probs_obs.gather(-1, labels).squeeze(-1)

            mask = attention_mask[:, 1:].float()
            ppl_obs = torch.exp(- (lprobs_selected * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8))

            # Cross-perplexity from Performer -> Observer
            logits_perf = perf_model(input_ids).logits
            probs_perf = torch.softmax(logits_perf[:, :-1, :], dim=-1)
            cross_entropy = - (probs_perf * log_probs_obs).sum(dim=-1)
            xppl = torch.exp((cross_entropy * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8))

            # Binoculars Score: PPL / XPPL (Lower -> AI; so negated so Higher -> AI)
            bino_score = -(ppl_obs / (xppl + 1e-8))
            scores.extend(bino_score.cpu().tolist())

    return np.array(scores)


# ==========================================
# 1. Error Categorization & Analysis Function
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

    # 1. Export Full Predictions CSV
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

    # 5. Error Breakdown by Synthetic Dataset Source
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

    # 6. High-Confidence Misclassifications
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
# 2. Main Test Evaluation & Plotting Function
# ==========================================
def evaluate_binoculars_test_set(scope: str, args, manager: DetectionDataManager):
    outputs_base = Path(args.outputs_dir) if args.outputs_dir else DEFAULT_OUTPUTS_DIR
    scope_dir = outputs_base / "binoculars" / scope
    scope_dir.mkdir(parents=True, exist_ok=True)

    cache_file = scope_dir / "raw_scores_cache.joblib"

    # Filter test split
    test_df = manager.filter_dataframe(DataFilter(splits=["test"], scopes=[scope]))
    texts = test_df["text"].tolist()
    labels = test_df["label"].values

    # Step 1: Compute or Load Cached Scores
    if cache_file.exists() and not args.recompute:
        print(f"[CACHE LOADED] Loading cached Binoculars raw scores from: {cache_file}")
        raw_scores = joblib.load(cache_file)
    else:
        if HAS_BINOCULARS_LIB:
            print("[BINOCULARS] Using installed Binoculars package...")
            bino = OfficialBinoculars(observer_name_or_path=args.observer_model, performer_name_or_path=args.performer_model)
            # Compute raw score (negated so higher value = higher likelihood of AI)
            raw_scores = np.array([-bino.compute_score(t) for t in tqdm(texts, desc="Binoculars Scoring")])
        else:
            raw_scores = compute_binoculars_scores(
                texts=texts,
                observer_model_name=args.observer_model,
                performer_model_name=args.performer_model,
                device=args.device,
                batch_size=args.batch_size,
            )
        joblib.dump(raw_scores, cache_file)
        print(f"[CACHE SAVED] Raw scores saved to: {cache_file}")

    # Standardize continuous scores to P(LLM) in range [0, 1] using standard sigmoid scaling
    z_scores = (raw_scores - np.mean(raw_scores)) / (np.std(raw_scores) + 1e-8)
    probs_llm = expit(z_scores)

    # Compute optimal threshold based on ROC curve (Youden's J statistic)
    fpr_tmp, tpr_tmp, thresholds_tmp = roc_curve(labels, probs_llm)
    optimal_idx = np.argmax(tpr_tmp - fpr_tmp)
    optimal_threshold = thresholds_tmp[optimal_idx]

    print("\n" + "=" * 70)
    print(f" EVALUATING BINOCULARS DETECTOR [{scope.upper()} SCOPE] ")
    print(f" Optimal Threshold (\u03c4*): {optimal_threshold:.6f}")
    print(f" Test Set Size           : {len(labels)} samples")
    print("=" * 70)

    # Inference predictions
    preds = (probs_llm >= optimal_threshold).astype(int)

    # Metrics
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

    # Generator breakdown
    gen_col = "model_name" if "model_name" in test_df.columns else ("generator_model" if "generator_model" in test_df.columns else None)
    per_model_results = []

    if gen_col and len(test_df) == len(labels):
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

    # Error Analysis and Export
    df_analysis = test_df.copy()
    df_analysis["prob_llm"] = probs_llm
    df_analysis["pred_llm"] = preds
    analyze_and_save_errors(df_analysis, scope_dir=scope_dir, scope=scope)

    # 4-Panel Publication Plot (300 DPI)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    plt.subplots_adjust(wspace=0.25, hspace=0.3)

    # Panel A: ROC Curve
    axes[0, 0].plot(fpr, tpr, color="#2b5c8f", lw=2, label=f"Binoculars ({scope.upper()})\npAUC@0.01={pauc_001_val:.4f}")
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

    # LaTeX Table Export
    latex_table_path = scope_dir / "test_metrics_table.tex"
    with open(latex_table_path, "w") as f:
        f.write("% Auto-generated LaTeX table for Binoculars test evaluation\n")
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write(f"\\caption{{Performance Evaluation of Binoculars Detector ({scope.upper()}) on Held-out Test Set. Threshold $\\tau^* = {optimal_threshold:.4f}$.}}\n")
        f.write("\\label{tab:binoculars_" + scope + "_test_results}\n")
        f.write("\\begin{tabular}{lcccccc}\n\\hline\n")
        f.write("\\textbf{Evaluation Group} & \\textbf{pAUC @ 0.01} & \\textbf{ROC-AUC} & \\textbf{Accuracy} & \\textbf{F1-Score} & \\textbf{TPR @ 1\\% FPR} \\\\\n\\hline\n")
        f.write(f"Overall Test Set & {pauc_001_val:.4f} & {roc_auc_val:.4f} & {acc:.4f} & {f1:.4f} & {tpr_at_1fpr:.4f} \\\\\n")
        f.write("\\hline\n")
        for row in per_model_results:
            f.write(f"vs. {row['Generator Model']} & -- & {row['ROC-AUC']:.4f} & {row['Accuracy']:.4f} & {row['F1-Score']:.4f} & -- \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

    print(f"[LATEX SAVED] Copy-pasteable LaTeX table saved to: '{latex_table_path}'")

    # Save JSON Summary
    summary_path = scope_dir / "test_paper_evaluation_summary.json"
    json_data = {
        "overall_test_metrics": overall_metrics,
        "per_model_metrics": per_model_results,
        "observer_model": args.observer_model,
        "performer_model": args.performer_model,
    }
    with open(summary_path, "w") as f:
        json.dump(json_data, f, indent=4)

    print(f"[JSON SAVED] Test evaluation summary saved to: '{summary_path}'\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Binoculars on held-out test split.")
    parser.add_argument("--scopes", nargs="+", default=["sentence", "full"], choices=["sentence", "full"])
    parser.add_argument("--outputs_dir", type=str, default=None)
    parser.add_argument("--observer_model", type=str, default="tiiuae/falcon-7b", help="Observer model for Binoculars")
    parser.add_argument("--performer_model", type=str, default="tiiuae/falcon-7b-instruct", help="Performer model for Binoculars")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for model inference")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--recompute", action="store_true", help="Force recomputation of raw scores instead of loading cached scores.")

    args = parser.parse_args()
    manager = DetectionDataManager()

    for scope in args.scopes:
        evaluate_binoculars_test_set(scope=scope, args=args, manager=manager)

    print("=" * 70)
    print("[ALL DONE] Binoculars evaluation complete for all requested scopes!")
    print("=" * 70)


if __name__ == "__main__":
    import torch
    main()