import json
import os
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
)

FILE_PATH = "/home/gderijck/detection/output/fdgpt/sentence/test_results.json"


def calculate_fpr_at_tpr(y_true, y_scores, target_tpr=0.95):
    """Calculates False Positive Rate (FPR) at a fixed True Positive Rate (TPR)."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    idx = np.where(tpr >= target_tpr)[0]
    if len(idx) == 0:
        return 1.0
    return fpr[idx[0]]


def evaluate_subset(y_true, y_pred, y_prob):
    """Computes all summary metrics for a given subset."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    acc = accuracy_score(y_true, y_pred) * 100
    prec = precision_score(y_true, y_pred, zero_division=0) * 100
    rec = recall_score(y_true, y_pred, zero_division=0) * 100
    spec = (tn / (tn + fp) * 100) if (tn + fp) > 0 else float("nan")
    f1 = f1_score(y_true, y_pred, zero_division=0) * 100

    # AUROC & AUPRC (require both classes to be present)
    unique_classes = np.unique(y_true)
    if len(unique_classes) > 1:
        auroc = roc_auc_score(y_true, y_prob) * 100
        auprc = average_precision_score(y_true, y_prob) * 100
        fpr95 = calculate_fpr_at_tpr(y_true, y_prob, 0.95) * 100
    else:
        auroc, auprc, fpr95 = float("nan"), float("nan"), float("nan")

    return {
        "N": len(y_true),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "Specificity": spec,
        "F1": f1,
        "AUROC": auroc,
        "AUPRC": auprc,
        "FPR95": fpr95,
    }


def format_num(val):
    if np.isnan(val):
        return "--"
    return f"{val:.2f}"


def main():
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"Cannot find results file: {FILE_PATH}")

    with open(FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    predictions = data["predictions"] if "predictions" in data else data

    y_true = np.array([p["label"] for p in predictions])
    y_pred = np.array([p["predicted_label"] for p in predictions])
    
    # Use calibrated probability if available; otherwise fall back to raw score
    y_prob = np.array([
        p.get("calibrated_ai_probability", p.get("raw_discrepancy_score", 0.0))
        for p in predictions
    ])

    # 1. Overall Metrics
    overall = evaluate_subset(y_true, y_pred, y_prob)

    # -------------------------------------------------------------
    # LaTeX Table 1: Main Overall Summary Table
    # -------------------------------------------------------------
    latex_overall = rf"""
\begin{{table}}[htbp]
\centering
\caption{{Fast-DetectGPT Overall Sentence-Level Detection Performance on Test Set}}
\label{{tab:fdgpt_overall_results}}
\begin{{tabular}}{{lrrrrrrrr}}
\toprule
\textbf{{Total ($N$)}} & \textbf{{Acc (\%)}} & \textbf{{Prec (\%)}} & \textbf{{Rec (\%)}} & \textbf{{Spec (\%)}} & \textbf{{$F_1$ (\%)}} & \textbf{{AUROC (\%)}} & \textbf{{AUPRC (\%)}} & \textbf{{FPR@95\% (\%)}} \\
\midrule
{overall['N']} & {format_num(overall['Accuracy'])} & {format_num(overall['Precision'])} & {format_num(overall['Recall'])} & {format_num(overall['Specificity'])} & {format_num(overall['F1'])} & {format_num(overall['AUROC'])} & {format_num(overall['AUPRC'])} & {format_num(overall['FPR95'])} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""

    # -------------------------------------------------------------
    # LaTeX Table 2: Breakdown by Metadata (Source / Generation Type)
    # -------------------------------------------------------------
    breakdown_rows = []
    
    for category in ["source", "model_name", "generation_type"]:
        groups = {}
        for p in predictions:
            key = p.get("metadata", {}).get(category, "unknown")
            groups.setdefault(key, []).append(p)
        
        breakdown_rows.append(f"\\midrule\n\\multicolumn{{7}}{{l}}{{\\textbf{{By {category.replace('_', ' ').title()}}}}} \\\\")
        
        for group_name, group_preds in groups.items():
            g_true = np.array([p["label"] for p in group_preds])
            g_pred = np.array([p["predicted_label"] for p in group_preds])
            g_prob = np.array([
                p.get("calibrated_ai_probability", p.get("raw_discrepancy_score", 0.0))
                for p in group_preds
            ])
            m = evaluate_subset(g_true, g_pred, g_prob)
            breakdown_rows.append(
                f"\\quad {group_name} & {m['N']} & {format_num(m['Accuracy'])} & {format_num(m['Precision'])} & {format_num(m['Recall'])} & {format_num(m['F1'])} & {format_num(m['AUROC'])} \\\\"
            )

    breakdown_str = "\n".join(breakdown_rows)

    latex_breakdown = rf"""
\begin{{table}}[htbp]
\centering
\caption{{Fast-DetectGPT Detection Performance Breakdown by Metadata Subgroups}}
\label{{tab:fdgpt_breakdown_results}}
\begin{{tabular}}{{lrrrrrr}}
\toprule
\textbf{{Subset}} & \textbf{{$N$}} & \textbf{{Acc (\%)}} & \textbf{{Prec (\%)}} & \textbf{{Rec (\%)}} & \textbf{{$F_1$ (\%)}} & \textbf{{AUROC (\%)}} \\
{breakdown_str}
\bottomrule
\end{{tabular}}
\end{{table}}
"""

    print("\n" + "=" * 60)
    print("TABLE 1: OVERALL PERFORMANCE")
    print("=" * 60)
    print(latex_overall)

    print("\n" + "=" * 60)
    print("TABLE 2: DETAILED SUBGROUP BREAKDOWN")
    print("=" * 60)
    print(latex_breakdown)

    # Save to .tex files
    with open("table_overall.tex", "w") as f:
        f.write(latex_overall.strip())
    with open("table_breakdown.tex", "w") as f:
        f.write(latex_breakdown.strip())
    
    print("\nSaved tables to 'table_overall.tex' and 'table_breakdown.tex'.")


if __name__ == "__main__":
    main()