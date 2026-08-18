import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_tpr_at_fpr(
    y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.01
) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    valid_tprs = tpr[fpr <= target_fpr]
    return float(valid_tprs[-1]) if len(valid_tprs) > 0 else 0.0


def generate_latex_table(json_path: str, output_tex_path: str = None) -> str:
    # 1. Load JSON
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.json_normalize(data["predictions"])

    y_true = df["label"].values
    y_pred = df["predicted_label"].values
    y_prob = (
        df["calibrated_ai_probability"].values
        if "calibrated_ai_probability" in df
        else df["raw_discrepancy_score"].values
    )

    # 2. Compute Global Metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total_samples = len(y_true)
    num_human = int((y_true == 0).sum())
    num_ai = int((y_true == 1).sum())

    acc = accuracy_score(y_true, y_pred) * 100
    bal_acc = balanced_accuracy_score(y_true, y_pred) * 100
    prec = precision_score(y_true, y_pred, zero_division=0) * 100
    rec = (
        recall_score(y_true, y_pred, zero_division=0) * 100
    )  # TPR / Sensitivity
    spec = (tn / (tn + fp) * 100) if (tn + fp) > 0 else 0.0  # TNR
    fpr_val = (fp / (fp + tn) * 100) if (fp + tn) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0) * 100
    f1_macro = (
        f1_score(y_true, y_pred, average="macro", zero_division=0) * 100
    )

    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    try:
        pauc_1pct = roc_auc_score(y_true, y_prob, max_fpr=0.01)
    except Exception:
        pauc_1pct = 0.0

    tpr_at_01 = compute_tpr_at_fpr(y_true, y_prob, target_fpr=0.001) * 100
    tpr_at_1 = compute_tpr_at_fpr(y_true, y_prob, target_fpr=0.01) * 100
    tpr_at_5 = compute_tpr_at_fpr(y_true, y_prob, target_fpr=0.05) * 100

    # 3. Model Breakdown (Panel B)
    breakdown_rows = []
    if "metadata.model_name" in df.columns:
        for model_name, group in df.groupby("metadata.model_name"):
            g_true = group["label"].values
            g_pred = group["predicted_label"].values
            g_total = len(group)
            g_acc = accuracy_score(g_true, g_pred) * 100
            g_f1 = f1_score(g_true, g_pred, average="macro", zero_division=0) * 100
            g_mean_prob = (
                group["calibrated_ai_probability"].mean()
                if "calibrated_ai_probability" in group
                else np.nan
            )

            # Clean LaTeX name
            clean_name = str(model_name).replace("_", "\\_")
            breakdown_rows.append(
                f"{clean_name} & {g_total:,} & {g_acc:.2f}\\% & {g_f1:.2f}\\% & {g_mean_prob:.4f} \\\\"
            )

    model_breakdown_latex = "\n".join(breakdown_rows)

    # 4. Construct LaTeX Table String
    latex_code = f"""\\begin{{table}}[H]
\\centering
\\small
\\caption{{Fast-DetectGPT Sentence-Level Test Performance Overview.}}
\\label{{tab:fdgpt_sentence_performance}}
\\begin{{tabular*}}{{\\linewidth}}{{@{{\\extracolsep{{\\fill}}}} l r r r @{{}}}}
\\toprule
\\multicolumn{{4}}{{l}}{{\\textbf{{Panel A: Overall Classification Metrics}}}} \\\\
\\midrule
\\textbf{{Metric}} & \\textbf{{Value}} & \\textbf{{Metric}} & \\textbf{{Value}} \\\\
\\midrule
Accuracy & {acc:.2f}\\% & Precision (AI) & {prec:.2f}\\% \\\\
Balanced Accuracy & {bal_acc:.2f}\\% & Recall / TPR (AI) & {rec:.2f}\\% \\\\
F1-Score (AI) & {f1:.2f}\\% & Specificity (TNR) & {spec:.2f}\\% \\\\
F1-Score (Macro) & {f1_macro:.2f}\\% & False Positive Rate (FPR) & {fpr_val:.2f}\\% \\\\
\\midrule
\\multicolumn{{4}}{{l}}{{\\textbf{{Panel B: Ranking and Low-FPR Detection Metrics}}}} \\\\
\\midrule
ROC-AUC & {roc_auc:.4f} & TPR @ 0.1\\% FPR & {tpr_at_01:.2f}\\% \\\\
PR-AUC (Avg. Precision) & {pr_auc:.4f} & TPR @ 1.0\\% FPR & {tpr_at_1:.2f}\\% \\\\
pAUC (FPR $\\le$ 1\\%) & {pauc_1pct:.4f} & TPR @ 5.0\\% FPR & {tpr_at_5:.2f}\\% \\\\
\\midrule
\\multicolumn{{4}}{{l}}{{\\textbf{{Panel C: Breakdown by Generator / Source Model}}}} \\\\
\\midrule
\\textbf{{Generator / Model}} & \\textbf{{Samples ($N$)}} & \\textbf{{Accuracy}} & \\textbf{{Macro F1}} \\\\
\\midrule
{model_breakdown_latex}
\\bottomrule
\\end{{tabular*}}
\\end{{table}}"""

    print(latex_code)

    if output_tex_path:
        with open(output_tex_path, "w", encoding="utf-8") as f:
            f.write(latex_code)
        print(f"\nSaved LaTeX code to {output_tex_path}")

    return latex_code


if __name__ == "__main__":
    JSON_PATH = (
        "/home/gderijck/detection/output/fdgpt/sentence/test_results.json"
    )
    generate_latex_table(JSON_PATH, output_tex_path="fdgpt_sentence_table.tex")