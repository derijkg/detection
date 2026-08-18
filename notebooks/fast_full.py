import argparse
import json
import os
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

DEFAULT_PATH = "/home/gderijck/detection/output/fdgpt/abstract/test_results.json"


def format_val(val, is_dagger=False):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "--"
    res = f"{val:.4f}"
    if is_dagger:
        res += r"$^\dagger$"
    return res


def escape_latex(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%")


def evaluate_set(y_true, y_pred, y_prob):
    """Evaluates metrics on a binary dataset."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    except Exception:
        auc = np.nan

    return {
        "N": len(y_true),
        "mean_p": np.mean(y_prob),
        "auc": auc,
        "acc": acc,
        "f1": f1,
        "prec": prec,
        "rec": rec,
    }


def evaluate_ai_subset(pos_samples, human_samples):
    """Evaluates an AI-positive subset against the human baseline negatives."""
    n_pos = len(pos_samples)
    if n_pos == 0:
        return None

    combined = pos_samples + human_samples
    y_true = np.array([p["label"] for p in combined])
    y_pred = np.array([p["predicted_label"] for p in combined])
    y_prob = np.array([
        p.get("calibrated_ai_probability", p.get("raw_discrepancy_score", 0.0))
        for p in combined
    ])

    m = evaluate_set(y_true, y_pred, y_prob)
    # The reported Mean P(LLM) represents the positive slice itself
    pos_probs = [
        p.get("calibrated_ai_probability", p.get("raw_discrepancy_score", 0.0))
        for p in pos_samples
    ]
    m["mean_p"] = np.mean(pos_probs)
    m["N"] = n_pos
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file", type=str, default=DEFAULT_PATH, help="Path to test_results.json"
    )
    parser.add_argument(
        "--out", type=str, default="table_abstract_performance.tex", help="Output .tex file"
    )
    args = parser.parse_args()

    if not os.path.exists(args.file):
        raise FileNotFoundError(f"Results file not found: {args.file}")

    with open(args.file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    predictions = raw_data.get("predictions", raw_data)

    # 1. Separate Human and AI predictions
    human_samples = [
        p for p in predictions
        if p.get("label") == 0 or p.get("metadata", {}).get("model_name") == "human" or p.get("metadata", {}).get("llm_ratio") == 0.0
    ]
    ai_samples = [
        p for p in predictions
        if p.get("label") == 1 and p not in human_samples
    ]

    # ---------------------------------------------------------
    # 2. Overall Performance Metrics
    # ---------------------------------------------------------
    y_true_all = np.array([p["label"] for p in predictions])
    y_pred_all = np.array([p["predicted_label"] for p in predictions])
    y_prob_all = np.array([
        p.get("calibrated_ai_probability", p.get("raw_discrepancy_score", 0.0))
        for p in predictions
    ])
    overall_all = evaluate_set(y_true_all, y_pred_all, y_prob_all)

    # Full Clean Abstracts (0% / 100% LLM ratio)
    clean_samples = [
        p for p in predictions
        if p.get("metadata", {}).get("llm_ratio") in [0.0, 1.0]
        or p.get("metadata", {}).get("generation_type") in ["human_single", "full_rewrite"]
    ]
    if clean_samples:
        y_true_cl = np.array([p["label"] for p in clean_samples])
        y_pred_cl = np.array([p["predicted_label"] for p in clean_samples])
        y_prob_cl = np.array([
            p.get("calibrated_ai_probability", p.get("raw_discrepancy_score", 0.0))
            for p in clean_samples
        ])
        overall_clean = evaluate_set(y_true_cl, y_pred_cl, y_prob_cl)
    else:
        overall_clean = None

    # ---------------------------------------------------------
    # 3. Human Baseline Row
    # ---------------------------------------------------------
    if human_samples:
        h_true = np.array([p["label"] for p in human_samples])
        h_pred = np.array([p["predicted_label"] for p in human_samples])
        h_prob = np.array([
            p.get("calibrated_ai_probability", p.get("raw_discrepancy_score", 0.0))
            for p in human_samples
        ])
        h_cm = confusion_matrix(h_true, h_pred, labels=[0, 1])
        tn, fp, fn, tp = h_cm.ravel() if h_cm.shape == (2, 2) else (len(human_samples), 0, 0, 0)

        human_acc = tn / len(human_samples) if len(human_samples) > 0 else 0.0  # Specificity (TNR)
        human_fpr = fp / len(human_samples) if len(human_samples) > 0 else 0.0  # FPR (reported in Recall column)
        human_mean_p = float(np.mean(h_prob))
        human_n = len(human_samples)
    else:
        human_n, human_mean_p, human_acc, human_fpr = 0, np.nan, np.nan, np.nan

    # ---------------------------------------------------------
    # 4. Bucketed Substitution Ratios
    # ---------------------------------------------------------
    buckets = [
        ("25\\% Substitution", lambda r: 0.15 <= r <= 0.35),
        ("50\\% Substitution", lambda r: 0.40 <= r <= 0.60),
        ("75\\% Substitution", lambda r: 0.65 <= r <= 0.85),
        ("100\\% (Full Rewrite)", lambda r: r >= 0.95),
    ]
    bucket_results = []
    for label_name, condition in buckets:
        subset = [
            p for p in ai_samples
            if condition(p.get("metadata", {}).get("llm_ratio", -1.0))
        ]
        if subset:
            m = evaluate_ai_subset(subset, human_samples)
            bucket_results.append((label_name, m))

    # ---------------------------------------------------------
    # 5. Breakdown by Generator Model
    # ---------------------------------------------------------
    model_groups = {}
    for p in ai_samples:
        m_name = p.get("metadata", {}).get("model_name", "unknown")
        model_groups.setdefault(m_name, []).append(p)

    model_results = []
    for m_name in sorted(model_groups.keys()):
        subset = model_groups[m_name]
        m = evaluate_ai_subset(subset, human_samples)
        model_results.append((escape_latex(m_name), m))

    # ---------------------------------------------------------
    # 6. Build LaTeX Table
    # ---------------------------------------------------------
    def format_row(name, m, rec_dagger=False):
        return (
            f"{name} & {m['N']:,} & {format_val(m['mean_p'])} & {format_val(m['auc'])} & "
            f"{format_val(m['acc'])} & {format_val(m['f1'])} & {format_val(m['prec'])} & "
            f"{format_val(m['rec'], is_dagger=rec_dagger)} \\\\"
        )

    rows = []

    # Section 1
    rows.append(r"\multicolumn{8}{l}{\textbf{Overall System Performance}} \\")
    rows.append(format_row("Overall Test Set", overall_all))
    if overall_clean:
        rows.append(format_row(r"Full Clean Abstracts (0\% / 100\%)", overall_clean))
    rows.append(r"\midrule")

    # Section 2
    rows.append(r"\multicolumn{8}{l}{\textbf{Human Baseline}} \\")
    rows.append(
        f"Human Text & {human_n:,} & {format_val(human_mean_p)} & -- & "
        f"{format_val(human_acc)} & -- & -- & {format_val(human_fpr, is_dagger=True)} \\\\"
    )
    rows.append(r"\midrule")

    # Section 3
    rows.append(r"\multicolumn{8}{l}{\textbf{Bucketed Substitution Ratios}} \\")
    for name, m in bucket_results:
        rows.append(format_row(name, m))
    rows.append(r"\midrule")

    # Section 4
    rows.append(r"\multicolumn{8}{l}{\textbf{Breakdown by Generator Model}} \\")
    for name, m in model_results:
        rows.append(format_row(name, m))

    body = "\n".join(rows)

    latex_table = rf"""\begin{{table}}[htbp]
\centering
\small
\setlength{{\tabcolsep}}{{3.5pt}}
\caption{{Fast-DetectGPT test set performance for full abstracts across substitution ratios and generator models.}}
\label{{tab:abstract_substitution_performance}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lrrrrrrr}}
\toprule
\textbf{{Category / Subset}} & \textbf{{Samples ($N$)}} & \textbf{{Mean $P(\text{{LLM}})$}} & \textbf{{ROC-AUC}} & \textbf{{Accuracy}} & \textbf{{F1-Score}} & \textbf{{Precision}} & \textbf{{Recall}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}%
}}
\vspace{{3pt}}
\hfill \footnotesize\textit{{$^\dagger$ Note: For Human Text, Recall represents the False Positive Rate (FPR), and Accuracy represents Specificity (TNR).}}
\end{{table}}
"""

    print(latex_table)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(latex_table.strip())
    print(f"\nSuccessfully generated and saved LaTeX table to: {args.out}")


if __name__ == "__main__":
    main()