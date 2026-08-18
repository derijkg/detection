#!/usr/bin/env python3
"""
Generates an exact-format LaTeX comparison table from mDeBERTa predictions
to match the SVM baseline performance table.
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

def compute_subset_metrics(df_subset, human_df, calibrated_tau=0.5, is_human_only=False):
    """
    Computes Mean P(LLM), ROC-AUC, Accuracy, F1, Precision, and Recall.
    For AI subsets, metrics are evaluated against the Human baseline.
    """
    n_samples = len(df_subset)
    mean_p_llm = float(np.mean(df_subset["prob_ai"]))

    if is_human_only:
        preds = (df_subset["prob_ai"] >= calibrated_tau).astype(int)
        fp = int((preds == 1).sum())
        tn = int((preds == 0).sum())
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return {
            "Samples ($N$)": f"{n_samples:,}",
            "Mean $P(\\text{LLM})$": f"{mean_p_llm:.4f}",
            "ROC-AUC": "--",
            "Accuracy": f"{specificity:.4f}",
            "F1-Score": "--",
            "Precision": "--",
            "Recall": f"{fpr:.4f}$^\\dagger$",
        }

    # If it's an overall or AI subset, pair with Human test samples to compute binary metrics
    if "label" in df_subset.columns and len(df_subset["label"].unique()) > 1:
        eval_df = df_subset
    else:
        eval_df = pd.concat([human_df, df_subset]).reset_index(drop=True)

    labels = eval_df["label"].values
    probs = eval_df["prob_ai"].values
    preds = (probs >= calibrated_tau).astype(int)

    try:
        roc_auc = roc_auc_score(labels, probs)
        roc_auc_str = f"{roc_auc:.4f}"
    except Exception:
        roc_auc_str = "--"

    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)

    return {
        "Samples ($N$)": f"{n_samples:,}",
        "Mean $P(\\text{LLM})$": f"{mean_p_llm:.4f}",
        "ROC-AUC": roc_auc_str,
        "Accuracy": f"{acc:.4f}",
        "F1-Score": f"{f1:.4f}",
        "Precision": f"{prec:.4f}",
        "Recall": f"{rec:.4f}",
    }


def generate_comparison_table(csv_path: str, summary_json_path: str, output_tex_path: str = "deberta_svm_comparison.tex"):
    df = pd.read_csv(csv_path)

    # Read calibrated tau from json summary
    if os.path.exists(summary_json_path):
        import json
        with open(summary_json_path, "r") as f:
            summary = json.load(f)
            calibrated_tau = summary["overall_metrics"].get("Calibrated tau", 0.5)
    else:
        calibrated_tau = 0.5

    human_df = df[df["label"] == 0]
    ai_df = df[df["label"] == 1]

    # Detect model and substitution columns
    model_col = next((c for c in ["model_name", "generator_model", "generator"] if c in df.columns), None)
    ratio_col = next((c for c in ["substitution_ratio", "replacement_ratio", "ratio", "perturbed_ratio"] if c in df.columns), None)

    table_data = []

    # 1. Overall System Performance
    table_data.append(("SECTION", "Overall System Performance"))
    table_data.append(("Overall Test Set", compute_subset_metrics(df, human_df, calibrated_tau)))

    # Clean Abstracts (0% / 100% full rewrites)
    if ratio_col:
        clean_ai = ai_df[ai_df[ratio_col] >= 0.99]
        clean_df = pd.concat([human_df, clean_ai]).reset_index(drop=True)
        table_data.append(("Full Clean Abstracts (0\\% / 100\\%)", compute_subset_metrics(clean_df, human_df, calibrated_tau)))
    else:
        # If no explicit ratio column, treat non-synthetic as clean
        clean_df = df[df[model_col] != "synthetic_multi"] if model_col else df
        table_data.append(("Full Clean Abstracts (0\\% / 100\\%)", compute_subset_metrics(clean_df, human_df, calibrated_tau)))

    # 2. Human Baseline
    table_data.append(("SECTION", "Human Baseline"))
    table_data.append(("Human Text", compute_subset_metrics(human_df, human_df, calibrated_tau, is_human_only=True)))

    # 3. Bucketed Substitution Ratios
    if ratio_col:
        table_data.append(("SECTION", "Bucketed Substitution Ratios"))
        for r_val, r_label in [(0.25, "25\\% Substitution"), (0.50, "50\\% Substitution"), (0.75, "75\\% Substitution"), (1.00, "100\\% (Full Rewrite)")]:
            sub_r = ai_df[np.isclose(ai_df[ratio_col], r_val, atol=0.05)]
            if len(sub_r) > 0:
                table_data.append((r_label, compute_subset_metrics(sub_r, human_df, calibrated_tau)))

    # 4. Breakdown by Generator Model
    if model_col:
        table_data.append(("SECTION", "Breakdown by Generator Model"))
        for gen in sorted(ai_df[model_col].unique()):
            gen_df = ai_df[ai_df[model_col] == gen]
            gen_name_escaped = str(gen).replace("_", r"\_")
            table_data.append((gen_name_escaped, compute_subset_metrics(gen_df, human_df, calibrated_tau)))

    # Build LaTeX Table
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\caption{mDeBERTa-v3 test set performance for full abstracts ($\tau = " + f"{calibrated_tau:.4f}" + r"$).}",
        r"\label{tab:deberta_full_performance}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"\textbf{Category / Subset} & \textbf{Samples ($N$)} & \textbf{Mean $P(\text{LLM})$} & \textbf{ROC-AUC} & \textbf{Accuracy} & \textbf{F1-Score} & \textbf{Precision} & \textbf{Recall} \\",
        r"\midrule",
    ]

    for item in table_data:
        if item[0] == "SECTION":
            lines.append(rf"\multicolumn{{8}}{{l}}{{\textbf{{{item[1]}}}}} \\")
        else:
            name, m = item
            row = f"{name} & {m['Samples ($N$)']} & {m['Mean $P(\\text{LLM})$']} & {m['ROC-AUC']} & {m['Accuracy']} & {m['F1-Score']} & {m['Precision']} & {m['Recall']} \\\\"
            lines.append(row)
            if "Clean Abstracts" in name or "Human Text" in name or "100%" in name:
                lines.append(r"\midrule")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\vspace{3pt}",
        r"\hfill \footnotesize\textit{$^\dagger$ Note: For Human Text, Recall represents the False Positive Rate (FPR), and Accuracy represents Specificity (TNR).}",
        r"\end{table}",
    ])

    full_tex = "\n".join(lines)
    with open(output_tex_path, "w") as f:
        f.write(full_tex)

    print("\n" + "=" * 80)
    print("                PUBLICATION LATEX COMPARISON TABLE")
    print("=" * 80)
    print(full_tex)
    print("=" * 80)
    print(f"\nLaTeX table saved to: {output_tex_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="./outputs_imbalanced/deberta_full_fast/predictions_test_standard.csv")
    parser.add_argument("--json_path", type=str, default="./outputs_imbalanced/deberta_full_fast/evaluation_summary_test_standard.json")
    parser.add_argument("--output_tex", type=str, default="deberta_svm_comparison.tex")
    args = parser.parse_args()

    generate_comparison_table(args.csv_path, args.json_path, args.output_tex)