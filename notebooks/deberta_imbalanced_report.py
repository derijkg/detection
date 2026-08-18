#!/usr/bin/env python3
"""
Robust Comparative LaTeX Table Generator: Full Abstract SVM vs. mDeBERTa-v3
Auto-detects substitution ratios and calculates exact subset performance metrics.
"""

import argparse
import json
import os
import re
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

# -------------------------------------------------------------------------
# 1. FIXED SVM BENCHMARK NUMBERS (From your paper)
# -------------------------------------------------------------------------
SVM_BENCHMARK = {
    "Overall Test Set": {
        "MeanP": 0.5222, "AUC": 0.8494, "Acc": 0.4556, "F1": 0.5560, "Prec": 0.9960, "Rec": 0.3856, "N": 14707
    },
    "Full Abstracts (0% / 100%)": {
        "MeanP": 0.7139, "AUC": 0.9917, "Acc": 0.9162, "F1": 0.9381, "Prec": 0.9948, "Rec": 0.8875, "N": 5992
    },
    "Human Text (0% LLM)": {
        "MeanP": 0.1079, "AUC": None, "Acc": 0.9883, "F1": None, "Prec": None, "Rec": 0.0117, "N": 1709
    },
    "25% Substitution": {
        "MeanP": 0.2211, "AUC": 0.6640, "Acc": 0.3836, "F1": 0.0779, "Prec": 0.8601, "Rec": 0.0408, "N": 3015
    },
    "50% Substitution": {
        "MeanP": 0.3933, "AUC": 0.8009, "Acc": 0.4330, "F1": 0.2089, "Prec": 0.9464, "Rec": 0.1174, "N": 3007
    },
    "75% Substitution": {
        "MeanP": 0.5771, "AUC": 0.8850, "Acc": 0.5507, "F1": 0.4263, "Prec": 0.9735, "Rec": 0.2729, "N": 2693
    },
    "100% (Full Rewrite)": {
        "MeanP": 0.9556, "AUC": 0.9917, "Acc": 0.9162, "F1": 0.9381, "Prec": 0.9948, "Rec": 0.8875, "N": 4283
    },
    "Syn_25% Substitution": {
        "MeanP": 0.1950, "AUC": 0.6364, "Acc": 0.5117, "F1": 0.0671, "Prec": 0.7500, "Rec": 0.0351, "N": 1709
    },
    "Syn_50% Substitution": {
        "MeanP": 0.3220, "AUC": 0.7611, "Acc": 0.5310, "F1": 0.1358, "Prec": 0.8630, "Rec": 0.0737, "N": 1709
    },
    "Syn_75% Substitution": {
        "MeanP": 0.4855, "AUC": 0.8535, "Acc": 0.5781, "F1": 0.2847, "Prec": 0.9349, "Rec": 0.1679, "N": 1709
    },
}


# -------------------------------------------------------------------------
# 2. INTELLIGENT RATIO & METADATA EXTRACTION
# -------------------------------------------------------------------------
def extract_and_normalize_ratio(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Finds or parses substitution ratios (0.25, 0.50, 0.75, 1.00) from any column."""
    df_clean = df.copy()

    # 1. Search for ratio column by keywords
    ratio_candidates = [
        "substitution_ratio", "replacement_ratio", "ratio", "perturbed_ratio",
        "sub_ratio", "pct_replaced", "edit_ratio", "perturbation_ratio", "fraction"
    ]
    found_col = next((c for c in df_clean.columns if any(k in c.lower() for k in ratio_candidates)), None)

    if found_col:
        print(f"[INFO] Detected substitution ratio column: '{found_col}'")
        df_clean["_norm_ratio"] = (
            df_clean[found_col]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.strip()
        )
        df_clean["_norm_ratio"] = pd.to_numeric(df_clean["_norm_ratio"], errors="coerce")
        
        # Scale down if in 0-100 percentage format
        if df_clean["_norm_ratio"].dropna().max() > 1.5:
            df_clean["_norm_ratio"] = df_clean["_norm_ratio"] / 100.0
        return df_clean, "_norm_ratio"

    # 2. Fallback: Parse ratio from metadata strings (_id, model_name, etc.)
    print("[INFO] Searching for substitution ratios inside text/ID metadata...")
    inferred_ratios = []
    for idx, row in df_clean.iterrows():
        meta_str = " ".join([str(val) for val in row.values])
        if re.search(r"(\b25%|\b0\.25\b|_25_|_25\b)", meta_str):
            inferred_ratios.append(0.25)
        elif re.search(r"(\b50%|\b0\.50\b|\b0\.5\b|_50_|_50\b)", meta_str):
            inferred_ratios.append(0.50)
        elif re.search(r"(\b75%|\b0\.75\b|_75_|_75\b)", meta_str):
            inferred_ratios.append(0.75)
        elif re.search(r"(\b100%|\b1\.0\b|_100_|_100\b|clean|rewrite)", meta_str):
            inferred_ratios.append(1.00)
        else:
            inferred_ratios.append(1.00 if row.get("label", 0) == 1 else 0.0)

    df_clean["_norm_ratio"] = inferred_ratios
    return df_clean, "_norm_ratio"


# ---------------------------------------------------------
# 3. METRIC COMPUTATION & COMPARISON FORMATTER
# ---------------------------------------------------------
def compute_metrics_dict(
    eval_df: pd.DataFrame,
    sub_ai_df: pd.DataFrame,
    human_df: pd.DataFrame,
    tau: float,
    is_overall: bool = False,
    is_human: bool = False
) -> Dict[str, Any]:
    if is_human:
        n_samples = len(human_df)
        mean_p = float(np.mean(human_df["prob_ai"]))
        preds = (human_df["prob_ai"] >= tau).astype(int)
        tn = int((preds == 0).sum())
        fp = int((preds == 1).sum())
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        return {"MeanP": mean_p, "AUC": None, "Acc": spec, "F1": None, "Prec": None, "Rec": fpr, "N": n_samples}

    n_samples = len(eval_df) if is_overall else len(sub_ai_df)
    mean_p = float(np.mean(eval_df["prob_ai"])) if is_overall else float(np.mean(sub_ai_df["prob_ai"]))

    y_true = eval_df["label"].values.astype(int)
    y_prob = eval_df["prob_ai"].values
    y_pred = (y_prob >= tau).astype(int)

    try:
        auc_val = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc_val = None

    acc_val = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

    return {
        "MeanP": mean_p,
        "AUC": auc_val,
        "Acc": acc_val,
        "F1": float(f1),
        "Prec": float(prec),
        "Rec": float(rec),
        "N": n_samples,
    }


def format_rows(subset_label: str, svm_dict: Dict[str, Any], deb_dict: Dict[str, Any], is_human: bool = False) -> Tuple[str, str]:
    metrics = ["AUC", "Acc", "F1", "Prec", "Rec"]
    svm_fmt = {}
    deb_fmt = {}

    svm_mean_p = f"{svm_dict['MeanP']:.4f}"
    deb_mean_p = f"{deb_dict['MeanP']:.4f}"
    svm_n = f"{svm_dict['N']:,}"
    deb_n = f"{deb_dict['N']:,}"

    for m in metrics:
        s_val = svm_dict.get(m)
        d_val = deb_dict.get(m)

        if s_val is None or d_val is None:
            svm_fmt[m] = "--"
            deb_fmt[m] = "--"
            continue

        s_str = f"{s_val:.4f}"
        d_str = f"{d_val:.4f}"

        if is_human and m == "Rec":
            # For Human Recall (FPR), LOWER is superior
            if d_val < s_val - 1e-4:
                deb_fmt[m] = f"\\textbf{{{d_str}}}" + r"$^\dagger$"
                svm_fmt[m] = s_str + r"$^\dagger$"
            elif s_val < d_val - 1e-4:
                svm_fmt[m] = f"\\textbf{{{s_str}}}" + r"$^\dagger$"
                deb_fmt[m] = d_str + r"$^\dagger$"
            else:
                svm_fmt[m] = s_str + r"$^\dagger$"
                deb_fmt[m] = d_str + r"$^\dagger$"
        else:
            # For standard metrics, HIGHER is superior
            if d_val > s_val + 1e-4:
                deb_fmt[m] = f"\\textbf{{{d_str}}}"
                svm_fmt[m] = s_str
            elif s_val > d_val + 1e-4:
                svm_fmt[m] = f"\\textbf{{{s_str}}}"
                deb_fmt[m] = d_str
            else:
                svm_fmt[m] = s_str
                deb_fmt[m] = d_str

    row_svm = f"{subset_label:<28} & SVM & {svm_mean_p} & {svm_fmt['AUC']} & {svm_fmt['Acc']} & {svm_fmt['F1']} & {svm_fmt['Prec']} & {svm_fmt['Rec']} & {svm_n:>6} \\\\"
    row_deb = f"{'':<28} & mDeBERTa-v3 & {deb_mean_p} & {deb_fmt['AUC']} & {deb_fmt['Acc']} & {deb_fmt['F1']} & {deb_fmt['Prec']} & {deb_fmt['Rec']} & {deb_n:>6} \\\\"

    return row_svm, row_deb


# ---------------------------------------------------------
# 4. MAIN COMPARISON TABLE BUILDER
# ---------------------------------------------------------
def generate_comparative_table(csv_path: str, output_tex_path: str, tau: Optional[float] = None) -> str:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Prediction CSV not found at: {csv_path}")

    raw_df = pd.read_csv(csv_path)
    df, ratio_col = extract_and_normalize_ratio(raw_df)

    human_df = df[df["label"] == 0].reset_index(drop=True)
    ai_df = df[df["label"] == 1].reset_index(drop=True)

    if len(human_df) == 0:
        raise ValueError("The predictions file contains 0 Human (label=0) samples!")

    # Resolve threshold tau
    if tau is None:
        if "calibrated_tau" in df.columns:
            tau = float(df["calibrated_tau"].iloc[0])
        else:
            json_file = os.path.join(os.path.dirname(csv_path), "evaluation_summary_test_standard.json")
            if os.path.exists(json_file):
                with open(json_file, "r") as f:
                    meta = json.load(f)
                tau = float(meta.get("overall_metrics", {}).get("Calibrated tau", 0.5))
            else:
                tau = float(np.quantile(human_df["prob_ai"], 0.99))

    print(f"[INFO] Using Calibrated Threshold: tau = {tau:.6f}")

    model_col = next((c for c in ["model_name", "generator_model", "generator"] if c in df.columns), None)

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.0pt}",
        r"\caption{Comparative detection performance between full abstract SVM and mDeBERTa-v3 models overall and per substitution percentage.}",
        r"\label{tab:svm_vs_deberta_substitution_comparison}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"\textbf{Category / Subset} & \textbf{Model} & \textbf{Mean $P(\text{LLM})$} & \textbf{ROC-AUC} & \textbf{Accuracy} & \textbf{F1-Score} & \textbf{Precision} & \textbf{Recall} & \textbf{Samples ($N$)} \\",
        r"\midrule",
        r"\multicolumn{9}{l}{\textbf{Overall System Performance}} \\",
    ]

    # --- 1. Overall System Performance ---
    deb_overall = compute_metrics_dict(df, ai_df, human_df, tau, is_overall=True)
    r1_svm, r1_deb = format_rows("Overall Test Set", SVM_BENCHMARK["Overall Test Set"], deb_overall)
    lines.extend([r1_svm, r1_deb, r"\addlinespace"])

    # Full Abstracts (0% / 100%) -> 1,709 Human + 4,283 Clean 100% AI = 5,992 Total
    clean_ai = ai_df[np.isclose(ai_df[ratio_col], 1.0, atol=0.06)].reset_index(drop=True)
    if len(clean_ai) == 0 and model_col:
        clean_ai = ai_df[~ai_df[model_col].astype(str).str.contains("synthetic")].reset_index(drop=True)

    clean_total_df = pd.concat([human_df, clean_ai]).reset_index(drop=True)
    deb_clean = compute_metrics_dict(clean_total_df, clean_ai, human_df, tau, is_overall=True)
    r2_svm, r2_deb = format_rows("Full Abstracts (0\\% / 100\\%)", SVM_BENCHMARK["Full Abstracts (0% / 100%)"], deb_clean)
    lines.extend([r2_svm, r2_deb, r"\midrule"])

    # --- 2. Human Baseline ---
    lines.append(r"\multicolumn{9}{l}{\textbf{Human Baseline}} \\")
    deb_human = compute_metrics_dict(df, ai_df, human_df, tau, is_human=True)
    r3_svm, r3_deb = format_rows("Human Text (0\\% LLM)", SVM_BENCHMARK["Human Text (0% LLM)"], deb_human, is_human=True)
    lines.extend([r3_svm, r3_deb, r"\midrule"])

    # --- 3. Overall Substitution Ratios (All Generators) ---
    lines.append(r"\multicolumn{9}{l}{\textbf{Overall Substitution Ratios (All Generators)}} \\")
    sub_configs = [
        (0.25, "25\\% Substitution", "25% Substitution"),
        (0.50, "50\\% Substitution", "50% Substitution"),
        (0.75, "75\\% Substitution", "75% Substitution"),
        (1.00, "100\\% (Full Rewrite)", "100% (Full Rewrite)")
    ]

    for idx, (r_val, r_label, key) in enumerate(sub_configs):
        sub_ai = ai_df[np.isclose(ai_df[ratio_col], r_val, atol=0.06)].reset_index(drop=True)
        eval_sub_df = pd.concat([human_df, sub_ai]).reset_index(drop=True)
        deb_sub = compute_metrics_dict(eval_sub_df, sub_ai, human_df, tau)
        r_svm, r_deb = format_rows(r_label, SVM_BENCHMARK[key], deb_sub)
        lines.extend([r_svm, r_deb])
        if idx < len(sub_configs) - 1:
            lines.append(r"\addlinespace")

    lines.append(r"\midrule")

    # --- 4. Synthetic Substitution Ratios (synthetic_multi) ---
    lines.append(r"\multicolumn{9}{l}{\textbf{Synthetic Substitution Ratios (\texttt{synthetic\_multi})}} \\")
    if model_col:
        syn_ai = ai_df[ai_df[model_col].astype(str).str.contains("synthetic")].reset_index(drop=True)
    else:
        syn_ai = ai_df

    syn_configs = [
        (0.25, "25\\% Substitution", "Syn_25% Substitution"),
        (0.50, "50\\% Substitution", "Syn_50% Substitution"),
        (0.75, "75\\% Substitution", "Syn_75% Substitution")
    ]

    for idx, (r_val, r_label, key) in enumerate(syn_configs):
        sub_syn_ai = syn_ai[np.isclose(syn_ai[ratio_col], r_val, atol=0.06)].reset_index(drop=True)
        eval_syn_df = pd.concat([human_df, sub_syn_ai]).reset_index(drop=True)
        deb_syn = compute_metrics_dict(eval_syn_df, sub_syn_ai, human_df, tau)
        r_svm, r_deb = format_rows(r_label, SVM_BENCHMARK[key], deb_syn)
        lines.extend([r_svm, r_deb])
        if idx < len(syn_configs) - 1:
            lines.append(r"\addlinespace")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\vspace{3pt}",
        r"\hfill \footnotesize\textit{$^\dagger$ Note: For Human Text, Recall represents False Positive Rate (FPR), and Accuracy represents Specificity (TNR). Bold values indicate superior performance between SVM and mDeBERTa-v3.}",
        r"\end{table}",
    ])

    full_tex = "\n".join(lines)
    os.makedirs(os.path.dirname(os.path.abspath(output_tex_path)), exist_ok=True)
    with open(output_tex_path, "w") as f:
        f.write(full_tex)

    print("\n" + "=" * 90)
    print("        RECALCULATED PUBLICATION TABLE: SVM VS. mDeBERTa-v3")
    print("=" * 90)
    print(full_tex)
    print("=" * 90)
    print(f"\n[SUCCESS] Saved LaTeX table to: {output_tex_path}\n")
    return full_tex


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recalculate SVM vs mDeBERTa-v3 LaTeX Comparison Table")
    parser.add_argument("--csv_path", type=str, default="./outputs_imbalanced/deberta_full_fast/predictions_test_standard.csv")
    parser.add_argument("--output_tex", type=str, default="./outputs_imbalanced/deberta_full_fast/table_svm_vs_deberta_substitution_comparison.tex")
    parser.add_argument("--tau", type=float, default=None)
    args = parser.parse_args()

    generate_comparative_table(args.csv_path, args.output_tex, tau=args.tau)