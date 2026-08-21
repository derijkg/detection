# src/visualization/latex_tables.py

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd


def export_deberta_hyperparameters_table(
    best_params: Dict[str, Any],
    search_spaces: Dict[str, str],
    scope: str,
    output_path: Path,
    tuning_sample_size: int,
    final_sample_size: int,
    n_trials: int = 10,
):
    tex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Hyperparameter optimization configuration and optimal values for mDeBERTa-v3 (" + scope.capitalize() + r" Abstracts).}",
        r"\label{tab:deberta_hyperparams_" + scope.lower() + r"}",
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"\textbf{Hyperparameter / Setting} & \textbf{Search Range / Specification} & \textbf{Selected Value} \\",
        r"\midrule",
        r"\multicolumn{3}{l}{\textit{\textbf{A. Optimization \& Loss Parameters}}} \\",
    ]

    param_display = {
        "learning_rate": ("Base Learning Rate (Classifier)", search_spaces.get("learning_rate", r"$[1\times 10^{-5}, 5\times 10^{-5}]$")),
        "llrd_decay": (r"Layer-wise LR Decay ($\gamma$)", search_spaces.get("llrd_decay", "$[0.80, 0.95]$")),
        "lambda_neg": (r"CVaR Negative Tail Weight ($\lambda_{\text{neg}}$)", search_spaces.get("lambda_neg", "$[1.0, 4.0]$")),
        "weight_decay": ("Weight Decay", search_spaces.get("weight_decay", r"$[1\times 10^{-3}, 1\times 10^{-1}]$")),
        "warmup_ratio": ("Linear Warmup Ratio", search_spaces.get("warmup_ratio", "$[0.05, 0.15]$")),
    }

    for k, (label, s_range) in param_display.items():
        val = best_params.get(k)
        if val is not None:
            val_str = f"{val:.2e}" if (isinstance(val, float) and val < 0.01) else (f"{val:.4f}" if isinstance(val, float) else str(val))
            tex.append(f"{label} & {s_range} & \\textbf{{{val_str}}} \\\\")

    tex.extend([
        r"\midrule",
        r"\multicolumn{3}{l}{\textit{\textbf{B. Architecture \& Search Budget}}} \\",
        r"Backbone Architecture & Microsoft mDeBERTa-v3-base & 12 layers, 768-dim \\",
        r"Classification Head & Multi-Sample Dropout + Fusion & [CLS] + Mean + Max \\",
        f"Optuna Search Budget & TPE Sampler (Median Pruner) & {n_trials} Trials \\\\",
        f"Tuning Sample Size ($N_{{\\text{{tune}}}}$) & Stratified Subsample & {tuning_sample_size:,} \\\\",
        f"Final Training Size ($N_{{\\text{{train}}}}$) & Full Split & {final_sample_size:,} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(tex), encoding="utf-8")


def export_hyperparameters_table(
    best_params: Dict[str, Any],
    search_spaces: Dict[str, str],
    scope: str,
    output_path: Path
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def f_val(val: Any) -> str:
        if isinstance(val, float):
            return f"{val:.4f}" if (abs(val) < 1e-2 or abs(val) > 1e2) else f"{val:.3f}"
        if isinstance(val, tuple):
            return f"({val[0]}, {val[1]})"
        return str(val)

    w_ngram = (best_params.get('word_min_ngram', 1), best_params.get('word_max_ngram', 2))
    c_ngram = (best_params.get('char_min_ngram', 3), best_params.get('char_max_ngram', 5))

    rows = [
        ("Word TF-IDF", "N-gram Range", search_spaces.get("word_ngram", ""), f_val(w_ngram)),
        ("", "Max Features", search_spaces.get("word_max_features", ""), f"{best_params.get('word_max_features', 50000):,}"),
        ("", "Min DF", search_spaces.get("word_min_df", ""), str(best_params.get('word_min_df', 2))),
        ("", "Sublinear TF", search_spaces.get("word_sublinear_tf", ""), str(best_params.get('word_sublinear_tf', True))),
        ("Char TF-IDF", "N-gram Range", search_spaces.get("char_ngram", ""), f_val(c_ngram)),
        ("", "Max Features", search_spaces.get("char_max_features", ""), f"{best_params.get('char_max_features', 50000):,}"),
        ("", "Min DF", search_spaces.get("char_min_df", ""), str(best_params.get('char_min_df', 2))),
        ("Stylometrics", "Use Stylometrics", search_spaces.get("use_stylometrics", ""), str(best_params.get('use_stylometrics', True))),
        ("", "Subspace Weight", search_spaces.get("sty_weight", ""), f_val(best_params.get('sty_weight', 1.0))),
        ("SVM Solver", "Kernel", search_spaces.get("kernel", ""), str(best_params.get('kernel', 'linear'))),
        ("", "Regularization (C)", search_spaces.get("C", ""), f_val(best_params.get('C', 1.0))),
        ("", "Loss Function", search_spaces.get("linear_loss", ""), str(best_params.get('linear_loss', 'squared_hinge'))),
        ("", "Class Weighting", search_spaces.get("class_weight", ""), str(best_params.get('weight_mode', 'balanced'))),
    ]

    latex = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        f"\\caption{{SVM Hyperparameter Search Space and Optimal Configuration ({scope.capitalize()}-level).}}",
        f"\\label{{tab:svm_hyperparams_{scope}}}",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"\textbf{Component} & \textbf{Hyperparameter} & \textbf{Search Space} & \textbf{Optimal Value} \\",
        r"\midrule"
    ]

    for comp, param, space, opt in rows:
        latex.append(f"{comp} & {param} & {space} & {opt} \\\\")

    latex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    output_path.write_text("\n".join(latex), encoding="utf-8")


def export_performance_table(summary: Dict[str, Any], scope: str, model_name: str, output_path: Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    latex = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        f"\\caption{{Test Performance Summary for {model_name.upper()} ({scope.capitalize()}-level).}}",
        f"\\label{{tab:perf_{model_name}_{scope}}}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{Value} \\",
        r"\midrule",
        f"ROC-AUC & {summary.get('overall_roc_auc', 0.0):.4f} \\\\",
        r"pAUC (FPR $\le$ 1\%) & " + f"{summary.get('overall_pauc', 0.0):.4f} \\\\",
        r"PR-AUC (Avg Precision) & " + f"{summary.get('overall_pr_auc', 0.0):.4f} \\\\",
        f"F1-Score (LLM) & {summary.get('f1_ai', 0.0):.4f} \\\\",
        f"F1-Score (Human) & {summary.get('f1_human', 0.0):.4f} \\\\",
        r"False Positive Rate (Human) & " + f"{summary.get('fpr_human', 0.0)*100:.2f}\\% \\\\",
        f"Matthews Correlation (MCC) & {summary.get('mcc', 0.0):.4f} \\\\",
        f"Brier Score (Calibration Loss) & {summary.get('brier_score', 0.0):.4f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ]
    output_path.write_text("\n".join(latex), encoding="utf-8")


def export_robustness_table(ratio_stats: List[Dict[str, Any]], scope: str, model_name: str, output_path: Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    latex = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        f"\\caption{{Detection Sensitivity across Gradual LLM Substitution Ratios ({model_name.upper()}, {scope.capitalize()}).}}",
        f"\\label{{tab:robustness_{model_name}_{scope}}}",
        r"\begin{tabular}{cccc}",
        r"\toprule",
        r"\textbf{Target Ratio} & \textbf{Actual Ratio (Avg)} & \textbf{Flagged as LLM (\%)} & \textbf{Avg LLM Score} \\",
        r"\midrule"
    ]

    for row in ratio_stats:
        latex.append(f"{row['target_ratio']} & {row['actual_ratio']:.3f} & {row['flagged_pct']:.2f}\\% & {row['avg_score']:.4f} \\\\")

    latex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    output_path.write_text("\n".join(latex), encoding="utf-8")


def export_multi_model_comparison_table(summary_json_paths: List[Path], scope: str, output_path: Path):
    summaries = []
    for p in summary_json_paths:
        p = Path(p)
        if p.exists():
            try:
                summaries.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                pass

    if not summaries:
        return

    model_display_names = {
        "svm": "Linear SVM (TF-IDF + Stylo)",
        "fdgpt": "Fast-DetectGPT (Zero-Shot)",
        "fast_detect_gpt": "Fast-DetectGPT (Zero-Shot)",
        "stat_trajectory": "LLM Trajectory (Ours)",
        "stat": "LLM Trajectory (Ours)",
        "mdeberta": "mDeBERTa-v3 (CVaR-DRO)",
        "deberta": "mDeBERTa-v3 (CVaR-DRO)"
    }

    tex = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Comprehensive Benchmark Comparison across Detection Paradigms (" + scope.capitalize() + r" Abstracts). Operational threshold $\tau$ calibrated on Dev split at $\text{FPR} \le 1.0\%$.}",
        r"\label{tab:model_comparison_" + scope.lower() + r"}",
        r"\begin{tabular}{l" + "c" * len(summaries) + r"}",
        r"\toprule",
        r"\textbf{Metric} & " + " & ".join([r"\textbf{" + model_display_names.get(s["model_name"].lower(), s["model_name"].upper()) + r"}" for s in summaries]) + r" \\",
        r"\midrule",
    ]

    metrics_to_show = [
        (r"Partial AUC (FPR $\le 1\%$)", "overall_pauc", "{:.4f}"),
        (r"TPR @ 1\% FPR", "tpr_at_1fpr", lambda v: f"{v*100:.2f}\\%"),
        (r"Overall ROC-AUC", "overall_roc_auc", "{:.4f}"),
        (r"AI F1-Score (@ $\tau$)", "f1_ai", "{:.4f}"),
        (r"Matthews Corr. (MCC)", "overall_mcc", "{:.4f}"),
        (r"Human FPR (@ $\tau$)", "fpr_human", lambda v: f"{v*100:.2f}\\%"),
        (r"Brier Score Loss", "brier_score", "{:.4f}"),
    ]

    for label, key, fmt in metrics_to_show:
        row_vals = []
        for s in summaries:
            v = s.get(key, 0.0)
            if callable(fmt):
                row_vals.append(fmt(v))
            else:
                row_vals.append(fmt.format(v))
        tex.append(f"{label} & " + " & ".join(row_vals) + r" \\")

    tex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}"
    ])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(tex), encoding="utf-8")