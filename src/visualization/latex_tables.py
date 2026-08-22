import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from src.models.registry import get_model_display_name

def escape_latex(s: Any) -> str:
    if s is None:
        return ''
    st = str(s)
    return st.replace('\\', '\\textbackslash ').replace('_', '\\_').replace('%', '\\%').replace('&', '\\&').replace('$', '\\$')

def load_evaluation_summaries(summary_json_paths: List[Union[str, Path]]) -> List[Dict[str, Any]]:
    summaries = []
    for p in summary_json_paths:
        path_obj = Path(p)
        if path_obj.exists():
            try:
                summaries.append(json.loads(path_obj.read_text(encoding='utf-8')))
            except Exception:
                pass
    return summaries

def export_deberta_hyperparameters_table(best_params: Dict[str, Any], search_spaces: Dict[str, str], scope: str, output_path: Union[str, Path], tuning_sample_size: int, final_sample_size: int, n_trials: int=15):
    clean_scope = escape_latex(scope)
    tex = [
        '\\begin{table}[htbp]',
        '\\centering',
        '\\small',
        f'\\caption{{Hyperparameter configuration and optimal values for mDeBERTa-v3 ({clean_scope.capitalize()} Abstracts).}}',
        f'\\label{{tab:deberta_hyperparams_{scope.lower()}}}',
        '\\begin{tabular}{lll}',
        '\\toprule',
        '\\textbf{Hyperparameter / Setting} & \\textbf{Search Range / Specification} & \\textbf{Selected Value} \\\\',
        '\\midrule',
        '\\multicolumn{3}{l}{\\textit{\\textbf{A. Tuned Optimizer Hyperparameters (Optuna TPE)}}} \\\\'
    ]
    param_display = {
        'learning_rate': ('Base Learning Rate', search_spaces.get('learning_rate', '$[1\\times 10^{-5}, 5\\times 10^{-5}]$')),
        'llrd_decay': ('Layer-wise LR Decay ($\\gamma$)', search_spaces.get('llrd_decay', '$[0.85, 0.95]$')),
        'weight_decay': ('Weight Decay', search_spaces.get('weight_decay', '$[1\\times 10^{-3}, 1\\times 10^{-1}]$')),
        'warmup_ratio': ('Linear Warmup Ratio', search_spaces.get('warmup_ratio', '$[0.05, 0.15]$'))
    }
    for (k, (label, s_range)) in param_display.items():
        val = best_params.get(k)
        if val is not None:
            if isinstance(val, float) and val < 0.01:
                val_str = f'{val:.2e}'
            elif isinstance(val, float):
                val_str = f'{val:.4f}'
            else:
                val_str = escape_latex(val)
            tex.append(f'{label} & {s_range} & \\textbf{{{val_str}}} \\\\')
    tex.extend([
        '\\midrule',
        '\\multicolumn{3}{l}{\\textit{\\textbf{B. Fixed Theoretical Loss Constants (Rockafellar-Uryasev CVaR)}}} \\\\',
        'Tail Risk Quantile ($\\alpha_{\\text{train}}$) & Grounded in Mini-Batch Stability & $\\alpha = 0.05$ \\\\',
        'Negative Tail Penalty ($\\lambda_{\\text{neg}}$) & Asymmetric False-Positive Weight & $\\lambda = 2.0$ \\\\',
        'Multi-Scale Balance ($w_{\\text{doc}} / w_{\\text{sent}}$) & Uniform Hierarchy Weighting & $1.0 / 1.0$ \\\\',
        '\\midrule',
        '\\multicolumn{3}{l}{\\textit{\\textbf{C. Architecture \\& Search Budget}}} \\\\',
        'Backbone Architecture & Microsoft mDeBERTa-v3-base & 12 layers, 768-dim \\\\',
        'Classification Head & Multi-Sample Dropout + Fusion & [CLS] + Mean + Max \\\\',
        f'Optuna Search Budget & TPE Sampler (Median Pruner) & {n_trials} Trials \\\\',
        f'Tuning Sample Size ($N_{{\\text{{tune}}}}$) & Stratified Subsample & {tuning_sample_size:,} \\\\',
        f'Final Training Size ($N_{{\\text{{train}}}}$) & Full Split & {final_sample_size:,} \\\\',
        '\\bottomrule',
        '\\end{tabular}',
        '\\end{table}'
    ])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(tex), encoding='utf-8')

def export_hyperparameters_table(best_params: Dict[str, Any], search_spaces: Dict[str, str], scope: str, output_path: Union[str, Path]):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def f_val(val: Any) -> str:
        if val is None:
            return '-'
        if isinstance(val, float):
            return f'{val:.4f}' if abs(val) < 0.01 or abs(val) > 100.0 else f'{val:.3f}'
        if isinstance(val, tuple):
            return f'({val[0]}, {val[1]})'
        return escape_latex(val)

    w_ngram = (best_params.get('word_min_ngram', 1), best_params.get('word_max_ngram', 2))
    c_ngram = (best_params.get('char_min_ngram', 3), best_params.get('char_max_ngram', 5))
    rows = [
        ('Word TF-IDF', 'N-gram Range', search_spaces.get('word_ngram', ''), f_val(w_ngram)),
        ('', 'Max Features', search_spaces.get('word_max_features', ''), f"{best_params.get('word_max_features', 50000):,}"),
        ('', 'Min DF', search_spaces.get('word_min_df', ''), str(best_params.get('word_min_df', 2))),
        ('', 'Sublinear TF', search_spaces.get('word_sublinear_tf', ''), str(best_params.get('word_sublinear_tf', True))),
        ('Char TF-IDF', 'N-gram Range', search_spaces.get('char_ngram', ''), f_val(c_ngram)),
        ('', 'Max Features', search_spaces.get('char_max_features', ''), f"{best_params.get('char_max_features', 50000):,}"),
        ('', 'Min DF', search_spaces.get('char_min_df', ''), str(best_params.get('char_min_df', 2))),
        ('Stylometrics', 'Use Stylometrics', search_spaces.get('use_stylometrics', ''), str(best_params.get('use_stylometrics', True))),
        ('', 'Subspace Weight', search_spaces.get('sty_weight', ''), f_val(best_params.get('sty_weight', 1.0))),
        ('SVM Solver', 'Kernel', search_spaces.get('kernel', ''), escape_latex(best_params.get('kernel', 'linear'))),
        ('', 'Regularization (C)', search_spaces.get('C', ''), f_val(best_params.get('C', 1.0))),
        ('', 'Loss Function', search_spaces.get('linear_loss', ''), escape_latex(best_params.get('linear_loss', 'squared_hinge'))),
        ('', 'Class Weighting', search_spaces.get('class_weight', ''), escape_latex(best_params.get('weight_mode', 'balanced')))
    ]
    latex = [
        '\\begin{table}[h]',
        '\\centering',
        '\\small',
        f'\\caption{{SVM Hyperparameter Search Space and Optimal Configuration ({escape_latex(scope).capitalize()}-level).}}',
        f'\\label{{tab:svm_hyperparams_{scope}}}',
        '\\begin{tabular}{llcc}',
        '\\toprule',
        '\\textbf{Component} & \\textbf{Hyperparameter} & \\textbf{Search Space} & \\textbf{Optimal Value} \\\\',
        '\\midrule'
    ]
    for (comp, param, space, opt) in rows:
        latex.append(f'{escape_latex(comp)} & {escape_latex(param)} & {space} & {opt} \\\\')
    latex.extend(['\\bottomrule', '\\end{tabular}', '\\end{table}'])
    output_path.write_text('\n'.join(latex), encoding='utf-8')

def export_performance_table(summary: Dict[str, Any], scope: str, model_name: str, output_path: Union[str, Path]):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean_model = escape_latex(model_name)
    pct_format = '{:.2f}\\%'
    std_format = '{:.4f}'

    def f_metric(key: str, fmt: str=std_format, multiplier: float=1.0) -> str:
        v = summary.get(key)
        if v is None:
            return '-'
        return fmt.format(v * multiplier)

    roc_str = f_metric('overall_roc_auc')
    pauc_str = f_metric('overall_pauc')
    tpr_str = f_metric('tpr_at_1fpr', pct_format, 100.0)
    pr_str = f_metric('overall_pr_auc')
    f1_ai_str = f_metric('f1_ai')
    f1_hu_str = f_metric('f1_human')
    fpr_str = f_metric('fpr_human', pct_format, 100.0)
    mcc_str = f_metric('overall_mcc')
    brier_str = f_metric('brier_score')
    latex = [
        '\\begin{table}[h]',
        '\\centering',
        '\\small',
        f'\\caption{{Test Performance Summary for {clean_model.upper()} ({escape_latex(scope).capitalize()}-level).}}',
        f'\\label{{tab:perf_{model_name}_{scope}}}',
        '\\begin{tabular}{lc}',
        '\\toprule',
        '\\textbf{Metric} & \\textbf{Value} \\\\',
        '\\midrule',
        f'ROC-AUC & {roc_str} \\\\',
        f'pAUC (FPR $\\le$ 1\\%) & {pauc_str} \\\\',
        f'TPR @ 1\\% FPR & {tpr_str} \\\\',
        f'PR-AUC (Avg Precision) & {pr_str} \\\\',
        f'F1-Score (LLM) & {f1_ai_str} \\\\',
        f'F1-Score (Human) & {f1_hu_str} \\\\',
        f'False Positive Rate (Human) & {fpr_str} \\\\',
        f'Matthews Correlation (MCC) & {mcc_str} \\\\',
        f'Brier Score (Calibration Loss) & {brier_str} \\\\',
        '\\bottomrule',
        '\\end{tabular}',
        '\\end{table}'
    ]
    output_path.write_text('\n'.join(latex), encoding='utf-8')

def export_robustness_table(ratio_stats: List[Dict[str, Any]], scope: str, model_name: str, output_path: Union[str, Path]):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean_model = escape_latex(model_name)
    clean_scope = escape_latex(scope)
    latex = [
        '\\begin{table}[htbp]',
        '\\centering',
        '\\small',
        f'\\caption{{Discrete Rewrite Robustness for {clean_model.upper()} ({clean_scope.capitalize()}-level).}}',
        f'\\label{{tab:robustness_{model_name}_{scope}}}',
        '\\begin{tabular}{lcccc}',
        '\\toprule',
        '\\textbf{Rewrite Bucket} & \\textbf{Actual LLM Ratio} & \\textbf{Detection Rate (\\%)} & \\textbf{Avg Predicted Prob} & \\textbf{Count} \\\\',
        '\\midrule'
    ]
    for row in ratio_stats:
        b_label = escape_latex(row.get('bucket', row.get('target_ratio', '-')))
        act_ratio = f"{row.get('actual_ratio', 0.0) * 100:.1f}\\%"
        flagged = f"{row.get('flagged_pct', 0.0):.2f}\\%"
        avg_sc = f"{row.get('avg_score', 0.0):.4f}"
        cnt = f"{row.get('count', 0):,}"
        latex.append(f'{b_label} & {act_ratio} & {flagged} & {avg_sc} & {cnt} \\\\')
    latex.extend(['\\bottomrule', '\\end{tabular}', '\\end{table}'])
    output_path.write_text('\n'.join(latex), encoding='utf-8')

def export_multi_model_robustness_table(summary_json_paths: List[Union[str, Path]], scope: str, output_path: Union[str, Path]):
    summaries = load_evaluation_summaries(summary_json_paths)
    if not summaries:
        return
    model_display_names = {
        'svm': 'Linear SVM (TF-IDF + Stylo)',
        'fdgpt': 'Fast-DetectGPT (Zero-Shot)',
        'fast_detect_gpt': 'Fast-DetectGPT (Zero-Shot)',
        'stat_trajectory': 'LLM Trajectory (Ours)',
        'stat': 'LLM Trajectory (Ours)',
        'mdeberta': 'mDeBERTa-v3 (CVaR-DRO)',
        'deberta': 'mDeBERTa-v3 (CVaR-DRO)'
    }
    clean_scope = escape_latex(scope)
    tex = [
        '\\begin{table*}[htbp]',
        '\\centering',
        '\\small',
        f'\\caption{{Comparative Detection Sensitivity across Discrete Rewrite Buckets (25\\%, 50\\%, 75\\%) on {clean_scope.capitalize()} Abstracts. Operational threshold $\\tau$ calibrated on Dev split at $\\text{{FPR}} \\le 1.0\\%$.}}',
        f'\\label{{tab:model_robustness_comparison_{scope.lower()}}}',
        '\\begin{tabular}{l' + 'c' * len(summaries) + '}',
        '\\toprule',
        '\\textbf{Rewrite Bucket / Metric} & ' + ' & '.join(['\\textbf{' + model_display_names.get(s['model_name'].lower(), escape_latex(s['model_name'])) + '}' for s in summaries]) + ' \\\\',
        '\\midrule'
    ]
    buckets = [('25pct', '25% LLM Rewrite'), ('50pct', '50% LLM Rewrite'), ('75pct', '75% LLM Rewrite')]
    for (b_key, b_label) in buckets:
        row_vals_flagged = []
        row_vals_score = []
        for s in summaries:
            rob = s.get('robustness_ratios', {})
            b_info = rob.get(b_key, {})
            if b_info and 'flagged_pct' in b_info and (b_info['flagged_pct'] is not None):
                row_vals_flagged.append(f"{b_info['flagged_pct']:.2f}\\%")
                row_vals_score.append(f"{b_info.get('avg_score', 0.0):.4f}")
            else:
                row_vals_flagged.append('-')
                row_vals_score.append('-')
        clean_label = escape_latex(b_label)
        tex.append(f'{clean_label} (Flagged \\%) & ' + ' & '.join(row_vals_flagged) + ' \\\\')
        tex.append(f'{clean_label} (Avg Prob) & ' + ' & '.join(row_vals_score) + ' \\\\')
        if b_key != '75pct':
            tex.append('\\addlinespace')
    tex.extend(['\\bottomrule', '\\end{tabular}', '\\end{table*}'])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(tex), encoding='utf-8')

def export_multi_model_comparison_table(summary_json_paths: List[Union[str, Path]], scope: str, output_path: Union[str, Path]):
    summaries = load_evaluation_summaries(summary_json_paths)
    if not summaries:
        return
    clean_scope = escape_latex(scope)
    tex = [
        '\\begin{table*}[htbp]',
        '\\centering',
        '\\small',
        f'\\caption{{Comprehensive Benchmark Comparison across Detection Paradigms ({clean_scope.capitalize()} Abstracts). Operational threshold $\\tau$ calibrated on Dev split at $\\text{{FPR}} \\le 1.0\\%$.}}',
        f'\\label{{tab:model_comparison_{scope.lower()}}}',
        '\\begin{tabular}{l' + 'c' * len(summaries) + '}',
        '\\toprule',
        '\\textbf{Metric} & ' + ' & '.join(['\\textbf{' + escape_latex(get_model_display_name(s['model_name'])) + '}' for s in summaries]) + ' \\\\',
        '\\midrule'
    ]
    metrics_to_show = [
        ('Partial AUC (FPR $\\le 1\\%$)', 'overall_pauc', '{:.4f}', 1.0),
        ('TPR @ 1\\% FPR', 'tpr_at_1fpr', '{:.2f}\\%', 100.0),
        ('Overall ROC-AUC', 'overall_roc_auc', '{:.4f}', 1.0),
        ('AI F1-Score (@ $\\tau$)', 'f1_ai', '{:.4f}', 1.0),
        ('Matthews Corr. (MCC)', 'overall_mcc', '{:.4f}', 1.0),
        ('Human FPR (@ $\\tau$)', 'fpr_human', '{:.2f}\\%', 100.0),
        ('Brier Score Loss', 'brier_score', '{:.4f}', 1.0)
    ]
    for (label, key, fmt_str, mult) in metrics_to_show:
        row_vals = []
        for s in summaries:
            v = s.get(key)
            if v is None:
                row_vals.append('-')
            else:
                row_vals.append(fmt_str.format(v * mult))
        tex.append(f'{label} & ' + ' & '.join(row_vals) + ' \\\\')
    has_robustness = any((bool(s.get('robustness_ratios')) for s in summaries))
    if has_robustness:
        tex.extend(['\\midrule', '\\multicolumn{' + str(len(summaries) + 1) + '}{l}{\\textit{\\textbf{Discrete Rewrite Sensitivity (Flagged \\% @ $\\tau$)}}} \\\\'])
        for (b_key, b_label) in [('25pct', '25\\% LLM Rewrite'), ('50pct', '50\\% LLM Rewrite'), ('75pct', '75\\% LLM Rewrite')]:
            row_vals = []
            for s in summaries:
                rob = s.get('robustness_ratios', {})
                b_info = rob.get(b_key, {})
                if b_info and 'flagged_pct' in b_info and (b_info['flagged_pct'] is not None):
                    row_vals.append(f"{b_info['flagged_pct']:.2f}\\%")
                else:
                    row_vals.append('-')
            tex.append(f'{b_label} & ' + ' & '.join(row_vals) + ' \\\\')
    tex.extend(['\\bottomrule', '\\end{tabular}', '\\end{table*}'])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(tex), encoding='utf-8')