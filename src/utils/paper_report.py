# detection/src/utils/paper_reporter.py

import os
import pandas as pd
from typing import Dict, Any, List


def generate_dataset_latex_table(df: pd.DataFrame, save_path: str) -> None:
    """Generates booktabs LaTeX table summarizing dataset composition across splits and scope."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df_copy = df.copy()
    df_copy['words'] = df_copy['text'].apply(lambda x: len(str(x).split()))

    grouped = df_copy.groupby(['split', 'scope', 'generation_type']).agg(
        sample_count=('text', 'count'),
        avg_words=('words', 'mean'),
        llm_ratio=('llm_ratio', 'mean')
    ).reset_index()

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("% Auto-generated LaTeX table for Dataset Overview\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Summary of Preprocessed Benchmark Dataset across Splits and Modalities.}\n")
        f.write("\\label{tab:dataset_overview}\n")
        f.write("\\begin{tabular}{lllccc}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Split} & \\textbf{Scope} & \\textbf{Generation Type} & \\textbf{Samples} & \\textbf{Avg. Words} & \\textbf{LLM Ratio} \\\\\n")
        f.write("\\midrule\n")

        for _, row in grouped.iterrows():
            f.write(
                f"{row['split'].capitalize()} & {row['scope']} & {row['generation_type']} & "
                f"{row['sample_count']:,} & {row['avg_words']:.1f} & {row['llm_ratio']*100:.0f}\\% \\\\\n"
            )

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def generate_results_latex_table(
    overall_metrics: Dict[str, float], 
    per_model_results: List[Dict[str, Any]], 
    model_name: str, 
    save_path: str
) -> None:
    """Generates booktabs LaTeX table summarizing classification performance."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(f"% Auto-generated LaTeX results table for {model_name}\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{Performance Evaluation of \\textbf{{{model_name.upper()}}} AI Text Detector on Held-out Test Set.}}\n")
        f.write(f"\\label{{tab:results_{model_name.lower()}}}\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Evaluation Group} & \\textbf{ROC-AUC} & \\textbf{PR-AUC} & \\textbf{Accuracy} & \\textbf{F1-Score} & \\textbf{Spec.} & \\textbf{TPR @ 1\\% FPR} \\\\\n")
        f.write("\\midrule\n")

        # Overall row
        f.write(
            f"\\textbf{{Overall Test Set}} & \\textbf{{{overall_metrics.get('ROC-AUC', 0):.4f}}} & "
            f"{overall_metrics.get('PR-AUC (AP)', 0):.4f} & {overall_metrics.get('Accuracy', 0):.4f} & "
            f"{overall_metrics.get('F1-Score', 0):.4f} & {overall_metrics.get('Specificity', 0):.4f} & "
            f"\\textbf{{{overall_metrics.get('TPR @ 1% FPR', 0):.4f}}} \\\\\n"
        )
        f.write("\\midrule\n")

        # Per generator model rows
        for row in per_model_results:
            f.write(
                f"vs. {row['Generator Model']} & {row['ROC-AUC']:.4f} & -- & "
                f"{row['Accuracy']:.4f} & {row['F1-Score']:.4f} & -- & -- \\\\\n"
            )

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")