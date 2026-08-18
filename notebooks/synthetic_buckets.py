import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

def escape_latex(text):
    """Escapes special LaTeX characters in strings."""
    return str(text).replace('\\', r'\textbackslash{}').replace('_', r'\_').replace('%', r'\%')

def assign_bucket(ratio):
    """Bins llm_ratio into 25%, 50%, 75%, and 100% buckets."""
    if ratio == 0.0:
        return "0% (Human Baseline)"
    elif ratio == 1.0:
        return "100% Substitution"
    elif ratio <= 0.375:
        return "25% Substitution"
    elif ratio <= 0.625:
        return "50% Substitution"
    else:
        return "75% Substitution"

def compute_subset_metrics(df_sub, human_df):
    """Computes detection metrics on a specific LLM subset evaluated against Human baseline."""
    combined_df = pd.concat([human_df, df_sub]) if not df_sub.equals(human_df) else human_df
    
    y_true = combined_df['label'].values
    y_prob = combined_df['prob_llm'].values
    y_pred = combined_df['pred'].values
    
    has_both = len(np.unique(y_true)) > 1
    mean_prob = df_sub['prob_llm'].mean()
    sample_count = len(df_sub)
    
    if not has_both:  # Human baseline
        fp_rate = (y_pred == 1).sum() / len(y_pred)
        specificity = 1.0 - fp_rate
        return {
            'count': sample_count,
            'mean_prob': mean_prob,
            'roc_auc': None,
            'accuracy': specificity,
            'f1': None,
            'precision': None,
            'recall': fp_rate  # Represents False Positive Rate (FPR)
        }
    
    return {
        'count': sample_count,
        'mean_prob': mean_prob,
        'roc_auc': roc_auc_score(y_true, y_prob),
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0)
    }

def generate_synthetic_multi_table(csv_path: str) -> str:
    df = pd.read_csv(csv_path)
    
    # 1. Filter Human Baseline and Synthetic Multi generator
    human_df = df[(df['label'] == 0) | (df['model_name'].str.lower() == 'human')]
    syn_df = df[df['model_name'].str.lower() == 'synthetic_multi']
    
    if syn_df.empty:
        raise ValueError("No records found for model_name 'synthetic_multi' in CSV.")
        
    human_metrics = compute_subset_metrics(human_df, human_df)
    
    # 2. Bin synthetic_multi records into buckets
    syn_df = syn_df.copy()
    syn_df['ratio_bucket'] = syn_df['llm_ratio'].apply(assign_bucket)
    
    bucket_order = [
        "25% Substitution",
        "50% Substitution",
        "75% Substitution",
        "100% Substitution"
    ]
    
    bucket_results = []
    for bucket in bucket_order:
        sub_df = syn_df[syn_df['ratio_bucket'] == bucket]
        if not sub_df.empty:
            m = compute_subset_metrics(sub_df, human_df)
            m['bucket_label'] = bucket
            bucket_results.append(m)

    # 3. Construct LaTeX Table
    scope_name = df['scope'].iloc[0] if 'scope' in df.columns else 'full'
    
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        f"\\caption{{Detection performance for the \\texttt{{synthetic\\_multi}} generator across substitution buckets ('{scope_name}' scope).}}",
        r"\label{tab:synthetic_multi_bucket_breakdown}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"\textbf{Substitution Bucket} & \textbf{Samples ($N$)} & \textbf{Mean $P(\text{LLM})$} & \textbf{ROC-AUC} & \textbf{Accuracy} & \textbf{F1-Score} & \textbf{Precision} & \textbf{Recall} \\",
        r"\midrule",
        r"\multicolumn{8}{l}{\textbf{Human Baseline}} \\",
        f"Human Text (0\\% LLM) & {human_metrics['count']:,} & {human_metrics['mean_prob']:.4f} & -- & {human_metrics['accuracy']:.4f} & -- & -- & {human_metrics['recall']:.4f}$^\dagger$ \\\\",
        r"\midrule",
        r"\multicolumn{8}{l}{\textbf{\texttt{synthetic\_multi} Substitution Ratios}} \\"
    ]

    for b in bucket_results:
        b_label = escape_latex(b['bucket_label'])
        latex_lines.append(
            f"{b_label} & {b['count']:,} & {b['mean_prob']:.4f} & {b['roc_auc']:.4f} & {b['accuracy']:.4f} & {b['f1']:.4f} & {b['precision']:.4f} & {b['recall']:.4f} \\\\"
        )

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{3pt}",
        r"\hfill \footnotesize\textit{$^\dagger$ Note: For Human Text, Recall represents False Positive Rate (FPR), and Accuracy represents Specificity (TNR).}",
        r"\end{table}"
    ])

    return "\n".join(latex_lines)

if __name__ == "__main__":
    csv_file = "/home/gderijck/detection/outputs/deberta_no_tune/deberta_full/mdeberta_predictions_full.csv"  # Path to your test predictions CSV
    latex_code = generate_synthetic_multi_table(csv_file)
    
    print("\n--- GENERATED LATEX TABLE FOR SYNTHETIC MULTI ---\n")
    print(latex_code)
    
    with open("synthetic_multi_buckets.tex", "w") as f:
        f.write(latex_code)
    print("\nSaved output to 'synthetic_multi_buckets.tex'!")