import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)

def compute_metrics(y_true, y_pred, y_prob=None):
    """Calculates all relevant binary classification metrics safely."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Specificity (True Negative Rate) = TN / (TN + FP)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    # ROC-AUC requires both positive (1) and negative (0) labels
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = np.nan
    else:
        auc = np.nan

    return {
        "Support": len(y_true),
        "Accuracy": acc,
        "Precision": prec,
        "Recall (TPR)": rec,
        "Specificity (TNR)": spec,
        "F1-Score": f1,
        "ROC-AUC": auc
    }

def generate_latex_report(csv_path_or_df):
    # 1. Load CSV (accepts file path or pandas DataFrame)
    if isinstance(csv_path_or_df, str):
        df = pd.read_csv(csv_path_or_df)
    else:
        df = csv_path_or_df.copy()

    # 2. Ensure binary data columns are numerical
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df['pred_llm'] = pd.to_numeric(df['pred_llm'], errors='coerce')
    df['prob_llm'] = pd.to_numeric(df['prob_llm'], errors='coerce')

    rows = []

    # --- Overall Metrics ---
    overall = compute_metrics(df['label'], df['pred_llm'], df['prob_llm'])
    rows.append({"Evaluation Context": "\\textbf{Overall Dataset}", **overall})

    # --- Per Generator Model (Paired vs. Human Baseline) ---
    if 'human' in df['model_name'].values:
        human_df = df[df['model_name'] == 'human']
        generators = [m for m in df['model_name'].unique() if m != 'human']

        for gen in sorted(generators):
            gen_df = df[df['model_name'] == gen]
            combined = pd.concat([human_df, gen_df])
            m_metrics = compute_metrics(combined['label'], combined['pred_llm'], combined['prob_llm'])
            rows.append({"Evaluation Context": f"{gen} (vs. Human)", **m_metrics})

    # --- Direct Per-Model Subset (Detection Rate on model subset alone) ---
    for model_name in sorted(df['model_name'].unique()):
        sub_df = df[df['model_name'] == model_name]
        m_metrics = compute_metrics(sub_df['label'], sub_df['pred_llm'], sub_df['prob_llm'])
        rows.append({"Evaluation Context": f"Subset: {model_name}", **m_metrics})

    # 3. Format metrics into DataFrame
    summary_df = pd.DataFrame(rows)

    # Format numbers to 4 decimal places for LaTeX
    formatted_df = summary_df.copy()
    float_cols = ["Accuracy", "Precision", "Recall (TPR)", "Specificity (TNR)", "F1-Score", "ROC-AUC"]
    for col in float_cols:
        formatted_df[col] = formatted_df[col].apply(
            lambda x: f"{x:.4f}" if pd.notnull(x) and not np.isnan(x) else "-"
        )
    formatted_df["Support"] = formatted_df["Support"].astype(int)

    # 4. Generate LaTeX Code
    latex_code = formatted_df.to_latex(
        index=False,
        caption="Binary Classification Metrics: Overall and Per Generator Model Performance",
        label="tab:llm_classification_metrics",
        column_format="l" + "c" * (len(formatted_df.columns) - 1),
        position="htbp",
        escape=False
    )

    return latex_code, summary_df

# Example usage:
latex_table, df_metrics = generate_latex_report("/home/gderijck/detection/outputs/fastdetectgpt_nl/sentence/test_predictions.csv")
print(latex_table)