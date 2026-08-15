import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load data
df = pd.read_csv("your_results.csv")
test_df = df[df['split'] == 'test'].copy()

def compute_row_metrics(group, name):
    y_true = group['label']
    y_pred = group['pred_llm']
    y_prob = group['prob_llm']
    
    n_samples = len(group)
    acc = accuracy_score(y_true, y_pred)
    
    # Check class distribution
    has_positives = (y_true == 1).any()
    has_negatives = (y_true == 0).any()
    
    if has_positives and has_negatives:
        prec = precision_score(y_true, y_pred, zero_division=np.nan)
        rec = recall_score(y_true, y_pred, zero_division=np.nan)
        f1 = f1_score(y_true, y_pred, zero_division=np.nan)
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = np.nan
    else:
        # Single-class slice metrics
        prec = np.nan
        rec = np.nan
        f1 = np.nan
        auc = np.nan

    return {
        'Generator': name,
        'N': f"{n_samples:,}",
        'Accuracy': f"{acc:.4f}",
        'Precision': f"{prec:.4f}" if not np.isnan(prec) else "--",
        'Recall': f"{rec:.4f}" if not np.isnan(rec) else "--",
        'F1-Score': f"{f1:.4f}" if not np.isnan(f1) else "--",
        'ROC-AUC': f"{auc:.4f}" if not np.isnan(auc) else "--"
    }

# 1. Panel A: Individual Generators
generator_rows = []
for gen, group in test_df.groupby('model_name'):
    generator_rows.append(compute_row_metrics(group, gen))

df_panel_a = pd.DataFrame(generator_rows)

# 2. Panel B: Overall Benchmark Summary
overall_row = compute_row_metrics(test_df, "Total / Overall Benchmark")
df_panel_b = pd.DataFrame([overall_row])

print("--- PANEL A: GENERATORS ---")
print(df_panel_a.to_latex(index=False))

print("\n--- PANEL B: OVERALL ---")
print(df_panel_b.to_latex(index=False))