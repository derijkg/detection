import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

# ============================================================
# Configuration
# ============================================================

INPUT_FILE = "/home/gderijck/detection/outputs/fastdetectgpt_nl/sentence/test_predictions.csv"
OUTPUT_FILE = "binary_classification_metrics.tex"

# Only evaluate the test split
SPLIT = "test"

# ============================================================
# Load data
# ============================================================

df = pd.read_csv(INPUT_FILE)

# Keep only the desired split
df = df[df["split"] == SPLIT].copy()

# Remove rows with missing values in relevant columns
df = df.dropna(
    subset=["label", "pred_llm", "prob_llm", "model_name"]
)

# Make sure the variables have the correct types
df["label"] = df["label"].astype(int)
df["pred_llm"] = df["pred_llm"].astype(int)
df["prob_llm"] = df["prob_llm"].astype(float)


# ============================================================
# Function to calculate metrics
# ============================================================

def calculate_metrics(group):
    """
    Calculate binary classification metrics for one model.
    """

    y_true = group["label"]
    y_pred = group["pred_llm"]
    y_prob = group["prob_llm"]

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    # Specificity = TN / (TN + FP)
    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    # F1
    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    # Negative predictive value
    npv = (
        tn / (tn + fn)
        if (tn + fn) > 0
        else np.nan
    )

    # False-positive rate
    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else np.nan
    )

    # False-negative rate
    fnr = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else np.nan
    )

    # Balanced accuracy
    balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred
    )

    # Matthews correlation coefficient
    mcc = matthews_corrcoef(
        y_true,
        y_pred
    )

    # Cohen's kappa
    kappa = cohen_kappa_score(
        y_true,
        y_pred
    )

    # ROC-AUC and PR-AUC require both classes to be present
    if y_true.nunique() == 2:
        roc_auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
    else:
        roc_auc = np.nan
        pr_auc = np.nan

    return {
        "N": len(group),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1": f1,
        "NPV": npv,
        "FPR": fpr,
        "FNR": fnr,
        "Balanced Accuracy": balanced_accuracy,
        "MCC": mcc,
        "Cohen's Kappa": kappa,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
    }


# ============================================================
# Calculate metrics for every model
# ============================================================

results = []

for model_name, group in df.groupby("model_name"):
    metrics = calculate_metrics(group)
    metrics["Model"] = model_name
    results.append(metrics)

results_df = pd.DataFrame(results)

# Put Model first
columns = ["Model"] + [
    c for c in results_df.columns if c != "Model"
]

results_df = results_df[columns]


# ============================================================
# Print results
# ============================================================

print("\nClassification metrics:")
print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# Generate LaTeX table
# ============================================================

latex_table = results_df.to_latex(
    index=False,
    escape=False,
    float_format="%.3f",
    caption=(
        "Performance of the binary classification models "
        "on the test set."
    ),
    label="tab:classification_metrics",
    column_format="l" + "r" * (len(results_df.columns) - 1),
)

# Save LaTeX
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(latex_table)

print(f"\nLaTeX table saved to: {OUTPUT_FILE}")