import numpy as np
import torch
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score
)


def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Computes binary classification metrics from ground truths and positive-class probabilities."""
    preds = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = 0.5

    pr_auc = average_precision_score(y_true, y_prob)

    cm = confusion_matrix(y_true, preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        specificity = 0.0

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    tpr_at_1fpr = tpr[np.where(fpr <= 0.01)[0][-1]] if len(np.where(fpr <= 0.01)[0]) > 0 else 0.0
    tpr_at_5fpr = tpr[np.where(fpr <= 0.05)[0][-1]] if len(np.where(fpr <= 0.05)[0]) > 0 else 0.0

    return {
        "ROC-AUC": float(roc_auc),
        "PR-AUC (AP)": float(pr_auc),
        "Accuracy": float(acc),
        "F1-Score": float(f1),
        "Precision": float(prec),
        "Recall (Sensitivity)": float(rec),
        "Specificity": float(specificity),
        "TPR @ 1% FPR": float(tpr_at_1fpr),
        "TPR @ 5% FPR": float(tpr_at_5fpr),
        "confusion_matrix": cm.tolist()
    }


def hf_compute_metrics(eval_pred):
    """Hugging Face Trainer compatible compute_metrics function."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)
    try:
        auc_score = roc_auc_score(labels, probs)
    except ValueError:
        auc_score = 0.5

    return {
        'roc_auc': auc_score,
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }