# detection/src/training/metrics.py

import os
import json
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, accuracy_score, precision_recall_fscore_support, roc_auc_score
)

def evaluate_predictions(df_pred: pd.DataFrame, output_dir: str = "./outputs") -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Computes overall and per-generator metrics from a DataFrame containing
    'label', 'pred_label', and 'prob_llm'.
    Works uniformly for DeBERTa, SVM, or any other model.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    labels = df_pred['label'].values
    preds = df_pred['pred_label'].values
    probs_llm = df_pred['prob_llm'].values

    # 1. Overall Metrics
    fpr, tpr, _ = roc_curve(labels, probs_llm)
    roc_auc_val = auc(fpr, tpr)
    pr_auc_val = average_precision_score(labels, probs_llm)

    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    tpr_at_1fpr = tpr[np.where(fpr <= 0.01)[0][-1]] if len(np.where(fpr <= 0.01)[0]) > 0 else 0.0
    tpr_at_5fpr = tpr[np.where(fpr <= 0.05)[0][-1]] if len(np.where(fpr <= 0.05)[0]) > 0 else 0.0

    overall_metrics = {
        "total_samples": len(labels),
        "roc_auc": round(float(roc_auc_val), 4),
        "pr_auc": round(float(pr_auc_val), 4),
        "accuracy": round(float(acc), 4),
        "f1_score": round(float(f1), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "specificity": round(float(specificity), 4),
        "tpr_at_1fpr": round(float(tpr_at_1fpr), 4),
        "tpr_at_5fpr": round(float(tpr_at_5fpr), 4)
    }

    # 2. Per-LLM Subgroup Breakdown
    per_model_results = []
    if 'model_name' in df_pred.columns:
        human_df = df_pred[df_pred['label'] == 0]
        generator_models = [m for m in df_pred['model_name'].unique() if m not in ["Human", "human", "unknown"]]

        for g_model in generator_models:
            llm_sub_df = df_pred[(df_pred['label'] == 1) & (df_pred['model_name'] == g_model)]
            if len(llm_sub_df) == 0:
                continue
            
            combined_sub = pd.concat([human_df, llm_sub_df])
            sub_labels = combined_sub['label'].values
            sub_probs = combined_sub['prob_llm'].values
            sub_preds = combined_sub['pred_label'].values

            try:
                sub_auc = roc_auc_score(sub_labels, sub_probs)
            except Exception:
                sub_auc = 0.5

            sub_acc = accuracy_score(sub_labels, sub_preds)
            sub_prec, sub_rec, sub_f1, _ = precision_recall_fscore_support(sub_labels, sub_preds, average='binary', zero_division=0)

            per_model_results.append({
                "generator_model": g_model,
                "samples": len(llm_sub_df),
                "roc_auc": round(float(sub_auc), 4),
                "accuracy": round(float(sub_acc), 4),
                "f1_score": round(float(sub_f1), 4),
                "precision": round(float(sub_prec), 4),
                "recall": round(float(sub_rec), 4)
            })

    per_model_df = pd.DataFrame(per_model_results)

    # Save summary metrics
    metrics_file = os.path.join(output_dir, "metrics_summary.json")
    with open(metrics_file, "w") as f:
        json.dump({"overall": overall_metrics, "per_model": per_model_results}, f, indent=4)

    return overall_metrics, per_model_df