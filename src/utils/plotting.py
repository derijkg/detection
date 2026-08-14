# detection/src/utils/plotting.py

import os
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, auc, average_precision_score

# Set publication style defaults
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8


def plot_dataset_overview(df: pd.DataFrame, save_dir: str) -> str:
    """Generates 300 DPI multi-panel EDA figure for dataset description section."""
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)

    # Calculate word counts
    df_plot = df.copy()
    df_plot['word_count'] = df_plot['text'].apply(lambda x: len(str(x).split()))

    # Panel A: Word Count Distribution by Scope
    sns.kdeplot(
        data=df_plot, x='word_count', hue='scope', common_norm=False, 
        ax=axes[0], palette=['#1b9e77', '#d95f02'], fill=True, alpha=0.4
    )
    axes[0].set_title('(A) Word Count Density by Scope', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Word Count', fontsize=10)
    axes[0].set_ylabel('Density', fontsize=10)
    axes[0].set_xlim(0, 500)
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Panel B: LLM Content Ratio Breakdown
    ratio_counts = df_plot['llm_ratio'].round(2).value_counts().sort_index()
    bars = axes[1].bar([f"{int(r*100)}%" for r in ratio_counts.index], ratio_counts.values, color='#7570b3', edgecolor='black', alpha=0.85)
    axes[1].set_title('(B) Samples by LLM Content Ratio', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('LLM Content Ratio', fontsize=10)
    axes[1].set_ylabel('Sample Count', fontsize=10)
    axes[1].grid(True, axis='y', linestyle='--', alpha=0.5)
    for bar in bars:
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, f"{bar.get_height():,}", ha='center', fontsize=8)

    # Panel C: Generator Model Distribution
    model_counts = df_plot[df_plot['model_name'] != 'human']['model_name'].value_counts()
    axes[2].barh(model_counts.index, model_counts.values, color='#e7298a', edgecolor='black', alpha=0.85)
    axes[2].set_title('(C) LLM Generator Samples', fontsize=11, fontweight='bold')
    axes[2].set_xlabel('Sample Count', fontsize=10)
    axes[2].grid(True, axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(save_dir, "dataset_overview_plots.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_4panel_evaluation(
    labels: np.ndarray, 
    probs_llm: np.ndarray, 
    generation_types: np.ndarray,
    model_name: str, 
    save_dir: str
) -> str:
    """Generates 300 DPI 4-panel evaluation diagnostic figure for results section."""
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    plt.subplots_adjust(wspace=0.25, hspace=0.3)

    preds = (probs_llm >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(labels, probs_llm)
    roc_auc_val = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(labels, probs_llm)
    pr_auc_val = average_precision_score(labels, probs_llm)

    # Panel A: ROC Curve
    axes[0, 0].plot(fpr, tpr, color='#2b5c8f', lw=2, label=f'{model_name} (AUC = {roc_auc_val:.4f})')
    axes[0, 0].plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5, label='Random Guess')
    axes[0, 0].set_xlabel('False Positive Rate (1 - Specificity)', fontsize=10)
    axes[0, 0].set_ylabel('True Positive Rate (Sensitivity)', fontsize=10)
    axes[0, 0].set_title('(A) Receiver Operating Characteristic (ROC)', fontsize=11, fontweight='bold')
    axes[0, 0].legend(loc='lower right', fontsize=9)
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)

    # Panel B: Precision-Recall Curve
    axes[0, 1].plot(recall, precision, color='#d95f02', lw=2, label=f'{model_name} (AP = {pr_auc_val:.4f})')
    axes[0, 1].set_xlabel('Recall', fontsize=10)
    axes[0, 1].set_ylabel('Precision', fontsize=10)
    axes[0, 1].set_title('(B) Precision-Recall Curve', fontsize=11, fontweight='bold')
    axes[0, 1].legend(loc='lower left', fontsize=9)
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)

    # Panel C: Normalized Confusion Matrix
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    im = axes[1, 0].imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[1, 0].set_title('(C) Normalized Confusion Matrix', fontsize=11, fontweight='bold')
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    classes = ['Human', 'LLM']
    axes[1, 0].set_xticks([0, 1])
    axes[1, 0].set_xticklabels(classes, fontsize=10)
    axes[1, 0].set_yticks([0, 1])
    axes[1, 0].set_yticklabels(classes, fontsize=10)
    axes[1, 0].set_ylabel('True Label', fontsize=10)
    axes[1, 0].set_xlabel('Predicted Label', fontsize=10)

    for i in range(2):
        for j in range(2):
            axes[1, 0].text(j, i, f"{cm[i, j]:,}\n({cm_norm[i, j]*100:.1f}%)",
                            ha="center", va="center",
                            color="white" if cm_norm[i, j] > 0.5 else "black",
                            fontsize=10, fontweight='bold')

    # Panel D: Probability Density Distribution
    df_prob = pd.DataFrame({'prob': probs_llm, 'gen_type': generation_types})
    for gen_type in ['human_full', 'full_rewrite', 'synthetic_partial', 'prompt_partial']:
        sub = df_prob[df_prob['gen_type'] == gen_type]
        if not sub.empty:
            sns.kdeplot(sub['prob'], ax=axes[1, 1], label=gen_type, fill=True, alpha=0.25)

    axes[1, 1].set_xlabel('Predicted Probability $P(\\text{LLM})$', fontsize=10)
    axes[1, 1].set_ylabel('Density', fontsize=10)
    axes[1, 1].set_title('(D) Output Probability Distribution', fontsize=11, fontweight='bold')
    axes[1, 1].legend(loc='upper center', fontsize=8)
    axes[1, 1].grid(True, linestyle='--', alpha=0.5)

    out_path = os.path.join(save_dir, f"{model_name}_evaluation_plots.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_partial_sensitivity(test_df: pd.DataFrame, probs_llm: np.ndarray, save_dir: str) -> str:
    """Plots detector detection rate vs. LLM ratio for synthetic vs. prompt partials."""
    os.makedirs(save_dir, exist_ok=True)
    df_plot = test_df.copy()
    df_plot['prob_llm'] = probs_llm
    df_plot['detected'] = (probs_llm >= 0.5).astype(int)

    partial_df = df_plot[df_plot['generation_type'].isin(['prompt_partial', 'synthetic_partial'])]
    
    if partial_df.empty:
        return ""

    summary = partial_df.groupby(['llm_ratio', 'generation_type'])['detected'].mean().reset_index()
    summary['detection_rate_pct'] = summary['detected'] * 100

    plt.figure(figsize=(7, 4.5), dpi=300)
    sns.lineplot(
        data=summary, x='llm_ratio', y='detection_rate_pct', hue='generation_type',
        marker='o', linewidth=2.5, palette=['#d95f02', '#7570b3']
    )
    plt.title('Detector Sensitivity across Partial LLM Ratios', fontsize=11, fontweight='bold')
    plt.xlabel('LLM Content Ratio', fontsize=10)
    plt.ylabel('Detection Rate (%)', fontsize=10)
    plt.xticks([0.25, 0.50, 0.75], ['25%', '50%', '75%'])
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title='Mixing Type', fontsize=9)

    out_path = os.path.join(save_dir, "partial_sensitivity_curve.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path