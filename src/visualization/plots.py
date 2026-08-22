import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from src.models.registry import get_model_color, get_model_display_name, normalize_model_name

def set_academic_plot_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'lines.linewidth': 1.8,
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })

def plot_zoomed_roc_curves(prediction_csvs: Dict[str, Union[str, Path]], scope: str, output_path: Union[str, Path], max_fpr: float=0.05, target_fpr: float=0.01):
    set_academic_plot_style()
    (fig, ax) = plt.subplots(figsize=(7, 5.5), dpi=300)
    for (model_key, csv_path) in prediction_csvs.items():
        p_path = Path(csv_path)
        if not p_path.exists():
            continue
        df = pd.read_csv(p_path)
        if 'label' not in df.columns or 'predicted_prob' not in df.columns:
            continue
        y_true = df['label'].astype(int).values
        y_score = df['predicted_prob'].astype(float).values
        if len(np.unique(y_true)) < 2:
            continue
        (fpr, tpr, _) = roc_curve(y_true, y_score)
        try:
            pauc = roc_auc_score(y_true, y_score, max_fpr=target_fpr)
            pauc_str = f'pAUC={pauc:.4f}'
        except Exception:
            pauc_str = 'pAUC=N/A'
        color = get_model_color(model_key)
        model_label = get_model_display_name(model_key)
        ax.plot(fpr, tpr, label=f'{model_label} ({pauc_str})', color=color)
    ax.axvline(x=target_fpr, color='black', linestyle=':', alpha=0.7, label=f'Target FPR = {target_fpr * 100:.0f}%')
    ax.set_xlim([0.0, max_fpr])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.set_title(f'Zoomed ROC Operating Regime [{scope.upper()} ABSTRACTS]')
    ax.grid(True)
    ax.legend(loc='lower right', frameon=True)
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_p, bbox_inches='tight')
    plt.close()
    print(f' -> [Saved Plot] Zoomed ROC exported to: {out_p}')

def plot_rewrite_sensitivity(summary_json_paths: List[Union[str, Path]], scope: str, output_path: Union[str, Path]):
    summaries = []
    for p in summary_json_paths:
        p = Path(p)
        if p.exists():
            try:
                summaries.append(json.loads(p.read_text(encoding='utf-8')))
            except Exception:
                pass
    if not summaries:
        return
    set_academic_plot_style()
    (fig, ax) = plt.subplots(figsize=(7, 5), dpi=300)
    bucket_keys = ['25pct', '50pct', '75pct']
    bucket_labels = ['25%', '50%', '75%']
    x = np.arange(len(bucket_labels))
    for s in summaries:
        raw_name = s.get('model_name', 'unknown')
        rob = s.get('robustness_ratios', {})
        rates = []
        valid = False
        for k in bucket_keys:
            if k in rob and 'flagged_pct' in rob[k] and (rob[k]['flagged_pct'] is not None):
                rates.append(rob[k]['flagged_pct'])
                valid = True
            else:
                rates.append(np.nan)
        if valid:
            color = get_model_color(raw_name)
            lbl = get_model_display_name(raw_name)
            ax.plot(x, rates, marker='o', linewidth=2.0, label=lbl, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels)
    ax.set_xlabel('LLM Rewrite Bucket')
    ax.set_ylabel('Detection Rate (% Flagged at $\\text{FPR} \\le 1\\%$)')
    ax.set_ylim([0, 105])
    ax.set_title(f'Sensitivity Across Rewrite Buckets [{scope.upper()} ABSTRACTS]')
    ax.grid(True)
    ax.legend(loc='lower right', frameon=True)
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_p, bbox_inches='tight')
    plt.close()
    print(f' -> [Saved Plot] Rewrite sensitivity plot exported to: {out_p}')

def plot_feature_importance(feature_csv: Union[str, Path], scope: str, output_path: Union[str, Path], top_n: int=15):
    f_path = Path(feature_csv)
    if not f_path.exists():
        return
    df = pd.read_csv(f_path)
    if df.empty or 'weight' not in df.columns or 'feature' not in df.columns:
        return
    set_academic_plot_style()
    top_llm = df.sort_values(by='weight', ascending=False).head(top_n)
    top_human = df.sort_values(by='weight', ascending=True).head(top_n)
    combined = pd.concat([top_human, top_llm]).drop_duplicates(subset=['feature']).sort_values('weight').reset_index(drop=True)
    (fig, ax) = plt.subplots(figsize=(8, max(5, int(len(combined) * 0.35))), dpi=300)
    colors = ['#2b8cbe' if w < 0 else '#e34a33' for w in combined['weight']]
    y_pos = np.arange(len(combined))
    ax.barh(y_pos, combined['weight'], color=colors, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(combined['feature'])
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Linear SVM Weight (Learned Margin Coefficient)')
    ax.set_title(f'Top Discriminative Feature Weights [{scope.upper()}]\n(Blue: Pushes Human | Red: Pushes LLM)')
    ax.grid(True, axis='x')
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_p, bbox_inches='tight')
    plt.close()
    print(f' -> [Saved Plot] Feature importance plot saved to: {out_p}')

def plot_cvar_trajectory(cvar_csv: Union[str, Path], scope: str, output_path: Union[str, Path]):
    csv_p = Path(cvar_csv)
    if not csv_p.exists():
        return
    df = pd.read_csv(csv_p)
    if df.empty or 'step' not in df.columns:
        return
    set_academic_plot_style()
    (fig, ax1) = plt.subplots(figsize=(7.5, 4.8), dpi=300)
    lines = []
    has_eta = 'eta' in df.columns and (df['eta'].dropna().count() > 0)
    has_multiscale = ('eta_doc' in df.columns and df['eta_doc'].dropna().count() > 0) or ('eta_sent' in df.columns and df['eta_sent'].dropna().count() > 0)
    ax1.set_xlabel('Global Optimization Steps')
    ax1.set_ylabel('Learned Cutoff $\\eta$ (Upper 1% Tail Threshold)')
    if has_multiscale:
        df_doc = df.dropna(subset=['eta_doc']).sort_values('step')
        df_sent = df.dropna(subset=['eta_sent']).sort_values('step')
        if not df_doc.empty:
            l_doc = ax1.plot(df_doc['step'], df_doc['eta_doc'], color='#d62728', linewidth=2.0, label='$\\eta_{\\text{doc}}$ (Document Tail)')
            lines.extend(l_doc)
        if not df_sent.empty:
            l_sent = ax1.plot(df_sent['step'], df_sent['eta_sent'], color='#ff7f0e', linestyle='--', linewidth=2.0, label='$\\eta_{\\text{sent}}$ (Sentence Tail)')
            lines.extend(l_sent)
    elif has_eta:
        df_clean = df.dropna(subset=['eta']).sort_values('step')
        l_eta = ax1.plot(df_clean['step'], df_clean['eta'], color='#d62728', linewidth=2.2, label='CVaR Cutoff $\\eta$')
        lines.extend(l_eta)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax2 = ax1.twinx()
    color_loss = '#1f77b4'
    ax2.set_ylabel('Batch Training Loss', color=color_loss)
    df_loss = df.dropna(subset=['train_loss']).sort_values('step')
    if not df_loss.empty:
        l_loss = ax2.plot(df_loss['step'], df_loss['train_loss'], color=color_loss, alpha=0.45, linewidth=1.2, label='Train Loss')
        lines.extend(l_loss)
        ax2.tick_params(axis='y', labelcolor=color_loss)
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', frameon=True)
    plt.title(f'mDeBERTa-v3 CVaR-DRO Parameter Dynamics [{scope.upper()} ABSTRACTS]')
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_p, bbox_inches='tight')
    plt.close()
    print(f' -> [Saved Plot] CVaR trajectory plot exported to: {out_p}')