# src/visualization/plots.py

from pathlib import Path
from typing import Dict, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score


def set_academic_plot_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "lines.linewidth": 1.8,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })


def plot_zoomed_roc_curves(
    prediction_csvs: Dict[str, Union[str, Path]],
    scope: str,
    output_path: Union[str, Path],
    max_fpr: float = 0.05,
    target_fpr: float = 0.01
):
    """
    Plots high-resolution ROC curves zoomed to FPR in [0, max_fpr] comparing all models.
    """
    set_academic_plot_style()
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=300)

    colors = {
        "svm": "#1f77b4",
        "mdeberta": "#d62728",
        "fdgpt": "#2ca02c",
        "fast_detectgpt": "#2ca02c",
        "stat_trajectory": "#9467bd",
        "stat": "#9467bd"
    }

    labels_map = {
        "svm": "Linear SVM + Stylometrics",
        "mdeberta": "mDeBERTa-v3 (CVaR-DRO)",
        "fdgpt": "Fast-DetectGPT",
        "stat_trajectory": "LLM Trajectory (Ours)",
        "stat": "LLM Trajectory (Ours)"
    }

    for model_key, csv_path in prediction_csvs.items():
        p_path = Path(csv_path)
        if not p_path.exists():
            continue

        df = pd.read_csv(p_path)
        if "label" not in df.columns or "predicted_prob" not in df.columns:
            continue

        y_true = df["label"].astype(int).values
        y_score = df["predicted_prob"].astype(float).values

        if len(np.unique(y_true)) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_true, y_score)
        pauc = roc_auc_score(y_true, y_score, max_fpr=target_fpr)
        
        color = colors.get(model_key.lower(), None)
        model_label = labels_map.get(model_key.lower(), model_key.upper())

        ax.plot(fpr, tpr, label=f"{model_label} (pAUC={pauc:.4f})", color=color)

    # Reference threshold line at target FPR
    ax.axvline(x=target_fpr, color="black", linestyle=":", alpha=0.7, label=f"Target FPR = {target_fpr*100:.0f}%")

    ax.set_xlim([0.0, max_fpr])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title(f"Zoomed ROC Operating Regime [{scope.upper()} ABSTRACTS]")
    ax.grid(True)
    ax.legend(loc="lower right", frameon=True)

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_p, bbox_inches="tight")
    plt.close()
    print(f" -> [Saved Plot] Zoomed ROC exported to: {out_p}")


def plot_feature_importance(
    feature_csv: Union[str, Path],
    scope: str,
    output_path: Union[str, Path],
    top_n: int = 15
):
    """
    Plots top N features pushing towards LLM vs Human for Linear SVM.
    """
    f_path = Path(feature_csv)
    if not f_path.exists():
        return

    df = pd.read_csv(f_path)
    if df.empty or "weight" not in df.columns or "feature" not in df.columns:
        return

    set_academic_plot_style()
    
    top_llm = df.sort_values(by="weight", ascending=False).head(top_n)
    top_human = df.sort_values(by="weight", ascending=True).head(top_n)
    combined = pd.concat([top_human, top_llm]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    colors = ["#2b8cbe" if w < 0 else "#e34a33" for w in combined["weight"]]

    y_pos = np.arange(len(combined))
    ax.barh(y_pos, combined["weight"], color=colors, align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(combined["feature"])
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)

    ax.set_xlabel("Linear SVM Weight (Learned Margin Coefficient)")
    ax.set_title(f"Top Discriminative Feature Weights [{scope.upper()}]\n(Blue: Pushes Human | Red: Pushes LLM)")
    ax.grid(True, axis="x")

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_p, bbox_inches="tight")
    plt.close()
    print(f" -> [Saved Plot] Feature importance plot saved to: {out_p}")


def plot_cvar_trajectory(cvar_csv: Union[str, Path], scope: str, output_path: Union[str, Path]):
    """
    Plots the learned CVaR cutoff parameter eta across optimization steps.
    """
    csv_p = Path(cvar_csv)
    if not csv_p.exists():
        return

    df = pd.read_csv(csv_p)
    df_clean = df.dropna(subset=["eta"]).sort_values("step")
    if df_clean.empty:
        return

    set_academic_plot_style()
    fig, ax1 = plt.subplots(figsize=(7.5, 4.8), dpi=300)

    color_eta = "#d62728"
    ax1.set_xlabel("Global Optimization Steps")
    ax1.set_ylabel(r"Learned Cutoff $\eta$ (Upper 1% Tail Threshold)", color=color_eta)
    line1 = ax1.plot(df_clean["step"], df_clean["eta"], color=color_eta, linewidth=2.2, label=r"CVaR Cutoff $\eta$")
    ax1.tick_params(axis="y", labelcolor=color_eta)
    ax1.grid(True, linestyle="--", alpha=0.3)

    ax2 = ax1.twinx()
    color_loss = "#1f77b4"
    ax2.set_ylabel("Batch Training Loss", color=color_loss)
    df_loss = df.dropna(subset=["train_loss"])
    line2 = ax2.plot(df_loss["step"], df_loss["train_loss"], color=color_loss, alpha=0.45, linewidth=1.2, label="Train Loss")
    ax2.tick_params(axis="y", labelcolor=color_loss)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", frameon=True)

    plt.title(f"mDeBERTa-v3 CVaR-DRO Parameter Dynamics [{scope.upper()} ABSTRACTS]")
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_p, bbox_inches="tight")
    plt.close()
    print(f" -> [Saved Plot] CVaR trajectory plot exported to: {out_p}")