#!/usr/bin/env python3
# scripts/train_svm.py

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# Calculate project root dynamically (~/detection)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_FEATURES_DIR = PROJECT_ROOT / "data_static" / "model_features"

from src.data.data_loader import DataFilter, DetectionDataManager


def _ensure_sliceable_X(X):
    """Converts sparse matrices to CSR and lists/arrays to numpy array for safe indexing."""
    if sp.issparse(X):
        return X.tocsr()
    if isinstance(X, list):
        return np.asarray(X)
    return X


def load_best_svm_hyperparameters(scope: str, outputs_dir: Path) -> dict:
    """Loads tuned hyperparameters from outputs/svm/<scope>/best_hyperparameters.json."""
    config_path = outputs_dir / "svm" / scope / "best_hyperparameters.json"
    if not config_path.exists():
        print(f"[WARNING] No tuned parameters found at '{config_path}'. Using default C=1.0.")
        return {"C": 1.0, "penalty": "l2", "loss": "squared_hinge", "class_weight": None}

    with open(config_path, "r") as f:
        data = json.load(f)
        print(f"[LOADED PARAMS] Loaded best hyperparameters from: {config_path}")
        return data.get("best_hyperparameters", data)


def build_linear_svm(best_params: dict):
    penalty = best_params.get("penalty", "l2")
    loss = best_params.get("loss", "squared_hinge")

    # Set dual parameter for backward and forward compatibility with scikit-learn
    if penalty == "l1":
        loss = "squared_hinge"
        dual = False
    elif loss == "hinge":
        dual = True
    else:
        dual = False

    return LinearSVC(
        C=float(best_params.get("C", 1.0)),
        penalty=penalty,
        loss=loss,
        dual=dual,
        class_weight=best_params.get("class_weight", None),
        random_state=42,
        max_iter=10000,
    )


def calculate_optimal_threshold_on_dev(model, dev_data, max_fpr=0.01):
    """Calculates optimal decision threshold directly on the dev set using the fitted model.
    
    Since the model was fitted on 'train', 'dev' is already unseen holdout data.
    """
    print(f"\nCalculating optimal decision threshold on 'dev' set (Target Max FPR: {max_fpr})...")

    X_dev = _ensure_sliceable_X(dev_data["X"])
    y_dev = np.asarray(dev_data["y"])

    # Get predicted probabilities directly from the model trained on 'train'
    probs_dev = model.predict_proba(X_dev)[:, 1]

    # Compute ROC curve to get exact mathematical thresholds
    fpr, tpr, thresholds = roc_curve(y_dev, probs_dev)

    # Filter candidate thresholds satisfying FPR constraint (<= max_fpr)
    valid_indices = np.where(fpr <= max_fpr)[0]

    if len(valid_indices) > 0:
        # Choose the threshold that maximizes TPR subject to FPR <= max_fpr
        best_idx = valid_indices[np.argmax(tpr[valid_indices])]
        best_threshold = float(thresholds[best_idx])
    else:
        best_threshold = 0.5

    # Evaluate exact stats at best_threshold
    preds = (probs_dev >= best_threshold).astype(int)
    cm = confusion_matrix(y_dev, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    actual_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    actual_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    acc = (tp + tn) / len(y_dev)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = actual_tpr
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    dev_stats = {
        "optimal_threshold": round(best_threshold, 6),
        "dev_fpr": round(float(actual_fpr), 6),
        "dev_tpr_at_1fpr": round(float(actual_tpr), 6),
        "dev_accuracy": round(float(acc), 4),
        "dev_f1": round(float(f1), 4),
        "dev_precision": round(float(prec), 4),
        "dev_recall": round(float(rec), 4),
        "dev_specificity": round(float(tn / (tn + fp)), 4) if (tn + fp) > 0 else 0.0,
    }

    print(f"Optimal Decision Threshold (\u03c4*): {best_threshold:.6f}")
    print(f"  - Dev FPR        : {dev_stats['dev_fpr']:.6f}")
    print(f"  - Dev TPR@1% FPR : {dev_stats['dev_tpr_at_1fpr']:.4f}")
    print(f"  - Dev F1 Score   : {dev_stats['dev_f1']:.4f}")

    return best_threshold, dev_stats


def evaluate_and_plot_results(
    split_data, df_raw, split_name, scope, model, optimal_threshold, save_dir, left_out_model=None
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    X_mat = _ensure_sliceable_X(split_data["X"])
    labels = np.asarray(split_data["y"])

    probs_llm = model.predict_proba(X_mat)[:, 1]
    preds = (probs_llm >= optimal_threshold).astype(int)

    if len(np.unique(labels)) > 1:
        fpr, tpr, _ = roc_curve(labels, probs_llm)
        roc_auc_val = float(auc(fpr, tpr))
        try:
            pauc_001_val = float(roc_auc_score(labels, probs_llm, max_fpr=0.01))
        except Exception:
            pauc_001_val = 0.0
        precision_curve, recall_curve, _ = precision_recall_curve(labels, probs_llm)
        pr_auc_val = float(average_precision_score(labels, probs_llm))
        tpr_at_1fpr = float(tpr[np.where(fpr <= 0.01)[0][-1]]) if len(np.where(fpr <= 0.01)[0]) > 0 else 0.0
    else:
        fpr, tpr = np.array([0, 1]), np.array([0, 1])
        precision_curve, recall_curve = np.array([1, 0]), np.array([0, 1])
        roc_auc_val, pauc_001_val, pr_auc_val, tpr_at_1fpr = 0.0, 0.0, 0.0, 0.0

    acc = float(accuracy_score(labels, preds))
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    prec, rec, f1 = float(prec), float(rec), float(f1)

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    metrics_summary = {
        "Scope": scope,
        "Split": split_name,
        "Left-Out Model": left_out_model if left_out_model else "None",
        "Total Samples": int(len(labels)),
        "Optimal Threshold (\u03c4*)": round(float(optimal_threshold), 6),
        "pAUC @ max FPR 0.01": round(pauc_001_val, 6),
        "ROC-AUC": round(roc_auc_val, 4),
        "PR-AUC (AP)": round(pr_auc_val, 4),
        "Accuracy": round(acc, 4),
        "F1-Score": round(f1, 4),
        "Precision": round(prec, 4),
        "Recall (Sensitivity)": round(rec, 4),
        "Specificity": round(specificity, 4),
        "TPR @ 1% FPR": round(tpr_at_1fpr, 4),
    }

    print(f"\n--- PERFORMANCE SUMMARY [{split_name.upper()}] ---")
    if left_out_model:
        print(f"  Held-Out Generator          : {left_out_model}")
    for k, v in metrics_summary.items():
        if k != "Left-Out Model":
            print(f"  {k:<28}: {v}")

    gen_col = None
    for col in ["model_name", "generator_model", "model"]:
        if col in df_raw.columns:
            gen_col = col
            break

    per_model_results = []

    if gen_col and len(df_raw) == len(labels):
        df_eval = df_raw.copy()
        df_eval["prob_llm"] = probs_llm
        df_eval["pred"] = preds
        df_eval["label"] = labels

        human_df = df_eval[df_eval["label"] == 0]

        for generator in df_eval[gen_col].unique():
            if str(generator).lower() == "human":
                continue

            is_left_out = bool(left_out_model and str(generator).lower() == left_out_model.lower())
            llm_sub = df_eval[df_eval[gen_col] == generator]
            combined = pd.concat([human_df, llm_sub])

            sub_labels = combined["label"].values
            sub_probs = combined["prob_llm"].values
            sub_preds = combined["pred"].values

            sub_auc = float(roc_auc_score(sub_labels, sub_probs)) if len(np.unique(sub_labels)) > 1 else 0.0
            sub_acc = float(accuracy_score(sub_labels, sub_preds))
            sub_prec, sub_rec, sub_f1, _ = precision_recall_fscore_support(
                sub_labels, sub_preds, average="binary", zero_division=0
            )

            gen_display = f"{generator} (HELD-OUT)" if is_left_out else str(generator)

            per_model_results.append({
                "Generator": gen_display,
                "LLM Samples": int(len(llm_sub)),
                "ROC-AUC": round(sub_auc, 4),
                "Accuracy": round(sub_acc, 4),
                "F1-Score": round(float(sub_f1), 4),
                "Precision": round(float(sub_prec), 4),
                "Recall": round(float(sub_rec), 4),
                "Is Held-Out": is_left_out,
            })

        if per_model_results:
            print("\n--- PER-GENERATOR BREAKDOWN ---")
            print(pd.DataFrame(per_model_results).to_string(index=False))

    # Generate Plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    plt.subplots_adjust(wspace=0.25, hspace=0.3)

    label_suffix = f" (LOO: {left_out_model})" if left_out_model else ""
    axes[0, 0].plot(fpr, tpr, color="#2b5c8f", lw=2, label=f"Linear SVM ({scope.upper()}){label_suffix}\npAUC@0.01={pauc_001_val:.4f}")
    axes[0, 0].axvline(x=0.01, color="red", linestyle=":", lw=1.5, label="FPR = 0.01")
    axes[0, 0].plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5, label="Random")
    axes[0, 0].set_xlabel("False Positive Rate", fontsize=11)
    axes[0, 0].set_ylabel("True Positive Rate", fontsize=11)
    axes[0, 0].set_title(f"(A) ROC Curve ({split_name.capitalize()})", fontsize=12, fontweight="bold")
    axes[0, 0].legend(loc="lower right", fontsize=10)
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    axes[0, 1].plot(recall_curve, precision_curve, color="#d95f02", lw=2, label=f"AP={pr_auc_val:.4f}")
    axes[0, 1].set_xlabel("Recall", fontsize=11)
    axes[0, 1].set_ylabel("Precision", fontsize=11)
    axes[0, 1].set_title(f"(B) Precision-Recall Curve ({split_name.capitalize()})", fontsize=12, fontweight="bold")
    axes[0, 1].legend(loc="lower left", fontsize=10)
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    cm_sum = cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.divide(cm.astype("float"), cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum != 0)
    
    im = axes[1, 0].imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    axes[1, 0].set_title(f"(C) Confusion Matrix (\u03c4* = {optimal_threshold:.4f})", fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    classes = ["Human", "LLM"]
    tick_marks = np.arange(len(classes))
    axes[1, 0].set_xticks(tick_marks)
    axes[1, 0].set_xticklabels(classes, fontsize=10)
    axes[1, 0].set_yticks(tick_marks)
    axes[1, 0].set_yticklabels(classes, fontsize=10)
    axes[1, 0].set_ylabel("True Label", fontsize=11)
    axes[1, 0].set_xlabel("Predicted Label", fontsize=11)

    for i in range(2):
        for j in range(2):
            axes[1, 0].text(
                j, i, f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > 0.5 else "black",
                fontsize=11, fontweight="bold",
            )

    human_probs = probs_llm[labels == 0]
    llm_probs = probs_llm[labels == 1]

    if len(human_probs) > 0:
        axes[1, 1].hist(human_probs, bins=25, alpha=0.6, color="#1b9e77", label="Human", density=True)
    if len(llm_probs) > 0:
        axes[1, 1].hist(llm_probs, bins=25, alpha=0.6, color="#7570b3", label="LLM", density=True)
        
    axes[1, 1].axvline(x=optimal_threshold, color="black", linestyle="--", lw=2, label=f"\u03c4*={optimal_threshold:.4f}")
    axes[1, 1].set_xlabel("Predicted Probability P(LLM)", fontsize=11)
    axes[1, 1].set_ylabel("Density", fontsize=11)
    axes[1, 1].set_title(f"(D) Probability Density ({split_name.capitalize()})", fontsize=12, fontweight="bold")
    axes[1, 1].legend(loc="upper center", fontsize=10)
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    plot_path = save_dir / f"{split_name}_evaluation_plots.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"[SAVED PLOT] Plot saved to: '{plot_path}'")
    plt.close()

    # Save LaTeX Table
    latex_table_path = save_dir / f"{split_name}_metrics_table.tex"
    loo_tex_info = f" (Held-out: {left_out_model})" if left_out_model else ""
    with open(latex_table_path, "w") as f:
        f.write("% Auto-generated LaTeX table\n")
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write(f"\\caption{{SVM {split_name.capitalize()} Split Performance ({scope.upper()}){loo_tex_info}. Threshold $\\tau^* = {optimal_threshold:.4f}$.}}\n")
        f.write("\\label{tab:svm_" + scope + "_" + split_name + "}\n")
        f.write("\\begin{tabular}{lcccccc}\n\\hline\n")
        f.write("\\textbf{Split} & \\textbf{pAUC @ 0.01} & \\textbf{ROC-AUC} & \\textbf{Accuracy} & \\textbf{F1-Score} & \\textbf{Precision} & \\textbf{Recall} \\\\\n\\hline\n")
        f.write(f"{split_name.capitalize()} & {pauc_001_val:.4f} & {roc_auc_val:.4f} & {acc:.4f} & {f1:.4f} & {prec:.4f} & {rec:.4f} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

    # Save Full Logits & Predictions (Full Data)
    if len(df_raw) == len(probs_llm):
        df_logits = df_raw.copy()
    else:
        df_logits = pd.DataFrame({"label": labels})

    df_logits["prob_llm"] = probs_llm
    df_logits["pred_llm"] = preds
    if left_out_model and gen_col in df_logits.columns:
        df_logits["is_held_out_generator"] = df_logits[gen_col].astype(str).str.lower() == left_out_model.lower()

    csv_path = save_dir / f"{split_name}_logits_analysis.csv"
    df_logits.to_csv(csv_path, index=False)
    print(f"[SAVED LOGITS] Full prediction data saved to: '{csv_path}'")

    return metrics_summary, per_model_results


def run_full_training_for_scope(scope: str, args, manager: DetectionDataManager):
    outputs_base = Path(args.outputs_dir) if args.outputs_dir else DEFAULT_OUTPUTS_DIR
    features_dir = Path(args.features_dir) if args.features_dir else DEFAULT_FEATURES_DIR / f"svm_{scope}"

    train_path = features_dir / "train.joblib"
    dev_path = features_dir / "dev.joblib"
    test_path = features_dir / "test.joblib"

    if not train_path.exists() or not dev_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Pre-extracted features for scope '{scope}' missing at: {features_dir}.\n"
            f"Please run feature extraction first."
        )

    # Determine scope-specific sample size
    if scope == "full" and args.sample_size_full is not None:
        sample_size = args.sample_size_full
    elif scope == "sentence" and args.sample_size_sentence is not None:
        sample_size = args.sample_size_sentence
    else:
        sample_size = args.sample_size

    # Load Data and Features
    train_data = joblib.load(train_path)
    dev_data = joblib.load(dev_path)
    test_data = joblib.load(test_path)

    X_train = _ensure_sliceable_X(train_data["X"])
    y_train = np.asarray(train_data["y"])
    X_dev = _ensure_sliceable_X(dev_data["X"])
    y_dev = np.asarray(dev_data["y"])
    X_test = _ensure_sliceable_X(test_data["X"])
    y_test = np.asarray(test_data["y"])

    train_df = manager.filter_dataframe(DataFilter(splits=["train"], scopes=[scope]))
    dev_df = manager.filter_dataframe(DataFilter(splits=["dev"], scopes=[scope]))
    test_df = manager.filter_dataframe(DataFilter(splits=["test"], scopes=[scope]))

    gen_col = None
    for col in ["model_name", "generator_model", "model"]:
        if col in train_df.columns:
            gen_col = col
            break

    # Handle Leave-One-Out (LOO) setup if --loo is passed
    left_out_model = None
    if args.loo:
        if gen_col is None:
            raise ValueError("Cannot perform LOO filtering: Generator model column not found in dataframe.")

        # Identify unique LLM generators (excluding human class)
        all_generators = train_df[gen_col].dropna().unique()
        llm_generators = sorted([str(g) for g in all_generators if str(g).lower() != "human"])

        if args.loo.lower() == "random":
            rng = np.random.default_rng(args.seed)
            left_out_model = str(rng.choice(llm_generators))
        else:
            matched = [g for g in llm_generators if g.lower() == args.loo.lower()]
            if matched:
                left_out_model = matched[0]
            else:
                raise ValueError(
                    f"Specified LOO model '{args.loo}' not found. Available LLM generators: {llm_generators}"
                )

        # Filter Left-Out Model out of TRAIN set
        train_keep_mask = (train_df[gen_col].astype(str) != left_out_model).values
        X_train = X_train[train_keep_mask]
        y_train = y_train[train_keep_mask]
        train_df = train_df[train_keep_mask].reset_index(drop=True)

        # Filter Left-Out Model out of DEV set
        dev_keep_mask = (dev_df[gen_col].astype(str) != left_out_model).values
        X_dev = X_dev[dev_keep_mask]
        y_dev = y_dev[dev_keep_mask]
        dev_df = dev_df[dev_keep_mask].reset_index(drop=True)

        # Configure Output Directory under outputs/svm_loo/
        scope_dir = outputs_base / "svm_loo" / scope / left_out_model
    else:
        # Standard Output Directory under outputs/svm/
        scope_dir = outputs_base / "svm" / scope

    scope_dir.mkdir(parents=True, exist_ok=True)

    dev_data = {"X": X_dev, "y": y_dev}
    test_data = {"X": X_test, "y": y_test}

    print("\n" + "=" * 70)
    print(f" FULL LINEAR SVM TRAINING FOR SCOPE: '{scope.upper()}' ")
    print(f" LOO Active       : {bool(args.loo)}")
    if left_out_model:
        print(f" Left-Out Model   : '{left_out_model}' (Excluded from TRAIN and DEV)")
    print(f" Features Dir    : {features_dir}")
    print(f" Balanced Split  : {args.balanced} (50% Human / 50% LLM)")
    print(f" Target Size     : {sample_size if sample_size > 0 else 'FULL Pre-extracted'}")
    print(f" Output Directory: {scope_dir}")
    print("=" * 70 + "\n")

    best_params = load_best_svm_hyperparameters(scope=scope, outputs_dir=outputs_base)

    # Balanced Subsampling / Standard Subsampling on Train Set
    if args.balanced or (sample_size > 0 and sample_size < X_train.shape[0]):
        human_idx = np.where(y_train == 0)[0]
        llm_idx = np.where(y_train == 1)[0]

        if args.balanced:
            if sample_size > 0:
                n_per_class = min(sample_size // 2, len(human_idx), len(llm_idx))
            else:
                n_per_class = min(len(human_idx), len(llm_idx))

            rng = np.random.default_rng(args.seed)
            sampled_human = rng.choice(human_idx, size=n_per_class, replace=False)
            sampled_llm = rng.choice(llm_idx, size=n_per_class, replace=False)

            selected_idx = np.concatenate([sampled_human, sampled_llm])
            rng.shuffle(selected_idx)

            X_train = X_train[selected_idx]
            y_train = y_train[selected_idx]
            if len(train_df) == len(train_data["y"]):
                train_df = train_df.iloc[selected_idx].reset_index(drop=True)

        elif sample_size > 0 and sample_size < X_train.shape[0]:
            X_train, _, y_train, _, idx_tr, _ = train_test_split(
                X_train, y_train, np.arange(len(y_train)),
                train_size=sample_size, stratify=y_train, random_state=args.seed
            )
            if len(train_df) == len(train_data["y"]):
                train_df = train_df.iloc[idx_tr].reset_index(drop=True)

    # Sample Statistics & Model Type Breakdown
    n_human = int((y_train == 0).sum())
    n_llm = int((y_train == 1).sum())
    total_samples = len(y_train)

    print("\n-------------------------------------------------------------")
    print(f" SAMPLE DISTRIBUTION & MODEL BREAKDOWN [{scope.upper()}]")
    print("-------------------------------------------------------------")
    print(f" Total Training Samples : {total_samples}")
    print(f"   - Human Samples (0)  : {n_human} ({n_human / total_samples * 100:.1f}%)")
    print(f"   - LLM Samples   (1)  : {n_llm} ({n_llm / total_samples * 100:.1f}%)")

    if gen_col and len(train_df) == total_samples:
        print("\n Training Model Breakdown:")
        breakdown = train_df[gen_col].value_counts()
        for model_type, count in breakdown.items():
            pct = count / total_samples * 100
            print(f"   - {str(model_type):<20}: {count:>6d} ({pct:>5.1f}%)")
    print("-------------------------------------------------------------\n")

    # Fit Calibrated Model
    print("Fitting Calibrated Linear SVM model on train split...")
    base_svm = build_linear_svm(best_params)
    min_class_samples = int(np.min(np.bincount(y_train))) if len(np.unique(y_train)) > 1 else 1
    cal_cv = min(5, max(2, min_class_samples))

    calibrated_svm = CalibratedClassifierCV(estimator=base_svm, method="sigmoid", cv=cal_cv)
    calibrated_svm.fit(X_train, y_train)

    model_path = scope_dir / "model.joblib"
    joblib.dump(calibrated_svm, model_path)
    print(f"[MODEL SAVED] Calibrated SVM saved to: '{model_path}'")

    # Calculate Optimal Threshold directly using calibrated_svm on dev_data
    optimal_threshold, dev_threshold_stats = calculate_optimal_threshold_on_dev(
        model=calibrated_svm, dev_data=dev_data, max_fpr=0.01
    )

    # Dev Evaluation using optimal_threshold
    dev_metrics, _ = evaluate_and_plot_results(
        split_data=dev_data,
        df_raw=dev_df,
        split_name="dev",
        scope=scope,
        model=calibrated_svm,
        optimal_threshold=optimal_threshold,
        save_dir=scope_dir,
        left_out_model=left_out_model,
    )

    # Test Evaluation using optimal_threshold (evaluated across all generators in test)
    test_metrics, per_gen_metrics = evaluate_and_plot_results(
        split_data=test_data,
        df_raw=test_df,
        split_name="test",
        scope=scope,
        model=calibrated_svm,
        optimal_threshold=optimal_threshold,
        save_dir=scope_dir,
        left_out_model=left_out_model,
    )

    # Save Summaries
    summary_path = scope_dir / "paper_evaluation_summary.json"
    summary_json = {
        "scope": scope,
        "loo_active": bool(args.loo),
        "left_out_model": left_out_model,
        "is_balanced": bool(args.balanced),
        "training_sample_size": int(total_samples),
        "optimal_decision_threshold": float(optimal_threshold),
        "dev_threshold_statistics": dev_threshold_stats,
        "dev_metrics": dev_metrics,
        "test_metrics": test_metrics,
        "per_generator_metrics": per_gen_metrics,
        "best_hyperparameters": best_params,
        "model_path": str(model_path),
    }

    with open(summary_path, "w") as f:
        json.dump(summary_json, f, indent=4)

    params_json_path = scope_dir / "best_hyperparameters.json"
    hyperparams_data = {
        "classifier": "linear_svm",
        "kernel": "linear",
        "scope": scope,
        "loo_active": bool(args.loo),
        "left_out_model": left_out_model,
        "is_balanced": bool(args.balanced),
        "optimal_decision_threshold": float(optimal_threshold),
        "best_hyperparameters": best_params,
        "dev_threshold_statistics": dev_threshold_stats,
        "dev_metrics": dev_metrics,
        "test_metrics": test_metrics,
        "features_dir": str(features_dir),
    }
    with open(params_json_path, "w") as f:
        json.dump(hyperparams_data, f, indent=4)

    print(f"\n[SUMMARY SAVED] Comprehensive summary saved to: '{summary_path}'")
    print(f"[PARAMS UPDATED] Best parameters & threshold updated in: '{params_json_path}'")


def main():
    parser = argparse.ArgumentParser(description="Full Linear SVM training setup.")

    parser.add_argument(
        "--scopes",
        nargs="+",
        default=["full", "sentence"],
        choices=["full", "sentence"],
        help="List of scopes to train (default: full sentence)."
    )
    parser.add_argument(
        "--loo",
        type=str,
        default=None,
        help="Leave-One-Out mode. Pass 'random' to leave out a random generator model, or pass a specific model name (e.g. 'gpt-4')."
    )
    parser.add_argument("--balanced", "--balance_dataset", action="store_true", help="Balance training dataset (50%% Human, 50%% LLM).")
    parser.add_argument("--sample_size", type=int, default=-1, help="Fallback sample size for training (-1 for full data).")
    parser.add_argument("--sample_size_full", type=int, default=None, help="Sample size for 'full' scope training.")
    parser.add_argument("--sample_size_sentence", type=int, default=None, help="Sample size for 'sentence' scope training.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--features_dir", type=str, default=None, help="Custom features directory path.")
    parser.add_argument("--outputs_dir", type=str, default=None, help="Custom outputs directory path.")

    args = parser.parse_args()
    manager = DetectionDataManager()

    for scope in args.scopes:
        run_full_training_for_scope(scope=scope, args=args, manager=manager)

    output_base_display = args.outputs_dir if args.outputs_dir else (DEFAULT_OUTPUTS_DIR / ("svm_loo" if args.loo else "svm"))
    print("\n" + "=" * 70)
    print("[ALL DONE] Full Linear SVM training & evaluation complete for all requested scopes!")
    print(f"Results saved under: {output_base_display}")
    print("=" * 70)


if __name__ == "__main__":
    main()