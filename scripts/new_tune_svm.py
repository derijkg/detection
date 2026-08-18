#!/usr/bin/env python3
# scripts/tune_svm.py

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Optional, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from scipy.sparse import hstack
from sklearn.preprocessing import normalize

# Suppress ConvergenceWarnings and UserWarnings

# Calculate project root dynamically (~/detection)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_FEATURES_DIR = PROJECT_ROOT / "data_static" / "model_features"

from src.data.data_loader import DataFilter, DetectionDataManager

# Suppress Optuna verbose logging per trial
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ==========================================
# Loader Helper for Downstream Usage
# ==========================================
def load_best_svm_config(
    scope: str, 
    outputs_dir: Optional[Union[str, Path]] = None
) -> dict:
    base_dir = Path(outputs_dir) if outputs_dir else DEFAULT_OUTPUTS_DIR
    config_path = base_dir / "svm" / scope / "best_hyperparameters.json"

    if not config_path.exists():
        raise FileNotFoundError(f"No saved best SVM configuration found for scope '{scope}' at: {config_path}")

    with open(config_path, "r") as f:
        return json.load(f)


# ==========================================
# Optuna Objective Function (Linear Kernel)
# ==========================================
def optuna_objective(trial, X_tr_tfidf, X_tr_style, y_train, X_dv_tfidf, X_dv_style, y_dev):
    # Hyperparameters
    C = trial.suggest_float("C", 1e-4, 20.0, log=True)
    penalty = trial.suggest_categorical("penalty", ["l2", "l1"])
    class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])
    
    # Tune Stylometric Weight (0.0 means feature disabled, up to e.g. 5.0)
    style_weight = trial.suggest_float("stylometric_weight", 0.0, 5.0)

    # Combine matrices dynamically on the fly
    if style_weight > 0:
        X_tr_combined = hstack([X_tr_tfidf, X_tr_style * style_weight])
        X_dv_combined = hstack([X_dv_tfidf, X_dv_style * style_weight])
    else:
        X_tr_combined = X_tr_tfidf
        X_dv_combined = X_dv_tfidf

    # Apply global L2 normalization across all features
    X_train = normalize(X_tr_combined, norm="l2")
    X_dev = normalize(X_dv_combined, norm="l2")

    n_samples, n_features = X_train.shape

    if penalty == "l1":
        loss = "squared_hinge"
        dual = False
    else:
        loss = trial.suggest_categorical("loss", ["squared_hinge", "hinge"])
        dual = True if loss == "hinge" else (False if n_samples >= n_features else True)

    trial.set_user_attr("loss", loss)
    trial.set_user_attr("dual", dual)

    try:
        clf = LinearSVC(
            C=C,
            penalty=penalty,
            loss=loss,
            dual=dual,
            class_weight=class_weight,
            random_state=42,
            max_iter=10000,
            tol=1e-4,
        )
        clf.fit(X_train, y_train)

        dev_scores = clf.decision_function(X_dev)
        pauc_001 = roc_auc_score(y_dev, dev_scores, max_fpr=0.01)
        return pauc_001
    except Exception:
        return 0.5


# ==========================================
# Optuna Trial Progress Callback
# ==========================================
def print_trial_callback(study, trial):
    if trial.value is not None:
        best_val = study.best_value
        print(f"   [Trial #{trial.number:02d}] Current pAUC@0.01: {trial.value:.6f} | Best So Far: {best_val:.6f}")


# ==========================================
# Plot Optuna Study Progress
# ==========================================
def plot_optuna_progress(study, save_path: Path, scope: str):
    trial_numbers = [t.number for t in study.trials if t.value is not None]
    trial_values = [t.value for t in study.trials if t.value is not None]

    best_so_far = []
    current_best = -1.0
    for val in trial_values:
        if val > current_best:
            current_best = val
        best_so_far.append(current_best)

    plt.figure(figsize=(10, 5), dpi=300)
    plt.plot(trial_numbers, trial_values, marker="o", linestyle="--", color="#2b5c8f", alpha=0.6, label="Trial pAUC @ 0.01")
    plt.plot(trial_numbers, best_so_far, marker="s", linestyle="-", color="#d95f02", linewidth=2.5, label="Best pAUC So Far")

    plt.xlabel("Trial Number", fontsize=11)
    plt.ylabel("Dev pAUC @ max FPR 0.01", fontsize=11)
    plt.title(f"Optuna Linear SVM Tuning Progress ({scope.upper()})", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[PLOT SAVED] Optuna study progress plot saved to: '{save_path}'")


# ==========================================
# Tune Single Scope Workflow
# ==========================================
def tune_svm_for_scope(scope: str, args, manager: DetectionDataManager):
    outputs_base = Path(args.outputs_dir) if args.outputs_dir else DEFAULT_OUTPUTS_DIR
    scope_dir = outputs_base / "svm" / scope
    scope_dir.mkdir(parents=True, exist_ok=True)

    if args.features_dir:
        base_feat = Path(args.features_dir)
        features_dir = base_feat / f"svm_{scope}" if (base_feat / f"svm_{scope}").exists() else base_feat
    else:
        features_dir = DEFAULT_FEATURES_DIR / f"svm_{scope}"

    train_path = features_dir / "train.joblib"
    dev_path = features_dir / "dev.joblib"

    if not train_path.exists() or not dev_path.exists():
        raise FileNotFoundError(
            f"Pre-extracted SVM features for scope '{scope}' not found at: {features_dir}.\n"
            f"Please run 'python scripts/features_svm.py' first to extract features."
        )

    # Determine sample size
    if scope == "full" and args.sample_size_full is not None:
        sample_size = args.sample_size_full
    elif scope == "sentence" and args.sample_size_sentence is not None:
        sample_size = args.sample_size_sentence
    else:
        sample_size = args.sample_size

    print("\n" + "=" * 70)
    print(f" TUNING LINEAR SVM FOR SCOPE: '{scope.upper()}' ")
    print(f" Features Source : {features_dir}")
    print(f" Target Metric   : pAUC @ max FPR <= 0.01")
    print(f" Balanced Split  : {args.balanced} (50% Human / 50% LLM)")
    print(f" Target Size     : {sample_size if sample_size > 0 else 'FULL Pre-extracted'}")
    print(f" Output Folder   : {scope_dir}")
    print("=" * 70 + "\n")

    # 1. Load pre-extracted feature matrices and raw DataFrame

    train_data = joblib.load(train_path)
    dev_data = joblib.load(dev_path)

    X_train_tfidf = train_data["X_tfidf"]
    X_train_style = train_data["X_style"]
    y_train = np.asarray(train_data["y"])

    X_dev_tfidf = dev_data["X_tfidf"]
    X_dev_style = dev_data["X_style"]
    y_dev = np.asarray(dev_data["y"])

    train_df = manager.filter_dataframe(DataFilter(splits=["train"], scopes=[scope]))
    gen_col = "model_name" if "model_name" in train_df.columns else ("generator_model" if "generator_model" in train_df.columns else None)

    # 2. Balanced Subsampling / Standard Subsampling
    if args.balanced or (sample_size > 0 and sample_size < X_train_tfidf.shape[0]):
        human_idx = np.where(y_train == 0)[0]
        llm_idx = np.where(y_train == 1)[0]

        if args.balanced:
            n_per_class = min(sample_size // 2, len(human_idx), len(llm_idx)) if sample_size > 0 else min(len(human_idx), len(llm_idx))
            rng = np.random.default_rng(args.seed)
            sampled_human = rng.choice(human_idx, size=n_per_class, replace=False)
            sampled_llm = rng.choice(llm_idx, size=n_per_class, replace=False)

            selected_idx = np.concatenate([sampled_human, sampled_llm])
            rng.shuffle(selected_idx)

            X_train_tfidf = X_train_tfidf[selected_idx]
            X_train_style = X_train_style[selected_idx]
            y_train = y_train[selected_idx]
            if len(train_df) == len(train_data["y"]):
                train_df = train_df.iloc[selected_idx].reset_index(drop=True)

        elif sample_size > 0 and sample_size < X_train_tfidf.shape[0]:
            idx_all = np.arange(len(y_train))
            _, _, y_train, _, idx_tr, _ = train_test_split(
                X_train_tfidf, y_train, idx_all,
                train_size=sample_size, stratify=y_train, random_state=args.seed
            )
            X_train_tfidf = X_train_tfidf[idx_tr]
            X_train_style = X_train_style[idx_tr]
            if len(train_df) == len(train_data["y"]):
                train_df = train_df.iloc[idx_tr].reset_index(drop=True)

    # 3. Print Sample Statistics & Generator Breakdown
    n_human = (y_train == 0).sum()
    n_llm = (y_train == 1).sum()
    total_samples = len(y_train)

    print("\n-------------------------------------------------------------")
    print(f" SAMPLE DISTRIBUTION & MODEL BREAKDOWN [{scope.upper()}]")
    print("-------------------------------------------------------------")
    print(f" Total Training Samples : {total_samples}")
    if total_samples > 0:
        print(f"   - Human Samples (0)  : {n_human} ({n_human / total_samples * 100:.1f}%)")
        print(f"   - LLM Samples   (1)  : {n_llm} ({n_llm / total_samples * 100:.1f}%)")

        if gen_col and len(train_df) == total_samples:
            print("\n Model Type Breakdown:")
            breakdown = train_df[gen_col].value_counts()
            for model_type, count in breakdown.items():
                pct = count / total_samples * 100
                print(f"   - {str(model_type):<20}: {count:>6d} ({pct:>5.1f}%)")
    print("-------------------------------------------------------------\n")

    # 4. Optuna Optimization

    study = optuna.create_study(direction="maximize", study_name=f"linear_svm_{scope}_pauc")

    print(f"Running Optuna Search ({args.n_trials} Trials)...")
    study.optimize(
        lambda trial: optuna_objective(
            trial=trial,
            X_tr_tfidf=X_train_tfidf,
            X_tr_style=X_train_style,
            y_train=y_train,
            X_dv_tfidf=X_dev_tfidf,
            X_dv_style=X_dev_style,
            y_dev=y_dev
        ),
        n_trials=args.n_trials,
        callbacks=[print_trial_callback]
    )

    best_trial = study.best_trial

    # Reconstruct complete hyperparameter dictionary including loss and dual
    best_hyperparams = dict(best_trial.params)
    best_hyperparams["loss"] = best_trial.user_attrs["loss"]
    best_hyperparams["dual"] = best_trial.user_attrs["dual"]

    print("\n" + "=" * 60)
    print(f"OPTUNA LINEAR SVM TUNING COMPLETE [{scope.upper()}]")
    print(f"Best Trial Number                : #{best_trial.number}")
    print(f"Best Validation pAUC @ FPR<=0.01 : {best_trial.value:.6f}")
    print("Best Linear SVM Hyperparameters  :")
    for k, v in best_hyperparams.items():
        print(f"  - {k}: {v}")
    print("=" * 60 + "\n")

    # 5. Save progress plot
    plot_path = scope_dir / "optuna_study_progress.png"
    plot_optuna_progress(study, save_path=plot_path, scope=scope)

    # 6. Save best hyperparameters to JSON
    json_path = scope_dir / "best_hyperparameters.json"
    hyperparams_json = {
        "classifier": "linear_svm",
        "kernel": "linear",
        "scope": scope,
        "is_balanced": args.balanced,
        "best_trial_number": best_trial.number,
        "best_val_pauc_001": round(best_trial.value, 6),
        "tuning_sample_size": total_samples,
        "best_hyperparameters": best_hyperparams,
        "features_dir": str(features_dir),
    }

    with open(json_path, "w") as f:
        json.dump(hyperparams_json, f, indent=4)

    print(f"[SAVED JSON] Best parameters saved to: '{json_path}'")


# ==========================================
# Main Execution Pipeline
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Tune Linear SVM using pre-extracted features.")

    parser.add_argument(
        "--scopes",
        nargs="+",
        default=["full", "sentence"],
        choices=["full", "sentence"],
        help="List of scopes to tune (default: full sentence)."
    )
    parser.add_argument("--balanced", "--balance_dataset", action="store_true", help="Balance dataset (50%% Human, 50%% LLM).")
    parser.add_argument("--sample_size", type=int, default=-1, help="Fallback sample size for tuning (-1 for full data).")
    parser.add_argument("--sample_size_full", type=int, default=10000, help="Sample size for 'full' scope tuning (default: 10000).")
    parser.add_argument("--sample_size_sentence", type=int, default=100000, help="Sample size for 'sentence' scope tuning (default: 100000).")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials per scope (default: 50).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--features_dir", type=str, default=None, help="Custom features directory path.")
    parser.add_argument("--outputs_dir", type=str, default=None, help="Custom outputs directory path (defaults to outputs folder).")

    args = parser.parse_args()
    manager = DetectionDataManager()

    for scope in args.scopes:
        tune_svm_for_scope(scope=scope, args=args, manager=manager)

    print("\n" + "=" * 70)
    print("[ALL DONE] Linear SVM tuning complete for all requested scopes!")
    print(f"Results saved in: {args.outputs_dir if args.outputs_dir else DEFAULT_OUTPUTS_DIR / 'svm'}")
    print("=" * 70)


if __name__ == "__main__":
    main()