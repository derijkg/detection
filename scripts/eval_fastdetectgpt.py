#!/usr/bin/env python3
# scripts/eval_fastdetectgpt.py

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

# Prevent CUDA memory fragmentation on Maxwell GPUs
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm
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
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Calculate project root dynamically (~/detection)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Add mgteval submodule path
MGTEVAL_ROOT = PROJECT_ROOT / "evals" / "mgteval"
if str(MGTEVAL_ROOT) not in sys.path:
    sys.path.append(str(MGTEVAL_ROOT))

DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"

from src.data.data_loader import DataFilter, DetectionDataManager
from evals.mgteval.src.detectors.metric.fast_detect_gpt import (
    FastDetectGPTDetector,
    _ensure_local_or_hf_target,
    _model_basename,
    _resolve_params_key,
)

# Optimized defaults for 2x 12GB Titan X Maxwell GPUs in FP32
LANGUAGE_DEFAULTS = {
    "en": {
        "scoring_model": "gpt2-xl",
        "reference_model": "gpt2",
    },
    "nl": {
        "scoring_model": "Qwen/Qwen2.5-3B-Instruct",
        "reference_model": "Qwen/Qwen2.5-3B",
    },
}


# ==========================================
# GPU Memory Helper
# ==========================================
def clear_gpu_memory():
    """Triggers Python garbage collection and clears PyTorch CUDA memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==========================================
# Failure Recording and Filtering
# ==========================================
def filter_and_record_failures(
    df: pd.DataFrame, scores: np.ndarray, split_name: str, scope_dir: Path
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
    """Filters out non-finite (NaN/Inf) scores and exports failed samples to CSV for reporting."""
    scores = np.array(scores, dtype=np.float64)
    valid_mask = np.isfinite(scores)

    num_total = len(scores)
    num_failed = int(np.sum(~valid_mask))
    num_valid = int(np.sum(valid_mask))
    fail_rate_pct = float((num_failed / num_total * 100.0)) if num_total > 0 else 0.0

    accounting_info = {
        f"{split_name}_total_samples": num_total,
        f"{split_name}_evaluated_samples": num_valid,
        f"{split_name}_failed_samples": num_failed,
        f"{split_name}_failure_rate_pct": round(fail_rate_pct, 4),
    }

    if num_failed > 0:
        print(
            f"[{split_name.upper()} ACCOUNTING] {num_failed}/{num_total} samples "
            f"({fail_rate_pct:.2f}%) failed (NaN/Inf). Saving to '{split_name}_failed_samples.csv'..."
        )
        failed_df = df[~valid_mask].copy()
        failed_df["raw_score"] = scores[~valid_mask]
        failed_csv_path = scope_dir / f"{split_name}_failed_samples.csv"
        failed_df.to_csv(failed_csv_path, index=False)
        print(f"  [CSV SAVED] Failed samples recorded at: '{failed_csv_path}'")
    else:
        print(f"[{split_name.upper()} ACCOUNTING] All {num_total} samples successfully computed (0 failures).")

    valid_df = df[valid_mask].copy().reset_index(drop=True)
    valid_scores = scores[valid_mask]

    return valid_df, valid_scores, accounting_info


# ==========================================
# Dual Titan X (2x12GB Maxwell) Optimized Detector
# ==========================================
class OptimizedFastDetectGPTDetector(FastDetectGPTDetector):
    """
    Dual-GPU FastDetectGPT detector optimized for 2x GTX TITAN X (12GB Maxwell CC 5.2).
    - Uses FP32 (torch.float32) for Maxwell CUDA kernel compatibility.
    - Isolates Scoring Model to GPU 0 and Reference Model to GPU 1 (+ CPU offloading).
    """

    def load(self):
        def _load_tokenizer(name: str):
            target, use_hf = _ensure_local_or_hf_target(name)
            tok = AutoTokenizer.from_pretrained(
                target,
                use_fast=True,
                trust_remote_code=True,
                local_files_only=not use_hf,
            )
            if tok.pad_token is None and getattr(tok, "eos_token", None) is not None:
                tok.pad_token = tok.eos_token
            return tok

        def _load_causallm(name: str, max_memory_budget: dict):
            target, use_hf = _ensure_local_or_hf_target(name)

            print(f"[OPT-LOAD] Loading '{name}' in FP32 mode with budget {max_memory_budget}...")
            try:
                mdl = AutoModelForCausalLM.from_pretrained(
                    target,
                    torch_dtype=torch.float32,
                    device_map="auto",
                    max_memory=max_memory_budget,
                    trust_remote_code=True,
                    local_files_only=not use_hf,
                    low_cpu_mem_usage=True,
                )
                mdl.eval()
                return mdl, True
            except Exception as e:
                print(f"[LOAD ERROR DETAILS] Failed to load model '{name}': {e}")
                return None, False

        print(f"[LOADER] Loading tokenizer '{self.tokenizer_name}'...")
        self._tokenizer = _load_tokenizer(self.tokenizer_name)

        same_model = (self.sampling_model_name == self.scoring_model_name)

        if same_model:
            print(f"[LOADER] Scoring and Reference models are identical ('{self.scoring_model_name}'). Loading single instance...")
            budget_shared = {0: "10.0GiB", 1: "10.0GiB", "cpu": "64GiB"}
            self._score_model, ok_s = _load_causallm(self.scoring_model_name, budget_shared)
            self._samp_model, ok_a = self._score_model, ok_s
        else:
            budget_scoring = {0: "10.0GiB", 1: "0GiB", "cpu": "64GiB"}
            print(f"[LOADER] Loading scoring model '{self.scoring_model_name}' on GPU 0...")
            self._score_model, ok_s = _load_causallm(self.scoring_model_name, budget_scoring)

            budget_ref = {0: "0GiB", 1: "10.0GiB", "cpu": "64GiB"}
            print(f"[LOADER] Loading reference model '{self.sampling_model_name}' on GPU 1...")
            self._samp_model, ok_a = _load_causallm(self.sampling_model_name, budget_ref)

        if not (ok_s and ok_a):
            raise RuntimeError(
                f"[ModelLoadError] Failed to load models.\n"
                f"  scoring_model_name = {self.scoring_model_name}\n"
                f"  sampling_model_name = {self.sampling_model_name}\n"
            )

        key = _resolve_params_key(self.sampling_model_name, self.scoring_model_name, self.distrib_params)
        self._params_key = key
        self._have_intrinsic_prob = key is not None

        s_show = _model_basename(self.sampling_model_name)
        c_show = _model_basename(self.scoring_model_name)
        self.name = f"FastDetectGPT[{s_show}_{c_show}]"
        self.DETECTOR_NAME = self.name
        self.is_loaded = True


# ==========================================
# Calibration and Probability Functions
# ==========================================
def prob_from_two_normals_vec(x: np.ndarray, mu0: float, s0: float, mu1: float, s1: float) -> np.ndarray:
    s0 = max(float(s0), 1e-4) if np.isfinite(s0) and s0 > 0 else 1.0
    s1 = max(float(s1), 1e-4) if np.isfinite(s1) and s1 > 0 else 1.0
    mu0 = float(mu0) if np.isfinite(mu0) else 0.0
    mu1 = float(mu1) if np.isfinite(mu1) else 1.0

    log_pdf0 = norm.logpdf(x, loc=mu0, scale=s0)
    log_pdf1 = norm.logpdf(x, loc=mu1, scale=s1)

    delta = log_pdf1 - log_pdf0
    delta = np.nan_to_num(delta, nan=0.0, posinf=20.0, neginf=-20.0)

    probs = expit(delta)
    return np.clip(probs, 1e-6, 1.0 - 1e-6)


# ==========================================
# Hierarchical Configuration-Balanced Sampler
# ==========================================
def subsample_balanced_configurations(
    df: pd.DataFrame,
    max_samples: int | None,
    seed: int = 42,
    config_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Subsamples `max_samples` from `df` such that:
    1. Label balance (50% Human / 50% LLM) is maintained.
    2. Within each class, samples are drawn as equally as possible across all
       configurations (generator models, domains, datasets, etc.).
    """
    if max_samples is None or len(df) <= max_samples or max_samples <= 0:
        return df

    if "label" not in df.columns:
        return df.sample(n=min(len(df), max_samples), random_state=seed)

    # Auto-detect configuration/metadata columns if not explicitly passed
    if config_cols is None:
        potential_cols = ["model_name", "generator_model", "generator", "model", "domain", "dataset", "source"]
        config_cols = [col for col in potential_cols if col in df.columns]

    target_per_class = max_samples // 2
    sampled_dfs = []

    for label_val in [0, 1]:
        class_df = df[df["label"] == label_val].copy()

        if len(class_df) <= target_per_class:
            sampled_dfs.append(class_df)
            continue

        active_cols = [col for col in config_cols if col in class_df.columns and class_df[col].nunique() > 1]

        if not active_cols:
            sampled_dfs.append(class_df.sample(n=target_per_class, random_state=seed))
            continue

        for col in active_cols:
            class_df[col] = class_df[col].fillna("UNKNOWN")

        groups = [group for _, group in class_df.groupby(active_cols)]
        n_groups = len(groups)
        target_per_config = max(1, target_per_class // n_groups)

        class_sampled = []
        budget = target_per_class

        # Pass 1: Draw up to target_per_config per group
        for group in groups:
            n_take = min(len(group), target_per_config)
            sampled_grp = group.sample(n=n_take, random_state=seed)
            class_sampled.append(sampled_grp)
            budget -= len(sampled_grp)

        # Pass 2: Distribute remaining budget evenly if any small group had < target_per_config
        if budget > 0:
            already_sampled_ids = set().union(*[set(s.index) for s in class_sampled])
            remaining_df = class_df.loc[~class_df.index.isin(already_sampled_ids)]
            if len(remaining_df) > 0:
                extra_sample = remaining_df.sample(n=min(len(remaining_df), budget), random_state=seed)
                class_sampled.append(extra_sample)

        sampled_dfs.append(pd.concat(class_sampled, ignore_index=True))

    final_df = pd.concat(sampled_dfs, ignore_index=True)

    print(f"\n[CONFIG-SAMPLER] Subsampled total: {len(df)} -> {len(final_df)} samples")
    for col in config_cols:
        if col in final_df.columns:
            counts = final_df.groupby(["label", col]).size().to_dict()
            print(f"  Configuration breakdown for '{col}':")
            for (lbl, cfg), count in counts.items():
                lbl_str = "Human" if lbl == 0 else "LLM"
                print(f"    - [{lbl_str}] {cfg}: {count} samples")

    return final_df


def analyze_and_save_errors(df_eval: pd.DataFrame, scope_dir: Path, scope: str):
    conditions = [
        (df_eval["label"] == 1) & (df_eval["pred_llm"] == 1),
        (df_eval["label"] == 0) & (df_eval["pred_llm"] == 0),
        (df_eval["label"] == 0) & (df_eval["pred_llm"] == 1),
        (df_eval["label"] == 1) & (df_eval["pred_llm"] == 0),
    ]
    choices = ["TP", "TN", "FP", "FN"]
    df_eval["error_type"] = np.select(conditions, choices, default="UNKNOWN")
    df_eval["is_error"] = df_eval["error_type"].isin(["FP", "FN"])

    # Save all test predictions (all dataset columns + prediction metadata)
    df_eval.to_csv(scope_dir / "test_predictions.csv", index=False)
    df_eval[df_eval["is_error"]].to_csv(scope_dir / "test_errors.csv", index=False)
    df_eval[df_eval["error_type"] == "FP"].to_csv(scope_dir / "test_false_positives.csv", index=False)
    df_eval[df_eval["error_type"] == "FN"].to_csv(scope_dir / "test_false_negatives.csv", index=False)

    gen_col = next((col for col in ["model_name", "generator_model", "generator", "model"] if col in df_eval.columns), None)
    if gen_col:
        gen_breakdown = []
        for gen, group in df_eval.groupby(gen_col):
            g_total = len(group)
            g_errors = int(group["is_error"].sum())
            g_err_rate = (g_errors / g_total * 100) if g_total > 0 else 0.0
            gen_breakdown.append({
                "Generator / Source": str(gen),
                "Total Samples": int(g_total),
                "Errors": g_errors,
                "Error Rate (%)": round(float(g_err_rate), 2),
                "False Positives": int((group["error_type"] == "FP").sum()),
                "False Negatives": int((group["error_type"] == "FN").sum()),
                "Mean Prob P(LLM)": round(float(group["prob_llm"].mean()), 4),
            })
        pd.DataFrame(gen_breakdown).sort_values(by="Error Rate (%)", ascending=False).to_csv(
            scope_dir / "test_error_breakdown.csv", index=False
        )


def evaluate_fastdetectgpt_scope(scope: str, detector: FastDetectGPTDetector, args, manager: DetectionDataManager):
    outputs_base = Path(args.outputs_dir) if args.outputs_dir else DEFAULT_OUTPUTS_DIR
    scope_dir = outputs_base / f"fastdetectgpt_{args.language}" / scope
    scope_dir.mkdir(parents=True, exist_ok=True)

    calib_params_file = scope_dir / "calibration_params.json"

    if scope == "sentence":
        max_train = args.max_train_sentence
        max_val = args.max_val_sentence
        max_test = args.max_test_sentence
    else:  # full / abstract
        max_train = args.max_train_full
        max_val = args.max_val_full
        max_test = args.max_test_full

    # ==========================================
    # CHECK FOR EXISTING CALIBRATION PARAMS
    # ==========================================
    if calib_params_file.exists() and not args.recompute:
        print(f"\n[{scope.upper()}] Loading cached calibration parameters from '{calib_params_file}'...")
        with open(calib_params_file, "r") as f:
            calib_params = json.load(f)

        mu0 = calib_params["mu0"]
        sigma0 = calib_params["sigma0"]
        mu1 = calib_params["mu1"]
        sigma1 = calib_params["sigma1"]
        optimal_threshold = calib_params["optimal_threshold_tau"]
        print(f"  [SKIPPED CALIBRATION] Parameters loaded: mu0={mu0}, sigma0={sigma0}, mu1={mu1}, sigma1={sigma1}, threshold={optimal_threshold}")
    else:
        print(f"\n[{scope.upper()}] No cached parameters found or --recompute passed. Running calibration...")
        train_cache = scope_dir / "raw_scores_train_cache.joblib"
        val_cache = scope_dir / "raw_scores_val_cache.joblib"

        # ------------------------------------------
        # STAGE 1: CALIBRATE (mu, sigma) ON TRAIN
        # ------------------------------------------
        raw_train_df = manager.filter_dataframe(DataFilter(splits=["train"], scopes=[scope]))
        sub_train_df = subsample_balanced_configurations(raw_train_df, max_samples=max_train, seed=args.seed)

        if train_cache.exists() and not args.recompute:
            print(f"[{scope.upper()}] Loading cached TRAIN scores...")
            train_raw_scores = joblib.load(train_cache)
        else:
            print(f"[{scope.upper()}] Scoring TRAIN split ({len(sub_train_df)} configuration-balanced samples out of {len(raw_train_df)} total)...")
            with torch.inference_mode():
                train_raw_scores = detector.score_batch(sub_train_df["text"].tolist())
                print("  [TRAIN SCORES] min/max/mean:", round(float(np.min(train_raw_scores)), 4), round(float(np.max(train_raw_scores)), 4), round(float(np.mean(train_raw_scores)), 4))
            joblib.dump(train_raw_scores, train_cache)
            clear_gpu_memory()

        train_df, train_raw_scores, train_accounting = filter_and_record_failures(
            sub_train_df, train_raw_scores, "train", scope_dir
        )

        human_train = train_raw_scores[train_df["label"].values == 0]
        llm_train = train_raw_scores[train_df["label"].values == 1]

        mu0, sigma0 = float(np.mean(human_train)), float(np.std(human_train))
        mu1, sigma1 = float(np.mean(llm_train)), float(np.std(llm_train))

        # ------------------------------------------
        # STAGE 2: FIND OPTIMAL THRESHOLD (tau*) ON VAL
        # ------------------------------------------
        raw_val_df = manager.filter_dataframe(DataFilter(splits=["val"], scopes=[scope]))
        sub_val_df = subsample_balanced_configurations(raw_val_df, max_samples=max_val, seed=args.seed)

        if val_cache.exists() and not args.recompute:
            print(f"[{scope.upper()}] Loading cached VAL scores...")
            val_raw_scores = joblib.load(val_cache)
        else:
            print(f"[{scope.upper()}] Scoring VAL split ({len(sub_val_df)} configuration-balanced samples out of {len(raw_val_df)} total)...")
            with torch.inference_mode():
                val_raw_scores = detector.score_batch(sub_val_df["text"].tolist())
                print("  [VAL SCORES] min/max/mean:", round(float(np.min(val_raw_scores)), 4), round(float(np.max(val_raw_scores)), 4), round(float(np.mean(val_raw_scores)), 4))
            joblib.dump(val_raw_scores, val_cache)
            clear_gpu_memory()

        val_df, val_raw_scores, val_accounting = filter_and_record_failures(
            sub_val_df, val_raw_scores, "val", scope_dir
        )

        val_probs = prob_from_two_normals_vec(val_raw_scores, mu0, sigma0, mu1, sigma1)

        fpr_v, tpr_v, thresholds_v = roc_curve(val_df["label"].values, val_probs)
        optimal_idx = int(np.argmax(tpr_v - fpr_v))

        valid_thresholds = np.copy(thresholds_v)
        if len(valid_thresholds) > 1 and valid_thresholds[0] > 1.0:
            valid_thresholds[0] = float(np.max(val_probs))

        optimal_threshold = float(np.clip(valid_thresholds[optimal_idx], 0.0, 1.0))

        calib_params = {
            "scope": scope,
            "language": args.language,
            "mu0": round(mu0, 6),
            "sigma0": round(sigma0, 6),
            "mu1": round(mu1, 6),
            "sigma1": round(sigma1, 6),
            "optimal_threshold_tau": round(optimal_threshold, 6),
            "sample_accounting": {
                **train_accounting,
                **val_accounting,
            },
        }
        with open(calib_params_file, "w") as f:
            json.dump(calib_params, f, indent=4)

    print("\n" + "=" * 70)
    print(f" FAST-DETECTGPT PARAMETERS [{scope.upper()} | LANG: {args.language.upper()}] ")
    print(f"  Parameters : mu0={mu0:.4f}, sigma0={sigma0:.4f} | mu1={mu1:.4f}, sigma1={sigma1:.4f}")
    print(f"  Threshold  : optimal_threshold_tau={optimal_threshold:.6f}")
    print("=" * 70)

    # ==========================================
    # STAGE 3: EVALUATE ON SUBSAMPLED TEST SPLIT
    # ==========================================
    test_cache = scope_dir / "raw_scores_test_cache.joblib"
    raw_test_df = manager.filter_dataframe(DataFilter(splits=["test"], scopes=[scope]))
    sub_test_df = subsample_balanced_configurations(raw_test_df, max_samples=max_test, seed=args.seed)

    # Validate cached test raw scores match current subsample length
    cached_scores_valid = False
    if test_cache.exists() and not args.recompute:
        test_raw_scores = joblib.load(test_cache)
        if len(test_raw_scores) == len(sub_test_df):
            print(f"[{scope.upper()}] Loading cached TEST scores ({len(test_raw_scores)} samples)...")
            cached_scores_valid = True
        else:
            print(f"[{scope.upper()}] Test cache size mismatch ({len(test_raw_scores)} vs {len(sub_test_df)}). Recomputing...")

    if not cached_scores_valid:
        print(f"[{scope.upper()}] Scoring TEST split ({len(sub_test_df)} configuration-balanced samples out of {len(raw_test_df)} total)...")
        with torch.inference_mode():
            test_raw_scores = detector.score_batch(sub_test_df["text"].tolist())
            print("  [TEST SCORES] min/max/mean:", round(float(np.min(test_raw_scores)), 4), round(float(np.max(test_raw_scores)), 4), round(float(np.mean(test_raw_scores)), 4))
        joblib.dump(test_raw_scores, test_cache)
        clear_gpu_memory()

    test_df, test_raw_scores, test_accounting = filter_and_record_failures(
        sub_test_df, test_raw_scores, "test", scope_dir
    )
    test_labels = test_df["label"].values

    probs_llm = prob_from_two_normals_vec(test_raw_scores, mu0, sigma0, mu1, sigma1)
    preds = (probs_llm >= optimal_threshold).astype(int)

    fpr, tpr, _ = roc_curve(test_labels, probs_llm)
    roc_auc_val = float(auc(fpr, tpr))
    pauc_001_val = float(roc_auc_score(test_labels, probs_llm, max_fpr=0.01))

    precision_curve, recall_curve, _ = precision_recall_curve(test_labels, probs_llm)
    pr_auc_val = float(average_precision_score(test_labels, probs_llm))

    acc = float(accuracy_score(test_labels, preds))
    prec, rec, f1, _ = precision_recall_fscore_support(test_labels, preds, average="binary", zero_division=0)
    prec, rec, f1 = float(prec), float(rec), float(f1)

    cm = confusion_matrix(test_labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    tpr_at_1fpr = float(np.interp(0.01, fpr, tpr))
    tpr_at_5fpr = float(np.interp(0.05, fpr, tpr))

    overall_metrics = {
        "Scope": scope,
        "Total Test Samples": int(test_accounting["test_total_samples"]),
        "Evaluated Test Samples": int(test_accounting["test_evaluated_samples"]),
        "Failed Test Samples": int(test_accounting["test_failed_samples"]),
        "Failure Rate (%)": float(test_accounting["test_failure_rate_pct"]),
        "Optimal Threshold (\u03c4*)": round(optimal_threshold, 6),
        "pAUC @ max FPR 0.01": round(pauc_001_val, 6),
        "ROC-AUC": round(roc_auc_val, 4),
        "PR-AUC (AP)": round(pr_auc_val, 4),
        "Accuracy": round(acc, 4),
        "F1-Score": round(f1, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "Specificity": round(specificity, 4),
        "TPR @ 1% FPR": round(tpr_at_1fpr, 4),
        "TPR @ 5% FPR": round(tpr_at_5fpr, 4),
    }

    print(f"\n--- OVERALL TEST SET PERFORMANCE [{scope.upper()} SCOPE] ---")
    for k, v in overall_metrics.items():
        print(f"  {k:<32}: {v}")

    df_analysis = test_df.copy()
    df_analysis["prob_llm"] = probs_llm
    df_analysis["pred_llm"] = preds
    analyze_and_save_errors(df_analysis, scope_dir=scope_dir, scope=scope)

    # 4-Panel Figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    plt.subplots_adjust(wspace=0.25, hspace=0.3)

    axes[0, 0].plot(fpr, tpr, color="#2b5c8f", lw=2, label=f"Fast-DetectGPT ({scope.upper()})\npAUC@0.01={pauc_001_val:.4f}")
    axes[0, 0].axvline(x=0.01, color="red", linestyle=":", lw=1.5, label="FPR = 0.01")
    axes[0, 0].plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5)
    axes[0, 0].set_xlabel("False Positive Rate")
    axes[0, 0].set_ylabel("True Positive Rate")
    axes[0, 0].set_title(f"(A) ROC Curve ({scope.upper()})", fontweight="bold")
    axes[0, 0].legend(loc="lower right")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    axes[0, 1].plot(recall_curve, precision_curve, color="#d95f02", lw=2, label=f"AP={pr_auc_val:.4f}")
    axes[0, 1].set_xlabel("Recall")
    axes[0, 1].set_ylabel("Precision")
    axes[0, 1].set_title(f"(B) Precision-Recall Curve ({scope.upper()})", fontweight="bold")
    axes[0, 1].legend(loc="lower left")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype("float"), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    im = axes[1, 0].imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    axes[1, 0].set_title(f"(C) Confusion Matrix (\u03c4* = {optimal_threshold:.4f})", fontweight="bold")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    classes = ["Human", "LLM"]
    axes[1, 0].set_xticks([0, 1])
    axes[1, 0].set_xticklabels(classes)
    axes[1, 0].set_yticks([0, 1])
    axes[1, 0].set_yticklabels(classes)
    axes[1, 0].set_ylabel("True Label")
    axes[1, 0].set_xlabel("Predicted Label")

    for i in range(2):
        for j in range(2):
            axes[1, 0].text(
                j, i, f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > 0.5 else "black",
                fontweight="bold",
            )

    axes[1, 1].hist(probs_llm[test_labels == 0], bins=25, alpha=0.6, color="#1b9e77", label="Human Text", density=True)
    axes[1, 1].hist(probs_llm[test_labels == 1], bins=25, alpha=0.6, color="#7570b3", label="LLM Text", density=True)
    axes[1, 1].axvline(x=optimal_threshold, color="black", linestyle="--", lw=2, label=f"Threshold (\u03c4*={optimal_threshold:.4f})")
    axes[1, 1].set_xlabel("Predicted Probability P(LLM)")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].set_title(f"(D) Probability Density ({scope.upper()})", fontweight="bold")
    axes[1, 1].legend(loc="upper center")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    plot_path = scope_dir / "test_evaluation_plots.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    summary_path = scope_dir / "test_paper_evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "test_sample_accounting": test_accounting,
            "overall_test_metrics": overall_metrics,
        }, f, indent=4)

    print(f"[COMPLETED] Predictions saved to: {scope_dir / 'test_predictions.csv'}\n")


def main():
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(description="3-Split Calibrate & Evaluate Fast-DetectGPT with sample filtering.")
    parser.add_argument("--scopes", nargs="+", default=["sentence", "full"], choices=["sentence", "full"])
    parser.add_argument("--language", type=str, default="nl", choices=["en", "nl"])

    # Calibration split max samples
    parser.add_argument("--max_train_sentence", type=int, default=10000, help="Max Train samples for sentence calibration")
    parser.add_argument("--max_train_full", type=int, default=2000, help="Max Train samples for full calibration")
    parser.add_argument("--max_val_sentence", type=int, default=5000, help="Max Val samples for sentence thresholding")
    parser.add_argument("--max_val_full", type=int, default=1000, help="Max Val samples for full thresholding")

    # Test split max samples
    parser.add_argument("--max_test_sentence", type=int, default=10000, help="Max Test samples for sentence evaluation")
    parser.add_argument("--max_test_full", type=int, default=5000, help="Max Test samples for full evaluation")

    parser.add_argument("--outputs_dir", type=str, default=None)
    parser.add_argument("--scoring_model", type=str, default=None)
    parser.add_argument("--reference_model", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recompute", action="store_true", help="Force recomputing calibration & scores even if files exist")

    args = parser.parse_args()

    lang_config = LANGUAGE_DEFAULTS.get(args.language, LANGUAGE_DEFAULTS["en"])
    scoring_model = args.scoring_model or lang_config["scoring_model"]
    reference_model = args.reference_model or lang_config["reference_model"]

    print(f"[INIT] Dual-GPU Fast-DetectGPT Detector ({args.language.upper()})")
    print(f"  Scoring Model   : {scoring_model}")
    print(f"  Reference Model : {reference_model}")

    detector = OptimizedFastDetectGPTDetector(
        scoring_model_name=scoring_model,
        sampling_model_name=reference_model,
        device=args.device,
        max_length=512,
    )
    detector.load()

    manager = DetectionDataManager()

    for scope in args.scopes:
        evaluate_fastdetectgpt_scope(scope=scope, detector=detector, args=args, manager=manager)


if __name__ == "__main__":
    main()