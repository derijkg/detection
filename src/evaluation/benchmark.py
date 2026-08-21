# src/evaluation/benchmark.py

import argparse
import gc
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, confusion_matrix, f1_score
import torch

from src.evaluation.metrics import MetricEvaluator
from src.visualization.latex_tables import export_performance_table, export_robustness_table


def load_detector_pipeline(
    model_type: str, 
    model_path: Union[str, Path], 
    scope: str = "full", 
    device: Optional[str] = None
) -> Any:
    """Universal Loader for all 4 detection paradigms."""
    model_path = Path(model_path)
    model_type = model_type.lower()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if "deberta" in model_type:
        from src.models.deberta import MDeBERTaDetector
        return MDeBERTaDetector.load(model_path, scope=scope, device=device)
    elif "svm" in model_type:
        from src.models.svm_pipeline import SVMDetector
        return SVMDetector.load(model_path, scope=scope)
    elif "fdgpt" in model_type or "fast_detect_gpt" in model_type:
        from src.models.fast_detect_gpt import FastDetectGPTDetector
        calib_file = model_path if str(model_path).endswith(".json") else (Path(model_path) / "model_calibration.json")
        return FastDetectGPTDetector.load(calib_file, device=device)
    elif "stat" in model_type:
        from src.models.statistical_detector import StatisticalTrajectoryDetector
        return StatisticalTrajectoryDetector.load(model_path, scope=scope)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def _get_positive_probs(model, data: Union[List[Dict], pd.DataFrame]) -> np.ndarray:
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(data)
    elif hasattr(model, 'decision_function'):
        scores = model.decision_function(data)
        probs = 1.0 / (1.0 + np.exp(-scores))
    else:
        raise AttributeError("Model must implement `predict_proba` or `decision_function`.")

    probs_arr = np.asarray(probs, dtype=np.float64)
    if probs_arr.ndim == 2:
        return probs_arr[:, 1]
    return probs_arr


class BenchmarkOrchestrator:


    @staticmethod
    def _attach_prediction_diagnostics(
        df: pd.DataFrame, 
        probs: np.ndarray, 
        threshold: float, 
        has_labels: bool = True
    ) -> pd.DataFrame:
        pred_df = df.copy().reset_index(drop=True)
        preds = (probs >= threshold).astype(int)
        pred_df['predicted_prob'] = np.round(probs, 6)
        pred_df['predicted_label'] = preds
        pred_df['threshold_used'] = round(float(threshold), 6)

        if has_labels and 'label' in pred_df.columns:
            y_true = pred_df['label'].astype(int).values
            pred_df['is_correct'] = (preds == y_true).astype(int)
            
            # Vectorized TP/TN/FP/FN assignment
            conditions = [
                (y_true == 1) & (preds == 1),
                (y_true == 0) & (preds == 0),
                (y_true == 0) & (preds == 1),
                (y_true == 1) & (preds == 0)
            ]
            choices = ["TP", "TN", "FP", "FN"]
            pred_df['error_type'] = np.select(conditions, choices, default="UNKNOWN")

        if 'text' in pred_df.columns:
            text_series = pred_df['text'].fillna("").astype(str)
            pred_df['word_count'] = text_series.str.split().str.len()
            pred_df['char_count'] = text_series.str.len()

        return pred_df
    
    @classmethod
    def run_full_benchmark(
        cls,
        model_pipeline,
        dev_df: pd.DataFrame,
        test_suites: Dict[str, Any],
        model_name: str,
        scope: str,
        output_dir: Union[str, Path],
        max_fpr: float = 0.01
    ) -> Dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        latex_dir = output_dir / "latex_tables"
        latex_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=======================================================")
        print(f"   RUNNING BENCHMARK: {model_name.upper()} ({scope.upper()})   ")
        print(f"=======================================================")

        # 1. Dev-Set Threshold Calibration
        print("\n[Step 1] Calibrating Decision Threshold on Dev Set...")
        y_dev = dev_df['label'].astype(int).values
        dev_probs = _get_positive_probs(model_pipeline, dev_df)
        
        calibrated_threshold = MetricEvaluator.find_threshold_for_max_fpr(y_dev, dev_probs, target_fpr=max_fpr)
        best_f1_threshold = MetricEvaluator.find_threshold_for_best_f1(y_dev, dev_probs)

        if hasattr(model_pipeline, 'calibrated_threshold'):
            model_pipeline.calibrated_threshold = calibrated_threshold

        print(f" -> Optimal Dev Threshold (FPR <= {max_fpr*100:.1f}%): {calibrated_threshold:.6f}")
        print(f" -> Best F1 Dev Threshold:                      {best_f1_threshold:.6f}")

        dev_pred_df = cls._attach_prediction_diagnostics(dev_df, dev_probs, calibrated_threshold)
        dev_pred_path = output_dir / "predictions_dev_calibrated.csv"
        dev_pred_df.to_csv(dev_pred_path, index=False)

        # 2. Standard Test Suite Evaluation
        print("\n[Step 2] Evaluating Standard Test Set (Pure Human vs Full LLM)...")
        std_df = test_suites['standard']
        y_std = std_df['label'].astype(int).values

        std_probs = _get_positive_probs(model_pipeline, std_df)
        std_preds = (std_probs >= calibrated_threshold).astype(int)

        roc_auc = MetricEvaluator.compute_metric(y_std, std_probs, 'roc_auc')
        pauc = MetricEvaluator.compute_metric(y_std, std_probs, 'pauc', max_fpr=max_fpr)
        tpr_at_1fpr = MetricEvaluator.compute_tpr_at_max_fpr(y_std, std_probs, target_fpr=max_fpr)
        pr_auc = MetricEvaluator.compute_metric(y_std, std_probs, 'pr_auc')
        mcc = MetricEvaluator.compute_metric(y_std, std_probs, 'mcc', threshold=calibrated_threshold)
        f1_ai = MetricEvaluator.compute_metric(y_std, std_probs, 'f1', threshold=calibrated_threshold)

        tn, fp, fn, tp = confusion_matrix(y_std, std_preds, labels=[0, 1]).ravel()
        fpr_human = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        f1_human = float(f1_score(y_std, std_preds, pos_label=0, zero_division=0))
        brier = float(brier_score_loss(y_std, std_probs)) if (0 <= np.min(std_probs) and np.max(std_probs) <= 1) else 0.0

        print(f" -> ROC-AUC:        {roc_auc:.4f}")
        print(f" -> pAUC (FPR<=1%): {pauc:.4f}")
        print(f" -> TPR @ 1% FPR:   {tpr_at_1fpr*100:.2f}%")
        print(f" -> FPR (Human):    {fpr_human*100:.2f}% ({fp}/{fp+tn})")
        print(f" -> F1-Score AI:    {f1_ai:.4f}")
        print(f" -> MCC:            {mcc:.4f}")

        std_pred_df = cls._attach_prediction_diagnostics(std_df, std_probs, calibrated_threshold)
        std_csv_path = output_dir / "predictions_test_standard.csv"
        std_pred_df.to_csv(std_csv_path, index=False)

        # 3. Gradual LLM Substitution Sensitivity
        print("\n[Step 3] Evaluating Gradual Substitution Sensitivity...")
        all_test_df = test_suites.get('all', pd.DataFrame())
        ratio_stats = []
        ratio_dict = {}

        if not all_test_df.empty and 'llm_ratio' in all_test_df.columns:
            all_probs = _get_positive_probs(model_pipeline, all_test_df)
            sub_pred_df = cls._attach_prediction_diagnostics(all_test_df, all_probs, calibrated_threshold, has_labels=False)
            
            sub_csv_path = output_dir / "predictions_test_substitution.csv"
            sub_pred_df.to_csv(sub_csv_path, index=False)

            for ratio, grp in sub_pred_df.groupby('llm_ratio'):
                if ratio == 0.0:
                    continue
                grp_probs = grp['predicted_prob'].values
                flagged_pct = float(np.mean(grp_probs >= calibrated_threshold) * 100)
                avg_score = float(np.mean(grp_probs))

                r_key = f"{int(round(ratio*100))}pct"
                ratio_dict[r_key] = {
                    "target_ratio": f"{int(round(ratio*100))}%",
                    "actual_ratio": float(ratio),
                    "flagged_pct": flagged_pct,
                    "avg_score": avg_score,
                    "count": len(grp)
                }
                ratio_stats.append(ratio_dict[r_key])
                print(f" -> Substitution {int(round(ratio*100))}%: {flagged_pct:.2f}% flagged (Avg Prob: {avg_score:.4f})")

        # 4. Per-Generator Model Breakdown
        generator_stats = {}
        gen_pred_dfs = []

        for gen_name, gen_df in test_suites.get("per_generator", {}).items():
            y_g = gen_df['label'].astype(int).values
            g_probs = _get_positive_probs(model_pipeline, gen_df)
            
            g_annotated = cls._attach_prediction_diagnostics(gen_df, g_probs, calibrated_threshold)
            g_annotated['generator_group'] = gen_name
            gen_pred_dfs.append(g_annotated)

            generator_stats[gen_name] = {
                "roc_auc": MetricEvaluator.compute_metric(y_g, g_probs, 'roc_auc'),
                "pauc": MetricEvaluator.compute_metric(y_g, g_probs, 'pauc', max_fpr=max_fpr),
                "tpr_at_1fpr": MetricEvaluator.compute_tpr_at_max_fpr(y_g, g_probs, target_fpr=max_fpr),
                "f1_ai": MetricEvaluator.compute_metric(y_g, g_probs, 'f1', threshold=calibrated_threshold),
                "count": len(gen_df)
            }

        if gen_pred_dfs:
            combined_gen_df = pd.concat(gen_pred_dfs, ignore_index=True)
            gen_csv_path = output_dir / "predictions_test_per_generator.csv"
            combined_gen_df.to_csv(gen_csv_path, index=False)

        # 5. Export Summary JSON & LaTeX Tables
        summary = {
            "model_name": model_name,
            "scope": scope,
            "calibrated_threshold": float(calibrated_threshold),
            "best_f1_threshold": float(best_f1_threshold),
            "overall_roc_auc": float(roc_auc),
            "overall_pauc": float(pauc),
            "tpr_at_1fpr": float(tpr_at_1fpr),
            "overall_pr_auc": float(pr_auc),
            "overall_mcc": float(mcc),
            "f1_ai": float(f1_ai),
            "f1_human": float(f1_human),
            "fpr_human": float(fpr_human),
            "brier_score": float(brier),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "robustness_ratios": ratio_dict,
            "per_generator_performance": generator_stats
        }

        summary_json = output_dir / "evaluation_summary.json"
        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\n[Exported Summary JSON] -> {summary_json}")

        export_performance_table(summary, scope=scope, model_name=model_name, output_path=latex_dir / "table_performance.tex")
        if ratio_stats:
            export_robustness_table(ratio_stats, scope=scope, model_name=model_name, output_path=latex_dir / "table_robustness_ratios.tex")

        return summary


def test_saved_model(
    model_type: str,
    model_path: Union[str, Path],
    scope: str = "full",
    output_dir: Optional[Union[str, Path]] = None,
    max_fpr: float = 0.01,
    seed: int = 42
) -> Dict[str, Any]:
    from src.data.data_loader import DataFilter, DetectionDataManager

    manager = DetectionDataManager()
    val_filter = DataFilter(splits=["dev"], scopes=[scope])
    dev_df = manager.filter_dataframe(val_filter, sample_size=10000, seed=seed)
    test_suites = manager.get_benchmark_test_suites(scope=scope)

    if output_dir is None:
        output_dir = Path(model_path).parent

    detector_pipeline = load_detector_pipeline(
        model_type=model_type,
        model_path=model_path,
        scope=scope
    )

    return BenchmarkOrchestrator.run_full_benchmark(
        model_pipeline=detector_pipeline,
        dev_df=dev_df,
        test_suites=test_suites,
        model_name=model_type,
        scope=scope,
        output_dir=Path(output_dir),
        max_fpr=max_fpr
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Universal Benchmark Evaluator")
    parser.add_argument("--model_type", type=str, required=True, choices=["mdeberta", "svm", "fdgpt", "stat_trajectory"])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--scope", type=str, required=True, choices=["full", "sentence"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--target_fpr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    test_saved_model(
        model_type=args.model_type,
        model_path=args.model_path,
        scope=args.scope,
        output_dir=args.output_dir,
        max_fpr=args.target_fpr,
        seed=args.seed
    )