"""
src/evaluation/benchmark.py
Standardized evaluation orchestrator: Dev threshold calibration (Conformal / Empirical / EVT),
multi-suite evaluation, discrete rewrite sensitivity analysis (25%, 50%, 75%), per-generator slicing,
and report generation.
"""

import argparse
import gc
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, confusion_matrix, f1_score
import torch

from src.evaluation.metrics import MetricEvaluator
from src.visualization.latex_tables import (
    export_performance_table,
    export_robustness_table
)


def load_detector_pipeline(
    model_type: str,
    model_path: Union[str, Path],
    scope: str = 'full',
    device: Optional[str] = None
) -> Any:
    model_path = Path(model_path)
    model_type = model_type.lower()
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    if 'deberta' in model_type:
        from src.models.deberta import MDeBERTaDetector
        return MDeBERTaDetector.load(model_path, scope=scope, device=device)
    elif 'svm' in model_type:
        from src.models.svm_pipeline import SVMDetector
        return SVMDetector.load(model_path, scope=scope)
    elif 'fdgpt' in model_type or 'fast_detect_gpt' in model_type:
        from src.models.fast_detect_gpt import FastDetectGPTDetector
        calib_file = model_path if str(model_path).endswith('.json') else Path(model_path) / 'model_calibration.json'
        return FastDetectGPTDetector.load(calib_file, device=device)
    elif 'stat' in model_type:
        from src.models.statistical_detector import StatisticalTrajectoryDetector
        return StatisticalTrajectoryDetector.load(model_path, scope=scope)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def _get_positive_probs(model, data: Union[List[Dict], pd.DataFrame]) -> np.ndarray:
    """Safely extracts calibrated P(AI | text) probabilities."""
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


def assign_discrete_rewrite_bucket(val: Any) -> Optional[str]:
    """
    Assigns continuous rewrite ratios to discrete robustness buckets (25%, 50%, 75%).
    Strictly filters out pure human (0%) and pure full rewrites (100%).
    """
    if val is None or pd.isna(val):
        return None
    try:
        r = float(val)
    except (ValueError, TypeError):
        return None

    if r > 1.0:
        r = r / 100.0

    if r <= 0.0 or r >= 0.875:
        return None

    if r <= 0.375:
        return '25%'
    elif r <= 0.625:
        return '50%'
    else:
        return '75%'


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
            conditions = [
                (y_true == 1) & (preds == 1),
                (y_true == 0) & (preds == 0),
                (y_true == 0) & (preds == 1),
                (y_true == 1) & (preds == 0)
            ]
            choices = ['TP', 'TN', 'FP', 'FN']
            pred_df['error_type'] = np.select(conditions, choices, default='UNKNOWN')

        if 'text' in pred_df.columns:
            text_series = pred_df['text'].fillna('').astype(str)
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
        max_fpr: float = 0.01,
        calibration_method: str = 'conformal'
    ) -> Dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        latex_dir = output_dir / "latex_tables"
        latex_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=======================================================")
        print(f"   RUNNING BENCHMARK: {model_name.upper()} ({scope.upper()})   ")
        print(f"=======================================================")

        # Step 1: Threshold Calibration on Dev
        print("\n[Step 1] Calibrating Decision Threshold on Dev Set...")
        y_dev = dev_df['label'].astype(int).values
        dev_probs = _get_positive_probs(model_pipeline, dev_df)
        n_dev_human = int(np.sum(y_dev == 0))

        calibrated_threshold_conf = MetricEvaluator.find_threshold_for_max_fpr(
            y_dev, dev_probs, target_fpr=max_fpr, method='conformal'
        )
        calibrated_threshold_emp = MetricEvaluator.find_threshold_for_max_fpr(
            y_dev, dev_probs, target_fpr=max_fpr, method='empirical'
        )
        calibrated_threshold_evt = MetricEvaluator.find_threshold_for_max_fpr(
            y_dev, dev_probs, target_fpr=max_fpr, method='evt'
        )
        best_f1_threshold = MetricEvaluator.find_threshold_for_best_f1(y_dev, dev_probs)

        # Select operational threshold (Default: Conformal)
        if calibration_method.lower() == 'conformal':
            calibrated_threshold = calibrated_threshold_conf
        elif calibration_method.lower() == 'evt':
            calibrated_threshold = calibrated_threshold_evt
        else:
            calibrated_threshold = calibrated_threshold_emp

        if hasattr(model_pipeline, 'calibrated_threshold'):
            model_pipeline.calibrated_threshold = calibrated_threshold

        print(f" -> Human Dev Count (N_neg):                     {n_dev_human:,}")
        print(f" -> Conformal Threshold (FPR <= {max_fpr * 100:.1f}%):     {calibrated_threshold_conf:.6f} [ACTIVE]")
        print(f" -> Empirical Dev Threshold (FPR <= {max_fpr * 100:.1f}%): {calibrated_threshold_emp:.6f}")
        print(f" -> EVT GPD Threshold (FPR <= {max_fpr * 100:.1f}%):       {calibrated_threshold_evt:.6f}")
        print(f" -> Best F1 Dev Threshold:                        {best_f1_threshold:.6f}")

        dev_pred_df = cls._attach_prediction_diagnostics(dev_df, dev_probs, calibrated_threshold)
        dev_pred_df.to_csv(output_dir / "predictions_dev_calibrated.csv", index=False)

        # Step 2: Standard Test Evaluation
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
        n_human_test = fp + tn
        fpr_human = float(fp / n_human_test) if n_human_test > 0 else 0.0
        fpr_ci_lower, fpr_ci_upper = MetricEvaluator.compute_wilson_ci(fp, n_human_test, confidence=0.95)

        f1_human = float(f1_score(y_std, std_preds, pos_label=0, zero_division=0))
        brier = float(brier_score_loss(y_std, np.clip(std_probs, 0.0, 1.0)))

        print(f" -> ROC-AUC:        {roc_auc:.4f}")
        print(f" -> pAUC (FPR<=1%): {pauc:.4f}")
        print(f" -> TPR @ 1% FPR:   {tpr_at_1fpr * 100:.2f}%")
        print(f" -> FPR (Human):    {fpr_human * 100:.2f}% [{fpr_ci_lower * 100:.2f}%, {fpr_ci_upper * 100:.2f}%] ({fp}/{n_human_test})")
        print(f" -> F1-Score AI:    {f1_ai:.4f}")
        print(f" -> MCC:            {mcc:.4f}")

        std_pred_df = cls._attach_prediction_diagnostics(std_df, std_probs, calibrated_threshold)
        std_pred_df.to_csv(output_dir / "predictions_test_standard.csv", index=False)

        # Step 3: Discrete Rewrite Sensitivity (25%, 50%, 75%)
        print("\n[Step 3] Evaluating Discrete Rewrite Sensitivity (25%, 50%, 75% Buckets)...")
        sub_df = test_suites.get('substitution', test_suites.get('all', pd.DataFrame()))
        ratio_stats = []
        ratio_dict = {}

        ratio_col = next((c for c in ['llm_ratio', 'rewrite_ratio', 'pct_rewrite', 'replacement_ratio', 'ratio'] if c in sub_df.columns), None)

        if not sub_df.empty and ratio_col is not None:
            all_probs = _get_positive_probs(model_pipeline, sub_df)
            sub_pred_df = cls._attach_prediction_diagnostics(sub_df, all_probs, calibrated_threshold, has_labels=False)
            sub_pred_df['rewrite_ratio_raw'] = sub_pred_df[ratio_col].astype(float)
            sub_pred_df['rewrite_bucket'] = sub_pred_df['rewrite_ratio_raw'].apply(assign_discrete_rewrite_bucket)
            sub_pred_df.to_csv(output_dir / "predictions_test_substitution.csv", index=False)

            bucket_definitions = [('25pct', '25%', 0.25), ('50pct', '50%', 0.50), ('75pct', '75%', 0.75)]
            for key, label, num_val in bucket_definitions:
                grp = sub_pred_df[sub_pred_df['rewrite_bucket'] == label]
                if grp.empty:
                    continue
                grp_probs = grp['predicted_prob'].values
                raw_ratios = grp['rewrite_ratio_raw'].values
                norm_raw_ratios = np.where(raw_ratios > 1.0, raw_ratios / 100.0, raw_ratios)

                flagged_pct = float(np.mean(grp_probs >= calibrated_threshold) * 100.0)
                avg_score = float(np.mean(grp_probs))
                avg_actual_ratio = float(np.mean(norm_raw_ratios))
                n_count = len(grp)

                ratio_dict[key] = {
                    'bucket': label,
                    'target_ratio': label,
                    'target_ratio_num': num_val,
                    'actual_ratio': round(avg_actual_ratio, 4),
                    'flagged_pct': round(flagged_pct, 2),
                    'avg_score': round(avg_score, 4),
                    'count': int(n_count)
                }
                ratio_stats.append(ratio_dict[key])
                print(f" -> Rewrite Bucket [{label}] (Mean Actual: {avg_actual_ratio * 100:.1f}%, N={n_count:,}): {flagged_pct:.2f}% flagged (Avg Score: {avg_score:.4f})")

        # Step 4: Per-Generator Performance
        generator_stats = {}
        gen_pred_dfs = []
        for gen_name, gen_df in test_suites.get('per_generator', {}).items():
            y_g = gen_df['label'].astype(int).values
            g_probs = _get_positive_probs(model_pipeline, gen_df)
            g_annotated = cls._attach_prediction_diagnostics(gen_df, g_probs, calibrated_threshold)
            g_annotated['generator_group'] = gen_name
            gen_pred_dfs.append(g_annotated)

            has_both = len(np.unique(y_g)) >= 2
            generator_stats[gen_name] = {
                'detection_rate_at_threshold': float(np.mean(g_probs >= calibrated_threshold)),
                'mean_predicted_prob': float(np.mean(g_probs)),
                'roc_auc': MetricEvaluator.compute_metric(y_g, g_probs, 'roc_auc') if has_both else None,
                'pauc': MetricEvaluator.compute_metric(y_g, g_probs, 'pauc', max_fpr=max_fpr) if has_both else None,
                'tpr_at_1fpr': MetricEvaluator.compute_tpr_at_max_fpr(y_g, g_probs, target_fpr=max_fpr) if has_both else None,
                'f1_ai': MetricEvaluator.compute_metric(y_g, g_probs, 'f1', threshold=calibrated_threshold) if has_both else None,
                'count': len(gen_df)
            }

        if gen_pred_dfs:
            combined_gen_df = pd.concat(gen_pred_dfs, ignore_index=True)
            combined_gen_df.to_csv(output_dir / "predictions_test_per_generator.csv", index=False)

        summary = {
            'model_name': model_name,
            'scope': scope,
            'calibration_method': calibration_method,
            'calibrated_threshold': float(calibrated_threshold),
            'conformal_threshold': float(calibrated_threshold_conf),
            'empirical_threshold': float(calibrated_threshold_emp),
            'evt_threshold': float(calibrated_threshold_evt),
            'best_f1_threshold': float(best_f1_threshold),
            'overall_roc_auc': float(roc_auc),
            'overall_pauc': float(pauc),
            'tpr_at_1fpr': float(tpr_at_1fpr),
            'overall_pr_auc': float(pr_auc),
            'overall_mcc': float(mcc),
            'f1_ai': float(f1_ai),
            'f1_human': float(f1_human),
            'fpr_human': float(fpr_human),
            'fpr_human_ci_95': [round(fpr_ci_lower, 6), round(fpr_ci_upper, 6)],
            'brier_score': float(brier),
            'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
            'robustness_ratios': ratio_dict,
            'per_generator_performance': generator_stats
        }

        summary_json = output_dir / "evaluation_summary.json"
        summary_json.write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(f"\n[Exported Summary JSON] -> {summary_json}")

        export_performance_table(summary, scope=scope, model_name=model_name, output_path=latex_dir / "table_performance.tex")
        if ratio_stats:
            export_robustness_table(ratio_stats, scope=scope, model_name=model_name, output_path=latex_dir / "table_robustness_ratios.tex")

        return summary


def test_saved_model(
    model_type: str,
    model_path: Union[str, Path],
    scope: str = 'full',
    output_dir: Optional[Union[str, Path]] = None,
    max_fpr: float = 0.01,
    calibration_method: str = 'conformal',
    seed: int = 42
) -> Dict[str, Any]:
    from src.data.data_loader import DataFilter, DetectionDataManager
    manager = DetectionDataManager()
    val_filter = DataFilter(splits=['dev'], scopes=[scope])
    dev_df = manager.filter_dataframe(val_filter, sample_size=-1, seed=seed)
    test_suites = manager.get_benchmark_test_suites(scope=scope)

    if output_dir is None:
        output_dir = Path(model_path).parent

    detector_pipeline = load_detector_pipeline(model_type=model_type, model_path=model_path, scope=scope)
    return BenchmarkOrchestrator.run_full_benchmark(
        model_pipeline=detector_pipeline,
        dev_df=dev_df,
        test_suites=test_suites,
        model_name=model_type,
        scope=scope,
        output_dir=Path(output_dir),
        max_fpr=max_fpr,
        calibration_method=calibration_method
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Standalone Universal Benchmark Evaluator")
    parser.add_argument('--model_type', type=str, required=True, choices=['mdeberta', 'svm', 'fdgpt', 'stat_trajectory'])
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--scope', type=str, required=True, choices=['full', 'sentence'])
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--target_fpr', type=float, default=0.01)
    parser.add_argument('--calibration_method', type=str, default='conformal', choices=['conformal', 'empirical', 'evt'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    test_saved_model(
        model_type=args.model_type,
        model_path=args.model_path,
        scope=args.scope,
        output_dir=args.output_dir,
        max_fpr=args.target_fpr,
        calibration_method=args.calibration_method,
        seed=args.seed
    )