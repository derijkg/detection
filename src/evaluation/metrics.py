# src/evaluation/metrics.py

from typing import Dict, Optional, Tuple
import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, matthews_corrcoef, precision_score, recall_score,
    roc_auc_score, roc_curve
)


class MetricEvaluator:
    @staticmethod
    def compute_tpr_at_max_fpr(
        y_true: np.ndarray, 
        y_score: np.ndarray, 
        target_fpr: float = 0.01
    ) -> float:
        """Interpolates True Positive Rate at exact target False Positive Rate."""
        y_true = np.asarray(y_true, dtype=int)
        y_score = np.asarray(y_score, dtype=float)
        if len(np.unique(y_true)) < 2:
            return 0.0
        fpr, tpr, _ = roc_curve(y_true, y_score)
        interp_fn = interp1d(fpr, tpr, bounds_error=False, fill_value=(0.0, 1.0))
        return float(interp_fn(target_fpr))

    @staticmethod
    def compute_metric(
        y_true: np.ndarray, 
        y_score: np.ndarray, 
        metric_name: str = 'pauc', 
        max_fpr: float = 0.01,
        threshold: float = 0.5
    ) -> float:
        metric = metric_name.lower().replace('-', '_')
        y_true = np.asarray(y_true, dtype=int)
        y_score = np.asarray(y_score, dtype=float)

        if len(np.unique(y_true)) < 2:
            return 0.5 if metric in ['pauc', 'roc_auc'] else 0.0

        if metric in ['pauc', 'partial_auc', 'p_auc']:
            try:
                return float(roc_auc_score(y_true, y_score, max_fpr=max_fpr))
            except Exception:
                return 0.5
        elif metric in ['roc_auc', 'rocauc']:
            try:
                return float(roc_auc_score(y_true, y_score))
            except Exception:
                return 0.5
        elif metric in ['tpr_at_1fpr', 'tpr_at_fpr']:
            return MetricEvaluator.compute_tpr_at_max_fpr(y_true, y_score, target_fpr=max_fpr)
        elif metric in ['pr_auc', 'average_precision']:
            try:
                return float(average_precision_score(y_true, y_score))
            except Exception:
                return 0.0
        
        y_pred = (y_score >= threshold).astype(int)

        if metric == 'mcc':
            return float(matthews_corrcoef(y_true, y_pred))
        elif metric == 'f1':
            return float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
        elif metric == 'precision':
            return float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
        elif metric == 'recall':
            return float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
            
        return float(roc_auc_score(y_true, y_score))

    @staticmethod
    def find_threshold_for_max_fpr(
        y_true: np.ndarray, 
        y_score: np.ndarray, 
        target_fpr: float = 0.01
    ) -> float:
        """
        Calculates the decision threshold enforcing FPR <= target_fpr on the negative (human) distribution.
        """
        y_true = np.asarray(y_true, dtype=int)
        y_score = np.asarray(y_score, dtype=float)
        
        neg_scores = y_score[y_true == 0]
        if len(neg_scores) == 0:
            return 0.5

        return float(np.quantile(neg_scores, 1.0 - target_fpr))

    @staticmethod
    def find_threshold_for_best_f1(
        y_true: np.ndarray, 
        y_score: np.ndarray
    ) -> float:
        """
        Calculates the threshold maximizing F1-score across empirical score quantiles.
        """
        y_true = np.asarray(y_true, dtype=int)
        y_score = np.asarray(y_score, dtype=float)

        threshold_candidates = np.unique(np.quantile(y_score, np.linspace(0.01, 0.99, 100)))
        best_f1 = -1.0
        best_thresh = float(np.median(y_score))

        for t in threshold_candidates:
            preds = (y_score >= t).astype(int)
            f1 = f1_score(y_true, preds, pos_label=1, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(t)

        return float(best_thresh)

    @staticmethod
    def compute_bootstrap_ci(
        y_true: np.ndarray,
        y_score: np.ndarray,
        metric_name: str = 'pauc',
        max_fpr: float = 0.01,
        threshold: float = 0.5,
        n_bootstraps: int = 1000,
        ci: float = 0.95,
        seed: int = 42
    ) -> Dict[str, float]:
        y_true = np.asarray(y_true, dtype=int)
        y_score = np.asarray(y_score, dtype=float)
        
        point_est = MetricEvaluator.compute_metric(
            y_true, y_score, metric_name=metric_name, max_fpr=max_fpr, threshold=threshold
        )

        neg_indices = np.where(y_true == 0)[0]
        pos_indices = np.where(y_true == 1)[0]
        
        if len(neg_indices) == 0 or len(pos_indices) == 0:
            return {"point_estimate": point_est, "ci_lower": point_est, "ci_upper": point_est}

        rng = np.random.RandomState(seed)
        bootstrapped_scores = []
        alpha = (1.0 - ci) / 2.0

        for _ in range(n_bootstraps):
            b_neg = rng.choice(neg_indices, size=len(neg_indices), replace=True)
            b_pos = rng.choice(pos_indices, size=len(pos_indices), replace=True)
            b_idx = np.concatenate([b_neg, b_pos])
            
            val = MetricEvaluator.compute_metric(
                y_true[b_idx], y_score[b_idx], metric_name=metric_name, max_fpr=max_fpr, threshold=threshold
            )
            bootstrapped_scores.append(val)

        ci_lower = float(np.percentile(bootstrapped_scores, alpha * 100))
        ci_upper = float(np.percentile(bootstrapped_scores, (1.0 - alpha) * 100))

        return {
            "point_estimate": float(point_est),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper
        }