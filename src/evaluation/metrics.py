# src/evaluation/metrics.py

from typing import Dict, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


class MetricEvaluator:
    @staticmethod
    def compute_tpr_at_max_fpr(
        y_true: np.ndarray, 
        y_score: np.ndarray, 
        target_fpr: float = 0.01
    ) -> float:
        """
        Interpolates True Positive Rate at exact target False Positive Rate
        handling duplicate FPR values monotonically without scipy exceptions.
        """
        y_true = np.asarray(y_true, dtype=int)
        y_score = np.asarray(y_score, dtype=float)
        
        if len(np.unique(y_true)) < 2:
            return 0.0
            
        fpr, tpr, _ = roc_curve(y_true, y_score)
        
        # Deduplicate FPR taking the maximum achievable TPR at each unique FPR
        unique_fpr, indices = np.unique(fpr, return_index=True)
        max_tpr = np.zeros_like(unique_fpr)
        for i, u_fpr in enumerate(unique_fpr):
            max_tpr[i] = np.max(tpr[fpr == u_fpr])
            
        # Ensure strict monotonicity for interpolation
        max_tpr = np.maximum.accumulate(max_tpr)
        return float(np.interp(target_fpr, unique_fpr, max_tpr, left=0.0, right=1.0))

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
            return 0.5 if metric in ['pauc', 'roc_auc', 'partial_auc'] else 0.0

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
            try:
                return float(matthews_corrcoef(y_true, y_pred))
            except Exception:
                return 0.0
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
        Calculates a conservative decision threshold enforcing empirical FPR <= target_fpr 
        on the negative (human) distribution without fractional boundary leakage.
        """
        y_true = np.asarray(y_true, dtype=int)
        y_score = np.asarray(y_score, dtype=float)
        
        neg_scores = np.sort(y_score[y_true == 0])
        n_neg = len(neg_scores)
        if n_neg == 0:
            return 0.5

        # Maximum allowed false positives
        max_fp = int(np.floor(target_fpr * n_neg))
        
        if max_fp == 0:
            # Strictest boundary: threshold is the maximum observed negative score
            return float(neg_scores[-1])
            
        # Index corresponding to top max_fp tail
        idx = max(0, n_neg - max_fp)
        return float(neg_scores[idx])

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

        if len(y_score) == 0:
            return 0.5

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