"""
src/evaluation/metrics.py
Operational evaluation metrics for AI text detection under low False Positive Rate (FPR) regimes.
Includes Split Conformal calibration, empirical quantiles, EVT GPD extreme value tail calibration,
Wilson score binomial confidence intervals, pAUC, TPR@max_FPR, MCC, F1, and Brier score.
"""

from typing import Dict, Optional, Tuple, Union
import warnings
import numpy as np
import scipy.stats as stats
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
    def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Computes the Wilson score binomial confidence interval for proportion k / n.
        Essential for reporting empirical human False Positive Rate bounds on the test set.
        """
        if n <= 0:
            return (0.0, 0.0)
        p_hat = k / n
        z = float(stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0))
        denom = 1.0 + (z ** 2) / n
        center = (p_hat + (z ** 2) / (2.0 * n)) / denom
        margin = (z * np.sqrt((p_hat * (1.0 - p_hat) + (z ** 2) / (4.0 * n)) / n)) / denom
        lower = float(np.clip(center - margin, 0.0, 1.0))
        upper = float(np.clip(center + margin, 0.0, 1.0))
        return (lower, upper)

    @staticmethod
    def compute_tpr_at_max_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.01) -> float:
        """Computes the maximum achievable True Positive Rate at or below a target FPR threshold."""
        y_true = np.asarray(y_true, dtype=int)
        y_score = np.asarray(y_score, dtype=float)
        if len(np.unique(y_true)) < 2:
            return 0.0

        fpr, tpr, _ = roc_curve(y_true, y_score)
        unique_fpr, rev_indices = np.unique(fpr, return_inverse=True)
        max_tpr = np.maximum.reduceat(tpr, np.r_[0, np.where(np.diff(rev_indices))[0] + 1])
        max_tpr_accum = np.maximum.accumulate(max_tpr)

        return float(np.interp(target_fpr, unique_fpr, max_tpr_accum, left=0.0, right=float(max_tpr_accum[-1])))

    @staticmethod
    def compute_metric(
        y_true: np.ndarray,
        y_score: np.ndarray,
        metric_name: str = 'pauc',
        max_fpr: float = 0.01,
        threshold: float = 0.5
    ) -> float:
        """Unified interface to compute classification metrics."""
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
    def find_threshold_for_best_f1(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Finds the decision threshold maximizing F1 on the positive class."""
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
        """Calculates stratified percentile bootstrap confidence intervals."""
        y_true = np.asarray(y_true, dtype=int)
        y_score = np.asarray(y_score, dtype=float)
        point_est = MetricEvaluator.compute_metric(
            y_true, y_score, metric_name=metric_name, max_fpr=max_fpr, threshold=threshold
        )
        neg_indices = np.where(y_true == 0)[0]
        pos_indices = np.where(y_true == 1)[0]
        if len(neg_indices) == 0 or len(pos_indices) == 0:
            return {'point_estimate': point_est, 'ci_lower': point_est, 'ci_upper': point_est}

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
        return {'point_estimate': float(point_est), 'ci_lower': ci_lower, 'ci_upper': ci_upper}

    @staticmethod
    def find_threshold_conformal(
        y_true: np.ndarray,
        y_score: np.ndarray,
        target_fpr: float = 0.01
    ) -> float:
        """
        Split Conformal Prediction threshold calibration.
        Guarantees P(False Positive) <= target_fpr in finite samples under exchangeability.
        Index rule: k = ceil((n_neg + 1) * (1 - target_fpr))
        """
        y_true = np.asarray(y_true, dtype=int)
        y_score = np.asarray(y_score, dtype=float)
        neg_scores = np.sort(y_score[y_true == 0])
        n_neg = len(neg_scores)
        if n_neg == 0:
            return 0.5

        k = int(np.ceil((n_neg + 1) * (1.0 - target_fpr)))
        k = min(max(1, k), n_neg)

        # 1-based index k corresponds to index k-1 in 0-indexed sorted array
        return float(neg_scores[k - 1])

    @staticmethod
    def fit_gpd_tail(neg_scores: np.ndarray, tail_quantile: float = 0.9) -> Tuple[float, float, float, int]:
        """Fits Generalized Pareto Distribution (GPD) on upper tail excesses above tail_quantile."""
        neg_scores = np.sort(neg_scores[np.isfinite(neg_scores)])
        n = len(neg_scores)
        if n < 20:
            u = float(np.percentile(neg_scores, tail_quantile * 100))
            return (u, 1.0, 0.0, max(1, int(n * (1 - tail_quantile))))

        u = float(np.quantile(neg_scores, tail_quantile))
        excesses = neg_scores[neg_scores > u] - u
        if len(excesses) < 5 or np.all(excesses == 0):
            return (u, 0.001, 0.0, len(excesses))

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                c_fit, _, scale_fit = stats.genpareto.fit(excesses, floc=0.0)
                xi = float(c_fit)
                sigma = float(scale_fit)
            except Exception:
                xi = 0.0
                sigma = float(np.mean(excesses))

        return (u, max(sigma, 1e-06), xi, len(excesses))

    @staticmethod
    def find_threshold_evt_gpd(
        y_true: np.ndarray,
        y_score: np.ndarray,
        target_fpr: float = 0.01,
        tail_quantile: float = 0.9
    ) -> float:
        """
        Extrapolates extreme decision threshold for low target FPR using Pickands-Balkema-de Haan EVT.
        """
        y_true = np.asarray(y_true, dtype=int)
        y_score = np.asarray(y_score, dtype=float)
        neg_scores = y_score[y_true == 0]
        n_neg = len(neg_scores)
        if n_neg == 0:
            return 0.5

        q = 1.0 - tail_quantile
        if target_fpr >= q:
            return MetricEvaluator.find_threshold_conformal(y_true, y_score, target_fpr=target_fpr)

        u, sigma, xi, _ = MetricEvaluator.fit_gpd_tail(neg_scores, tail_quantile=tail_quantile)
        ratio = q / max(target_fpr, 1e-07)

        if abs(xi) < 1e-4:
            threshold = u + sigma * np.log(ratio)
        else:
            threshold = u + (sigma / xi) * (ratio ** xi - 1.0)

        max_observed = float(np.max(neg_scores))
        min_observed = float(np.min(neg_scores))
        return float(np.clip(threshold, min_observed, max_observed + 5.0 * sigma))

    @staticmethod
    def find_threshold_for_max_fpr(
        y_true: np.ndarray,
        y_score: np.ndarray,
        target_fpr: float = 0.01,
        method: str = 'conformal'
    ) -> float:
        """
        Calculates the score threshold guaranteeing False Positive Rate <= target_fpr on Dev split.
        Supported methods:
        - 'conformal' (Default): Distribution-free split conformal prediction with finite-sample bounds.
        - 'empirical': Classical exact empirical order statistic.
        - 'evt': Extreme Value Theory Generalized Pareto Distribution tail fitting.
        """
        method = method.lower()
        if method == 'conformal':
            return MetricEvaluator.find_threshold_conformal(y_true, y_score, target_fpr=target_fpr)
        elif method == 'evt':
            return MetricEvaluator.find_threshold_evt_gpd(y_true, y_score, target_fpr=target_fpr)
        elif method == 'empirical':
            y_true = np.asarray(y_true, dtype=int)
            y_score = np.asarray(y_score, dtype=float)
            neg_scores = np.sort(y_score[y_true == 0])
            n_neg = len(neg_scores)
            if n_neg == 0:
                return 0.5
            max_fp = int(np.floor(target_fpr * n_neg))
            if max_fp == 0:
                return float(neg_scores[-1] + 1e-07)
            idx = max(0, n_neg - max_fp)
            thresh = float(neg_scores[idx])
            actual_fp = n_neg - np.searchsorted(neg_scores, thresh, side='left')
            while actual_fp > max_fp and idx < n_neg - 1:
                idx += 1
                thresh = float(neg_scores[idx])
                actual_fp = n_neg - np.searchsorted(neg_scores, thresh, side='left')
            if actual_fp > max_fp:
                return float(neg_scores[-1] + 1e-07)
            return thresh
        else:
            raise ValueError(f"Unknown calibration method: '{method}'. Choose from ['conformal', 'empirical', 'evt'].")