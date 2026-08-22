"""
src/models/statistical_detector.py
Tabular thermodynamic trajectory detector using RobustScaler and Platt-calibrated LinearSVC.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVC

from src.models.base import BaseDetector
from src.models.statistical_features import extract_or_load_statistical_dataset

META_COLS = ['_id', 'id', 'text', 'source', 'keywords', 'year', 'split', 'scope', 'generation_type', 'model_name', 'llm_ratio']


class StatisticalTrajectoryDetector(BaseDetector):

    def __init__(
        self,
        pipeline: Optional[Pipeline] = None,
        feature_names: Optional[List[str]] = None,
        scope: str = 'full',
        t_critical: float = 0.88,
        seed: int = 42,
        log_dir: Optional[Union[str, Path]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        **kwargs
    ):
        super().__init__(model_name='stat_trajectory', scope=scope, seed=seed, log_dir=log_dir)
        self.pipeline = pipeline
        self.feature_names = feature_names or []
        self.t_critical = float(t_critical)
        self.cache_dir = Path(cache_dir or "data_static/preprocessed/stat_cache")

    def _ensure_feature_df(self, data: Union[pd.DataFrame, List[Dict[str, Any]]], split_tag: str = 'custom') -> pd.DataFrame:
        df = pd.DataFrame(data)
        stat_cols = [c for c in df.columns if any(k in c for k in ['surprisal', 'entropy', 'gini', 'zipf', 'log_prob', 'spec_heat', 'elasticity'])]
        if len(stat_cols) >= 8:
            return df

        self.logger.info(f"Extracting/loading thermodynamic trajectory features ({split_tag}, Tc={self.t_critical:.2f})...")
        return extract_or_load_statistical_dataset(
            df, scope=self.scope, split_name=split_tag, cache_dir=self.cache_dir, t_critical=self.t_critical
        )

    def fit(
        self,
        train_data: Union[pd.DataFrame, List[Dict[str, Any]]],
        y_train: Optional[np.ndarray] = None,
        **kwargs
    ) -> 'StatisticalTrajectoryDetector':
        df = self._ensure_feature_df(train_data, split_tag='train')
        if 'label' in df.columns:
            y = df['label'].astype(int).values
            X = df.drop(columns=['label'])
        else:
            y = y_train
            X = df

        drop_cols = [c for c in META_COLS if c in X.columns]
        if drop_cols:
            X = X.drop(columns=drop_cols)

        X = X.select_dtypes(include=[np.number]).fillna(0.0)
        self.feature_names = list(X.columns)

        self.logger.info(f"Fitting Thermodynamic Trajectory Classifier on {len(X)} samples ({len(self.feature_names)} features)...")
        scaler = RobustScaler()
        clf = CalibratedClassifierCV(
            estimator=LinearSVC(C=1.0, loss='squared_hinge', dual=False, random_state=self.seed, max_iter=5000),
            cv=3,
            method='sigmoid'
        )
        self.pipeline = Pipeline([('scaler', scaler), ('classifier', clf)])
        self.pipeline.fit(X.values, y)
        self.logger.info("Classifier fitted and calibrated via Platt scaling.")
        return self

    def predict_proba(self, X_input: Union[pd.DataFrame, List[Dict[str, Any]], np.ndarray]) -> np.ndarray:
        if self.pipeline is None:
            raise ValueError("StatisticalTrajectoryDetector is not fitted.")

        if isinstance(X_input, np.ndarray):
            return self.pipeline.predict_proba(np.asarray(X_input, dtype=np.float64))[:, 1]

        df = self._ensure_feature_df(X_input, split_tag='inference')
        df_clean = df.fillna(0.0)
        missing = [c for c in self.feature_names if c not in df_clean.columns]
        for m in missing:
            df_clean[m] = 0.0

        X_vals = df_clean[self.feature_names].astype(np.float64).values
        return self.pipeline.predict_proba(X_vals)[:, 1]

    def save(self, path: Union[str, Path]):
        save_p = Path(path)
        if save_p.is_dir() or not save_p.name.endswith('.joblib'):
            save_p = save_p / 'model.joblib'
        save_p.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            'feature_names': self.feature_names,
            'scope': self.scope,
            't_critical': self.t_critical,
            'calibrated_threshold': self.calibrated_threshold
        }
        joblib.dump({'pipeline': self.pipeline, 'metadata': meta}, save_p)
        self.logger.info(f"Saved Thermodynamic Trajectory Detector to: {save_p}")

    @classmethod
    def load(cls, path: Union[str, Path], scope: str = 'full', seed: int = 42, log_dir: Optional[Path] = None, **kwargs) -> 'StatisticalTrajectoryDetector':
        load_p = Path(path)
        if load_p.is_dir():
            load_p = load_p / 'model.joblib'
        if not load_p.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {load_p}")

        data = joblib.load(load_p)
        detector = cls(
            pipeline=data['pipeline'],
            feature_names=data['metadata']['feature_names'],
            scope=scope,
            t_critical=data['metadata'].get('t_critical', 0.88),
            seed=seed,
            log_dir=log_dir
        )
        detector.calibrated_threshold = data['metadata'].get('calibrated_threshold', 0.5)
        return detector