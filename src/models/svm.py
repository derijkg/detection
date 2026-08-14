# detection/src/models/svm.py

import os
import re
import string
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_curve

from src.models.base_model import BaseDetector
from src.training.metrics import compute_classification_metrics

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FEATURE_CACHE_DIR = PROJECT_ROOT / "data_static" / "model_features"

DUTCH_TRANSITIONS = {
    "echter", "bovendien", "daarnaast", "desalniettemin", "kortom",
    "tevens", "daardoor", "derhalve", "bijgevolg", "namelijk"
}


# =============================================================================
# FEATURE EXTRACTION TRANSFORMERS WITH FITTED ATTRIBUTES
# =============================================================================

class TextExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        output = []
        items = [X] if isinstance(X, (str, dict)) else X
        for item in items:
            if isinstance(item, str):
                output.append(item)
            elif isinstance(item, dict):
                output.append(item.get('text', ''))
            else:
                output.append(str(item))
        return output


class StylometricExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, granularity: str = 'full'):
        self.granularity = granularity

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def _extract_single(self, text: str) -> np.ndarray:
        words = re.findall(r'\w+', text.lower())
        total_chars = max(1, len(text))

        if not words:
            num_features = 8 if self.granularity in ['single', 'sentence'] else 11
            return np.zeros(num_features)

        word_lengths = [len(w) for w in words]
        mean_word_len = float(np.mean(word_lengths))
        var_word_len = float(np.var(word_lengths))

        ttr = len(set(words)) / len(words)
        counts = Counter(words)
        hapax_ratio = sum(1 for w, c in counts.items() if c == 1) / len(words)

        transition_count = sum(1 for w in words if w in DUTCH_TRANSITIONS)
        transition_ratio = transition_count / len(words)

        spaces_count = text.count(' ')
        double_spaces = text.count('  ')
        space_ratio = spaces_count / total_chars
        double_space_ratio = double_spaces / total_chars

        punc_count = sum(1 for c in text if c in string.punctuation)
        punc_ratio = punc_count / total_chars

        base_features = [
            mean_word_len,
            np.log1p(var_word_len),
            ttr,
            hapax_ratio,
            transition_ratio,
            space_ratio,
            double_space_ratio,
            punc_ratio
        ]

        if self.granularity in ['single', 'sentence']:
            return np.array(base_features)

        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        sent_lengths = [len(re.findall(r'\w+', s)) for s in sents if len(re.findall(r'\w+', s)) > 0]
        
        if not sent_lengths or len(sent_lengths) <= 1:
            mean_sent_len = float(len(words))
            var_sent_len = 0.0
            burstiness = 0.0
        else:
            mean_sent_len = float(np.mean(sent_lengths))
            var_sent_len = float(np.var(sent_lengths))
            std_sent_len = float(np.std(sent_lengths))
            burstiness = (std_sent_len - mean_sent_len) / (std_sent_len + mean_sent_len) if (std_sent_len + mean_sent_len) > 0 else 0.0

        return np.array([np.log1p(mean_sent_len), np.log1p(var_sent_len), burstiness] + base_features)

    def transform(self, X):
        items = [X] if isinstance(X, (str, dict)) else X
        text_extractor = TextExtractor()
        texts = text_extractor.transform(items)
        return np.array([self._extract_single(t) for t in texts])


class StylometricScaler(BaseEstimator, TransformerMixin):
    def __init__(self, weight: float = 1.0):
        self.weight = weight

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        return X * self.weight


# =============================================================================
# SVM DETECTOR WITH FEATURE CACHING & OOF THRESHOLD CALIBRATION
# =============================================================================

class SVMDetector(BaseDetector):
    def __init__(self, granularity: str = 'full', calibrate: bool = True, max_fpr: float = 0.01, **kwargs):
        self.granularity = granularity
        self.calibrate = calibrate
        self.max_fpr = max_fpr
        self.feature_pipeline: Optional[Pipeline] = None
        self.classifier: Optional[Any] = None
        self.optimal_threshold: float = 0.5

    def _build_feature_pipeline(self, params: Dict[str, Any]) -> Pipeline:
        default_max_df = 1.0 if self.granularity in ['single', 'sentence'] else 0.95

        w_params = {
            'ngram_range': (params.get('word_min_ngram', 1), params.get('word_max_ngram', 3)),
            'max_features': params.get('word_max_features', 50000),
            'min_df': params.get('word_min_df', 2),
            'max_df': params.get('word_max_df', default_max_df),
            'sublinear_tf': params.get('word_sublinear_tf', True),
            'binary': params.get('word_binary', False),
            'norm': 'l2',
            'analyzer': 'word'
        }

        c_params = {
            'ngram_range': (params.get('char_min_ngram', 3), params.get('char_max_ngram', 5)),
            'max_features': params.get('char_max_features', 50000),
            'min_df': params.get('char_min_df', 2),
            'max_df': params.get('char_max_df', default_max_df),
            'sublinear_tf': params.get('char_sublinear_tf', True),
            'binary': params.get('char_binary', False),
            'norm': 'l2',
            'analyzer': 'char'
        }

        transformers = [
            ('word_ngrams', Pipeline([('extract', TextExtractor()), ('tfidf', TfidfVectorizer(**w_params))])),
            ('char_ngrams', Pipeline([('extract', TextExtractor()), ('tfidf', TfidfVectorizer(**c_params))]))
        ]

        if params.get('use_stylometrics', True):
            sty_weight = params.get('sty_weight', 0.05)
            transformers.append(
                ('stylometrics', Pipeline([
                    ('extractor', StylometricExtractor(granularity=self.granularity)),
                    ('scaler', StandardScaler()),
                    ('subspace_norm', Normalizer(norm='l2')),
                    ('weight', StylometricScaler(weight=sty_weight))
                ]))
            )

        return Pipeline([('union', FeatureUnion(transformers)), ('normalizer', Normalizer(norm='l2'))])

    def _get_or_compute_features(self, X: List[str], split_name: str, params: Dict[str, Any]) -> Any:
        cache_folder = FEATURE_CACHE_DIR / f"svm_{self.granularity}"
        cache_folder.mkdir(parents=True, exist_ok=True)
        
        n_samples = len(X)
        cache_file = cache_folder / f"{split_name}_{n_samples}_features.joblib"
        legacy_cache_file = cache_folder / f"{split_name}_features.joblib"

        # Check size-specific cache first
        if cache_file.exists():
            print(f"-> [FEATURE CACHE HIT] Loading '{split_name}' features ({n_samples} samples) from: {cache_file}")
            cached_data = joblib.load(cache_file)
            self.feature_pipeline = cached_data['pipeline']
            return cached_data['X_feats']
        
        # Check legacy cache file with length validation
        if legacy_cache_file.exists():
            cached_data = joblib.load(legacy_cache_file)
            if cached_data['X_feats'].shape[0] == n_samples:
                print(f"-> [FEATURE CACHE HIT] Loading legacy '{split_name}' features ({n_samples} samples) from: {legacy_cache_file}")
                self.feature_pipeline = cached_data['pipeline']
                return cached_data['X_feats']
            else:
                print(f"-> [FEATURE CACHE MISMATCH] Legacy '{split_name}' feature size ({cached_data['X_feats'].shape[0]}) != input size ({n_samples}). Recomputing...")

        print(f"-> [FEATURE CACHE MISS] Extracting '{split_name}' features ({n_samples} samples) for scope '{self.granularity}'...")
        if self.feature_pipeline is None or split_name == "train":
            self.feature_pipeline = self._build_feature_pipeline(params)
            X_feats = self.feature_pipeline.fit_transform(X)
        else:
            X_feats = self.feature_pipeline.transform(X)

        joblib.dump({'X_feats': X_feats, 'pipeline': self.feature_pipeline}, cache_file)
        print(f"-> Saved extracted features to: {cache_file}")
        return X_feats

    def _compute_oof_optimal_threshold(self, X_feats: Any, y_train: np.ndarray, doc_ids: Optional[np.ndarray]) -> float:
        print("Calculating Out-Of-Fold (OOF) cross-validation predictions for optimal operating point...")
        
        if doc_ids is None or len(np.unique(doc_ids)) < 2:
            doc_ids = np.arange(len(y_train))

        sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
        oof_scores = np.zeros(len(y_train))

        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_feats, y_train, groups=doc_ids)):
            X_tr, y_tr = X_feats[train_idx], y_train[train_idx]
            X_va = X_feats[val_idx]

            base_clf = LinearSVC(C=1.0, dual=False, random_state=42, class_weight='balanced', max_iter=10000)
            clf = CalibratedClassifierCV(estimator=base_clf, cv=3, method='sigmoid')
            clf.fit(X_tr, y_tr)
            oof_scores[val_idx] = clf.predict_proba(X_va)[:, 1]

        fpr, tpr, thresholds = roc_curve(y_train, oof_scores)
        valid_indices = np.where(fpr <= self.max_fpr)[0]

        if len(valid_indices) > 0:
            optimal_thresh = float(thresholds[valid_indices[-1]])
        else:
            optimal_thresh = 0.5

        print(f"-> [OOF OPERATING POINT] Optimal threshold for FPR <= {self.max_fpr*100:.1f}%: {optimal_thresh:.6f}")
        return optimal_thresh

    def train(self, train_ds: Any, val_ds: Any, training_args_dict: Dict[str, Any]) -> Dict[str, Any]:
        if hasattr(train_ds, 'to_pandas'):
            train_df = train_ds.to_pandas()
            X_train, y_train = train_df['text'].tolist(), train_df['label'].values
            doc_ids = train_df['_id'].values if '_id' in train_df.columns else None
        elif isinstance(train_ds, tuple):
            X_train, y_train = train_ds
            doc_ids = None
        else:
            X_train = [sample.text for sample in train_ds]
            y_train = np.array([sample.label for sample in train_ds])
            doc_ids = np.array([getattr(sample, 'id', i) for i, sample in enumerate(train_ds)])

        # 1. Feature Extraction / Cache
        X_train_feats = self._get_or_compute_features(X_train, "train", training_args_dict)

        # 2. OOF Optimal Operating Point Calculation
        self.optimal_threshold = self._compute_oof_optimal_threshold(X_train_feats, y_train, doc_ids)

        # 3. Train Final Model on 100% Training Data
        kernel = training_args_dict.get('kernel', 'linear')
        c_val = training_args_dict.get('C', 1.0)
        class_weight = training_args_dict.get('class_weight', 'balanced')

        if kernel == 'linear':
            base_clf = LinearSVC(C=c_val, loss='squared_hinge', dual=False, random_state=42, class_weight=class_weight, max_iter=10000)
        else:
            base_clf = SVC(C=c_val, kernel=kernel, gamma='scale', random_state=42, class_weight=class_weight, probability=True)

        if self.calibrate and kernel == 'linear':
            self.classifier = CalibratedClassifierCV(estimator=base_clf, cv=3, method='sigmoid')
        else:
            self.classifier = base_clf

        print(f"Fitting final SVM model on 100% of training feature matrix...")
        self.classifier.fit(X_train_feats, y_train)

        # 4. Evaluate on Validation Set
        if val_ds is not None:
            if hasattr(val_ds, 'to_pandas'):
                val_df = val_ds.to_pandas()
                X_val, y_val = val_df['text'].tolist(), val_df['label'].values
            elif isinstance(val_ds, tuple):
                X_val, y_val = val_ds
            else:
                X_val = [sample.text for sample in val_ds]
                y_val = np.array([sample.label for sample in val_ds])

            X_val_feats = self._get_or_compute_features(X_val, "dev", training_args_dict)
            val_probs = self.classifier.predict_proba(X_val_feats)[:, 1] if hasattr(self.classifier, 'predict_proba') else self.classifier.decision_function(X_val_feats)
            
            val_metrics = compute_classification_metrics(y_val, val_probs, threshold=self.optimal_threshold)
            val_metrics["optimal_operating_threshold"] = self.optimal_threshold
            return val_metrics

        return {"optimal_operating_threshold": self.optimal_threshold}

    def predict_proba(self, texts: List[str], batch_size: int = 256) -> np.ndarray:
        if self.classifier is None or self.feature_pipeline is None:
            raise RuntimeError("Model is not fitted. Call train() or load() first.")

        X_feats = self.feature_pipeline.transform(texts)
        if hasattr(self.classifier, 'predict_proba'):
            probs = self.classifier.predict_proba(X_feats)
        else:
            scores = self.classifier.decision_function(X_feats)
            p1 = 1.0 / (1.0 + np.exp(-scores))
            probs = np.column_stack([1.0 - p1, p1])

        return probs

    def save(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "svm_model_bundle.joblib")
        
        bundle = {
            'feature_pipeline': self.feature_pipeline,
            'classifier': self.classifier,
            'optimal_threshold': self.optimal_threshold,
            'granularity': self.granularity,
            'calibrate': self.calibrate
        }
        joblib.dump(bundle, save_path)
        print(f"[MODEL SAVED] SVM model bundle saved to '{save_path}'")

    def load(self, input_dir: str) -> None:
        load_path = os.path.join(input_dir, "svm_model_bundle.joblib") if os.path.isdir(input_dir) else input_dir
        if not os.path.exists(load_path):
            load_path = os.path.join(input_dir, "svm_pipeline.pkl") if os.path.isdir(input_dir) else input_dir

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"SVM checkpoint not found at: {load_path}")

        loaded = joblib.load(load_path)
        if isinstance(loaded, dict) and 'classifier' in loaded:
            self.feature_pipeline = loaded['feature_pipeline']
            self.classifier = loaded['classifier']
            self.optimal_threshold = loaded.get('optimal_threshold', 0.5)
            self.granularity = loaded.get('granularity', 'full')
        else:
            self.feature_pipeline = loaded.named_steps['features']
            self.classifier = loaded.named_steps['classifier']
            self.optimal_threshold = getattr(loaded, 'optimal_threshold', 0.5)

        print(f"[MODEL LOADED] SVM model loaded from '{load_path}' (Optimal Threshold: {self.optimal_threshold:.6f})")