# src/models/svm_pipeline.py
import json
import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import nltk
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import LinearSVC

from src.data.stylometrics import extract_stylometric_features, normalize_text, RE_SENT_SPLIT
from src.models.base import BaseDetector

try:
    from optuna.trial import Trial
except ImportError:
    Trial = Any

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
from nltk.corpus import stopwords

_nlp = None
_dutch_stopwords_lemmatized = None

def get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            try:
                _nlp = spacy.load('nl_core_news_sm', disable=['parser', 'ner'])
            except Exception:
                import spacy.cli
                spacy.cli.download('nl_core_news_sm')
                _nlp = spacy.load('nl_core_news_sm', disable=['parser', 'ner'])
            if 'sentencizer' not in _nlp.pipe_names:
                _nlp.add_pipe('sentencizer')
        except Exception:
            _nlp = None
    return _nlp

def get_dutch_stopwords_lemmatized() -> List[str]:
    global _dutch_stopwords_lemmatized
    if _dutch_stopwords_lemmatized is None:
        try:
            raw_stops = stopwords.words('dutch')
        except Exception:
            raw_stops = []
        nlp_model = get_nlp()
        if nlp_model is not None:
            try:
                _dutch_stopwords_lemmatized = list(set([
                    token.lemma_.lower()
                    for doc in nlp_model.pipe(raw_stops, batch_size=1000)
                    for token in doc
                ]))
            except Exception:
                _dutch_stopwords_lemmatized = list(set([w.lower() for w in raw_stops]))
        else:
            _dutch_stopwords_lemmatized = list(set([w.lower() for w in raw_stops]))
    return _dutch_stopwords_lemmatized

class TextExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, key: str = 'text'):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X: Any) -> List[str]:
        if isinstance(X, pd.DataFrame):
            records = X.to_dict(orient='records')
        elif isinstance(X, list) and len(X) > 0 and isinstance(X[0], dict):
            records = X
        elif isinstance(X, (list, np.ndarray)):
            if len(X) == 0:
                return []
            records = [{'text': str(t)} for t in X]
        else:
            records = [{'text': str(X)}]

        output = []
        missing_lemma_indices = []
        raw_texts_to_lemma = []
        for idx, item in enumerate(records):
            if self.key == 'text_lemmatized':
                val = item.get('text_lemmatized')
                if val and str(val).strip():
                    output.append(str(val))
                else:
                    norm_t = item.get('normalized_text', normalize_text(item.get('text', '')))
                    output.append('')
                    missing_lemma_indices.append(idx)
                    raw_texts_to_lemma.append(norm_t)
            elif self.key == 'normalized_text':
                val = item.get('normalized_text')
                output.append(str(val) if val else normalize_text(item.get('text', '')))
            else:
                output.append(normalize_text(item.get('text', '')))

        if missing_lemma_indices:
            nlp = get_nlp()
            if nlp is not None:
                try:
                    docs = nlp.pipe(raw_texts_to_lemma, batch_size=2000, disable=['parser', 'ner'])
                    for orig_idx, doc in zip(missing_lemma_indices, docs):
                        output[orig_idx] = ' '.join([t.lemma_.lower() for t in doc if not t.is_punct])
                except Exception:
                    for orig_idx, raw_t in zip(missing_lemma_indices, raw_texts_to_lemma):
                        output[orig_idx] = raw_t.lower()
            else:
                for orig_idx, raw_t in zip(missing_lemma_indices, raw_texts_to_lemma):
                    output[orig_idx] = raw_t.lower()
        return output

class StylometricExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, granularity: str = 'full'):
        self.granularity = granularity

    def fit(self, X, y=None):
        return self

    def transform(self, X: Any) -> np.ndarray:
        is_sent = ('sentence' in self.granularity or self.granularity == 'single')
        feat_dim = 8 if is_sent else 11

        if isinstance(X, pd.DataFrame):
            records = X.to_dict(orient='records')
        elif isinstance(X, list) and len(X) > 0 and isinstance(X[0], dict):
            records = X
        elif isinstance(X, (list, np.ndarray)):
            if len(X) == 0:
                return np.empty((0, feat_dim), dtype=np.float32)
            records = [{'text': str(t)} for t in X]
        else:
            records = [{'text': str(X)}]

        if not records:
            return np.empty((0, feat_dim), dtype=np.float32)

        features = []
        for item in records:
            if 'stylometrics_vec' in item and item['stylometrics_vec'] is not None:
                features.append(item['stylometrics_vec'])
            else:
                raw_text = str(item.get('text', ''))
                cleaned_text = item.get('normalized_text', normalize_text(raw_text))
                sents = item.get('sentences', None)
                if sents is None:
                    if is_sent:
                        sents = [cleaned_text]
                    else:
                        sents = [s.strip() for s in RE_SENT_SPLIT.split(cleaned_text) if s.strip()] or [cleaned_text]
                feat = extract_stylometric_features(
                    text=cleaned_text,
                    sentences=sents,
                    granularity=self.granularity,
                    raw_text=raw_text
                )
                features.append(feat)
        return np.array(features, dtype=np.float32)

class StylometricScaler(BaseEstimator, TransformerMixin):

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X * self.weight

class TFIDFParamBuilder:
    SEARCH_SPACES = {
        'word_ngram': '$[1, 2]$ to $[1, 3]$ (Full) | $[1, 1]$ to $[1, 2]$ (Sent)',
        'word_max_features': '10,000 to 80,000',
        'word_min_df': '1 to 4',
        'word_sublinear_tf': r'\{\text{True}, \text{False}\}',
        'char_ngram': '$[2, 3]$ to $[3, 5]$',
        'char_max_features': '10,000 to 80,000',
        'char_min_df': '1 to 4',
        'use_stylometrics': r'\{\text{True}, \text{False}\}',
        'sty_weight': '$[0.01, 10.0]$ (Log-uniform)',
        'kernel': r'\text{Linear}',
        'C': '$[10^{-2}, 10^{2}]$ (Log-uniform)',
        'linear_loss': r'\{\text{squared\_hinge}, \text{hinge}\}',
        'class_weight': r'\{\text{balanced}, \text{custom}\}'
    }

    DEFAULT_PRIORS = {
        'word_min_ngram': 1,
        'word_max_ngram': 2,
        'word_max_features': 50000,
        'word_min_df': 2,
        'word_max_df': 0.95,
        'word_sublinear_tf': True,
        'char_min_ngram': 3,
        'char_max_ngram': 5,
        'char_max_features': 50000,
        'char_min_df': 2,
        'char_max_df': 0.95,
        'char_sublinear_tf': True,
        'use_stylometrics': True,
        'sty_weight': 1.0,
        'C': 1.0,
        'linear_loss': 'squared_hinge',
        'weight_mode': 'balanced'
    }

    @staticmethod
    def sample_tfidf(trial: Trial, prefix: str, granularity: str = 'full') -> Dict[str, Any]:
        is_sent = ('sentence' in granularity or granularity == 'single')
        min_n_low = 1 if prefix == 'word' else 2
        min_n_high = 2 if prefix == 'word' else 3
        max_n_limit = 2 if prefix == 'word' and is_sent else 3 if prefix == 'word' else 5
        raw_min_n = trial.suggest_int(f'{prefix}_min_ngram', min_n_low, min_n_high)
        raw_max_n = trial.suggest_int(f'{prefix}_max_ngram', min_n_low, max_n_limit)
        min_n = min(raw_min_n, raw_max_n)
        max_n = max(raw_min_n, raw_max_n)
        max_feat_low, max_feat_high = (10000, 60000) if is_sent else (20000, 80000)
        return {
            'ngram_range': (min_n, max_n),
            'max_features': trial.suggest_int(f'{prefix}_max_features', max_feat_low, max_feat_high, step=10000),
            'min_df': trial.suggest_int(f'{prefix}_min_df', 1, 4),
            'max_df': trial.suggest_float(f'{prefix}_max_df', 0.85, 0.99),
            'norm': 'l2',
            'sublinear_tf': trial.suggest_categorical(f'{prefix}_sublinear_tf', [True, False]),
            'binary': False,
            'analyzer': 'word' if prefix == 'word' else 'char',
            'stop_words': get_dutch_stopwords_lemmatized() if prefix == 'word' else None
        }

    @staticmethod
    def sample_stylometrics(trial: Trial) -> Dict[str, Any]:
        use_sty = trial.suggest_categorical('use_stylometrics', [True, False])
        sty_weight = trial.suggest_float('sty_weight', 0.01, 10.0, log=True) if use_sty else 0.0
        return {'use_stylometrics': use_sty, 'sty_weight': sty_weight}

    @staticmethod
    def sample_model_params(trial: Trial) -> Dict[str, Any]:
        c_val = trial.suggest_float('C', 0.01, 100.0, log=True)
        linear_loss = trial.suggest_categorical('linear_loss', ['squared_hinge', 'hinge'])
        weight_mode = trial.suggest_categorical('weight_mode', ['balanced', 'custom'])
        class_weight = {0: trial.suggest_float('human_class_weight', 1.0, 5.0), 1: 1.0} if weight_mode == 'custom' else 'balanced'
        return {
            'C': c_val,
            'linear_loss': linear_loss,
            'class_weight': class_weight,
            'weight_mode': weight_mode
        }

    @staticmethod
    def parse_best_tfidf(best_params: Dict[str, Any], prefix: str, granularity: str = 'full') -> Dict[str, Any]:
        min_n = int(best_params.get(f'{prefix}_min_ngram', 1 if prefix == 'word' else 3))
        max_n = int(best_params.get(f'{prefix}_max_ngram', 2 if prefix == 'word' else 5))
        if min_n > max_n:
            min_n = max_n
        return {
            'ngram_range': (min_n, max_n),
            'max_features': int(best_params.get(f'{prefix}_max_features', 50000)),
            'min_df': int(best_params.get(f'{prefix}_min_df', 2)),
            'max_df': float(best_params.get(f'{prefix}_max_df', 0.95)),
            'norm': 'l2',
            'sublinear_tf': bool(best_params.get(f'{prefix}_sublinear_tf', True)),
            'binary': False,
            'analyzer': 'word' if prefix == 'word' else 'char',
            'stop_words': get_dutch_stopwords_lemmatized() if prefix == 'word' else None
        }

def build_svm_feature_pipeline(
    word_params: Optional[Dict[str, Any]] = None,
    char_params: Optional[Dict[str, Any]] = None,
    sty_params: Optional[Dict[str, Any]] = None,
    granularity: str = 'full'
) -> Pipeline:
    w_params = word_params or {
        'ngram_range': (1, 2),
        'max_features': 50000,
        'min_df': 2,
        'max_df': 0.95,
        'sublinear_tf': True,
        'norm': 'l2',
        'analyzer': 'word',
        'stop_words': get_dutch_stopwords_lemmatized()
    }
    c_params = char_params or {
        'ngram_range': (3, 5),
        'max_features': 50000,
        'min_df': 2,
        'max_df': 0.95,
        'sublinear_tf': True,
        'norm': 'l2',
        'analyzer': 'char'
    }
    sty_config = sty_params or {'use_stylometrics': True, 'sty_weight': 1.0}

    transformers = [
        ('word_ngrams', Pipeline([
            ('extract', TextExtractor(key='text_lemmatized')),
            ('tfidf', TfidfVectorizer(**w_params))
        ])),
        ('char_ngrams', Pipeline([
            ('extract', TextExtractor(key='normalized_text')),
            ('tfidf', TfidfVectorizer(**c_params))
        ]))
    ]
    if sty_config.get('use_stylometrics', True):
        sty_weight = float(sty_config.get('sty_weight', 1.0))
        transformers.append((
            'stylometrics', Pipeline([
                ('extractor', StylometricExtractor(granularity=granularity)),
                ('scaler', StandardScaler()),
                ('subspace_norm', Normalizer(norm='l2')),
                ('weight', StylometricScaler(weight=sty_weight))
            ])
        ))
    return Pipeline([
        ('union', FeatureUnion(transformers)),
        ('normalizer', Normalizer(norm='l2'))
    ])

class SVMPipelineFactory:

    @staticmethod
    def create_classifier(
        c_val: float = 1.0,
        linear_loss: str = 'squared_hinge',
        class_weight: Any = 'balanced',
        calibrate: bool = True,
        cv_splits: Optional[Any] = None,
        max_iter: int = 5000,
        tol: float = 0.0001
    ):
        dual = linear_loss == 'hinge'
        base_clf = LinearSVC(
            C=c_val,
            loss=linear_loss,
            dual=dual,
            random_state=42,
            class_weight=class_weight,
            max_iter=max_iter,
            tol=tol
        )
        if calibrate:
            cv = cv_splits if cv_splits is not None else 3
            return CalibratedClassifierCV(estimator=base_clf, cv=cv, method='sigmoid')
        return base_clf

class SVMDetector(BaseDetector):

    def __init__(self, pipeline: Optional[Pipeline] = None, scope: str = 'full', seed: int = 42, log_dir: Optional[Union[str, Path]] = None, **kwargs):
        super().__init__(model_name='svm', scope=scope, seed=seed, log_dir=log_dir)
        if pipeline is None:
            feat_pipe = build_svm_feature_pipeline(granularity=scope)
            clf = SVMPipelineFactory.create_classifier(calibrate=True)
            self.pipeline = Pipeline([('features', feat_pipe), ('classifier', clf)])
        else:
            self.pipeline = pipeline

    def _apply_hyperparameters(self, best_params: Dict[str, Any], out_path: Path, cv_splits: Optional[Any] = None):
        from src.visualization.latex_tables import export_hyperparameters_table
        w_params = TFIDFParamBuilder.parse_best_tfidf(best_params, 'word', granularity=self.scope)
        c_params = TFIDFParamBuilder.parse_best_tfidf(best_params, 'char', granularity=self.scope)
        sty_params = {
            'use_stylometrics': bool(best_params.get('use_stylometrics', True)),
            'sty_weight': float(best_params.get('sty_weight', 1.0))
        }
        c_val = float(best_params.get('C', 1.0))
        linear_loss = str(best_params.get('linear_loss', 'squared_hinge'))
        weight_mode = best_params.get('weight_mode', 'balanced')
        class_weight = {0: float(best_params.get('human_class_weight', 1.0)), 1: 1.0} if weight_mode == 'custom' else 'balanced'

        feat_pipe = build_svm_feature_pipeline(w_params, c_params, sty_params, granularity=self.scope)
        clf = SVMPipelineFactory.create_classifier(
            c_val=c_val,
            linear_loss=linear_loss,
            class_weight=class_weight,
            calibrate=True,
            cv_splits=cv_splits
        )
        self.pipeline = Pipeline([('features', feat_pipe), ('classifier', clf)])
        latex_dir = out_path / 'latex_tables'
        latex_dir.mkdir(parents=True, exist_ok=True)
        export_hyperparameters_table(
            best_params=best_params,
            search_spaces=TFIDFParamBuilder.SEARCH_SPACES,
            scope=self.scope,
            output_path=latex_dir / f'table_hyperparams_svm_{self.scope}.tex'
        )

    @classmethod
    def from_config(cls, config: Any, log_dir: Optional[Union[str, Path]] = None, **kwargs) -> 'SVMDetector':
        return cls(scope=config.scope, seed=config.seed, log_dir=log_dir, **kwargs)

    def fit(
        self,
        train_data: Union[pd.DataFrame, List[Dict[str, Any]], List[str]],
        y_train: Optional[np.ndarray] = None,
        dev_data: Optional[pd.DataFrame] = None,
        config: Optional[Any] = None,
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> 'SVMDetector':
        df_train = pd.DataFrame(train_data)
        if 'label' not in df_train.columns and y_train is not None:
            df_train['label'] = y_train

        out_path = Path(output_dir or f'./output/svm_{self.scope}')
        out_path.mkdir(parents=True, exist_ok=True)
        params_file = out_path / 'best_params.json'

        if config is not None:
            tune = config.tuning.enabled
            n_trials = config.tuning.n_trials
            tuning_sample_size = config.tuning.sample_size
            target_fpr = config.target_fpr
        else:
            tune = kwargs.get('tune', False)
            n_trials = kwargs.get('n_trials', 10)
            tuning_sample_size = kwargs.get('tuning_sample_size', 15000)
            target_fpr = kwargs.get('target_fpr', 0.01)

        y = df_train['label'].astype(int).values
        X = df_train.to_dict(orient='records')

        id_col = next((c for c in ['_id', 'doc_id', 'id'] if c in df_train.columns), None)
        if id_col and len(df_train[id_col].unique()) >= 3:
            groups = df_train[id_col].values
            sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=self.seed)
            cv_splits = list(sgkf.split(X, y, groups=groups))
        else:
            cv_splits = 3

        if tune:
            from src.training.tune_svm import SVMOptunaTuner
            best_params, _ = SVMOptunaTuner.run(
                train_df=df_train,
                scope=self.scope,
                n_trials=n_trials,
                tuning_sample_size=tuning_sample_size,
                target_fpr=target_fpr,
                seed=self.seed,
                output_dir=out_path
            )
            self._apply_hyperparameters(best_params, out_path, cv_splits=cv_splits)
        elif params_file.exists():
            self.logger.info(f'Loaded existing best hyperparameters from: {params_file}')
            best_params = json.loads(params_file.read_text(encoding='utf-8'))
            self._apply_hyperparameters(best_params, out_path, cv_splits=cv_splits)
        else:
            self._apply_hyperparameters(TFIDFParamBuilder.DEFAULT_PRIORS, out_path, cv_splits=cv_splits)

        self.logger.info(f'Fitting calibrated SVM pipeline on {len(X)} samples [Scope: {self.scope.upper()}]...')
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, texts: Union[List[str], List[Dict[str, Any]], pd.DataFrame, np.ndarray]) -> np.ndarray:
        if isinstance(texts, pd.DataFrame):
            recs = texts.to_dict(orient='records')
        elif isinstance(texts, np.ndarray):
            recs = texts.tolist()
        else:
            recs = texts

        if len(recs) == 0:
            return np.array([], dtype=np.float32)

        if hasattr(self.pipeline, 'predict_proba'):
            return self.pipeline.predict_proba(recs)[:, 1]
        else:
            scores = self.pipeline.decision_function(recs)
            return 1.0 / (1.0 + np.exp(-scores))

    def extract_feature_importances(self) -> pd.DataFrame:
        if self.pipeline is None:
            return pd.DataFrame()
        try:
            feat_step = self.pipeline.named_steps.get('features')
            if hasattr(feat_step, 'named_steps') and 'union' in feat_step.named_steps:
                feat_union = feat_step.named_steps['union']
            elif isinstance(feat_step, FeatureUnion):
                feat_union = feat_step
            else:
                return pd.DataFrame()

            clf_step = self.pipeline.named_steps['classifier']
            feature_names = []
            for trans_name, trans_pipe in feat_union.transformer_list:
                if trans_name == 'word_ngrams':
                    vec = trans_pipe.named_steps['tfidf']
                    feature_names.extend([f'word:{f}' for f in vec.get_feature_names_out()])
                elif trans_name == 'char_ngrams':
                    vec = trans_pipe.named_steps['tfidf']
                    feature_names.extend([f'char:{f}' for f in vec.get_feature_names_out()])
                elif trans_name == 'stylometrics':
                    sty_names = [
                        'mean_word_len', 'var_word_len', 'ttr', 'hapax',
                        'transitions', 'space_ratio', 'double_space', 'punc_ratio'
                    ]
                    if not ('sentence' in self.scope or self.scope == 'single'):
                        sty_names = ['mean_sent_len', 'var_sent_len', 'burstiness'] + sty_names
                    feature_names.extend([f'stylo:{n}' for n in sty_names])

            if isinstance(clf_step, CalibratedClassifierCV):
                estimators = [getattr(cc, 'estimator', getattr(cc, 'base_estimator', None)) for cc in clf_step.calibrated_classifiers_]
                valid_ests = [est for est in estimators if est is not None and hasattr(est, 'coef_')]
                if not valid_ests:
                    return pd.DataFrame()
                weights = np.mean([est.coef_.flatten() for est in valid_ests], axis=0)
            else:
                weights = clf_step.coef_.flatten()

            min_len = min(len(feature_names), len(weights))
            return pd.DataFrame({
                'feature': feature_names[:min_len],
                'weight': weights[:min_len]
            })
        except Exception as e:
            self.logger.warning(f'Could not extract feature importances: {e}')
            return pd.DataFrame()

    def save(self, path: Union[str, Path]):
        save_p = Path(path)
        if save_p.is_dir() or not str(save_p).endswith('.joblib'):
            save_p = save_p / 'model.joblib'
        save_p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'pipeline': self.pipeline,
            'scope': self.scope,
            'calibrated_threshold': self.calibrated_threshold
        }, save_p)
        self.logger.info(f'Saved SVM pipeline to: {save_p}')

    @classmethod
    def load(cls, path: Union[str, Path], scope: str = 'full', seed: int = 42, log_dir: Optional[Path] = None, **kwargs) -> 'SVMDetector':
        load_p = Path(path)
        if load_p.is_dir():
            load_p = load_p / 'model.joblib'
        if not load_p.exists():
            checkpoint_alt = load_p.parent / 'checkpoint' / 'model.joblib'
            if checkpoint_alt.exists():
                load_p = checkpoint_alt
            else:
                raise FileNotFoundError(f'SVM checkpoint not found at: {load_p}')
        loaded = joblib.load(load_p)
        if isinstance(loaded, dict) and 'pipeline' in loaded:
            pipeline = loaded['pipeline']
            thresh = loaded.get('calibrated_threshold', 0.5)
            loaded_scope = loaded.get('scope', scope)
        else:
            pipeline = loaded
            thresh = 0.5
            loaded_scope = scope
        detector = cls(pipeline=pipeline, scope=loaded_scope, seed=seed, log_dir=log_dir)
        detector.calibrated_threshold = thresh
        return detector