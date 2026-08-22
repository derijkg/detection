# src/training/tune_svm.py
import gc
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import optuna
import pandas as pd
from optuna.trial import Trial
from scipy.sparse import csr_matrix, hstack
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import Normalizer, StandardScaler

from src.data.data_loader import group_stratified_sample
from src.evaluation.metrics import MetricEvaluator
from src.models.svm_pipeline import (
    SVMPipelineFactory,
    StylometricExtractor,
    TextExtractor,
    TFIDFParamBuilder
)
from src.utils.optuna_utils import TqdmOptunaCallback

class MergedSVMObjective:

    def __init__(
        self,
        lemma_texts: List[str],
        raw_texts: List[str],
        stylometrics_matrix: np.ndarray,
        labels: np.ndarray,
        groups: np.ndarray,
        granularity: str = 'full',
        score_metric: str = 'pauc',
        max_fpr: float = 0.01,
        seed: int = 42
    ):
        self.lemma_texts = np.array(lemma_texts, dtype=object)
        self.raw_texts = np.array(raw_texts, dtype=object)
        self.sty_mat = stylometrics_matrix
        self.labels = labels
        self.groups = groups
        self.granularity = granularity
        self.score_metric = score_metric
        self.max_fpr = max_fpr
        self.cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)

    def __call__(self, trial: Trial) -> float:
        word_params = TFIDFParamBuilder.sample_tfidf(trial, 'word', granularity=self.granularity)
        char_params = TFIDFParamBuilder.sample_tfidf(trial, 'char', granularity=self.granularity)
        sty_params = TFIDFParamBuilder.sample_stylometrics(trial)
        model_params = TFIDFParamBuilder.sample_model_params(trial)

        fold_scores = []
        for train_idx, val_idx in self.cv.split(self.lemma_texts, self.labels, groups=self.groups):
            w_vec = TfidfVectorizer(**word_params)
            X_tr_w = w_vec.fit_transform(self.lemma_texts[train_idx])
            X_va_w = w_vec.transform(self.lemma_texts[val_idx])

            c_vec = TfidfVectorizer(**char_params)
            X_tr_c = c_vec.fit_transform(self.raw_texts[train_idx])
            X_va_c = c_vec.transform(self.raw_texts[val_idx])

            blocks_tr = [X_tr_w.tocsr(), X_tr_c.tocsr()]
            blocks_va = [X_va_w.tocsr(), X_va_c.tocsr()]

            if sty_params.get('use_stylometrics', True) and self.sty_mat is not None and (self.sty_mat.shape[1] > 0):
                sty_tr_raw = self.sty_mat[train_idx]
                sty_va_raw = self.sty_mat[val_idx]
                scaler = StandardScaler()
                normer = Normalizer(norm='l2')
                weight = float(sty_params.get('sty_weight', 1.0))
                sty_tr = csr_matrix(normer.fit_transform(scaler.fit_transform(sty_tr_raw)) * weight)
                sty_va = csr_matrix(normer.transform(scaler.transform(sty_va_raw)) * weight)
                blocks_tr.append(sty_tr)
                blocks_va.append(sty_va)

            global_norm = Normalizer(norm='l2')
            X_tr = global_norm.fit_transform(hstack(blocks_tr).tocsr())
            X_va = global_norm.transform(hstack(blocks_va).tocsr())

            clf = SVMPipelineFactory.create_classifier(
                c_val=model_params['C'],
                linear_loss=model_params['linear_loss'],
                class_weight=model_params['class_weight'],
                calibrate=False,
                max_iter=4000
            )

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', ConvergenceWarning)
                clf.fit(X_tr, self.labels[train_idx])

            decision_scores = clf.decision_function(X_va)
            score = MetricEvaluator.compute_metric(
                self.labels[val_idx],
                decision_scores,
                metric_name=self.score_metric,
                max_fpr=self.max_fpr
            )
            fold_scores.append(score)

        return float(np.mean(fold_scores))

class SVMOptunaTuner:

    @classmethod
    def run(
        cls,
        train_df: pd.DataFrame,
        scope: str = 'full',
        n_trials: int = 10,
        tuning_sample_size: int = 15000,
        target_fpr: float = 0.01,
        seed: int = 42,
        output_dir: Optional[Union[str, Path]] = None
    ) -> Tuple[Dict[str, Any], int]:
        df = group_stratified_sample(train_df, sample_size=tuning_sample_size, seed=seed)
        id_col = next((c for c in ['_id', 'doc_id', 'id'] if c in df.columns), None)

        out_path = Path(output_dir or f'./output/svm_{scope}')
        out_path.mkdir(parents=True, exist_ok=True)
        params_file = out_path / 'best_params.json'
        db_path = out_path / 'optuna_study.db'
        storage_url = f'sqlite:///{db_path.resolve()}'
        actual_sz = len(df)

        print(f'\n==================================================================')
        print(f'   RUNNING LINEAR SVM OPTUNA TUNING [{scope.upper()}] ({n_trials} Trials)   ')
        print(f'   Database Storage : {storage_url}')
        print(f'   Live Params File : {params_file}')
        print(f'   Search Budget    : {actual_sz:,} Subsampled Training Rows')
        print(f'==================================================================')

        print(' -> Pre-computing text normalizations and lemmatizations...')
        lemma_texts = TextExtractor(key='text_lemmatized').transform(df)
        raw_texts = TextExtractor(key='normalized_text').transform(df)

        print(' -> Pre-computing stylometric base matrices...')
        sty_mat = StylometricExtractor(granularity=scope).transform(df)
        labels = df['label'].astype(int).values
        groups = df[id_col].values if id_col and id_col in df.columns else np.arange(len(df))

        objective = MergedSVMObjective(
            lemma_texts=lemma_texts,
            raw_texts=raw_texts,
            stylometrics_matrix=sty_mat,
            labels=labels,
            groups=groups,
            granularity=scope,
            score_metric='pauc',
            max_fpr=target_fpr,
            seed=seed
        )

        study = optuna.create_study(
            study_name=f'svm_{scope}',
            storage=storage_url,
            load_if_exists=True,
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=seed)
        )

        initial_params = TFIDFParamBuilder.DEFAULT_PRIORS.copy()
        if 'sentence' in scope or scope == 'single':
            initial_params['word_max_ngram'] = 2
            initial_params['char_max_features'] = 40000

        if params_file.exists():
            try:
                saved = json.loads(params_file.read_text(encoding='utf-8'))
                clean_saved = {k: v for k, v in saved.items() if not k.startswith('_')}
                if clean_saved:
                    initial_params.update(clean_saved)
            except Exception:
                pass

        try:
            study.enqueue_trial(initial_params, skip_if_exists=True)
        except TypeError:
            study.enqueue_trial(initial_params)

        with TqdmOptunaCallback(n_trials=n_trials, desc=f'Tuning SVM [{scope.upper()}]', save_path=params_file) as opt_cb:
            study.optimize(objective, n_trials=n_trials, callbacks=[opt_cb])

        try:
            best_clean_params = {k: v for k, v in study.best_params.items() if not k.startswith('_')}
            best_val = study.best_value
        except Exception:
            best_clean_params = TFIDFParamBuilder.DEFAULT_PRIORS.copy()
            best_val = 0.5

        print(f'\n-> Best Linear SVM Parameters ({scope.upper()}): {best_clean_params} | Best pAUC={best_val:.4f}')
        return best_clean_params, actual_sz