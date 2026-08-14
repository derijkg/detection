# models/svm.py
import sys
import os
import optuna
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from typing import Dict, Any, Optional, List

from optuna.study import Study
from optuna.trial import Trial, FrozenTrial, TrialState

from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, f1_score, roc_auc_score, precision_score, fbeta_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold

from src.detection.svm.features import (
    get_feature_extraction_pipeline,
    get_cached_split_features,
    clear_optuna_cache,
    get_dutch_stopwords_lemmatized
)

optuna.logging.set_verbosity(optuna.logging.INFO)


# ==========================================
# 1. Classifier & Parameter Builders
# ==========================================
class ClassifierFactory:
    """Instantiates SVM classifiers with optimized convergence settings."""

    @staticmethod
    def create(kernel: str, c_val: float, gamma: str = 'scale', calibrate: bool = False):
        if kernel == 'linear':
            base_clf = LinearSVC(
                C=c_val,
                random_state=42,
                class_weight='balanced',
                dual=False,
                max_iter=5000
            )
        else:
            base_clf = SVC(
                C=c_val,
                kernel=kernel,
                gamma=gamma,
                random_state=42,
                class_weight='balanced',
                cache_size=500
            )

        if calibrate:
            return CalibratedClassifierCV(estimator=base_clf, cv=3, method='sigmoid')
        return base_clf


class TFIDFParamBuilder:
    """Constructs complete dictionary parameter blocks for Word and Char TF-IDF Extractors."""

    @staticmethod
    def sample_from_trial(trial: Trial, prefix: str) -> Dict[str, Any]:
        min_ngram = trial.suggest_int(f'{prefix}_min_ngram', 1, 2 if prefix == 'word' else 3)
        max_ngram_limit = 3 if prefix == 'word' else 5
        max_ngram = trial.suggest_int(f'{prefix}_max_ngram', max(min_ngram, 2 if prefix == 'word' else 3), max_ngram_limit)

        max_features = trial.suggest_int(f'{prefix}_max_features', 20000, 120000, step=10000)
        min_df = trial.suggest_int(f'{prefix}_min_df', 1, 5)

        params = {
            'ngram_range': (min_ngram, max_ngram),
            'max_features': max_features,
            'min_df': min_df,
            'max_df': 0.95,
            'sublinear_tf': True
        }

        if prefix == 'word':
            params['analyzer'] = 'word'
            params['stop_words'] = get_dutch_stopwords_lemmatized()
        else:
            params['analyzer'] = 'char'

        return params

    @staticmethod
    def from_best_params(best_params: Dict[str, Any], prefix: str) -> Dict[str, Any]:
        """Reconstructs full TF-IDF params dictionary from Optuna's best_params dictionary."""
        params = {}

        if f'{prefix}_min_ngram' in best_params and f'{prefix}_max_ngram' in best_params:
            params['ngram_range'] = (
                int(best_params[f'{prefix}_min_ngram']),
                int(best_params[f'{prefix}_max_ngram'])
            )

        key_mapping = {
            'max_features': 'max_features',
            'min_df': 'min_df',
            'max_df': 'max_df',
            'sublinear_tf': 'sublinear_tf',
            'analyzer': 'analyzer',
            'norm': 'norm',
            'smooth_idf': 'smooth_idf',
            'use_idf': 'use_idf'
        }

        for param_name, tfidf_arg in key_mapping.items():
            optuna_key = f'{prefix}_{param_name}'
            if optuna_key in best_params:
                params[tfidf_arg] = best_params[optuna_key]

        params.setdefault('sublinear_tf', True)
        params.setdefault('max_df', 0.95)

        if prefix == 'word':
            params.setdefault('analyzer', 'word')
            params.setdefault('stop_words', get_dutch_stopwords_lemmatized())
        else:
            params.setdefault('analyzer', 'char')

        return params


# ==========================================
# 2. Metric & Threshold Evaluators
# ==========================================
class ScoreEvaluator:
    """Calculates classification evaluation metrics cleanly."""

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_score: Optional[np.ndarray], metric_name: str) -> float:
        metric = metric_name.lower().replace('-', '_')

        if metric in ['roc_auc', 'rocauc']:
            if y_score is None:
                return 0.0
            return float(roc_auc_score(y_true, y_score))
        elif metric == 'precision':
            return float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
        elif metric in ['f0.5', 'f0_5']:
            return float(fbeta_score(y_true, y_pred, beta=0.5, pos_label=1, zero_division=0))
        elif metric == 'f1':
            return float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
        elif metric == 'set_fp':
            if y_score is None:
                return 0.0
            fpr, tpr, _ = roc_curve(y_true, y_score)
            valid_indices = np.where(fpr <= 0.01)[0]
            return float(tpr[valid_indices[-1]]) if len(valid_indices) > 0 else 0.0
        else:
            return float(f1_score(y_true, y_pred, average='macro'))

    @staticmethod
    def find_threshold_for_max_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.01) -> float:
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        valid_indices = np.where(fpr <= target_fpr)[0]
        if len(valid_indices) > 0:
            return float(thresholds[valid_indices[-1]])
        return 0.5


# ==========================================
# 3. Optuna Objectives
# ==========================================
class Stage1Objective:
    """Optuna objective function for TF-IDF feature extraction tuning."""

    def __init__(self, X_raw: List[Dict], y: np.ndarray, cv: StratifiedKFold, kernel_choice: str, metric_name: str):
        self.X_raw = X_raw
        self.y = y
        self.cv = cv
        self.kernel_choice = kernel_choice
        self.metric_name = metric_name

    def __call__(self, trial: Trial) -> float:
        word_params = TFIDFParamBuilder.sample_from_trial(trial, 'word')
        char_params = TFIDFParamBuilder.sample_from_trial(trial, 'char')

        fold_scores = []
        eval_kernel = self.kernel_choice if self.kernel_choice != 'all' else 'linear'

        for fold, (train_idx, val_idx) in enumerate(self.cv.split(self.X_raw, self.y)):
            X_tr_raw = [self.X_raw[i] for i in train_idx]
            X_va_raw = [self.X_raw[i] for i in val_idx]
            y_tr, y_va = self.y[train_idx], self.y[val_idx]

            X_tr, X_va = get_cached_split_features(X_tr_raw, X_va_raw, word_params, char_params, use_pre_lemmatized=True)

            clf = ClassifierFactory.create(kernel=eval_kernel, c_val=1.0, calibrate=False)
            clf.fit(X_tr, y_tr)

            preds = clf.predict(X_va)
            decision_scores = clf.decision_function(X_va)

            score = ScoreEvaluator.evaluate(y_va, preds, decision_scores, self.metric_name)
            fold_scores.append(score)

            intermediate_mean = float(np.mean(fold_scores))
            trial.report(intermediate_mean, step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))


class Stage2Objective:
    """Optuna objective function for SVM hyperparameter tuning."""

    def __init__(self, X_raw: List[Dict], y: np.ndarray, cv: StratifiedKFold,
                 kernel_choice: str, metric_name: str, tuning_strategy: str,
                 best_tfidf_params: Optional[Dict] = None):
        self.X_raw = X_raw
        self.y = y
        self.cv = cv
        self.kernel_choice = kernel_choice
        self.metric_name = metric_name
        self.tuning_strategy = tuning_strategy
        self.best_tfidf_params = best_tfidf_params or {}

    def __call__(self, trial: Trial) -> float:
        if self.tuning_strategy == '2stage':
            word_params = TFIDFParamBuilder.from_best_params(self.best_tfidf_params, 'word')
            char_params = TFIDFParamBuilder.from_best_params(self.best_tfidf_params, 'char')
        elif self.tuning_strategy == 'model':
            word_params, char_params = None, None
        else:  # 'merged'
            word_params = TFIDFParamBuilder.sample_from_trial(trial, 'word')
            char_params = TFIDFParamBuilder.sample_from_trial(trial, 'char')

        c_val = trial.suggest_float('C', 1e-2, 1e2, log=True)
        kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'sigmoid']) if self.kernel_choice == 'all' else self.kernel_choice
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto']) if kernel != 'linear' else 'scale'

        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(self.X_raw, self.y)):
            X_tr_raw = [self.X_raw[i] for i in train_idx]
            X_va_raw = [self.X_raw[i] for i in val_idx]
            y_tr, y_va = self.y[train_idx], self.y[val_idx]

            X_tr, X_va = get_cached_split_features(X_tr_raw, X_va_raw, word_params, char_params, use_pre_lemmatized=True)

            clf = ClassifierFactory.create(kernel=kernel, c_val=c_val, gamma=gamma, calibrate=False)
            clf.fit(X_tr, y_tr)

            preds = clf.predict(X_va)
            decision_scores = clf.decision_function(X_va)

            score = ScoreEvaluator.evaluate(y_va, preds, decision_scores, self.metric_name)
            fold_scores.append(score)

            intermediate_mean = float(np.mean(fold_scores))
            trial.report(intermediate_mean, step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))


# ==========================================
# 4. Optuna Tuning Orchestrator
# ==========================================
class OptunaTuner:
    """Manages study creation, execution, and parameter selection across tuning stages."""

    @staticmethod
    def print_best_trial_callback(study: Study, trial: FrozenTrial):
        if trial.state == TrialState.COMPLETE:
            best = study.best_trial
            print(f"-> [Optuna Progress] Best Trial {best.number} | Score ({study.direction.name}): {best.value:.4f}")

    @classmethod
    def get_or_create_study(cls, study_name: str, storage: str, reset: bool = False) -> Study:
        if reset:
            try:
                optuna.delete_study(study_name=study_name, storage=storage)
                print(f"-> Cleared existing Optuna study: '{study_name}'")
            except Exception:
                pass

        pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)
        return optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction='maximize',
            pruner=pruner,
            load_if_exists=True
        )

    @classmethod
    def run(cls, train_df: pd.DataFrame, granularity: str, kernel_choice: str = 'linear',
            tuning_strategy: str = '2stage', tuning_sample_size: int = 3000,
            trials: int = 15, trials_stage1: int = 10, trials_stage2: int = 10,
            reset_study: bool = False, score_metric: str = 'roc_auc',
            study_name: Optional[str] = None, n_jobs_optuna: int = 1) -> Dict[str, Any]:

        db_path = "sqlite:///optuna_svm.db?timeout=60"
        clean_metric = score_metric.replace('-', '_').replace('.', '')
        if study_name is None:
            study_name = f"svm_{kernel_choice}_{granularity}_{clean_metric}_{tuning_strategy}"

        sample_size = max(1, int(len(train_df) * tuning_sample_size)) if isinstance(tuning_sample_size, float) else min(tuning_sample_size, len(train_df))
        stratify_train = train_df['label'] if ('label' in train_df.columns and train_df['label'].value_counts().min() > 1) else None

        if len(train_df) > sample_size:
            print(f"Subsampling training set from {len(train_df)} down to {sample_size} for CV tuning...")
            train_sub, _ = train_test_split(train_df, train_size=sample_size, random_state=42, stratify=stratify_train)
        else:
            train_sub = train_df

        X_raw_all = train_sub[['text', 'sentences', 'text_lemmatized']].to_dict(orient='records')
        y_all = train_sub['label'].values
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        best_tfidf_params = {}

        # Stage 1: Feature Extraction Tuning
        if tuning_strategy == '2stage':
            print(f"\n>>> [Stage 1] Tuning Preprocessing and TF-IDF via 3-Fold CV ({trials_stage1} trials)...")
            study_s1 = cls.get_or_create_study(f"{study_name}_stage1", db_path, reset=reset_study)
            objective_s1 = Stage1Objective(X_raw_all, y_all, cv, kernel_choice, score_metric)
            study_s1.optimize(objective_s1, n_trials=trials_stage1, n_jobs=max(1, n_jobs_optuna), callbacks=[cls.print_best_trial_callback])
            best_tfidf_params = study_s1.best_params
            print(f"-> Best Preprocessing parameters found: {best_tfidf_params}")

        # Stage 2: SVM Parameter Tuning
        stage2_trials = trials_stage2 if tuning_strategy in ['2stage', 'model'] else trials
        print(f"\n>>> [Stage 2] Tuning SVM Parameters via 3-Fold CV ({stage2_trials} trials)...")

        study_s2 = cls.get_or_create_study(study_name, db_path, reset=reset_study)
        objective_s2 = Stage2Objective(X_raw_all, y_all, cv, kernel_choice, score_metric, tuning_strategy, best_tfidf_params)
        study_s2.optimize(objective_s2, n_trials=stage2_trials, n_jobs=max(1, n_jobs_optuna), callbacks=[cls.print_best_trial_callback])

        completed_trials = [t for t in study_s2.trials if t.state == TrialState.COMPLETE]
        if completed_trials:
            best_value = study_s2.best_value
            top_trials = [t for t in completed_trials if abs(t.value - best_value) < 5e-3]
            best_trial = min(top_trials, key=lambda t: t.params.get('C', float('inf')))
            best_s2_params = best_trial.params
            print(f"\n[Tie-Breaker Applied] Best trial chosen (lowest C within tolerance of {best_value:.4f}): {best_s2_params}")
        else:
            best_s2_params = study_s2.best_params

        best_overall = {}
        if tuning_strategy == '2stage':
            best_overall.update(best_tfidf_params)
        best_overall.update(best_s2_params)
        if kernel_choice != 'all':
            best_overall['kernel'] = kernel_choice

        return best_overall


# ==========================================
# 5. Logging Utility
# ==========================================
class ExperimentLogger:
    """Logs metadata and experimental performance into a persistent registry CSV."""

    @staticmethod
    def log(record: Dict[str, Any], registry_path: str = "experiment_results.csv"):
        df_new = pd.DataFrame([record])
        if os.path.exists(registry_path):
            try:
                df_existing = pd.read_csv(registry_path)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            except Exception:
                df_combined = df_new
        else:
            df_combined = df_new

        df_combined.to_csv(registry_path, index=False)
        print(f"-> Experiment metrics registered successfully in '{registry_path}'")


# ==========================================
# 6. Main Pipeline Runner
# ==========================================
def compute_oof_scores(X_train_raw: List[Dict], y_train: np.ndarray, word_params: Optional[Dict],
                       char_params: Optional[Dict], c_val: float, kernel: str, gamma: str,
                       calibrate: bool = True, n_splits: int = 3) -> np.ndarray:
    """Computes unbiased Out-of-Fold (OOF) scores across the training set for threshold calibration."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_scores = np.zeros(len(y_train))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_raw, y_train)):
        X_tr_raw = [X_train_raw[i] for i in train_idx]
        X_va_raw = [X_train_raw[i] for i in val_idx]
        y_tr_fold = y_train[train_idx]

        X_tr, X_va = get_cached_split_features(X_tr_raw, X_va_raw, word_params, char_params, use_pre_lemmatized=True)
        clf = ClassifierFactory.create(kernel=kernel, c_val=c_val, gamma=gamma, calibrate=calibrate)
        clf.fit(X_tr, y_tr_fold)

        if calibrate or not hasattr(clf, 'decision_function'):
            oof_scores[val_idx] = clf.predict_proba(X_va)[:, 1]
        else:
            oof_scores[val_idx] = clf.decision_function(X_va)

    return oof_scores


def train_svm(train_df: pd.DataFrame,
              test_df: pd.DataFrame,
              c_val: float,
              kernel: str,
              save_path: str,
              granularity: str,
              val_df: Optional[pd.DataFrame] = None,
              run_optuna: bool = False,
              reset_study: bool = False,
              trials: int = 15,
              trials_stage1: int = 10,
              trials_stage2: int = 10,
              tuning_strategy: str = '2stage',
              tuning_sample_size: int = 3000,
              score_metric: str = 'roc_auc',
              study_name: Optional[str] = None,
              n_jobs_optuna: int = 1):

    word_params, char_params = None, None
    gamma = 'scale'
    best_params = {}

    if run_optuna:
        print("Running Hyperparameter Optimization via Optuna...")
        best_params = OptunaTuner.run(
            train_df=train_df,
            granularity=granularity,
            kernel_choice=kernel,
            tuning_strategy=tuning_strategy,
            tuning_sample_size=tuning_sample_size,
            trials=trials,
            trials_stage1=trials_stage1,
            trials_stage2=trials_stage2,
            reset_study=reset_study,
            score_metric=score_metric,
            study_name=study_name,
            n_jobs_optuna=n_jobs_optuna
        )
        c_val = best_params.get('C', c_val)
        kernel = best_params.get('kernel', kernel)
        gamma = best_params.get('gamma', 'scale')

        if any(k.startswith('word_') for k in best_params):
            word_params = TFIDFParamBuilder.from_best_params(best_params, 'word')
        if any(k.startswith('char_') for k in best_params):
            char_params = TFIDFParamBuilder.from_best_params(best_params, 'char')

    X_train_raw = train_df[['text', 'sentences', 'text_lemmatized']].to_dict(orient='records')
    X_test_raw = test_df[['text', 'sentences', 'text_lemmatized']].to_dict(orient='records')
    y_train = train_df['label'].values
    y_test = test_df['label'].values

    feature_pipeline = get_feature_extraction_pipeline(word_params, char_params, stylometrics_n_jobs=1, use_pre_lemmatized=True)
    calibrated_clf = ClassifierFactory.create(kernel=kernel, c_val=c_val, gamma=gamma, calibrate=True)

    full_pipeline = Pipeline([
        ('features', feature_pipeline),
        ('classifier', calibrated_clf)
    ])

    optimal_threshold = 0.5
    if score_metric == 'set_fp':
        print("\nCalculating Out-of-Fold (OOF) probability scores across training set for threshold calibration...")
        oof_scores = compute_oof_scores(X_train_raw, y_train, word_params, char_params, c_val, kernel, gamma, calibrate=True)
        optimal_threshold = ScoreEvaluator.find_threshold_for_max_fpr(y_train, oof_scores, target_fpr=0.01)
        print(f"-> Calibrated Threshold (OOF 1% Max FPR Probability): {optimal_threshold:.6f}")

    full_pipeline.optimal_threshold = optimal_threshold

    print(f"Training final probability-calibrated SVM pipeline on 100% of training data...")
    full_pipeline.fit(X_train_raw, y_train)

    # Evaluate on Test Set
    test_scores = full_pipeline.predict_proba(X_test_raw)[:, 1]
    preds = (test_scores >= optimal_threshold).astype(int) if score_metric == 'set_fp' else full_pipeline.predict(X_test_raw)

    print("\n" + "=" * 50)
    print("      OVERALL TEST PERFORMANCE EVALUATION      ")
    print("=" * 50)
    print(classification_report(y_test, preds, digits=4))

    overall_auc = 0.0
    try:
        overall_auc = float(roc_auc_score(y_test, test_scores))
        print(f"Overall Test ROC-AUC Score: {overall_auc:.4f}\n")
    except Exception as e:
        print(f"Could not calculate ROC-AUC: {e}")

    # Performance split diagnosis on Full Abstracts vs Sentences
    full_auc, sent_auc = None, None
    full_f1, sent_f1 = None, None

    if 'task_type' in test_df.columns:
        full_mask = (test_df['task_type'] == 'full').values
        sent_mask = (test_df['task_type'] == 'sentence').values

        if full_mask.sum() > 0:
            print("\n" + "-" * 50)
            print("  DIAGNOSIS: FULL ABSTRACTS ONLY PERFORMANCE  ")
            print("-" * 50)
            print(classification_report(y_test[full_mask], preds[full_mask], digits=4))
            full_f1 = float(f1_score(y_test[full_mask], preds[full_mask], pos_label=1, zero_division=0))
            if len(np.unique(y_test[full_mask])) > 1:
                full_auc = float(roc_auc_score(y_test[full_mask], test_scores[full_mask]))
                print(f"Full Abstracts ROC-AUC: {full_auc:.4f}")

        if sent_mask.sum() > 0:
            print("\n" + "-" * 50)
            print("  DIAGNOSIS: SENTENCES ONLY PERFORMANCE      ")
            print("-" * 50)
            print(classification_report(y_test[sent_mask], preds[sent_mask], digits=4))
            sent_f1 = float(f1_score(y_test[sent_mask], preds[sent_mask], pos_label=1, zero_division=0))
            if len(np.unique(y_test[sent_mask])) > 1:
                sent_auc = float(roc_auc_score(y_test[sent_mask], test_scores[sent_mask]))
                print(f"Sentences ROC-AUC: {sent_auc:.4f}\n")

    # Record Metadata & Metrics
    record = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'study_name': study_name or f"svm_{granularity}",
        'save_path': save_path,
        'granularity': granularity,
        'tuning_strategy': tuning_strategy,
        'kernel': kernel,
        'score_metric': score_metric,
        'tuning_sample_size': tuning_sample_size,
        'calibrated_threshold': optimal_threshold,

        # Best Hyperparameters
        'C': c_val,
        'word_ngram': f"({best_params.get('word_min_ngram')},{best_params.get('word_max_ngram')})" if 'word_min_ngram' in best_params else None,
        'word_max_features': best_params.get('word_max_features', None),
        'word_min_df': best_params.get('word_min_df', None),
        'char_ngram': f"({best_params.get('char_min_ngram')},{best_params.get('char_max_ngram')})" if 'char_min_ngram' in best_params else None,
        'char_max_features': best_params.get('char_max_features', None),
        'char_min_df': best_params.get('char_min_df', None),

        # Test Metrics
        'overall_f1_ai': float(f1_score(y_test, preds, pos_label=1, zero_division=0)),
        'overall_precision_ai': float(precision_score(y_test, preds, pos_label=1, zero_division=0)),
        'overall_roc_auc': overall_auc,
        'full_abstract_f1_ai': full_f1,
        'full_abstract_roc_auc': full_auc,
        'sentence_f1_ai': sent_f1,
        'sentence_roc_auc': sent_auc
    }

    ExperimentLogger.log(record)

    joblib.dump(full_pipeline, save_path)
    print(f"Deployable probability-calibrated pipeline saved successfully to {save_path}")

    clear_optuna_cache()