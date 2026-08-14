# models/svm.py
import sys
import os
import optuna
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, f1_score, roc_auc_score, precision_score, fbeta_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold

from src.detection.svm.features import get_feature_extraction_pipeline, generate_df_hash, get_cached_split_features, clear_optuna_cache

optuna.logging.set_verbosity(optuna.logging.INFO)


# #explained Optuna callback function to report the overall best trial and score live as each trial finishes.
def print_best_trial_callback(study, trial):
    if trial.state == optuna.trial.TrialState.COMPLETE:
        best = study.best_trial
        print(f"-> [Optuna Progress] Current Best: Trial {best.number} | Value ({study.direction.name}): {best.value:.4f}")


def get_classifier(kernel, c_val, gamma='scale', calibrate=False):
    """Instantiates appropriate classifier (LinearSVC for linear, SVC for non-linear)."""
    if kernel == 'linear':
        base_clf = LinearSVC(C=c_val, random_state=42, class_weight='balanced', dual='auto', max_iter=5000)
    else:
        base_clf = SVC(C=c_val, kernel=kernel, gamma=gamma, random_state=42, class_weight='balanced', cache_size=500)

    if calibrate:
        return CalibratedClassifierCV(estimator=base_clf, cv=3, method='sigmoid')
    return base_clf


def safe_create_or_reset_study(study_name, storage, direction='maximize', reset_study=False, pruner=None):
    """Safely handles study creation or resetting in SQLite storage with pruning support."""
    if reset_study:
        try:
            optuna.delete_study(study_name=study_name, storage=storage)
            print(f"-> Cleared existing Optuna study: '{study_name}'")
        except Exception:
            pass

    if pruner is None:
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)

    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        pruner=pruner,
        load_if_exists=True
    )


def find_threshold_for_max_fpr(y_true, y_score, target_fpr=0.01):
    """Finds decision threshold corresponding to target maximum FPR."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    valid_indices = np.where(fpr <= target_fpr)[0]
    if len(valid_indices) > 0:
        best_index = valid_indices[-1]
        return thresholds[best_index]
    return 0.5


def evaluate_metric(y_true, y_pred, y_score, metric_name):
    """Calculates target metric score based on argparse configuration."""
    if metric_name == 'precision':
        return precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    elif metric_name == 'f0.5':
        return fbeta_score(y_true, y_pred, beta=0.5, pos_label=1, zero_division=0)
    elif metric_name == 'f1':
        return f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    elif metric_name in ['roc-auc', 'roc_auc']:
        return roc_auc_score(y_true, y_score)
    elif metric_name == 'set_fp':
        if y_score is None:
            return 0.0
        fpr, tpr, _ = roc_curve(y_true, y_score)
        valid_indices = np.where(fpr <= 0.01)[0]
        return tpr[valid_indices[-1]] if len(valid_indices) > 0 else 0.0
    else:
        return f1_score(y_true, y_pred, average='macro')


def compute_oof_scores(X_train_raw, y_train, word_params, char_params, c_val, kernel, gamma, calibrate=True, n_splits=3):
    """Computes unbiased Out-of-Fold (OOF) scores across training set."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_scores = np.zeros(len(y_train))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_raw, y_train)):
        X_tr_raw = [X_train_raw[i] for i in train_idx]
        X_va_raw = [X_train_raw[i] for i in val_idx]
        y_tr_fold = y_train[train_idx]

        X_tr, X_va = get_cached_split_features(X_tr_raw, X_va_raw, word_params, char_params, use_pre_lemmatized=True)
        clf = get_classifier(kernel=kernel, c_val=c_val, gamma=gamma, calibrate=calibrate)
        clf.fit(X_tr, y_tr_fold)

        if calibrate or not hasattr(clf, 'decision_function'):
            oof_scores[val_idx] = clf.predict_proba(X_va)[:, 1]
        else:
            oof_scores[val_idx] = clf.decision_function(X_va)

    return oof_scores


def optimize_svm_with_optuna(
        train_df,
        granularity, kernel_choice='rbf',
        tuning_strategy='2stage',
        tuning_sample_size=3000,
        trials=15,
        trials_stage1=10,
        trials_stage2=10,
        reset_study=False,
        score_metric='f1',
        study_name=None,
        n_jobs_optuna=1
):
    """Finds best SVM and TF-IDF parameters using 3-Fold Stratified CV with Trial Pruning."""
    print(f"\n--- Optuna Optimization Initialized (3-Fold Stratified CV with Pruning) ---")

    if n_jobs_optuna <= 0:
        n_jobs_optuna = 1

    db_path = "sqlite:///optuna_svm.db?timeout=60"

    if study_name is None:
        clean_metric = score_metric.replace('-', '_').replace('.', '')
        study_name = f"svm_{kernel_choice}_{granularity}_{clean_metric}_{tuning_strategy}"

    study_s1_name = f"{study_name}_stage1"

    if isinstance(tuning_sample_size, float):
        sample_size = max(1, int(len(train_df) * tuning_sample_size))
    else:
        sample_size = min(tuning_sample_size, len(train_df))

    stratify_train = train_df['label'] if ('label' in train_df.columns and train_df['label'].value_counts().min() > 1) else None
    if len(train_df) > sample_size:
        print(f"Subsampling training set from {len(train_df)} down to {sample_size} for 3-fold tuning...")
        train_sub, _ = train_test_split(
            train_df,
            train_size=sample_size,
            random_state=42,
            stratify=stratify_train
        )
    else:
        train_sub = train_df

    X_train_raw_all = train_sub[['text', 'sentences', 'text_lemmatized']].to_dict(orient='records')
    y_train_all = train_sub['label'].values

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    best_tfidf_params = {}

    # ==========================================
    # STAGE 1: Preprocessing & TF-IDF Parameter Optimization
    # ==========================================
    if tuning_strategy == '2stage':
        print(f"\n>>> [Stage 1] Tuning Preprocessing and TF-IDF Parameters via 3-Fold CV ({trials_stage1} trials)...")

        def objective_stage1(trial):
            word_min = trial.suggest_int('word_min_ngram', 1, 2)
            word_max = trial.suggest_int('word_max_ngram', max(word_min, 2), 3)
            word_ngram = (word_min, word_max)
            word_max_feat = trial.suggest_int('word_max_features', 20000, 100000, step=10000)
            word_min_df = trial.suggest_int('word_min_df', 1, 5)

            char_min = trial.suggest_int('char_min_ngram', 1, 3)
            char_max = trial.suggest_int('char_max_ngram', max(char_min, 3), 5)
            char_ngram = (char_min, char_max)
            char_max_feat = trial.suggest_int('char_max_features', 20000, 100000, step=10000)
            char_min_df = trial.suggest_int('char_min_df', 1, 5)

            word_params = {
                'ngram_range': word_ngram,
                'max_features': word_max_feat,
                'min_df': word_min_df,
                'max_df': 0.95,
                'sublinear_tf': True
            }
            char_params = {
                'ngram_range': char_ngram,
                'max_features': char_max_feat,
                'min_df': char_min_df,
                'max_df': 0.95,
                'sublinear_tf': True
            }

            print(f"\n[Trial {trial.number}] Stage 1 - Evaluating Extraction Parameters (3 Folds):")
            print(f"  -> Word NGrams: {word_ngram} | Max Feat: {word_max_feat} | Min DF: {word_min_df}")
            print(f"  -> Char NGrams: {char_ngram} | Max Feat: {char_max_feat} | Min DF: {char_min_df}")

            fold_scores = []
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_raw_all, y_train_all)):
                X_tr_raw = [X_train_raw_all[i] for i in train_idx]
                X_va_raw = [X_train_raw_all[i] for i in val_idx]
                y_tr_fold = y_train_all[train_idx]
                y_va_fold = y_train_all[val_idx]

                X_tr, X_va = get_cached_split_features(X_tr_raw, X_va_raw, word_params, char_params, use_pre_lemmatized=True)

                eval_kernel = kernel_choice if kernel_choice != 'all' else 'linear'
                clf = get_classifier(kernel=eval_kernel, c_val=1.0, calibrate=False)
                clf.fit(X_tr, y_tr_fold)
                preds = clf.predict(X_va)
                decision_scores = clf.decision_function(X_va)

                fold_score = evaluate_metric(y_va_fold, preds, decision_scores, score_metric)
                fold_scores.append(fold_score)

                intermediate_mean = float(np.mean(fold_scores))
                trial.report(intermediate_mean, step=fold)
                if trial.should_prune():
                    print(f"-> [Trial {trial.number}] Stage 1 PRUNED at Fold {fold + 1} (Score: {intermediate_mean:.4f})")
                    raise optuna.TrialPruned()

            mean_score = float(np.mean(fold_scores))
            print(f"-> [Trial {trial.number}] Stage 1 Mean 3-Fold Score: {mean_score:.4f}")
            return mean_score

        study_s1 = safe_create_or_reset_study(study_s1_name, db_path, 'maximize', reset_study)
        # #explained Added print_best_trial_callback to report the overall best trial live upon trial completion.
        study_s1.optimize(objective_stage1, n_trials=trials_stage1, n_jobs=n_jobs_optuna, callbacks=[print_best_trial_callback])
        best_tfidf_params = study_s1.best_params
        print(f"-> Best Preprocessing parameters found: {best_tfidf_params}")

    # ==========================================
    # STAGE 2: SVM Classifier Parameter Optimization
    # ==========================================
    stage2_trials = trials_stage2 if tuning_strategy in ['2stage', 'model'] else trials
    print(f"\n>>> [Stage 2] Tuning SVM Parameters via 3-Fold CV ({stage2_trials} trials)...")

    def objective_stage2(trial):
        if tuning_strategy == '2stage':
            word_params = {
                'ngram_range': (best_tfidf_params['word_min_ngram'], best_tfidf_params['word_max_ngram']),
                'max_features': best_tfidf_params['word_max_features'],
                'min_df': best_tfidf_params.get('word_min_df', 1),
                'max_df': 0.95,
                'sublinear_tf': True
            }
            char_params = {
                'ngram_range': (best_tfidf_params['char_min_ngram'], best_tfidf_params['char_max_ngram']),
                'max_features': best_tfidf_params['char_max_features'],
                'min_df': best_tfidf_params.get('char_min_df', 1),
                'max_df': 0.95,
                'sublinear_tf': True
            }
        elif tuning_strategy == 'model':
            word_params = None
            char_params = None
        else:  # 'merged'
            word_min = trial.suggest_int('word_min_ngram', 1, 2)
            word_max = trial.suggest_int('word_max_ngram', max(word_min, 2), 3)
            word_params = {
                'ngram_range': (word_min, word_max),
                'max_features': trial.suggest_int('word_max_features', 10000, 100000, step=10000),
                'min_df': trial.suggest_int('word_min_df', 1, 5),
                'max_df': 0.95,
                'sublinear_tf': True
            }
            char_min = trial.suggest_int('char_min_ngram', 1, 3)
            char_max = trial.suggest_int('char_max_ngram', max(char_min, 3), 5)
            char_params = {
                'ngram_range': (char_min, char_max),
                'max_features': trial.suggest_int('char_max_features', 10000, 100000, step=10000),
                'min_df': trial.suggest_int('char_min_df', 1, 5),
                'max_df': 0.95,
                'sublinear_tf': True
            }

        c_val = trial.suggest_float('C', 1e-2, 1e2, log=True)
        kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'sigmoid']) if kernel_choice == 'all' else kernel_choice
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto']) if kernel != 'linear' else 'scale'

        print(f"\n[Trial {trial.number}] Stage 2 - Evaluating SVM Parameters (3 Folds):")
        print(f"  -> SVM: C: {c_val:.4f} | Kernel: {kernel} | Gamma: {gamma}")

        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_raw_all, y_train_all)):
            X_tr_raw = [X_train_raw_all[i] for i in train_idx]
            X_va_raw = [X_train_raw_all[i] for i in val_idx]
            y_tr_fold = y_train_all[train_idx]
            y_va_fold = y_train_all[val_idx]

            X_tr, X_va = get_cached_split_features(X_tr_raw, X_va_raw, word_params, char_params, use_pre_lemmatized=True)

            clf = get_classifier(kernel=kernel, c_val=c_val, gamma=gamma, calibrate=False)
            clf.fit(X_tr, y_tr_fold)

            preds = clf.predict(X_va)
            decision_scores = clf.decision_function(X_va)

            fold_score = evaluate_metric(y_va_fold, preds, decision_scores, score_metric)
            fold_scores.append(fold_score)

            intermediate_mean = float(np.mean(fold_scores))
            trial.report(intermediate_mean, step=fold)
            if trial.should_prune():
                print(f"-> [Trial {trial.number}] Stage 2 PRUNED at Fold {fold + 1} (Score: {intermediate_mean:.4f})")
                raise optuna.TrialPruned()

        mean_score = float(np.mean(fold_scores))
        print(f"-> [Trial {trial.number}] Stage 2 Mean 3-Fold Score: {mean_score:.4f}")
        return mean_score

    current_hash = generate_df_hash(train_df)
    study_s2 = safe_create_or_reset_study(study_name, db_path, 'maximize', reset_study)

    stored_hash = study_s2.user_attrs.get("dataset_hash")
    if stored_hash is not None and stored_hash != current_hash:
        print("\n⚠️ WARNING: Training dataset changed since last Optuna run!")
        if not reset_study and sys.stdin.isatty():
            try:
                response = input("Would you like to reset the database? [y/N]: ").strip().lower()
                if response in ['y', 'yes']:
                    optuna.delete_study(study_name=study_name, storage=db_path)
                    if tuning_strategy == '2stage':
                        try:
                            optuna.delete_study(study_name=study_s1_name, storage=db_path)
                        except Exception:
                            pass
                    study_s2 = optuna.create_study(study_name=study_name, storage=db_path, direction='maximize')
            except Exception:
                pass

    study_s2.set_user_attr("dataset_hash", current_hash)
    # #explained Added print_best_trial_callback to Stage 2 optimization.
    study_s2.optimize(objective_stage2, n_trials=stage2_trials, n_jobs=n_jobs_optuna, callbacks=[print_best_trial_callback])

    completed_trials = [t for t in study_s2.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed_trials) > 0:
        best_value = study_s2.best_value
        tolerance = 5e-3
        top_trials = [t for t in completed_trials if abs(t.value - best_value) < tolerance]
        best_trial = min(top_trials, key=lambda t: t.params.get('C', float('inf')))
        best_s2_params = best_trial.params

        print(f"\n[Tie-Breaker Applied] Best trial chosen (lowest C within tolerance of {best_value:.4f}):")
        for key, value in best_s2_params.items():
            print(f"  {key}: {value}")
    else:
        best_s2_params = study_s2.best_params

    best_overall_params = {}
    if tuning_strategy == '2stage':
        best_overall_params.update(best_tfidf_params)

    best_overall_params.update(best_s2_params)
    if kernel_choice != 'all':
        best_overall_params['kernel'] = kernel_choice

    return best_overall_params


# #explained Helper function to log complete experiment details, parameters, and multi-granularity performance into experiment_results.csv.
def log_experiment_to_registry(record_dict, registry_path="experiment_results.csv"):
    df_new = pd.DataFrame([record_dict])
    if os.path.exists(registry_path):
        try:
            df_existing = pd.read_csv(registry_path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        except Exception:
            df_combined = df_new
    else:
        df_combined = df_new

    df_combined.to_csv(registry_path, index=False)
    print(f"-> Experiment metrics and metadata successfully registered in '{registry_path}'")


def train_svm(train_df,
              test_df, c_val,
              kernel,
              save_path,
              granularity,
              val_df=None,
              run_optuna=False,
              reset_study=False,
              trials=15,
              trials_stage1=10,
              trials_stage2=10,
              tuning_strategy='2stage',
              tuning_sample_size=3000,
              score_metric='f1',
              study_name=None,
              n_jobs_optuna=1
              ):
    word_params, char_params = None, None
    gamma = 'scale'
    best_params = {}

    if run_optuna:
        print("Running Hyperparameter Optimization via Optuna...")
        best_params = optimize_svm_with_optuna(
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

        if 'word_min_ngram' in best_params:
            word_params = {
                'ngram_range': (best_params['word_min_ngram'], best_params['word_max_ngram']),
                'max_features': best_params['word_max_features'],
                'min_df': best_params.get('word_min_df', 1),
                'max_df': 0.95,
                'sublinear_tf': True
            }
        if 'char_min_ngram' in best_params:
            char_params = {
                'ngram_range': (best_params['char_min_ngram'], best_params['char_max_ngram']),
                'max_features': best_params['char_max_features'],
                'min_df': best_params.get('char_min_df', 1),
                'max_df': 0.95,
                'sublinear_tf': True
            }

    X_train_raw = train_df[['text', 'sentences', 'text_lemmatized']].to_dict(orient='records')
    X_test_raw = test_df[['text', 'sentences', 'text_lemmatized']].to_dict(orient='records')
    y_train = train_df['label'].values
    y_test = test_df['label'].values

    feature_pipeline = get_feature_extraction_pipeline(word_params, char_params, stylometrics_n_jobs=1, use_pre_lemmatized=True)
    calibrated_clf = get_classifier(kernel=kernel, c_val=c_val, gamma=gamma, calibrate=True)

    full_pipeline = Pipeline([
        ('features', feature_pipeline),
        ('classifier', calibrated_clf)
    ])

    optimal_threshold = 0.5
    if score_metric == 'set_fp':
        print("\nCalculating Out-of-Fold (OOF) probability scores across training set for threshold calibration...")
        oof_scores = compute_oof_scores(X_train_raw, y_train, word_params, char_params, c_val, kernel, gamma, calibrate=True)
        optimal_threshold = find_threshold_for_max_fpr(y_train, oof_scores, target_fpr=0.01)
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
        overall_auc = roc_auc_score(y_test, test_scores)
        print(f"Overall Test ROC-AUC Score: {overall_auc:.4f}\n")
    except Exception as e:
        print(f"Could not calculate ROC-AUC: {e}")

    # =============================================================
    # #explained Diagnosis & Report: Performance split on Abstracts vs Sentences
    # =============================================================
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
            full_f1 = f1_score(y_test[full_mask], preds[full_mask], pos_label=1, zero_division=0)
            if len(np.unique(y_test[full_mask])) > 1:
                full_auc = roc_auc_score(y_test[full_mask], test_scores[full_mask])
                print(f"Full Abstracts ROC-AUC: {full_auc:.4f}")

        if sent_mask.sum() > 0:
            print("\n" + "-" * 50)
            print("  DIAGNOSIS: SENTENCES ONLY PERFORMANCE      ")
            print("-" * 50)
            print(classification_report(y_test[sent_mask], preds[sent_mask], digits=4))
            sent_f1 = f1_score(y_test[sent_mask], preds[sent_mask], pos_label=1, zero_division=0)
            if len(np.unique(y_test[sent_mask])) > 1:
                sent_auc = roc_auc_score(y_test[sent_mask], test_scores[sent_mask])
                print(f"Sentences ROC-AUC: {sent_auc:.4f}\n")

    # =============================================================
    # #explained Log metadata and performance metrics to experiment registry CSV
    # =============================================================
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

        # Metrics
        'overall_f1_ai': f1_score(y_test, preds, pos_label=1, zero_division=0),
        'overall_precision_ai': precision_score(y_test, preds, pos_label=1, zero_division=0),
        'overall_roc_auc': overall_auc,
        'full_abstract_f1_ai': full_f1,
        'full_abstract_roc_auc': full_auc,
        'sentence_f1_ai': sent_f1,
        'sentence_roc_auc': sent_auc
    }

    log_experiment_to_registry(record)

    joblib.dump(full_pipeline, save_path)
    print(f"Deployable probability-calibrated pipeline saved successfully to {save_path}")

    clear_optuna_cache()