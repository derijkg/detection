# evaluate_importance.py
import sys
import os
import warnings

# Suppress spurious BeautifulSoup url/markup locator warnings
try:
    from bs4 import MarkupResemblesLocatorWarning
    warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
except ImportError:
    pass

# Ensure custom module search path is added before any local imports
sys.path.append(os.path.abspath('src/detection'))

import argparse
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score

from sklearn.metrics.pairwise import euclidean_distances
import scipy.sparse as sp

from src.detection.svm.features import prepare_classification_dataset, pre_lemmatize_dataset

# Define default models as they were originally set up
DEFAULT_MODELS = ['qwen3.5:4b', 'qwen3.6:27b', 'gemma4:e4b', 'gemma4:26b']

def evaluate_distance_update_chunk(chunk_id, col_indices, X_dense, X_sv, D_base, dual_coef, clf_intercept, clf_classes, 
                                   baseline_score, y_test, gamma, n_repeats, random_state, invert_class_mapping, sv_class_labels):
    """Evaluates a chunk of columns using the vectorized distance update trick in parallel, avoiding oversubscription."""
    import os
    import json
    import numpy as np
    from sklearn.metrics import f1_score
    import threadpoolctl
    
    M = X_dense.shape[0]
    checkpoint_file = f"checkpoint_chunk_{chunk_id}.json"
    
    progress = {}
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                progress = json.load(f)
            print(f"[Worker {chunk_id}] Resuming from checkpoint: {len(progress)} features already completed.")
        except Exception as e:
            print(f"[Worker {chunk_id}] Failed to load checkpoint, starting fresh: {e}")
            
    closest_sv_base = np.argmin(D_base, axis=1)
    sv_class_base = sv_class_labels[closest_sv_base]
    
    dec_base = (np.exp(-gamma * D_base) @ dual_coef.T + clf_intercept).ravel()
    
    with threadpoolctl.threadpool_limits(limits=1, user_api='blas'):
        for i, col_idx in enumerate(col_indices):
            col_str = str(col_idx)
            
            if col_str in progress:
                continue
                
            f1_drops = []
            margin_shifts = []
            rel_losses = []
            drifts = []
            inter_drifts = []
            helpful_pcts = []
            
            for rep in range(n_repeats):
                rng = np.random.default_rng(random_state + rep if random_state is not None else None)
                
                x_j = X_dense[:, col_idx][:, np.newaxis]
                perm_idx = rng.permutation(M)
                x_j_perm = X_dense[perm_idx, col_idx][:, np.newaxis]
                sv_j = X_sv[:, col_idx][np.newaxis, :]
                
                delta = - (x_j - sv_j)**2 + (x_j_perm - sv_j)**2
                D_new = D_base + delta
                
                K_new = np.exp(-gamma * D_new)
                dec_new = (K_new @ dual_coef.T + clf_intercept).ravel()
                
                if invert_class_mapping:
                    preds_new = np.where(dec_new > 0, clf_classes[0], clf_classes[1])
                else:
                    preds_new = np.where(dec_new > 0, clf_classes[1], clf_classes[0])
                
                score_new = f1_score(y_test, preds_new, average='binary')
                f1_drops.append(float(baseline_score - score_new))
                
                margin_shifts.append(float(np.mean(np.abs(dec_base - dec_new))))
                
                rel_loss = np.mean(np.abs(dec_base - dec_new) / (np.abs(dec_base) + 1e-9))
                rel_losses.append(float(rel_loss))
                
                closest_sv_new = np.argmin(D_new, axis=1)
                drift_ratio = np.mean(closest_sv_base != closest_sv_new)
                drifts.append(float(drift_ratio))
                
                sv_class_new = sv_class_labels[closest_sv_new]
                inter_drift_ratio = np.mean(sv_class_base != sv_class_new)
                inter_drifts.append(float(inter_drift_ratio))
                
                helpful_pct = np.mean(np.abs(dec_new) < np.abs(dec_base))
                helpful_pcts.append(float(helpful_pct))
                
            progress[col_str] = {
                "f1_drop": f1_drops,
                "margin_shift": margin_shifts,
                "rel_loss": rel_losses,
                "drift": drifts,
                "inter_drift": inter_drifts,
                "helpful": helpful_pcts
            }
            
            with open(checkpoint_file, 'w') as f:
                json.dump(progress, f)
                
    return progress

def parse_arguments():
    """Defines and parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate feature importances for a trained SVM model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--model_path', type=str, default="svm_full.pkl",
                        help="Path to the trained pipeline .pkl file.")
    parser.add_argument('--data_path', type=str, default="/home/gderijck/internship/data/gold/llm_added.parquet",
                        help="Path to the raw parquet dataset file.")
    
    parser.add_argument('--sample_size', type=int, default=350,
                        help="Number of test samples to use for fast evaluation. Use -1 for the full test split.")
    parser.add_argument('--n_repeats', type=int, default=2,
                        help="Number of times to shuffle each feature during permutation testing.")
    
    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS,
                        help="List of LLM model tags used to reconstruct columns.")
    parser.add_argument('--granularity', type=str, choices=['full', 'sentence', 'both'], default='full',
                        help="Granularity context to load and shape.")
    parser.add_argument('--llm_ratio', type=int, default=1,
                        help="Target Human-to-LLM ratio during dataset preparation.")
    parser.add_argument('--source_filter', nargs='+', default=None,
                        help="List of sources to filter (e.g., UG, SB, HBO). Leaves unfiltered if None.")
    
    parser.add_argument('--scoring', type=str, default='f1',
                        help="Scoring metric to use (e.g., 'f1' or 'accuracy').")
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help="Number of parallel jobs to run for sklearn's permutation_importance.")
    parser.add_argument('--random_state', type=int, default=42,
                        help="Random seed for data splitting and permutations.")
    parser.add_argument('--top_n', type=int, default=20,
                        help="Number of top features to display in the console printout.")
    
    parser.add_argument('--method', type=str, choices=['standard', 'custom_loop', 'distance_update', 'gradient', 'linear'], default='linear',
                        help="Evaluation method: 'linear' (exact linear weights), 'distance_update' (RBF trick), 'gradient' (RBF analytical), 'standard' (sklearn permutation), or 'custom_loop'.")

    parser.add_argument('--use_custom_loop', action='store_true',
                        help="Enables the optimized, single-threaded active-feature loop.")

    parser.add_argument('--output_report_path', type=str, default="svm_diagnostic_report.csv",
                        help="Path to output the final full multi-dimensional diagnostic report (CSV format).")
    parser.add_argument('--clear_checkpoints', action='store_true',
                        help="If set, deletes all old chunk checkpoint files before launching the evaluation.")

    return parser.parse_args()

def evaluate_feature_chunk(col_indices, X_data, baseline_score, y_test, clf):
    """Evaluates a chunk of features sequentially using in-place mutation."""
    X_worker = X_data.copy()
    chunk_results = []
    
    for col_idx in col_indices:
        original_col = X_worker[:, col_idx].copy()
        np.random.shuffle(X_worker[:, col_idx])
        
        preds = clf.predict(X_worker)
        score = f1_score(y_test, preds, average='binary')
        
        chunk_results.append((col_idx, baseline_score - score))
        X_worker[:, col_idx] = original_col
        
    return chunk_results

def unwrap_classifier(clf):
    """Unwraps CalibratedClassifierCV or GridSearch wrappers to access the core estimator."""
    if hasattr(clf, 'calibrated_classifiers_') and len(clf.calibrated_classifiers_) > 0:
        cal_obj = clf.calibrated_classifiers_[0]
        return getattr(cal_obj, 'estimator', getattr(cal_obj, 'base_estimator', clf))
    if hasattr(clf, 'best_estimator_'):
        return clf.best_estimator_
    return clf

def main():
    args = parse_arguments()

    if args.use_custom_loop and args.method == 'distance_update':
        args.method = 'custom_loop'

    n_jobs = args.n_jobs if args.n_jobs > 0 else os.cpu_count()

    if args.clear_checkpoints:
        print("Clearing temporary checkpoint files...")
        for chunk_id in range(n_jobs):
            checkpoint_file = f"checkpoint_chunk_{chunk_id}.json"
            if os.path.exists(checkpoint_file):
                try:
                    os.remove(checkpoint_file)
                except OSError:
                    pass

    # 1. Load the saved pipeline and raw test data
    print(f"Loading pipeline from: {args.model_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found at: {args.model_path}")
    pipeline = joblib.load(args.model_path)
    
    print(f"Loading dataset from: {args.data_path}")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found at: {args.data_path}")
    raw_df = pd.read_parquet(Path(args.data_path))
    
    # 2. Split raw data to isolate the test set exactly as done in training
    train_val_raw_df, test_raw_df = train_test_split(
        raw_df, 
        test_size=0.2, 
        random_state=args.random_state, 
        stratify=raw_df['source'] if 'source' in raw_df.columns else None
    )

    print("Formatting test split...")
    test_df = prepare_classification_dataset(
        test_raw_df, 
        selected_models=args.models, 
        granularity=args.granularity, 
        source_filter=args.source_filter, 
        llm_ratio=args.llm_ratio,
        random_state=args.random_state
    )

    print("Pre-lemmatizing test split...")
    test_df = pre_lemmatize_dataset(test_df, text_column='text')

    # Resolve evaluation sample size
    if args.sample_size > 0:
        eval_sample_size = min(len(test_df), args.sample_size)
        print(f"Subsampling test set down to {eval_sample_size} rows for fast evaluation...")
        eval_df, _ = train_test_split(
            test_df,
            train_size=eval_sample_size,
            random_state=args.random_state,
            stratify=test_df['label'] if 'label' in test_df.columns else None
        )
    else:
        eval_df = test_df
        print(f"Using full test split of {len(eval_df)} rows...")

    y_test = eval_df['label'].values
    X_test_raw = eval_df.to_dict(orient='records')

    # 3. Transform raw test data into sparse feature matrix
    print("Extracting features from evaluation set...")
    features_step = pipeline.named_steps['features']
    X_test_transformed = features_step.transform(X_test_raw)

    raw_clf = pipeline.named_steps['classifier']
    clf = unwrap_classifier(raw_clf)

    # 4. Robust Step-by-Step Feature Name Reconstruction
    all_feature_names = None

    if hasattr(features_step, 'get_feature_names_out'):
        try:
            names = features_step.get_feature_names_out()
            if len(names) == X_test_transformed.shape[1]:
                all_feature_names = np.array(names)
        except Exception:
            all_feature_names = None

    if all_feature_names is None:
        fu = None
        if hasattr(features_step, 'transformer_list'):
            fu = features_step
        elif hasattr(features_step, 'named_steps'):
            for step_obj in features_step.named_steps.values():
                if hasattr(step_obj, 'transformer_list'):
                    fu = step_obj
                    break

        if fu is not None:
            transformer_dict = dict(fu.transformer_list)

            word_names = []
            if 'word_ngrams' in transformer_dict:
                wp = transformer_dict['word_ngrams']
                tfidf = wp.named_steps['tfidf'] if hasattr(wp, 'named_steps') else wp
                if hasattr(tfidf, 'get_feature_names_out'):
                    word_names = [f"word_tfidf__{name}" for name in tfidf.get_feature_names_out()]

            char_names = []
            if 'char_ngrams' in transformer_dict:
                cp = transformer_dict['char_ngrams']
                tfidf = cp.named_steps['tfidf'] if hasattr(cp, 'named_steps') else cp
                if hasattr(tfidf, 'get_feature_names_out'):
                    char_names = [f"char_tfidf__{name}" for name in tfidf.get_feature_names_out()]

            stylometrics_names = [
                'style__mean_sent_len', 'style__var_sent_len', 'style__burstiness', 
                'style__mean_word_len', 'style__var_word_len', 'style__ttr', 'style__hapax_ratio', 
                'style__transition_ratio', 'style__space_ratio', 'style__double_space_ratio', 
                'style__punc_ratio', 'style__total_chars'
            ]

            all_feature_names = np.concatenate([word_names, char_names, stylometrics_names])

    n_features = X_test_transformed.shape[1]
    if all_feature_names is None or len(all_feature_names) != n_features:
        print(f"Warning: Reconstructed {len(all_feature_names) if all_feature_names is not None else 0} feature names, "
              f"but matrix has {n_features} columns. Falling back to generic feature indices.")
        all_feature_names = np.array([f"feature_{i}" for i in range(n_features)])

    # Automatic Method Fallback Inspection
    kernel_type = getattr(clf, 'kernel', None)
    
    if args.method == 'linear':
        if not hasattr(clf, 'coef_') or clf.coef_ is None:
            print(f"\n[INFO] Model kernel is '{kernel_type}' (Optuna selected non-linear kernel during tuning).")
            print(" -> Automatically switching to '--method distance_update' for non-linear SVM evaluation...\n")
            args.method = 'distance_update'

    D = len(all_feature_names)

    # =========================================================================
    # METHOD: LINEAR KERNEL DIRECT WEIGHT INSPECTION
    # =========================================================================
    if args.method == 'linear':
        print("\n=== Running Linear Kernel Coefficient Inspection ===")
        start_time = time.time()

        weights = clf.coef_
        if sp.issparse(weights):
            weights = weights.toarray()
        weights = weights.ravel()

        if len(weights) != D:
            raise ValueError(f"Mismatch: Extracted {len(weights)} feature weights, but expected {D} feature names.")

        if sp.issparse(X_test_transformed):
            feature_means = np.array(X_test_transformed.mean(axis=0)).ravel()
            
            X_sq = X_test_transformed.copy()
            X_sq.data **= 2
            feature_stds = np.sqrt(np.maximum(0, np.array(X_sq.mean(axis=0)).ravel() - feature_means**2))
        else:
            feature_means = np.mean(X_test_transformed, axis=0)
            feature_stds = np.std(X_test_transformed, axis=0)

        std_impact = weights * feature_stds
        abs_std_impact = np.abs(std_impact)

        class_1_label = clf.classes_[1] if hasattr(clf, 'classes_') else 1
        class_0_label = clf.classes_[0] if hasattr(clf, 'classes_') else 0
        direction = np.where(weights > 0, f"Pushes_to_{class_1_label}", f"Pushes_to_{class_0_label}")

        importance_df = pd.DataFrame({
            'feature': all_feature_names,
            'weight': weights,
            'abs_weight': np.abs(weights),
            'std_impact': std_impact,
            'abs_std_impact': abs_std_impact,
            'feature_mean': feature_means,
            'feature_std': feature_stds,
            'direction': direction
        }).sort_values(by='abs_weight', ascending=False)

        print(f"Calculation complete. Instantaneous extraction elapsed: {time.time() - start_time:.3f}s")

    elif args.method == 'distance_update':
        print("\n=== Running Parallelized Vectorized Distance-Update (Method 1) ===")
        start_time = time.time()
        
        if getattr(clf, 'kernel', None) != 'rbf':
            raise ValueError(f"The 'distance_update' method requires an SVM with an 'rbf' kernel, but got '{kernel_type}'.")
            
        if len(clf.classes_) != 2:
            raise ValueError("Optimized RBF SVM methods are designed for binary classification.")
            
        if hasattr(clf, '_gamma'):
            gamma = clf._gamma
        elif hasattr(clf, 'gamma') and not isinstance(clf.gamma, str):
            gamma = clf.gamma
        else:
            X_dense = X_test_transformed.toarray()
            gamma = 1.0 / (X_dense.shape[1] * X_dense.var()) if X_dense.shape[1] > 0 else 0.1

        dual_coef = clf.dual_coef_
        if sp.issparse(dual_coef):
            dual_coef = dual_coef.toarray()

        X_sv = clf.support_vectors_
        if sp.issparse(X_sv):
            X_sv = X_sv.toarray()
            
        X_dense = X_test_transformed.toarray()
        M, D = X_dense.shape
        
        D_base = euclidean_distances(X_dense, X_sv, squared=True)
        dec_base = (np.exp(-gamma * D_base) @ dual_coef.T + clf.intercept_).ravel()
        actual_preds = clf.predict(X_dense)
        preds_mapped = np.where(dec_base > 0, clf.classes_[1], clf.classes_[0])
        
        invert_class_mapping = not np.array_equal(preds_mapped, actual_preds)
        
        sv_class_labels = np.concatenate([
            np.full(clf.n_support_[0], clf.classes_[0]),
            np.full(clf.n_support_[1], clf.classes_[1])
        ])
            
        baseline_score = f1_score(y_test, actual_preds, average='binary')
        print(f"Baseline Test F1-Score: {baseline_score:.4f}")
        
        active_cols = np.where(X_dense.any(axis=0))[0]
        print(f"Active features in this subset: {len(active_cols)} / {D}")
        
        from joblib import Parallel, delayed
        print(f"Dividing {len(active_cols)} features across {n_jobs} parallel workers...")
        
        chunks = np.array_split(active_cols, n_jobs)
        
        Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(evaluate_distance_update_chunk)(
                chunk_id, chunk, X_dense, X_sv, D_base, dual_coef, clf.intercept_, clf.classes_, 
                baseline_score, y_test, gamma, args.n_repeats, args.random_state, invert_class_mapping, sv_class_labels
            )
            for chunk_id in range(len(chunks))
            for chunk in [chunks[chunk_id]]
        )
        
        print("Assembling completed metrics from checkpoints...")
        compiled_results = {}
        for chunk_id in range(n_jobs):
            checkpoint_file = f"checkpoint_chunk_{chunk_id}.json"
            if os.path.exists(checkpoint_file):
                try:
                    with open(checkpoint_file, 'r') as f:
                        data = json.load(f)
                        compiled_results.update(data)
                except Exception as e:
                    print(f"Warning: could not read {checkpoint_file}: {e}")

        importances_f1_mean = np.zeros(D)
        importances_f1_std = np.zeros(D)
        margin_shift_mean = np.zeros(D)
        rel_loss_mean = np.zeros(D)
        drift_mean = np.zeros(D)
        inter_drift_mean = np.zeros(D)
        helpful_mean = np.zeros(D)
        
        for col_str, metrics in compiled_results.items():
            col_idx = int(col_str)
            importances_f1_mean[col_idx] = np.mean(metrics["f1_drop"])
            importances_f1_std[col_idx] = np.std(metrics["f1_drop"])
            margin_shift_mean[col_idx] = np.mean(metrics["margin_shift"])
            rel_loss_mean[col_idx] = np.mean(metrics["rel_loss"])
            drift_mean[col_idx] = np.mean(metrics["drift"])
            inter_drift_mean[col_idx] = np.mean(metrics["inter_drift"])
            helpful_mean[col_idx] = np.mean(metrics["helpful"])
            
        importance_df = pd.DataFrame({
            'feature': all_feature_names,
            'importance_f1_mean': importances_f1_mean,
            'importance_f1_std': importances_f1_std,
            'margin_shift_mean': margin_shift_mean,
            'rel_confidence_loss_mean': rel_loss_mean,
            'neighborhood_drift_mean': drift_mean,
            'inter_class_drift_mean': inter_drift_mean,
            'helpful_direction_mean': helpful_mean
        }).sort_values(by='margin_shift_mean', ascending=False)

        if len(compiled_results) >= len(active_cols):
            print("Cleaning up temporary chunk checkpoints...")
            for chunk_id in range(n_jobs):
                checkpoint_file = f"checkpoint_chunk_{chunk_id}.json"
                if os.path.exists(checkpoint_file):
                    try:
                        os.remove(checkpoint_file)
                    except OSError:
                        pass
        print(f"Calculation complete. (Elapsed: {time.time() - start_time:.1f}s)")

    elif args.method == 'gradient':
        print("\n=== Running Analytical Gradients (Method 2) ===")
        start_time = time.time()
        
        if getattr(clf, 'kernel', None) != 'rbf':
            raise ValueError("The 'gradient' method requires an SVM with an 'rbf' kernel.")
            
        if len(clf.classes_) != 2:
            raise ValueError("Optimized RBF SVM methods are designed for binary classification.")
            
        if hasattr(clf, '_gamma'):
            gamma = clf._gamma
        elif hasattr(clf, 'gamma') and not isinstance(clf.gamma, str):
            gamma = clf.gamma
        else:
            X_dense = X_test_transformed.toarray()
            gamma = 1.0 / (X_dense.shape[1] * X_dense.var()) if X_dense.shape[1] > 0 else 0.1

        dual_coef = clf.dual_coef_
        if sp.issparse(dual_coef):
            dual_coef = dual_coef.toarray()

        X_sv = clf.support_vectors_
        if sp.issparse(X_sv):
            X_sv = X_sv.toarray()
            
        X_dense = X_test_transformed.toarray()
        M, D = X_dense.shape
        
        D_base = euclidean_distances(X_dense, X_sv, squared=True)
        K = np.exp(-gamma * D_base)
        
        W = -2 * gamma * K * dual_coef
        S = np.sum(W, axis=1, keepdims=True)
        G = (S * X_dense) - (W @ X_sv)
        
        importances_mean = np.mean(np.abs(G * X_dense), axis=0)
        importances_std = np.std(np.abs(G * X_dense), axis=0)
        
        importance_df = pd.DataFrame({
            'feature': all_feature_names,
            'gradient_input_importance_mean': importances_mean,
            'gradient_input_importance_std': importances_std
        }).sort_values(by='gradient_input_importance_mean', ascending=False)
        print(f"Calculation complete. (Elapsed: {time.time() - start_time:.1f}s)")

    elif args.method == 'custom_loop':
        from joblib import Parallel, delayed
        
        X_dense = X_test_transformed.toarray()
        active_cols = np.where(X_dense.any(axis=0))[0]
        
        print(f"\nTotal features in model: {X_dense.shape[1]}")
        print(f"Active features in this subset: {len(active_cols)}")
        
        baseline_preds = clf.predict(X_dense)
        baseline_score = f1_score(y_test, baseline_preds, average='binary')
        print(f"Baseline Test F1-Score: {baseline_score:.4f}")
        
        chunks = np.array_split(active_cols, n_jobs)
        start_time = time.time()
        
        results_nested = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(evaluate_feature_chunk)(chunk, X_dense, baseline_score, y_test, clf)
            for chunk in chunks
        )
        
        importances_mean = np.zeros(D)
        for chunk_results in results_nested:
            for col_idx, imp in chunk_results:
                importances_mean[col_idx] = imp

        importance_df = pd.DataFrame({
            'feature': all_feature_names,
            'importance_mean': importances_mean
        }).sort_values(by='importance_mean', ascending=False)
        print(f"Calculation complete. (Elapsed: {time.time() - start_time:.1f}s)")

    else:
        print(f"Calculating feature importances using Scikit-Learn (repeats={args.n_repeats}, jobs={args.n_jobs})...")
        start_time = time.time()
        result = permutation_importance(
            clf, 
            X_test_transformed.toarray(), 
            y_test, 
            scoring=args.scoring,       
            n_repeats=args.n_repeats,        
            random_state=args.random_state,
            n_jobs=args.n_jobs           
        )
        
        importance_df = pd.DataFrame({
            'feature': all_feature_names,
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std
        }).sort_values(by='importance_mean', ascending=False)
        print(f"Calculation complete. (Elapsed: {time.time() - start_time:.1f}s)")

    # Save Output Report
    output_path = Path(args.output_report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(output_path, index=False)
    print(f"\nDiagnostic report saved to: {output_path}")

    print(f"\n=== Top {args.top_n} Most Important Features ===")
    print(importance_df.head(args.top_n).to_string(index=False))


if __name__ == '__main__':
    main()