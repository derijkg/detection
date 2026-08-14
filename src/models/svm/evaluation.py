# evaluation.py
import os
import re
import zlib
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from sklearn.metrics import (
    classification_report, f1_score, roc_auc_score, precision_score, 
    recall_score, roc_curve, confusion_matrix, average_precision_score, 
    brier_score_loss, matthews_corrcoef
)
from sklearn.svm import LinearSVC

from features import clean_and_normalize_value, pre_lemmatize_dataset, safe_parse_list


# ==========================================
# 1. Synthetic Mixed Data Generation
# ==========================================
def split_sentences(text: str) -> list:
    if not isinstance(text, str) or not text.strip():
        return []
    try:
        import nltk
        return nltk.sent_tokenize(text)
    except Exception:
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def generate_mixed_test_dataset(test_raw_df: pd.DataFrame, selected_models: list, ratios: list, seed: int = 42) -> pd.DataFrame:
    mixed_records = []
    human_col = 'text' if 'text' in test_raw_df.columns else ('abstract' if 'abstract' in test_raw_df.columns else 'original_text')

    for idx, row in test_raw_df.iterrows():
        human_text = clean_and_normalize_value(row.get(human_col, ""))
        abstract_id = row.get('_id', row.get('doc_id', row.get('id', idx)))
        
        human_sents = []
        for col in ['abstract_sentences', 'abstract_sentence', 'human_sentences']:
            if col in row:
                parsed = safe_parse_list(row[col])
                if parsed:
                    human_sents = [clean_and_normalize_value(s) for s in parsed if s]
                    break

        if not human_sents and human_text:
            human_sents = [clean_and_normalize_value(s) for s in split_sentences(human_text) if s]

        if len(human_sents) < 4:
            continue

        for model_name in selected_models:
            llm_text = ""
            for col in [f"{model_name}_full", model_name]:
                if col in row and pd.notna(row[col]):
                    llm_text = clean_and_normalize_value(row[col])
                    break

            llm_sents = []
            for col in [f"{model_name}_sentences", f"{model_name}_sentence", f"{model_name}_single"]:
                if col in row:
                    parsed = safe_parse_list(row[col])
                    if parsed:
                        llm_sents = [clean_and_normalize_value(s) for s in parsed if s]
                        break

            if not llm_sents and llm_text:
                llm_sents = [clean_and_normalize_value(s) for s in split_sentences(llm_text) if s]

            min_len = min(len(human_sents), len(llm_sents))
            if min_len < 4:
                continue

            h_aligned = human_sents[:min_len]
            l_aligned = llm_sents[:min_len]

            for ratio in ratios:
                k = int(round(ratio * min_len))
                k = max(1, min(min_len - 1, k))

                seed_str = f"{seed}_{abstract_id}_{model_name}_{ratio}"
                pair_seed = zlib.crc32(seed_str.encode('utf-8'))
                rng = random.Random(pair_seed)
                llm_indices = set(rng.sample(range(min_len), k))

                mixed_sents = [
                    l_aligned[i] if i in llm_indices else h_aligned[i]
                    for i in range(min_len)
                ]

                mixed_records.append({
                    '_id': abstract_id,
                    'doc_id': abstract_id,
                    'llm_model': model_name,
                    'target_ratio': ratio,
                    'actual_ratio': k / min_len,
                    'num_sentences': min_len,
                    'sentences': mixed_sents,
                    'text': " ".join(mixed_sents)
                })

    return pd.DataFrame(mixed_records)


# ==========================================
# 2. Feature Importance Inspector (Averaged across CV Folds)
# ==========================================
def extract_feature_importance(pipeline, top_n: int = 15) -> Tuple[str, pd.DataFrame]:
    try:
        features_step = pipeline.named_steps['features']
        raw_clf = pipeline.named_steps['classifier']

        weights = None

        if hasattr(raw_clf, 'calibrated_classifiers_') and len(raw_clf.calibrated_classifiers_) > 0:
            coef_list = []
            for cal_obj in raw_clf.calibrated_classifiers_:
                est = getattr(cal_obj, 'estimator', getattr(cal_obj, 'base_estimator', None))
                if est is not None and hasattr(est, 'coef_') and est.coef_ is not None:
                    w = est.coef_.toarray() if sp.issparse(est.coef_) else est.coef_
                    coef_list.append(w.ravel())
            if coef_list:
                weights = np.mean(coef_list, axis=0)
        elif hasattr(raw_clf, 'best_estimator_'):
            est = raw_clf.best_estimator_
            if hasattr(est, 'coef_') and est.coef_ is not None:
                weights = est.coef_.toarray() if sp.issparse(est.coef_) else est.coef_.ravel()
        elif hasattr(raw_clf, 'coef_') and raw_clf.coef_ is not None:
            weights = raw_clf.coef_.toarray() if sp.issparse(raw_clf.coef_) else raw_clf.coef_.ravel()

        if weights is None:
            return "N/A", pd.DataFrame()

        # Extract feature names across FeatureUnion blocks
        feature_names = []
        union_step = features_step.named_steps.get('union', features_step) if hasattr(features_step, 'named_steps') else features_step

        if hasattr(union_step, 'transformer_list'):
            # Pre-calculate TF-IDF feature count to derive stylometrics feature count
            tfidf_feature_count = 0
            for name, trans_pipe in union_step.transformer_list:
                if hasattr(trans_pipe, 'named_steps') and 'tfidf' in trans_pipe.named_steps:
                    if hasattr(trans_pipe.named_steps['tfidf'], 'get_feature_names_out'):
                        tfidf_feature_count += len(trans_pipe.named_steps['tfidf'].get_feature_names_out())

            sty_feature_count = len(weights) - tfidf_feature_count

            for name, trans_pipe in union_step.transformer_list:
                if hasattr(trans_pipe, 'named_steps'):
                    if 'tfidf' in trans_pipe.named_steps and hasattr(trans_pipe.named_steps['tfidf'], 'get_feature_names_out'):
                        names = [f"{name}_{fn}" for fn in trans_pipe.named_steps['tfidf'].get_feature_names_out()]
                        feature_names.extend(names)
                    elif name == 'stylometrics':
                        # FIX: Match exact 8 vs 11 stylometric feature length
                        if sty_feature_count == 8:
                            sty_cols = ['mean_word_len', 'var_word_len', 'ttr', 'hapax_ratio', 
                                        'transition_ratio', 'space_ratio', 'double_space_ratio', 'punc_ratio']
                        else:
                            sty_cols = ['mean_sent_len', 'var_sent_len', 'burstiness', 'mean_word_len', 
                                        'var_word_len', 'ttr', 'hapax_ratio', 'transition_ratio', 
                                        'space_ratio', 'double_space_ratio', 'punc_ratio']
                            
                        feature_names.extend([f"sty_{col}" for col in sty_cols])

        if len(feature_names) == len(weights):
            feature_names = np.array(feature_names)
        else:
            feature_names = np.array([f"feature_{i}" for i in range(len(weights))])

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'weight': weights,
            'abs_weight': np.abs(weights),
            'direction': np.where(weights > 0, 'Pushes_to_LLM', 'Pushes_to_Human')
        }).sort_values(by='abs_weight', ascending=False)

        top_df = importance_df.head(top_n)
        summary_str = "; ".join([
            f"{row['feature']} ({row['weight']:+.2f})" 
            for _, row in top_df.iterrows()
        ])
        return summary_str, importance_df
    except Exception as e:
        print(f"Warning: Could not extract feature importances: {e}")

    return "N/A", pd.DataFrame()


# ==========================================
# 3. Consolidated Master Evaluation Runner
# ==========================================
def save_or_update_csv(df_new: pd.DataFrame, csv_path: str, match_cols: Optional[List[str]] = None):
    if match_cols is None:
        match_cols = ['study_name']

    if os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path)

            valid_match = all(c in existing_df.columns for c in match_cols) and \
                          all(c in df_new.columns for c in match_cols)

            if valid_match:
                keys_to_replace = set(zip(*[df_new[c] for c in match_cols]))
                existing_keys = zip(*[existing_df[c] for c in match_cols])

                keep_mask = [k not in keys_to_replace for k in existing_keys]
                filtered_df = existing_df[keep_mask]

                combined_df = pd.concat([filtered_df, df_new], ignore_index=True)
            else:
                combined_df = pd.concat([existing_df, df_new], ignore_index=True)
        except Exception:
            combined_df = df_new
    else:
        combined_df = df_new

    combined_df.to_csv(csv_path, index=False)


def run_full_evaluation(
    model_pipeline,
    test_raw_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metadata: Dict[str, Any],
    selected_models: list,
    mixed_ratios: list = [0.25, 0.50, 0.75],
    eval_mode: str = 'both',
    experiments_dir: str = "experiments"
) -> Dict[str, Any]:
    os.makedirs(experiments_dir, exist_ok=True)
    study_name = metadata.get('study_name', f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    clf = model_pipeline.named_steps.get('classifier', model_pipeline)
    is_calibrated = hasattr(clf, 'calibrated_classifiers_') or (
        hasattr(clf, 'predict_proba') and not isinstance(clf, LinearSVC)
    )

    default_threshold = 0.5 if is_calibrated else 0.0
    optimal_threshold = getattr(model_pipeline, "optimal_threshold", default_threshold)
    print(f"-> Active Decision Threshold: {optimal_threshold:.6f} (Is Calibrated: {is_calibrated})")
    run_standard = eval_mode in ['both', 'standard']
    run_synth = eval_mode in ['both', 'synth']

    test_record = {}
    ratio_flagged_map = {}

    if run_standard:
        print("\n" + "=" * 60)
        print("      PART 1: STANDARD TEST SET EVALUATION      ")
        print("=" * 60)

        cols = [c for c in ['text', 'sentences', 'text_lemmatized'] if c in test_df.columns]
        X_test_raw = test_df[cols].to_dict(orient='records')
        y_test = test_df['label'].values

        if is_calibrated:
            test_scores = model_pipeline.predict_proba(X_test_raw)[:, 1]
            brier_loss = float(brier_score_loss(y_test, test_scores))
        else:
            test_scores = model_pipeline.decision_function(X_test_raw)
            brier_loss = None

        preds = (test_scores >= optimal_threshold).astype(int)

        print(classification_report(y_test, preds, digits=4))

        tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
        fpr_human = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        overall_auc = float(roc_auc_score(y_test, test_scores)) if len(np.unique(y_test)) > 1 else 0.0
        overall_pr_auc = float(average_precision_score(y_test, test_scores)) if len(np.unique(y_test)) > 1 else 0.0
        mcc = float(matthews_corrcoef(y_test, preds))

        print(f"Overall Test ROC-AUC:              {overall_auc:.4f}")
        print(f"Overall Test PR-AUC (Average Prec): {overall_pr_auc:.4f}")
        print(f"False Positive Rate on Human Text: {fpr_human:.4f} ({fp}/{fp+tn})")
        if brier_loss is not None:
            print(f"Calibration Loss (Brier Score):    {brier_loss:.4f}")
        else:
            print(f"Calibration Loss (Brier Score):    N/A (Uncalibrated Decision Function)")
        print(f"Matthews Corr Coef (MCC):          {mcc:.4f}\n")

        full_auc, sent_auc = None, None
        full_f1_ai, sent_f1_ai = None, None
        full_fpr, sent_fpr = None, None

        if 'task_type' in test_df.columns:
            full_mask = (test_df['task_type'] == 'full').values
            sent_mask = (test_df['task_type'] == 'sentence').values

            if full_mask.sum() > 0:
                tn_f, fp_f, fn_f, tp_f = confusion_matrix(y_test[full_mask], preds[full_mask], labels=[0, 1]).ravel()
                full_fpr = float(fp_f / (fp_f + tn_f)) if (fp_f + tn_f) > 0 else 0.0
                full_f1_ai = float(f1_score(y_test[full_mask], preds[full_mask], pos_label=1, zero_division=0))
                if len(np.unique(y_test[full_mask])) > 1:
                    full_auc = float(roc_auc_score(y_test[full_mask], test_scores[full_mask]))

            if sent_mask.sum() > 0:
                tn_s, fp_s, fn_s, tp_s = confusion_matrix(y_test[sent_mask], preds[sent_mask], labels=[0, 1]).ravel()
                sent_fpr = float(fp_s / (fp_s + tn_s)) if (fp_s + tn_s) > 0 else 0.0
                sent_f1_ai = float(f1_score(y_test[sent_mask], preds[sent_mask], pos_label=1, zero_division=0))
                if len(np.unique(y_test[sent_mask])) > 1:
                    sent_auc = float(roc_auc_score(y_test[sent_mask], test_scores[sent_mask]))

        test_record = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'study_name': study_name,
            'calibrated_threshold': optimal_threshold,
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            'overall_f1_ai': float(f1_score(y_test, preds, pos_label=1, zero_division=0)),
            'overall_precision_ai': float(precision_score(y_test, preds, pos_label=1, zero_division=0)),
            'overall_recall_ai': float(recall_score(y_test, preds, pos_label=1, zero_division=0)),
            'overall_f1_human': float(f1_score(y_test, preds, pos_label=0, zero_division=0)),
            'overall_precision_human': float(precision_score(y_test, preds, pos_label=0, zero_division=0)),
            'overall_recall_human': float(recall_score(y_test, preds, pos_label=0, zero_division=0)),
            'overall_fpr_human': fpr_human,
            'overall_roc_auc': overall_auc,
            'overall_pr_auc': overall_pr_auc,
            'overall_brier_score': brier_loss if brier_loss is not None else np.nan,
            'overall_mcc': mcc,
            'full_abstract_f1_ai': full_f1_ai,
            'full_abstract_roc_auc': full_auc,
            'full_abstract_fpr_human': full_fpr,
            'sentence_f1_ai': sent_f1_ai,
            'sentence_roc_auc': sent_auc,
            'sentence_fpr_human': sent_fpr
        }
        
        test_csv_path = os.path.join(experiments_dir, "experiment_test.csv")
        save_or_update_csv(pd.DataFrame([test_record]), test_csv_path, match_cols=['study_name'])
        print(f"-> Standard Test Set metrics saved to '{test_csv_path}'")

    if run_synth:
        print("\n" + "=" * 60)
        print("      PART 2: SYNTHETIC MIXED DATASET EVALUATION      ")
        print("=" * 60)

        mixed_test_df = generate_mixed_test_dataset(test_raw_df, selected_models, mixed_ratios, seed=42)
        print(f"Generated {len(mixed_test_df)} synthetic mixed abstracts across ratios {mixed_ratios}.")

        mixed_test_df = pre_lemmatize_dataset(mixed_test_df, text_column='text')
        
        cols = [c for c in ['text', 'sentences', 'text_lemmatized'] if c in mixed_test_df.columns]
        X_mixed = mixed_test_df[cols].to_dict(orient='records')

        if is_calibrated:
            synth_scores = model_pipeline.predict_proba(X_mixed)[:, 1]
        else:
            synth_scores = model_pipeline.decision_function(X_mixed)

        synth_preds = (synth_scores >= optimal_threshold).astype(int)
        
        mixed_test_df['llm_score'] = synth_scores
        mixed_test_df['predicted_label'] = synth_preds

        synth_summary_rows = []

        for ratio, group in mixed_test_df.groupby('target_ratio'):
            flagged_pct = (group['predicted_label'] == 1).mean() * 100
            avg_llm_score = group['llm_score'].mean()
            ratio_key = f"{int(ratio*100)}pct"
            ratio_flagged_map[f"synth_flagged_{ratio_key}"] = round(float(flagged_pct), 2)
            ratio_flagged_map[f"synth_avg_score_{ratio_key}"] = round(float(avg_llm_score), 4)

            synth_summary_rows.append({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'study_name': study_name,
                'target_ratio': f"{int(ratio*100)}%",
                'actual_ratio_avg': round(float(group['actual_ratio'].mean()), 4),
                'sample_count': len(group),
                'flagged_as_llm_pct': round(float(flagged_pct), 2),
                'avg_llm_score': round(float(avg_llm_score), 4)
            })

        synth_summary_df = pd.DataFrame(synth_summary_rows)
        print("\n--- Synthetic Substitution Sensitivity Summary ---")
        print(synth_summary_df[['target_ratio', 'actual_ratio_avg', 'sample_count', 'flagged_as_llm_pct', 'avg_llm_score']].to_string(index=False))

        synth_csv_path = os.path.join(experiments_dir, "experiment_synth.csv")
        save_or_update_csv(synth_summary_df, synth_csv_path, match_cols=['study_name', 'target_ratio'])
        print(f"\n-> Synthetic Mixed metrics saved to '{synth_csv_path}'")

    print("\n" + "=" * 60)
    print("      PART 3: FEATURE IMPORTANCE EXTRACTION      ")
    print("=" * 60)

    top_features_str, full_importance_df = extract_feature_importance(model_pipeline, top_n=15)
    if not full_importance_df.empty:
        feat_csv_path = os.path.join(experiments_dir, f"feature_importance_{study_name}.csv")
        full_importance_df.to_csv(feat_csv_path, index=False)
        print(f"Top 15 Features: {top_features_str}")
        print(f"-> Full feature importance report saved to '{feat_csv_path}'")

    consolidated_record = {}
    consolidated_record.update(metadata)
    consolidated_record.update({
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'calibrated_threshold': optimal_threshold,
    })
    if run_standard and test_record:
        consolidated_record.update({
            'overall_f1_ai': test_record['overall_f1_ai'],
            'overall_precision_ai': test_record['overall_precision_ai'],
            'overall_recall_ai': test_record['overall_recall_ai'],
            'overall_fpr_human': test_record['overall_fpr_human'],
            'overall_roc_auc': test_record['overall_roc_auc'],
            'overall_pr_auc': test_record['overall_pr_auc'],
            'overall_brier_score': test_record['overall_brier_score'],
            'overall_mcc': test_record['overall_mcc'],
        })
    if run_synth:
        consolidated_record.update(ratio_flagged_map)

    consolidated_record['top_features'] = top_features_str

    results_csv_path = os.path.join(experiments_dir, "experiment_results.csv")
    save_or_update_csv(pd.DataFrame([consolidated_record]), results_csv_path, match_cols=['study_name'])
    print(f"\n-> Master consolidated experiment report saved to '{results_csv_path}'\n")

    return consolidated_record