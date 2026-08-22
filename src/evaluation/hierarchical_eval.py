from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from src.data.stylometrics import normalize_text, RE_SENT_SPLIT
from src.evaluation.metrics import MetricEvaluator
from src.visualization.latex_tables import export_performance_table

@dataclass
class AbstractSentenceStructure:
    doc_id: str
    doc_label: int
    generation_type: str
    llm_ratio: float
    sentences: List[str]
    sentence_labels: List[int]
    sentence_scores: List[float] = field(default_factory=list)

class DocumentSentenceAligner:

    @staticmethod
    def build_aligned_abstracts(df_full: pd.DataFrame, df_human_sentences: Optional[pd.DataFrame]=None) -> List[AbstractSentenceStructure]:
        id_col = next((c for c in ['_id', 'doc_id', 'id'] if c in df_full.columns), '_id')
        human_sent_lookup: Dict[str, set] = {}
        if df_human_sentences is not None and (not df_human_sentences.empty):
            for (d_id, group) in df_human_sentences.groupby(id_col):
                human_sent_lookup[str(d_id)] = set(group['text'].apply(normalize_text).values)
        aligned_docs: List[AbstractSentenceStructure] = []
        for (_, row) in df_full.iterrows():
            doc_id = str(row.get(id_col, 'unknown'))
            doc_label = int(row.get('label', 0))
            gen_type = str(row.get('generation_type', 'unknown'))
            llm_ratio = float(row.get('llm_ratio', 0.0 if doc_label == 0 else 1.0))
            if 'sentences' in row and isinstance(row['sentences'], (list, np.ndarray)) and (len(row['sentences']) > 0):
                sents = [normalize_text(s) for s in row['sentences'] if normalize_text(s)]
                s_labels = list(row.get('sentence_labels', [doc_label] * len(sents)))
            else:
                raw_text = str(row.get('text', '')).strip()
                sents = [normalize_text(s) for s in RE_SENT_SPLIT.split(raw_text) if len(normalize_text(s)) > 5]
                if not sents:
                    sents = [normalize_text(raw_text)]
                if doc_label == 0:
                    s_labels = [0] * len(sents)
                elif gen_type == 'full_rewrite':
                    s_labels = [1] * len(sents)
                else:
                    known_human = human_sent_lookup.get(doc_id, set())
                    s_labels = [0 if s in known_human else 1 for s in sents]
            aligned_docs.append(AbstractSentenceStructure(
                doc_id=doc_id,
                doc_label=doc_label,
                generation_type=gen_type,
                llm_ratio=llm_ratio,
                sentences=sents,
                sentence_labels=s_labels
            ))
        return aligned_docs

class SentenceAggregator:

    @staticmethod
    def aggregate_scores(sentence_scores: List[float], method: str='top_k_mean', tau_sentence: float=0.5, k_ratio: float=0.25) -> float:
        if not sentence_scores:
            return 0.0
        scores = np.asarray(sentence_scores, dtype=np.float64)
        if method == 'mean':
            return float(np.mean(scores))
        elif method == 'max':
            return float(np.max(scores))
        elif method == 'top_k_mean':
            k = max(1, int(np.ceil(k_ratio * len(scores))))
            top_k = np.sort(scores)[-k:]
            return float(np.mean(top_k))
        elif method == 'flagged_ratio':
            return float(np.mean(scores >= tau_sentence))
        elif method == 'softmax_weighted':
            temp = 0.1
            exp_scores = np.exp((scores - np.max(scores)) / temp)
            weights = exp_scores / np.sum(exp_scores)
            return float(np.sum(weights * scores))
        else:
            raise ValueError(f'Unknown aggregation method: {method}')

class HierarchicalDocumentEvaluator:

    @classmethod
    def evaluate_sentence_model_on_abstracts(cls, detector, df_full_dev: pd.DataFrame, df_full_test: pd.DataFrame, df_human_sentences: Optional[pd.DataFrame]=None, output_dir: Optional[Union[str, Path]]=None, target_fpr: float=0.01, aggregation_methods: Tuple[str, ...]=('mean', 'top_k_mean', 'flagged_ratio')) -> Dict[str, Any]:
        print('\n' + '=' * 80)
        print('   RUNNING HIERARCHICAL DOCUMENT EVALUATION (SENTENCE -> ABSTRACT)')
        print('=' * 80)
        print('\n[1/4] Aligning sentence structures and ground truth...')
        dev_docs = DocumentSentenceAligner.build_aligned_abstracts(df_full_dev, df_human_sentences)
        test_docs = DocumentSentenceAligner.build_aligned_abstracts(df_full_test, df_human_sentences)
        print('\n[2/4] Scoring individual sentences across all abstracts...')
        for (doc_list, split_name) in [(dev_docs, 'DEV'), (test_docs, 'TEST')]:
            all_sents = [s for doc in doc_list for s in doc.sentences]
            all_probs = detector.predict_proba(all_sents)
            ptr = 0
            for doc in doc_list:
                n_s = len(doc.sentences)
                doc.sentence_scores = all_probs[ptr:ptr + n_s].tolist()
                ptr += n_s
            print(f' -> Scored {len(all_sents):,} sentences across {len(doc_list):,} {split_name} abstracts.')
        test_all_labels = np.array([lbl for doc in test_docs for lbl in doc.sentence_labels])
        test_all_scores = np.array([sc for doc in test_docs for sc in doc.sentence_scores])
        sent_tau = getattr(detector, 'calibrated_threshold', 0.5)
        sent_pauc = MetricEvaluator.compute_metric(test_all_labels, test_all_scores, 'pauc', max_fpr=target_fpr)
        sent_tpr_1fpr = MetricEvaluator.compute_tpr_at_max_fpr(test_all_labels, test_all_scores, target_fpr=target_fpr)
        sent_roc_auc = MetricEvaluator.compute_metric(test_all_labels, test_all_scores, 'roc_auc')
        print('\n[3/4] Sentence-Level Attribution Performance on Full/Mixed Abstracts:')
        print(f' -> Sentence ROC-AUC:      {sent_roc_auc:.4f}')
        print(f' -> Sentence pAUC (<=1%):  {sent_pauc:.4f}')
        print(f' -> Sentence TPR @ 1% FPR: {sent_tpr_1fpr * 100:.2f}%')
        print('\n[4/4] Document-Level Aggregated Benchmark Results:')
        results_by_method = {}
        y_dev_doc = np.array([doc.doc_label for doc in dev_docs])
        y_test_doc = np.array([doc.doc_label for doc in test_docs])

        def _calc_subset_flagged_rate(df_sub: pd.DataFrame, tau: float) -> float:
            if df_sub.empty:
                return 0.0
            return float(np.mean(df_sub['score'] >= tau) * 100.0)

        for method in aggregation_methods:
            dev_doc_scores = np.array([SentenceAggregator.aggregate_scores(doc.sentence_scores, method=method, tau_sentence=sent_tau) for doc in dev_docs])
            test_doc_scores = np.array([SentenceAggregator.aggregate_scores(doc.sentence_scores, method=method, tau_sentence=sent_tau) for doc in test_docs])
            doc_tau = MetricEvaluator.find_threshold_for_max_fpr(y_dev_doc, dev_doc_scores, target_fpr=target_fpr, method='conformal')
            doc_roc_auc = MetricEvaluator.compute_metric(y_test_doc, test_doc_scores, 'roc_auc')
            doc_pauc = MetricEvaluator.compute_metric(y_test_doc, test_doc_scores, 'pauc', max_fpr=target_fpr)
            doc_tpr_1fpr = MetricEvaluator.compute_tpr_at_max_fpr(y_test_doc, test_doc_scores, target_fpr=target_fpr)
            test_df_diag = pd.DataFrame([{'label': doc.doc_label, 'score': sc, 'gen_type': doc.generation_type, 'llm_ratio': doc.llm_ratio} for (doc, sc) in zip(test_docs, test_doc_scores)])
            full_ai_rate = _calc_subset_flagged_rate(test_df_diag[test_df_diag['gen_type'] == 'full_rewrite'], doc_tau)
            p75_rate = _calc_subset_flagged_rate(test_df_diag[np.isclose(test_df_diag['llm_ratio'], 0.75)], doc_tau)
            p50_rate = _calc_subset_flagged_rate(test_df_diag[np.isclose(test_df_diag['llm_ratio'], 0.50)], doc_tau)
            p25_rate = _calc_subset_flagged_rate(test_df_diag[np.isclose(test_df_diag['llm_ratio'], 0.25)], doc_tau)
            human_fpr = _calc_subset_flagged_rate(test_df_diag[test_df_diag['label'] == 0], doc_tau)
            print(f'\n --- Aggregation Strategy: [{method.upper()}] (Calibrated tau_doc = {doc_tau:.4f}) ---')
            print(f'  • Document ROC-AUC:        {doc_roc_auc:.4f}')
            print(f'  • Document pAUC (FPR<=1%): {doc_pauc:.4f}')
            print(f'  • Document TPR @ 1% FPR:   {doc_tpr_1fpr * 100:.2f}%')
            print(f'  • Human Test FPR:          {human_fpr:.2f}%')
            print(f'  • 100% Full AI Detected:   {full_ai_rate:.1f}%')
            print(f'  • 75% Rewrite Detected:    {p75_rate:.1f}%')
            print(f'  • 50% Rewrite Detected:    {p50_rate:.1f}%')
            print(f'  • 25% Rewrite Detected:    {p25_rate:.1f}%')
            results_by_method[method] = {
                'doc_tau': doc_tau,
                'roc_auc': doc_roc_auc,
                'pauc': doc_pauc,
                'tpr_at_1fpr': doc_tpr_1fpr,
                'human_fpr': human_fpr,
                'detect_100pct': full_ai_rate,
                'detect_75pct': p75_rate,
                'detect_50pct': p50_rate,
                'detect_25pct': p25_rate
            }
        return {'sentence_attribution': {'roc_auc': sent_roc_auc, 'pauc': sent_pauc, 'tpr_at_1fpr': sent_tpr_1fpr}, 'document_aggregation_results': results_by_method}