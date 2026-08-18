#!/usr/bin/env python3
# scripts/svm_sent_on_full.py

import json
import os
import random
import re
import string
import sys
import unicodedata
import zlib
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup
import joblib
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from tqdm import tqdm

# --- Project Paths & Imports ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data.data_loader import DetectionDataManager, DataFilter

RAW_DATA_PATH = PROJECT_ROOT / "data_static" / "raw" / "llm_added.parquet"
PREPROCESSED_CSV = PROJECT_ROOT / "data_static" / "preprocessed" / "preprocessed_dataset.csv"
FULL_TEST_PRED_PATH = PROJECT_ROOT / "outputs" / "svm" / "full" / "test_predictions.csv"
SENTENCE_MODEL_PATH = PROJECT_ROOT / "outputs" / "svm" / "sentence" / "model.joblib"
SENTENCE_FEATURES_DIR = PROJECT_ROOT / "data_static" / "model_features" / "svm_sentence"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "svm" / "sentence_on_full_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Visual styling
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

# Ensure NLTK Dutch Stopwords
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
try:
    dutch_stopwords = stopwords.words('dutch')
except Exception:
    dutch_stopwords = []


# =========================================================
# Feature Extraction Components (from gen_features_svm.py)
# =========================================================
_nlp = None
_dutch_stopwords_lemmatized = None

DUTCH_TRANSITIONS = {
    "echter", "bovendien", "daarnaast", "desalniettemin", "kortom",
    "tevens", "daardoor", "derhalve", "bijgevolg", "namelijk"
}

RE_MD_IMG = re.compile(r'!\[(.*?)\]\(.*?\)')
RE_MD_LINK = re.compile(r'\[(.*?)\]\(.*?\)')
RE_MD_BOLD = re.compile(r'(\*\*|__)(.*?)\1')
RE_MD_ITALIC = re.compile(r'(\*|_)(.*?)\1')
RE_MD_STRIKE = re.compile(r'(~~)(.*?)\1')
RE_MD_CODE = re.compile(r'(`)(.*?)\1')
RE_MD_HEADER = re.compile(r'^\s*[#>]+\s+', flags=re.MULTILINE)
RE_MD_HR = re.compile(r'^\s*[-*_]{3,}\s*$', flags=re.MULTILINE)
RE_WORDS = re.compile(r'\w+')


def get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("nl_core_news_sm", disable=["parser", "ner"])
        except Exception:
            import spacy.cli
            spacy.cli.download('nl_core_news_sm')
            _nlp = spacy.load("nl_core_news_sm", disable=["parser", "ner"])
        if "sentencizer" not in _nlp.pipe_names:
            _nlp.add_pipe("sentencizer")
    return _nlp


def get_dutch_stopwords_lemmatized():
    global _dutch_stopwords_lemmatized
    if _dutch_stopwords_lemmatized is None:
        nlp_model = get_nlp()
        _dutch_stopwords_lemmatized = list(set([
            token.lemma_.lower() for doc in nlp_model.pipe(dutch_stopwords) for token in doc
        ]))
    return _dutch_stopwords_lemmatized


def strip_markdown(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = RE_MD_IMG.sub(r'\1', text)
    text = RE_MD_LINK.sub(r'\1', text)
    text = RE_MD_BOLD.sub(r'\2', text)
    text = RE_MD_ITALIC.sub(r'\2', text)
    text = RE_MD_STRIKE.sub(r'\2', text)
    text = RE_MD_CODE.sub(r'\2', text)
    text = RE_MD_HEADER.sub('', text)
    text = RE_MD_HR.sub('', text)
    return text


def clean_html_markdown(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    try:
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text(separator=" ")
    except Exception:
        pass
    return strip_markdown(text)


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='ignore')
        else:
            return ""
    text = clean_html_markdown(text)
    text = unicodedata.normalize('NFKC', text)
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    text = text.replace('“', '"').replace('”', '"').replace('’', "'").replace('‘', "'")
    text = text.replace('—', '-').replace('–', '-')
    return " ".join(text.split())


def preprocess_records_in_batch(records: list, scope: str = 'sentence') -> list:
    nlp = get_nlp()
    for rec in tqdm(records, desc="[1/2] Normalizing text", leave=False):
        if 'normalized_text' not in rec:
            rec['normalized_text'] = normalize_text(rec.get('text', ''))

    missing_lemma_indices = [i for i, r in enumerate(records) if not r.get('text_lemmatized')]
    if missing_lemma_indices:
        texts_to_lemmatize = [records[i]['normalized_text'] for i in missing_lemma_indices]
        docs = nlp.pipe(texts_to_lemmatize, batch_size=1000, disable=["parser", "ner"])
        for idx, doc in zip(missing_lemma_indices, tqdm(docs, total=len(missing_lemma_indices), desc="[2/2] Lemmatizing with spaCy", leave=False)):
            records[idx]['text_lemmatized'] = " ".join([token.lemma_.lower() for token in doc if not token.is_punct])

    for rec in records:
        if not rec.get('sentences'):
            rec['sentences'] = [rec['normalized_text']]

    return records


def calculate_ttr(words):
    return len(set(words)) / len(words) if words else 0.0


def calculate_hapax_ratio(words):
    if not words:
        return 0.0
    counts = Counter(words)
    return sum(1 for w, c in counts.items() if c == 1) / len(words)


def extract_stylometric_features(text, sentences, granularity='sentence'):
    words = RE_WORDS.findall(text.lower())
    total_chars = len(text)
    num_features = 8 if granularity == 'sentence' else 11
    if not words or not sentences:
        return np.zeros(num_features)

    word_lengths = [len(w) for w in words]
    mean_word_len = float(np.mean(word_lengths))
    var_word_len = float(np.var(word_lengths))

    ttr = calculate_ttr(words)
    hapax_ratio = calculate_hapax_ratio(words)
    transition_count = sum(1 for w in words if w in DUTCH_TRANSITIONS)
    transition_ratio = transition_count / len(words)

    spaces_count = text.count(' ')
    double_spaces = text.count('  ')
    space_ratio = spaces_count / total_chars if total_chars > 0 else 0.0
    double_space_ratio = double_spaces / total_chars if total_chars > 0 else 0.0

    punc_count = sum(1 for c in text if c in string.punctuation)
    punc_ratio = punc_count / total_chars if total_chars > 0 else 0.0

    word_char_features = [
        mean_word_len,
        np.log1p(var_word_len),
        ttr,
        hapax_ratio,
        transition_ratio,
        space_ratio,
        double_space_ratio,
        punc_ratio
    ]
    return np.array(word_char_features)


# Pipeline Transformers
class TextExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, key='text'):
        self.key = key
    def fit(self, X, y=None): return self
    def transform(self, X):
        items = [X] if isinstance(X, (str, dict)) else X
        output = []
        for item in items:
            if isinstance(item, dict):
                if self.key == 'text_lemmatized' and item.get('text_lemmatized'):
                    output.append(item['text_lemmatized'])
                else:
                    output.append(item.get('normalized_text', normalize_text(item.get('text', ''))))
            else:
                output.append(normalize_text(str(item)))
        return output


class StylometricExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, granularity='sentence'):
        self.granularity = granularity
    def fit(self, X, y=None): return self
    def transform(self, X):
        items = [X] if isinstance(X, (str, dict)) else X
        features = []
        for item in items:
            if isinstance(item, dict):
                cleaned_text = item.get('normalized_text', normalize_text(item.get('text', '')))
                sentences = item.get('sentences') or [cleaned_text]
            else:
                cleaned_text = normalize_text(str(item))
                sentences = [cleaned_text]
            features.append(extract_stylometric_features(cleaned_text, sentences, granularity=self.granularity))
        return np.array(features)


class StylometricScaler(BaseEstimator, TransformerMixin):
    def __init__(self, weight=1.0):
        self.weight = weight
    def fit(self, X, y=None): return self
    def transform(self, X): return X * self.weight


# Alias classes to __main__ so joblib unpickles cleanly
sys.modules['__main__'].TextExtractor = TextExtractor
sys.modules['__main__'].StylometricExtractor = StylometricExtractor
sys.modules['__main__'].StylometricScaler = StylometricScaler


# =========================================================
# 1. Load Test Split Metadata & Full SVM Predictions
# =========================================================
print("\n" + "=" * 75)
print(" 1. LOADING TEST METADATA & FULL SVM PREDICTIONS ")
print("=" * 75)

df_prep = pd.read_csv(PREPROCESSED_CSV, usecols=["_id", "split", "scope"])
test_ids = set(df_prep[df_prep["split"] == "test"]["_id"].astype(str).unique())
print(f"Found {len(test_ids):,} unique test abstract IDs.")

print(f"Loading full abstract predictions from: {FULL_TEST_PRED_PATH}")
df_full_preds = pd.read_csv(FULL_TEST_PRED_PATH)
df_full_preds["_id"] = df_full_preds["_id"].astype(str)
if df_full_preds["llm_ratio"].max() > 1.0:
    df_full_preds["llm_ratio"] = df_full_preds["llm_ratio"] / 100.0


# =========================================================
# 2. Reconstruct Sentence Ground Truth for Test Abstracts
# =========================================================
print("\n" + "=" * 75)
print(" 2. RECONSTRUCTING SENTENCE-LEVEL TEST DATA ")
print("=" * 75)

df_raw = pd.read_parquet(RAW_DATA_PATH)
df_raw["_id"] = df_raw["_id"].astype(str)
df_raw_test = df_raw[df_raw["_id"].isin(test_ids)].copy()

sentence_records = []
target_ratios = [0.25, 0.50, 0.75]

for _, row in df_raw_test.iterrows():
    doc_id = row["_id"]

    raw_human_sents = row.get("abstract_sentence", [])
    if isinstance(raw_human_sents, np.ndarray):
        human_sents = [str(s).strip() for s in raw_human_sents if str(s).strip()]
    elif isinstance(raw_human_sents, list):
        human_sents = [str(s).strip() for s in raw_human_sents if str(s).strip()]
    else:
        continue

    n_sentences = len(human_sents)
    if n_sentences < 3:
        continue

    # Pure Human Abstract (0% LLM)
    for s_idx, s_text in enumerate(human_sents):
        sentence_records.append({
            "abstract_id": doc_id,
            "abstract_type": "human_full",
            "target_ratio": 0.0,
            "actual_ratio": 0.0,
            "abstract_label": 0,
            "sentence_index": s_idx,
            "text": s_text,
            "true_sentence_label": 0,
            "generator_origin": "human"
        })

    # Available single-sentence models
    valid_models = {}
    for col in row.index:
        if col.endswith("_single"):
            model_sents = row[col]
            if model_sents is not None and len(model_sents) > 0:
                clean_sents = [str(s).strip() for s in list(model_sents) if str(s).strip()]
                if clean_sents:
                    valid_models[col.rsplit("_single", 1)[0]] = clean_sents

    if not valid_models:
        continue

    model_names = list(valid_models.keys())

    # Partial Substitutions (25%, 50%, 75%)
    for ratio in target_ratios:
        k = max(1, min(n_sentences - 1, int(round(ratio * n_sentences))))
        actual_ratio = k / n_sentences

        seed_str = f"42_{doc_id}_{ratio}"
        pair_seed = zlib.crc32(seed_str.encode("utf-8"))
        rng = random.Random(pair_seed)
        replace_indices = set(rng.sample(range(n_sentences), k))

        for s_idx in range(n_sentences):
            if s_idx in replace_indices:
                chosen_model = rng.choice(model_names)
                m_sents = valid_models[chosen_model]
                s_text = m_sents[s_idx] if s_idx < len(m_sents) else m_sents[s_idx % len(m_sents)]
                s_label = 1
                origin = chosen_model
            else:
                s_text = human_sents[s_idx]
                s_label = 0
                origin = "human"

            sentence_records.append({
                "abstract_id": doc_id,
                "abstract_type": "synthetic_partial",
                "target_ratio": ratio,
                "actual_ratio": actual_ratio,
                "abstract_label": 1,
                "sentence_index": s_idx,
                "text": s_text,
                "true_sentence_label": s_label,
                "generator_origin": origin
            })

df_sentences = pd.DataFrame(sentence_records)
print(f"Reconstructed {len(df_sentences):,} test sentences across {df_sentences['abstract_id'].nunique():,} unique abstracts.")


# =========================================================
# 3. Load Precomputed Sentence Feature Pipeline
# =========================================================
print("\n" + "=" * 75)
print(" 3. LOADING SENTENCE FEATURE PIPELINE & MODEL ")
print("=" * 75)

# Check both pipelines.joblib (plural) and pipeline.joblib
if (SENTENCE_FEATURES_DIR / "pipelines.joblib").exists():
    SENTENCE_PIPELINE_PATH = SENTENCE_FEATURES_DIR / "pipelines.joblib"
elif (SENTENCE_FEATURES_DIR / "pipeline.joblib").exists():
    SENTENCE_PIPELINE_PATH = SENTENCE_FEATURES_DIR / "pipeline.joblib"
else:
    raise FileNotFoundError(f"Sentence pipeline not found in {SENTENCE_FEATURES_DIR}")

print(f"[CACHE HIT] Loading sentence feature pipeline from: '{SENTENCE_PIPELINE_PATH}'...")
pipeline_obj = joblib.load(SENTENCE_PIPELINE_PATH)

if isinstance(pipeline_obj, dict):
    if "pipeline" in pipeline_obj:
        pipeline = pipeline_obj["pipeline"]
    elif "sentence" in pipeline_obj:
        pipeline = pipeline_obj["sentence"]
    else:
        pipeline = list(pipeline_obj.values())[0]
else:
    pipeline = pipeline_obj

print(f"Loading sentence SVM classifier from: '{SENTENCE_MODEL_PATH}'...")
sentence_model = joblib.load(SENTENCE_MODEL_PATH)


# =========================================================
# 4. Transform Sentences & Predict via Sentence Model
# =========================================================
print("\n" + "=" * 75)
print(" 4. SENTENCE FEATURE TRANSFORMATION & SVM INFERENCE ")
print("=" * 75)

print("Pre-processing reconstructed test sentences (normalization + lemmatization)...")
records_to_process = [{"text": t} for t in df_sentences["text"].tolist()]
processed_records = preprocess_records_in_batch(records_to_process, scope='sentence')

print("Transforming sentences through fitted pipeline...")
X_sentences = pipeline.transform(processed_records)

print("Running sentence model inference...")
probs_llm_sent = sentence_model.predict_proba(X_sentences)[:, 1]
df_sentences["prob_llm"] = probs_llm_sent
df_sentences["pred_sentence"] = (probs_llm_sent >= 0.5).astype(int)

# Save granular sentence test predictions
sent_pred_file = OUTPUT_DIR / "all_sentence_test_predictions.csv"
df_sentences.to_csv(sent_pred_file, index=False)
print(f"[SAVED] Granular sentence predictions saved to: '{sent_pred_file}'")


# =========================================================
# 5. Aggregate to Abstract Level & Merge with Full SVM
# =========================================================
print("\n" + "=" * 75)
print(" 5. ABSTRACT-LEVEL AGGREGATION & ALIGNMENT ")
print("=" * 75)

abstract_agg = df_sentences.groupby(["abstract_id", "target_ratio"]).agg(
    abstract_type=("abstract_type", "first"),
    actual_ratio=("actual_ratio", "first"),
    abstract_label=("abstract_label", "first"),
    total_sentences=("sentence_index", "count"),
    sent_mean_prob=("prob_llm", "mean"),
    sent_max_prob=("prob_llm", "max"),
    sent_detected_ratio=("pred_sentence", "mean"),
    true_llm_sentence_ratio=("true_sentence_label", "mean")
).reset_index()

# Merge with Full SVM test predictions
df_merged = df_full_preds.merge(
    abstract_agg,
    left_on=["_id", "llm_ratio"],
    right_on=["abstract_id", "target_ratio"],
    how="inner"
)

df_merged["pred_full_svm"] = (df_merged["prob_llm"] >= 0.5).astype(int)
df_merged["pred_sent_mean"] = (df_merged["sent_mean_prob"] >= 0.5).astype(int)
df_merged["pred_sent_max"] = (df_merged["sent_max_prob"] >= 0.5).astype(int)

abstract_pred_file = OUTPUT_DIR / "aligned_abstract_predictions.csv"
df_merged.to_csv(abstract_pred_file, index=False)
print(f"[SAVED] Aligned abstract predictions saved to: '{abstract_pred_file}'")


# =========================================================
# 6. Compute Metrics & Breakdown per Substitution Ratio
# =========================================================
def eval_subset(y_true, y_probs, threshold=0.5):
    y_preds = (y_probs >= threshold).astype(int)
    auc_val = roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else np.nan
    acc = accuracy_score(y_true, y_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_preds, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return {
        "ROC-AUC": round(auc_val, 4) if not np.isnan(auc_val) else "N/A",
        "Accuracy": round(acc, 4),
        "F1-Score": round(f1, 4),
        "Precision": round(prec, 4),
        "Recall (TPR)": round(rec, 4),
        "Specificity": round(spec, 4) if not np.isnan(spec) else "N/A"
    }

# A. Global Performance
global_rows = [
    {"Model / Pipeline": "Full Abstract SVM", **eval_subset(df_merged["abstract_label"], df_merged["prob_llm"])},
    {"Model / Pipeline": "Sentence SVM (Mean Agg.)", **eval_subset(df_merged["abstract_label"], df_merged["sent_mean_prob"])},
    {"Model / Pipeline": "Sentence SVM (Max Agg.)", **eval_subset(df_merged["abstract_label"], df_merged["sent_max_prob"])},
]
df_global = pd.DataFrame(global_rows)
print("\n--- GLOBAL PERFORMANCE COMPARISON ---")
print(df_global.to_string(index=False))
df_global.to_csv(OUTPUT_DIR / "global_metrics_comparison.csv", index=False)

# B. Ratio Breakdown
ratio_rows = []
for ratio in [0.0, 0.25, 0.50, 0.75]:
    sub = df_merged[np.isclose(df_merged["target_ratio"], ratio)]
    if len(sub) == 0:
        continue

    ratio_label = "0% (Pure Human)" if ratio == 0.0 else f"{int(ratio*100)}% LLM Mix"
    
    full_mean_p = sub["prob_llm"].mean()
    full_tpr = (sub["pred_full_svm"] == (1 if ratio > 0 else 0)).mean()

    sent_mean_p = sub["sent_mean_prob"].mean()
    sent_mean_tpr = (sub["pred_sent_mean"] == (1 if ratio > 0 else 0)).mean()

    sent_max_p = sub["sent_max_prob"].mean()
    sent_max_tpr = (sub["pred_sent_max"] == (1 if ratio > 0 else 0)).mean()

    ratio_rows.append({
        "Substitution Level": ratio_label,
        "Samples (N)": len(sub),
        "Full SVM Mean P": f"{full_mean_p:.4f}",
        "Full SVM Acc/TPR": f"{full_tpr:.2%}",
        "Sent Mean P": f"{sent_mean_p:.4f}",
        "Sent Mean Acc/TPR": f"{sent_mean_tpr:.2%}",
        "Sent Max P": f"{sent_max_p:.4f}",
        "Sent Max Acc/TPR": f"{sent_max_tpr:.2%}"
    })

df_ratio_summary = pd.DataFrame(ratio_rows)
print("\n--- BREAKDOWN BY SUBSTITUTION RATIO ---")
print(df_ratio_summary.to_string(index=False))
df_ratio_summary.to_csv(OUTPUT_DIR / "ratio_breakdown_comparison.csv", index=False)


# =========================================================
# 7. Generate Publication-Ready LaTeX Tables
# =========================================================
print("\n" + "=" * 75)
print(" 7. GENERATING PUBLICATION-READY LATEX TABLES ")
print("=" * 75)

# LaTeX Table 1: Global Comparison
latex_global_path = OUTPUT_DIR / "table_global_model_comparison.tex"
with open(latex_global_path, "w") as f:
    f.write("% Auto-generated: Global Full vs. Sentence SVM Comparison\n")
    f.write("\\begin{table}[htbp]\n\\centering\n\\small\n")
    f.write("\\caption{Detection performance comparison on test abstracts across full and sentence-aggregated SVM models.}\n")
    f.write("\\label{tab:full_vs_sentence_global}\n")
    f.write("\\begin{tabular}{lcccccc}\n\\toprule\n")
    f.write("\\textbf{Model Pipeline} & \\textbf{ROC-AUC} & \\textbf{Accuracy} & \\textbf{F1-Score} & \\textbf{Precision} & \\textbf{Recall (TPR)} & \\textbf{Specificity} \\\\\n\\midrule\n")
    for r in global_rows:
        acc_s = f"{r['Accuracy']:.4f}" if isinstance(r['Accuracy'], (int, float)) else str(r['Accuracy'])
        f1_s = f"{r['F1-Score']:.4f}" if isinstance(r['F1-Score'], (int, float)) else str(r['F1-Score'])
        pr_s = f"{r['Precision']:.4f}" if isinstance(r['Precision'], (int, float)) else str(r['Precision'])
        rec_s = f"{r['Recall (TPR)']:.4f}" if isinstance(r['Recall (TPR)'], (int, float)) else str(r['Recall (TPR)'])
        f.write(f"{r['Model / Pipeline']} & {r['ROC-AUC']} & {acc_s} & {f1_s} & {pr_s} & {rec_s} & {r['Specificity']} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

print(f"[SAVED] Global LaTeX table saved to: '{latex_global_path}'")

# LaTeX Table 2: Ratio Breakdown Comparison
latex_ratio_path = OUTPUT_DIR / "table_ratio_model_comparison.tex"
with open(latex_ratio_path, "w") as f:
    f.write("% Auto-generated: Performance Breakdown Across LLM Substitution Ratios\n")
    f.write("\\begin{table}[htbp]\n\\centering\n\\small\n")
    f.write("\\caption{Mean predicted probability and accuracy across synthetic substitution percentages.}\n")
    f.write("\\label{tab:full_vs_sentence_by_ratio}\n")
    f.write("\\begin{tabular}{lcccccc}\n\\toprule\n")
    f.write("\\multirow{2}{*}{\\textbf{Substitution}} & \\multicolumn{2}{c}{\\textbf{Full Abstract SVM}} & \\multicolumn{2}{c}{\\textbf{Sentence SVM (Mean)}} & \\multicolumn{2}{c}{\\textbf{Sentence SVM (Max)}} \\\\\n")
    f.write("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}\n")
    f.write(" & $\\bar{P}(\\text{LLM})$ & Acc/TPR & $\\bar{P}(\\text{LLM})$ & Acc/TPR & $\\bar{P}(\\text{LLM})$ & Acc/TPR \\\\\n\\midrule\n")
    for r in ratio_rows:
        full_acc = str(r["Full SVM Acc/TPR"]).replace("%", r"\%")
        sent_mean_acc = str(r["Sent Mean Acc/TPR"]).replace("%", r"\%")
        sent_max_acc = str(r["Sent Max Acc/TPR"]).replace("%", r"\%")
        f.write(
            f"{r['Substitution Level']} & {r['Full SVM Mean P']} & {full_acc} & "
            f"{r['Sent Mean P']} & {sent_mean_acc} & {r['Sent Max P']} & {sent_max_acc} \\\\\n"
        )
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

print(f"[SAVED] Ratio Breakdown LaTeX table saved to: '{latex_ratio_path}'")


# =========================================================
# 8. Diagnostic Visualizations
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

plot_data = df_ratio_summary.copy()
ratios_x = ["0%", "25%", "50%", "75%"]
axes[0].plot(ratios_x, [float(x) for x in plot_data["Full SVM Mean P"]], marker="o", lw=2.5, color="#2b5c8f", label="Full Abstract SVM")
axes[0].plot(ratios_x, [float(x) for x in plot_data["Sent Mean P"]], marker="s", lw=2.5, color="#1b9e77", label="Sentence SVM (Mean)")
axes[0].plot(ratios_x, [float(x) for x in plot_data["Sent Max P"]], marker="^", lw=2.5, color="#d95f02", label="Sentence SVM (Max)")
axes[0].axhline(0.5, color="gray", linestyle="--", alpha=0.6, label="Threshold (0.5)")
axes[0].set_title("(A) Mean P(LLM) Across Substitution Levels", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Synthetic Text Ratio in Abstract", fontsize=11)
axes[0].set_ylabel("Mean P(LLM | Text)", fontsize=11)
axes[0].set_ylim(-0.05, 1.05)
axes[0].legend(loc="upper left", frameon=True)

for name, probs, color, ls in [
    ("Full Abstract SVM", df_merged["prob_llm"], "#2b5c8f", "-"),
    ("Sentence SVM (Mean)", df_merged["sent_mean_prob"], "#1b9e77", "-."),
    ("Sentence SVM (Max)", df_merged["sent_max_prob"], "#d95f02", ":")
]:
    fpr, tpr, _ = roc_curve(df_merged["abstract_label"], probs)
    roc_val = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, label=f"{name} (AUC={roc_val:.4f})", lw=2, color=color, linestyle=ls)

axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--", alpha=0.5)
axes[1].set_title("(B) ROC Curve on Test Abstracts", fontsize=12, fontweight="bold")
axes[1].set_xlabel("False Positive Rate", fontsize=11)
axes[1].set_ylabel("True Positive Rate", fontsize=11)
axes[1].legend(loc="lower right", frameon=True)

plot_path = OUTPUT_DIR / "full_vs_sentence_evaluation_plots.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"[SAVED PLOT] Comparison plot saved to: '{plot_path}'")
print("\n" + "=" * 75)
print(f"[COMPLETE] All evaluation artifacts successfully saved to:\n'{OUTPUT_DIR}'")
print("=" * 75)