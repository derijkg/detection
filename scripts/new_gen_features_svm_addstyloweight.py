#!/usr/bin/env python3
# scripts/features_svm.py

import argparse
import os
import re
import string
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup
import joblib
import nltk
import numpy as np
import pandas as pd
import spacy
from scipy.sparse import hstack, csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from tqdm import tqdm
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import normalize

# Calculate project root dynamically relative to this script (~/detection)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data.data_loader import DetectionDataManager, DataFilter

# Ensure NLTK Dutch stopwords are downloaded
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
from nltk.corpus import stopwords

try:
    dutch_stopwords = stopwords.words('dutch')
except Exception:
    dutch_stopwords = []

_nlp = None
_dutch_stopwords_lemmatized = None

DUTCH_TRANSITIONS = {
    "echter", "bovendien", "daarnaast", "desalniettemin", "kortom",
    "tevens", "daardoor", "derhalve", "bijgevolg", "namelijk"
}

# Pre-compiled Regexes for maximum speed
RE_MD_IMG = re.compile(r'!\[(.*?)\]\(.*?\)')
RE_MD_LINK = re.compile(r'\[(.*?)\]\(.*?\)')
RE_MD_BOLD = re.compile(r'(\*\*|__)(.*?)\1')
RE_MD_ITALIC = re.compile(r'(\*|_)(.*?)\1')
RE_MD_STRIKE = re.compile(r'(~~)(.*?)\1')
RE_MD_CODE = re.compile(r'(`)(.*?)\1')
RE_MD_HEADER = re.compile(r'^\s*[#>]+\s+', flags=re.MULTILINE)
RE_MD_HR = re.compile(r'^\s*[-*_]{3,}\s*$', flags=re.MULTILINE)
RE_WORDS = re.compile(r'\w+')


# ==========================================
# Lazy NLP & Helper Functions
# ==========================================
def get_nlp():
    """Lazy loads spaCy Dutch model with fast sentencizer enabled."""
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
    """Lazy computes lemmatized Dutch stopwords."""
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


# ==========================================
# Fast Batched Preprocessing
# ==========================================
def preprocess_records_in_batch(records: list, scope: str) -> list:
    """
    Pre-computes text normalization, lemmatization, and sentence segmentation
    using spaCy's C-accelerated nlp.pipe in batches with tqdm progress bars.
    """
    nlp = get_nlp()

    # 1. Normalize Raw Texts
    for rec in tqdm(records, desc="[1/3] Normalizing text", leave=False):
        if 'normalized_text' not in rec:
            rec['normalized_text'] = normalize_text(rec.get('text', ''))

    # 2. Batched Lemmatization with spaCy (if missing)
    missing_lemma_indices = [i for i, r in enumerate(records) if not r.get('text_lemmatized')]
    if missing_lemma_indices:
        texts_to_lemmatize = [records[i]['normalized_text'] for i in missing_lemma_indices]
        docs = nlp.pipe(texts_to_lemmatize, batch_size=1000, disable=["parser", "ner"])
        
        for idx, doc in zip(missing_lemma_indices, tqdm(docs, total=len(missing_lemma_indices), desc="[2/3] Batched spaCy Lemmatization", leave=False)):
            records[idx]['text_lemmatized'] = " ".join([token.lemma_.lower() for token in doc if not token.is_punct])

    # 3. Batched Sentence Segmentation with spaCy (ONLY needed for full documents)
    if scope != 'sentence':
        missing_sents_indices = [i for i, r in enumerate(records) if not r.get('sentences')]
        if missing_sents_indices:
            texts_to_segment = [records[i]['normalized_text'] for i in missing_sents_indices]
            docs = nlp.pipe(texts_to_segment, batch_size=1000)
            
            for idx, doc in zip(missing_sents_indices, tqdm(docs, total=len(missing_sents_indices), desc="[3/3] Batched Sentence Segmentation", leave=False)):
                records[idx]['sentences'] = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    else:
        # For single sentences, the text itself is the only sentence
        for rec in records:
            if not rec.get('sentences'):
                rec['sentences'] = [rec['normalized_text']]

    return records


# ==========================================
# Stylometric Feature Extraction
# ==========================================
def calculate_ttr(words):
    return len(set(words)) / len(words) if words else 0.0


def calculate_hapax_ratio(words):
    if not words:
        return 0.0
    counts = Counter(words)
    return sum(1 for w, c in counts.items() if c == 1) / len(words)


def extract_stylometric_features(text, sentences, granularity='full'):
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

    if granularity == 'sentence':
        return np.array(word_char_features)

    sent_lengths = [len(RE_WORDS.findall(s)) for s in sentences if len(RE_WORDS.findall(s)) > 0]
    if not sent_lengths or len(sent_lengths) <= 1:
        mean_sent_len = float(len(words))
        var_sent_len = 0.0
        burstiness = 0.0
    else:
        mean_sent_len = float(np.mean(sent_lengths))
        var_sent_len = float(np.var(sent_lengths))
        std_sent_len = float(np.std(sent_lengths))
        burstiness = (std_sent_len - mean_sent_len) / (std_sent_len + mean_sent_len) if (std_sent_len + mean_sent_len) > 0 else 0.0

    sentence_features = [
        np.log1p(mean_sent_len),
        np.log1p(var_sent_len),
        burstiness
    ]

    return np.array(sentence_features + word_char_features)


# ==========================================
# Custom Scikit-Learn Transformers
# ==========================================
class TextExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, key='text'):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        items = [X] if isinstance(X, (str, dict)) else X
        output = []

        for item in tqdm(items, desc=f"Extracting text ({self.key})", leave=False):
            if isinstance(item, dict):
                if self.key == 'text_lemmatized' and item.get('text_lemmatized'):
                    output.append(item['text_lemmatized'])
                else:
                    output.append(item.get('normalized_text', normalize_text(item.get('text', ''))))
            else:
                output.append(normalize_text(str(item)))

        return output


class StylometricExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, granularity='full'):
        self.granularity = granularity

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        items = [X] if isinstance(X, (str, dict)) else X
        features = []

        for item in tqdm(items, desc=f"Extracting stylometrics ({self.granularity})", leave=False):
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

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X * self.weight


# ==========================================
# Feature Extraction Pipeline Builder
# ==========================================
def get_feature_extraction_pipeline(word_tfidf_params, char_tfidf_params, granularity='full'):
    word_extractor_key = 'text_lemmatized'

    transformers = [
        ('word_ngrams', Pipeline([
            ('extract', TextExtractor(key=word_extractor_key)),
            ('tfidf', TfidfVectorizer(**word_tfidf_params))
        ])),
        ('char_ngrams', Pipeline([
            ('extract', TextExtractor(key='text')),
            ('tfidf', TfidfVectorizer(**char_tfidf_params))
        ])),
        ('stylometrics', Pipeline([
            ('extractor', StylometricExtractor(granularity=granularity)),
            ('scaler', StandardScaler()),
            ('subspace_norm', Normalizer(norm='l2')),
            ('weight', StylometricScaler(weight=1.0))
        ]))
    ]

    return Pipeline([
        ('union', FeatureUnion(transformers)),
        ('normalizer', Normalizer(norm='l2'))
    ])


# ==========================================
# TF-IDF Parameters Configurations
# ==========================================
TFIDF_CONFIGS = {
    "full": {
        "word": {
            "ngram_range": (1, 4),
            "max_features": 30000,
            "min_df": 5,
            "max_df": 0.873586,
            "binary": True,
            "sublinear_tf": False,
            "analyzer": "word",
            "norm": "l2",
            "stop_words": get_dutch_stopwords_lemmatized(),
        },
        "char": {
            "ngram_range": (3, 6),
            "max_features": 40000,
            "min_df": 4,
            "max_df": 0.885702,
            "binary": True,
            "sublinear_tf": False,
            "analyzer": "char",
            "norm": "l2",
        }
    },
    "sentence": {
        "word": {
            "ngram_range": (1, 1),
            "max_features": 10000,
            "min_df": 1,
            "max_df": 0.982572,
            "binary": True,
            "sublinear_tf": True,
            "analyzer": "word",
            "norm": "l2",
            "stop_words": get_dutch_stopwords_lemmatized(),
        },
        "char": {
            "ngram_range": (3, 3),
            "max_features": 20000,
            "min_df": 1,
            "max_df": 0.986664,
            "binary": True,
            "sublinear_tf": True,
            "analyzer": "char",
            "norm": "l2",
        }
    }
}


# ==========================================
# Main Processing Logic
# ==========================================
def process_svm_features_for_scope(
    manager: DetectionDataManager,
    scope: str,
    splits: list,
    force_reprocess: bool = False
):
    features_dir = PROJECT_ROOT / "data_static" / "model_features" / f"svm_{scope}"
    features_dir.mkdir(parents=True, exist_ok=True)

    existing_files = [features_dir / f"{split}.joblib" for split in splits]
    if all(p.exists() for p in existing_files) and not force_reprocess:
        print(f"[Cache Hit] SVM features for 'svm_{scope}' already exist in: {features_dir}")
        return

    print(f"\n==================================================")
    print(f"   Extracting SVM Features for Scope: '{scope}'   ")
    print(f"==================================================")

    # 1. Load splits & preprocess text
    split_data = {}
    for split in splits:
        df_split = manager.filter_dataframe(DataFilter(splits=[split], scopes=[scope]))
        if df_split.empty:
            raise ValueError(f"No records found for scope='{scope}' and split='{split}'.")
        
        id_col = '_id' if '_id' in df_split.columns else ('doc_id' if 'doc_id' in df_split.columns else 'id')
        records = df_split.to_dict(orient='records')
        labels = df_split['label'].astype(int).values
        ids = df_split[id_col].values if id_col in df_split.columns else df_split.index.values

        print(f"\nPre-processing split '{split}' ({len(records)} records)...")
        records = preprocess_records_in_batch(records, scope=scope)

        split_data[split] = {
            "records": records,
            "labels": labels,
            "ids": ids
        }

    config = TFIDF_CONFIGS[scope]

    # 2. Define individual component pipelines
    word_pipeline = Pipeline([
        ('extract', TextExtractor(key='text_lemmatized')),
        ('tfidf', TfidfVectorizer(**config["word"]))
    ])

    char_pipeline = Pipeline([
        ('extract', TextExtractor(key='text')),
        ('tfidf', TfidfVectorizer(**config["char"]))
    ])

    style_pipeline = Pipeline([
        ('extractor', StylometricExtractor(granularity=scope)),
        ('scaler', StandardScaler()),
        ('subspace_norm', Normalizer(norm='l2'))
    ])

    # 3. Fit components on train split
    print(f"\n[Fitting Pipeline] Fitting Word/Char TF-IDF & Stylometric Scalers on 'train'...")
    train_recs = split_data['train']['records']
    word_pipeline.fit(train_recs)
    char_pipeline.fit(train_recs)
    style_pipeline.fit(train_recs)

    # 4. Transform and save component matrices separately
    for split in splits:
        print(f"[Transforming] Generating feature components for split '{split}'...")
        recs = split_data[split]['records']

        X_word = word_pipeline.transform(recs)
        X_char = char_pipeline.transform(recs)
        X_tfidf = hstack([X_word, X_char]).tocsr()
        
        X_style = csr_matrix(style_pipeline.transform(recs))

        out_data = {
            "X_tfidf": X_tfidf,
            "X_style": X_style,
            "y": split_data[split]['labels'],
            "ids": split_data[split]['ids'],
            "scope": scope,
            "split": split
        }

        save_path = features_dir / f"{split}.joblib"
        joblib.dump(out_data, save_path)
        print(f" -> Saved '{split}': X_tfidf shape {X_tfidf.shape}, X_style shape {X_style.shape} to: {save_path}")

    # 5. Save fitted pipelines
    joblib.dump({
        "word": word_pipeline,
        "char": char_pipeline,
        "style": style_pipeline
    }, features_dir / "pipelines.joblib")


def main():
    parser = argparse.ArgumentParser(description="Generate and cache SVM TF-IDF & Stylometric features.")
    parser.add_argument(
        "--scopes",
        nargs="+",
        default=["full", "sentence"],
        help="Scopes to generate features for (default: full sentence)."
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "dev", "test"],
        help="Splits to process (default: train dev test)."
    )
    parser.add_argument(
        "--force_reprocess",
        action="store_true",
        help="Force re-extraction and overwrite existing cached feature files."
    )

    args = parser.parse_args()

    manager = DetectionDataManager()

    for scope in args.scopes:
        process_svm_features_for_scope(
            manager=manager,
            scope=scope,
            splits=args.splits,
            force_reprocess=args.force_reprocess
        )

    print("\n[SUCCESS] All SVM feature sets are ready!")


if __name__ == "__main__":
    main()