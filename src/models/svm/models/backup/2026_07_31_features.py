# features.py
import ast
import hashlib
import json
import os
import re
import shutil
import string
import unicodedata
from collections import Counter

from bs4 import BeautifulSoup
import joblib
import nltk
import numpy as np
import pandas as pd
import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler
from scipy.sparse import hstack, csr_matrix

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
dutch_stopwords = stopwords.words('dutch')

_nlp = None
_dutch_stopwords_lemmatized = None


def get_nlp():
    """Lazy loads spaCy model only when needed to save memory."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("nl_core_news_sm", disable=["parser", "ner"])
        except Exception:
            import spacy.cli
            spacy.cli.download('nl_core_news_sm')
            _nlp = spacy.load("nl_core_news_sm", disable=["parser", "ner"])
    return _nlp


def get_dutch_stopwords_lemmatized():
    """Lazy loads and computes lemmatized Dutch stopwords on demand."""
    global _dutch_stopwords_lemmatized
    if _dutch_stopwords_lemmatized is None:
        nlp_model = get_nlp()
        _dutch_stopwords_lemmatized = list(set([
            token.lemma_.lower() for doc in nlp_model.pipe(dutch_stopwords) for token in doc
        ]))
    return _dutch_stopwords_lemmatized


def lemmatizing_tokenizer(text):
    nlp_model = get_nlp()
    return [token.lemma_ for token in nlp_model(text.lower()) if not token.is_punct]


DUTCH_TRANSITIONS = {
    "echter", "bovendien", "daarnaast", "desalniettemin", "kortom",
    "tevens", "daardoor", "derhalve", "bijgevolg", "namelijk"
}


# ==========================================
# Text Cleaning & Normalization Helpers
# ==========================================
def pre_lemmatize_dataset(df, text_column='text', n_process=1):
    """Lemmatizes dataset sequentially for non-parallel execution."""
    df = df.copy()
    print(f"Pre-lemmatizing {len(df)} texts sequentially...")

    nlp_model = get_nlp()
    docs = nlp_model.pipe(df[text_column].astype(str).tolist(), batch_size=256)
    lemmatized_texts = [" ".join([token.lemma_ for token in doc if not token.is_punct]) for doc in docs]

    df['text_lemmatized'] = lemmatized_texts
    return df


def strip_markdown(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'!\[(.*?)\]\(.*?\)', r'\1', text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', text)
    text = re.sub(r'(\*|_)(.*?)\1', r'\2', text)
    text = re.sub(r'(~~)(.*?)\1', r'\2', text)
    text = re.sub(r'(`)(.*?)\1', r'\2', text)
    text = re.sub(r'^\s*[#>]+\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
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


def clean_and_normalize_value(val):
    if isinstance(val, str):
        return normalize_text(val)
    elif isinstance(val, list):
        return [clean_and_normalize_value(v) for v in val]
    elif isinstance(val, np.ndarray):
        return np.array([clean_and_normalize_value(v) for v in val], dtype=object)
    elif isinstance(val, pd.Series):
        return val.apply(clean_and_normalize_value)
    return val


# ==========================================
# 1. Stylometric Helper Functions
# ==========================================
def calculate_ttr(words):
    return len(set(words)) / len(words) if words else 0.0


def calculate_hapax_ratio(words):
    if not words:
        return 0.0
    counts = Counter(words)
    return sum(1 for w, c in counts.items() if c == 1) / len(words)


def extract_stylometric_features(text, sentences):
    """Extracts a 12-dimensional array of statistical and syntactic metrics."""
    words = re.findall(r'\w+', text.lower())
    total_chars = len(text)

    if not words or not sentences:
        return np.zeros(12)

    if len(sentences) <= 1:
        mean_sent_len = float(len(words))
        var_sent_len = 0.0
        burstiness = 0.0
    else:
        sent_lengths = [len(re.findall(r'\w+', s)) for s in sentences if len(re.findall(r'\w+', s)) > 0]
        mean_sent_len = np.mean(sent_lengths) if sent_lengths else 0.0
        var_sent_len = np.var(sent_lengths) if sent_lengths else 0.0
        burstiness = (np.std(sent_lengths) / mean_sent_len) if mean_sent_len > 0 else 0.0

    word_lengths = [len(w) for w in words]
    mean_word_len = np.mean(word_lengths)
    var_word_len = np.var(word_lengths)

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

    return np.array([
        mean_sent_len, var_sent_len, burstiness,
        mean_word_len, var_word_len,
        ttr, hapax_ratio,
        transition_ratio, space_ratio, double_space_ratio, punc_ratio,
        float(total_chars)
    ])


# ==========================================
# 2. Custom Scikit-Learn Transformers
# ==========================================
class TextExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, key='text'):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [item[self.key] for item in X]


class StylometricExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, n_jobs=1):
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = [extract_stylometric_features(item['text'], item['sentences']) for item in X]
        return np.array(features)


# ==========================================
# 3. Core Preprocessing & Caching Utilities
# ==========================================
def generate_df_hash(df):
    serialized = f"{len(df)}_" + "".join(df['text'].head(50).astype(str))
    return hashlib.md5(serialized.encode('utf-8')).hexdigest()


def safe_parse_list(val):
    """Safely converts numpy arrays, list strings, single strings, or lists into a Python list."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (list, tuple)):
        return list(val)
    if isinstance(val, str):
        val = val.strip()
        if val.startswith('[') and val.endswith(']'):
            try:
                res = ast.literal_eval(val)
                if isinstance(res, list):
                    return res
            except (ValueError, SyntaxError):
                try:
                    res = json.loads(val)
                    if isinstance(res, list):
                        return res
                except Exception:
                    pass
        return [val] if val else []
    return [val]


def prepare_classification_dataset(df, selected_models, granularity='full', source_filter=None, llm_ratio=4, random_state=42):
    import random
    local_rng = random.Random(random_state) if random_state is not None else random

    if source_filter:
        df = df[df['source'].isin(source_filter)].copy()

    records = []
    active_granularities = ['full', 'sentence'] if granularity == 'both' else [granularity]

    for _, row in df.iterrows():
        source = row.get('source', 'unknown')
        row_human_records = []
        row_llm_records = []

        for gran in active_granularities:
            raw_human_sents = []
            for col in ['abstract_sentence', 'abstract_sentences']:
                if col in row:
                    raw_human_sents = safe_parse_list(row.get(col, []))
                    if isinstance(raw_human_sents, list) and len(raw_human_sents) > 0:
                        break

            human_sents = raw_human_sents if isinstance(raw_human_sents, (list, tuple)) else []
            cleaned_human_sents = [s for s in [clean_and_normalize_value(s) for s in human_sents] if s]

            if gran == 'full':
                human_text = row.get('abstract', '')
                if pd.notna(human_text) and human_text != "":
                    cleaned_human_text = clean_and_normalize_value(human_text)
                    if cleaned_human_text != "":
                        row_human_records.append({
                            'text': cleaned_human_text,
                            'sentences': cleaned_human_sents,
                            'label': 0,
                            'source': source,
                            'task_type': 'full'
                        })
            else:
                for sent in cleaned_human_sents:
                    row_human_records.append({
                        'text': sent,
                        'sentences': [sent],
                        'label': 0,
                        'source': source,
                        'task_type': 'sentence'
                    })

            valid_models = []
            for model in selected_models:
                if gran == 'full':
                    col_name = f"{model}_full"
                    if col_name in row and pd.notna(row[col_name]) and row[col_name] != "":
                        valid_models.append(model)
                else:
                    sent_col = None
                    for col_var in [f"{model}_single", f"{model}_sentence", f"{model}_sentences"]:
                        if col_var in row:
                            sent_col = col_var
                            break
                    if sent_col:
                        raw_sent_list = safe_parse_list(row[sent_col])
                        if isinstance(raw_sent_list, (list, tuple)) and any(s for s in raw_sent_list if s):
                            valid_models.append(model)

            if not valid_models:
                continue

            models_to_process = local_rng.sample(valid_models, k=llm_ratio) if (llm_ratio > 0 and len(valid_models) > llm_ratio) else valid_models

            for model in models_to_process:
                sent_col = None
                for col_var in [f"{model}_single", f"{model}_sentence", f"{model}_sentences"]:
                    if col_var in row:
                        sent_col = col_var
                        break

                raw_ai_sents = safe_parse_list(row.get(sent_col, [])) if sent_col else []
                ai_sents = raw_ai_sents if isinstance(raw_ai_sents, list) else []
                cleaned_ai_sents = [s for s in [clean_and_normalize_value(s) for s in ai_sents] if s]

                if gran == 'full':
                    col_name = f"{model}_full"
                    cleaned_ai_text = clean_and_normalize_value(row[col_name])
                    if cleaned_ai_text != "":
                        row_llm_records.append({
                            'text': cleaned_ai_text,
                            'sentences': cleaned_ai_sents,
                            'label': 1,
                            'source': source,
                            'task_type': 'full'
                        })
                else:
                    for sent in cleaned_ai_sents:
                        row_llm_records.append({
                            'text': sent,
                            'sentences': [sent],
                            'label': 1,
                            'source': source,
                            'task_type': 'sentence'
                        })

        if row_human_records:
            records.extend(row_human_records)
        if row_llm_records:
            records.extend(row_llm_records)

    final_df = pd.DataFrame(records)

    if not final_df.empty and 'task_type' in final_df.columns:
        num_full = (final_df['task_type'] == 'full').sum()
        num_sentence = (final_df['task_type'] == 'sentence').sum()
        total_tasks = len(final_df)
        print(f"[Task Proportions] Granularity: {granularity} | "
              f"Full Abstracts: {num_full} ({num_full/total_tasks*100:.1f}%) | "
              f"Sentences: {num_sentence} ({num_sentence/total_tasks*100:.1f}%)")

    if not final_df.empty and 'label' in final_df.columns:
        num_human = (final_df['label'] == 0).sum()
        num_llm = (final_df['label'] == 1).sum()
        if num_human > 0:
            print(f"[Dataset Stats] Counts: {num_human} Human, {num_llm} AI | Empirical Ratio -> 1:{num_llm/num_human:.2f}")

    return final_df


def get_feature_extraction_pipeline(word_tfidf_params=None, char_tfidf_params=None, stylometrics_n_jobs=1, use_pre_lemmatized=True):
    """Defines feature extraction blueprint for Sklearn pipelines, ensuring all custom trial params are passed."""
    if word_tfidf_params is None:
        word_tfidf_params = {
            'ngram_range': (1, 3),
            'max_features': 50000,
            'min_df': 2,
            'max_df': 0.95,
            'sublinear_tf': True,
            'analyzer': 'word',
            'token_pattern': r'(?u)\b\w\w+\b',
            'stop_words': get_dutch_stopwords_lemmatized()
        }
    else:
        word_tfidf_params = word_tfidf_params.copy()
        word_tfidf_params.setdefault('analyzer', 'word')
        if not use_pre_lemmatized:
            word_tfidf_params.setdefault('tokenizer', lemmatizing_tokenizer)
            word_tfidf_params.setdefault('token_pattern', None)
        word_tfidf_params.setdefault('sublinear_tf', True)
        word_tfidf_params.setdefault('max_df', 0.95)
        word_tfidf_params.setdefault('stop_words', get_dutch_stopwords_lemmatized())

    if char_tfidf_params is None:
        char_tfidf_params = {
            'analyzer': 'char',
            'ngram_range': (2, 5),
            'max_features': 50000,
            'min_df': 2,
            'max_df': 0.95,
            'sublinear_tf': True
        }
    else:
        char_tfidf_params = char_tfidf_params.copy()
        char_tfidf_params.setdefault('analyzer', 'char')
        char_tfidf_params.setdefault('sublinear_tf', True)
        char_tfidf_params.setdefault('max_df', 0.95)

    word_extractor_key = 'text_lemmatized' if use_pre_lemmatized else 'text'

    return FeatureUnion([
        ('word_ngrams', Pipeline([
            ('extract', TextExtractor(key=word_extractor_key)),
            ('tfidf', TfidfVectorizer(**word_tfidf_params))
        ])),
        ('char_ngrams', Pipeline([
            ('extract', TextExtractor(key='text')),
            ('tfidf', TfidfVectorizer(**char_tfidf_params))
        ])),
        ('stylometrics', Pipeline([
            ('extractor', StylometricExtractor(n_jobs=1)),
            ('scaler', MaxAbsScaler())
        ]))
    ])


def clear_optuna_cache(cache_dir="./.optuna_temp_cache"):
    """Deletes temporary Optuna trial feature cache."""
    if os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir)
            print(f"-> Cleared temporary Optuna trial feature cache at '{cache_dir}'")
        except Exception as e:
            print(f"Warning: Could not clear temporary Optuna cache: {e}")


def get_cached_split_features(X_train_raw, X_val_raw, word_params, char_params, cache_dir="./.optuna_temp_cache", use_pre_lemmatized=True):
    """Extracts features cleanly, unpacking all trial TF-IDF parameters into TfidfVectorizer."""
    os.makedirs(cache_dir, exist_ok=True)

    def get_data_hash(X_raw):
        sample = [x['text'][:30] for x in X_raw[:50]] + [x['text'][:30] for x in X_raw[-50:]]
        serialized = f"{len(X_raw)}_" + "".join(sample)
        return hashlib.md5(serialized.encode('utf-8')).hexdigest()

    train_hash = get_data_hash(X_train_raw)
    val_hash = get_data_hash(X_val_raw)

    train_sty_path = os.path.join(cache_dir, f"tr_sty_{train_hash}.joblib")
    val_sty_path = os.path.join(cache_dir, f"val_sty_tr_{train_hash}_val_{val_hash}.joblib")

    has_lemmatized = len(X_train_raw) > 0 and 'text_lemmatized' in X_train_raw[0]
    actual_use_pre_lemmatized = use_pre_lemmatized and has_lemmatized

    # Word TF-IDF parameter unpacking
    w_params = word_params.copy() if word_params else {}
    extractor = TextExtractor(key='text_lemmatized') if actual_use_pre_lemmatized else TextExtractor(key='text')
    w_params.setdefault('analyzer', 'word')
    if not actual_use_pre_lemmatized:
        w_params.setdefault('tokenizer', lemmatizing_tokenizer)
        w_params.setdefault('token_pattern', None)
    w_params.setdefault('sublinear_tf', True)
    w_params.setdefault('max_df', 0.95)
    w_params.setdefault('stop_words', get_dutch_stopwords_lemmatized())

    vectorizer = TfidfVectorizer(**w_params)
    X_tr_word = vectorizer.fit_transform(extractor.transform(X_train_raw))
    X_va_word = vectorizer.transform(extractor.transform(X_val_raw))

    # Char TF-IDF parameter unpacking
    c_params = char_params.copy() if char_params else {}
    c_params.setdefault('analyzer', 'char')
    c_params.setdefault('sublinear_tf', True)
    c_params.setdefault('max_df', 0.95)

    vectorizer_char = TfidfVectorizer(**c_params)
    extractor_char = TextExtractor(key='text')

    X_tr_char = vectorizer_char.fit_transform(extractor_char.transform(X_train_raw))
    X_va_char = vectorizer_char.transform(extractor_char.transform(X_val_raw))

    # Stylometrics
    if os.path.exists(train_sty_path) and os.path.exists(val_sty_path):
        X_tr_sty = joblib.load(train_sty_path)
        X_va_sty = joblib.load(val_sty_path)
    else:
        extractor_sty = StylometricExtractor(n_jobs=1)
        scaler = MaxAbsScaler()

        raw_tr = extractor_sty.transform(X_train_raw)
        raw_va = extractor_sty.transform(X_val_raw)

        X_tr_sty = csr_matrix(scaler.fit_transform(raw_tr))
        X_va_sty = csr_matrix(scaler.transform(raw_va))

        joblib.dump(X_tr_sty, train_sty_path)
        joblib.dump(X_va_sty, val_sty_path)

    # Concatenate features
    X_tr_combined = hstack([X_tr_word, X_tr_char, X_tr_sty]).tocsr()
    X_va_combined = hstack([X_va_word, X_va_char, X_va_sty]).tocsr()

    return X_tr_combined, X_va_combined