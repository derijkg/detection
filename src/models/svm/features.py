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
from sklearn.preprocessing import Normalizer, StandardScaler
from scipy.sparse import hstack, csr_matrix

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
from nltk.corpus import stopwords
dutch_stopwords = stopwords.words('dutch')

_nlp = None
_dutch_stopwords_lemmatized = None


def get_nlp():
    """Lazy loads spaCy model with fast sentencizer enabled."""
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
    return [token.lemma_.lower() for token in nlp_model(text.lower()) if not token.is_punct]


def lemmatize_text_string(text: str) -> str:
    """Helper for on-the-fly lemmatization during pipeline deployment on raw text."""
    if not isinstance(text, str) or not text.strip():
        return ""
    nlp_model = get_nlp()
    doc = nlp_model(text.lower())
    return " ".join([token.lemma_.lower() for token in doc if not token.is_punct])


DUTCH_TRANSITIONS = {
    "echter", "bovendien", "daarnaast", "desalniettemin", "kortom",
    "tevens", "daardoor", "derhalve", "bijgevolg", "namelijk"
}


# ==========================================
# Text Cleaning, Normalization & Sentence Helpers
# ==========================================
def tokenize_sentences(text: str) -> list:
    """Tokenizes Dutch text into sentences using spaCy's sentencizer."""
    if not isinstance(text, str) or not text.strip():
        return []
    nlp_model = get_nlp()
    doc = nlp_model(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def pre_lemmatize_dataset(df, text_column='text', n_process=1):
    """Lemmatizes dataset sequentially with lowercase normalized lemmas."""
    df = df.copy()
    print(f"Pre-lemmatizing {len(df)} texts sequentially...")

    nlp_model = get_nlp()
    texts_to_process = df[text_column].astype(str).str.lower().tolist()
    docs = nlp_model.pipe(texts_to_process, batch_size=256)
    
    lemmatized_texts = [" ".join([token.lemma_.lower() for token in doc if not token.is_punct]) for doc in docs]
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


def extract_stylometric_features(text, sentences, granularity='full'):
    """
    Extracts stylometric features.
    - 'full' granularity: Returns 11 features (including sentence length variance & burstiness).
    - 'sentence' granularity: Returns 8 features (omitting invalid sentence-level aggregation metrics).
    """
    words = re.findall(r'\w+', text.lower())
    total_chars = len(text)

    num_features = 8 if granularity == 'sentence' else 11
    if not words or not sentences:
        return np.zeros(num_features)

    # --- 8 Features valid for BOTH sentences and full abstracts ---
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

    # --- 3 Multi-Sentence Features (ONLY for 'full' abstracts) ---
    sent_lengths = [len(re.findall(r'\w+', s)) for s in sentences if len(re.findall(r'\w+', s)) > 0]
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
# 2. Custom Scikit-Learn Transformers
# ==========================================
class TextExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, key='text'):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        output = []
        items = [X] if isinstance(X, (str, dict)) else X

        for item in items:
            if isinstance(item, str):
                raw_text = item
                lemmatized_text = None
            elif isinstance(item, dict):
                raw_text = item.get('text', '')
                lemmatized_text = item.get('text_lemmatized', None)
            else:
                raw_text = str(item)
                lemmatized_text = None

            cleaned_text = normalize_text(raw_text)

            if self.key == 'text_lemmatized':
                if lemmatized_text is not None and str(lemmatized_text).strip():
                    output.append(str(lemmatized_text))
                else:
                    output.append(lemmatize_text_string(cleaned_text))
            else:
                output.append(cleaned_text)

        return output


class StylometricExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, n_jobs=1, granularity='full'):
        self.n_jobs = n_jobs
        self.granularity = granularity

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        items = [X] if isinstance(X, (str, dict)) else X

        for item in items:
            if isinstance(item, str):
                raw_text = item
                provided_sents = None
            elif isinstance(item, dict):
                raw_text = item.get('text', '')
                provided_sents = item.get('sentences', None)
            else:
                raw_text = str(item)
                provided_sents = None

            cleaned_text = normalize_text(raw_text)

            if provided_sents and isinstance(provided_sents, (list, tuple)) and len(provided_sents) > 0:
                sentences = [normalize_text(str(s)) for s in provided_sents if s and str(s).strip()]
            else:
                sentences = tokenize_sentences(cleaned_text)

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
# 3. Core Preprocessing & Caching Utilities
# ==========================================
def generate_df_hash(df):
    serialized = f"{len(df)}_" + "".join(df['text'].head(50).astype(str))
    return hashlib.md5(serialized.encode('utf-8')).hexdigest()


def safe_parse_list(val):
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


def prepare_classification_dataset(
    df,
    selected_models,
    granularity='full',
    source_filter=None,
    llm_ratio=4,
    random_state=42,
):
  import random

  if source_filter:
    df = df[df['source'].isin(source_filter)].copy()

  records = []
  active_granularities = (
      ['full', 'sentence'] if granularity == 'both' else [granularity]
  )

  for idx, row in df.iterrows():
    source = row.get('source', 'unknown')
    abstract_id = row.get('_id', row.get('doc_id', row.get('id', idx)))
    seed_val = (
        int(hashlib.md5(str(abstract_id).encode('utf-8')).hexdigest(), 16)
        % (2**32)
    )
    local_rng = random.Random(seed_val)

    row_human_records = []
    row_llm_records = []

    for gran in active_granularities:
      # --- Human Record Processing ---
      raw_human_sents = []
      for col in ['abstract_sentence', 'abstract_sentences']:
        if col in row:
          raw_human_sents = safe_parse_list(row.get(col, []))
          if isinstance(raw_human_sents, list) and len(raw_human_sents) > 0:
            break

      human_sents = (
          raw_human_sents if isinstance(raw_human_sents, (list, tuple)) else []
      )
      cleaned_human_sents = [
          s for s in [clean_and_normalize_value(s) for s in human_sents] if s
      ]

      if gran == 'full':
        human_text = row.get('abstract', '')
        if pd.notna(human_text) and human_text != '':
          cleaned_human_text = clean_and_normalize_value(human_text)
          if cleaned_human_text != '':
            rec = {
                '_id': abstract_id,
                'doc_id': abstract_id,
                'text': cleaned_human_text,
                'sentences': cleaned_human_sents,
                'label': 0,
                'source': source,
                'task_type': 'full',
            }
            if 'text_lemmatized' in row and pd.notna(row['text_lemmatized']):
              rec['text_lemmatized'] = row['text_lemmatized']
            row_human_records.append(rec)
      else:
        for sent in cleaned_human_sents:
          rec = {
              '_id': abstract_id,
              'doc_id': abstract_id,
              'text': sent,
              'sentences': [sent],
              'label': 0,
              'source': source,
              'task_type': 'sentence',
          }
          if 'text_lemmatized' in row and pd.notna(row['text_lemmatized']):
            rec['text_lemmatized'] = row['text_lemmatized']
          row_human_records.append(rec)

      # --- LLM Model Selection ---
      valid_models = []
      for model in selected_models:
        if gran == 'full':
          col_name = f'{model}_full'
          if col_name in row and pd.notna(row[col_name]) and row[col_name] != '':
            valid_models.append(model)
        else:
          sent_col = None
          for col_var in [
              f'{model}_single',
              f'{model}_sentence',
              f'{model}_sentences',
          ]:
            if col_var in row:
              sent_col = col_var
              break
          if sent_col:
            raw_sent_list = safe_parse_list(row[sent_col])
            if isinstance(raw_sent_list, (list, tuple)) and any(
                s for s in raw_sent_list if s
            ):
              valid_models.append(model)

      models_to_process = (
          local_rng.sample(valid_models, k=llm_ratio)
          if (llm_ratio > 0 and len(valid_models) > llm_ratio)
          else valid_models
      )

      for model in models_to_process:
        sent_col = None
        for col_var in [
            f'{model}_single',
            f'{model}_sentence',
            f'{model}_sentences',
        ]:
          if col_var in row:
            sent_col = col_var
            break

        raw_ai_sents = (
            safe_parse_list(row.get(sent_col, [])) if sent_col else []
        )
        ai_sents = raw_ai_sents if isinstance(raw_ai_sents, list) else []
        cleaned_ai_sents = [
            s for s in [clean_and_normalize_value(s) for s in ai_sents] if s
        ]

        if gran == 'full':
          col_name = f'{model}_full'
          cleaned_ai_text = clean_and_normalize_value(row[col_name])
          if cleaned_ai_text != '':
            rec = {
                '_id': abstract_id,
                'doc_id': abstract_id,
                'text': cleaned_ai_text,
                'sentences': cleaned_ai_sents,
                'label': 1,
                'source': source,
                'task_type': 'full',
            }
            if 'text_lemmatized' in row and pd.notna(row['text_lemmatized']):
              rec['text_lemmatized'] = row['text_lemmatized']
            row_llm_records.append(rec)
        else:
          for sent in cleaned_ai_sents:
            rec = {
                '_id': abstract_id,
                'doc_id': abstract_id,
                'text': sent,
                'sentences': [sent],
                'label': 1,
                'source': source,
                'task_type': 'sentence',
            }
            if 'text_lemmatized' in row and pd.notna(row['text_lemmatized']):
              rec['text_lemmatized'] = row['text_lemmatized']
            row_llm_records.append(rec)

    if row_human_records:
      records.extend(row_human_records)
    if row_llm_records:
      records.extend(row_llm_records)

  return pd.DataFrame(records)


def get_feature_extraction_pipeline(
    word_tfidf_params=None,
    char_tfidf_params=None,
    sty_params=None,
    stylometrics_n_jobs=1,
    use_pre_lemmatized=True,
    granularity='full',
):
    sty_config = sty_params or {'use_stylometrics': True, 'sty_weight': 1.0}
    default_max_df = 1.0 if granularity == 'sentence' else 0.95  # <--- DYNAMIC DEFAULT

    w_params = (word_tfidf_params or {}).copy()
    w_params.setdefault('ngram_range', (1, 3))
    w_params.setdefault('max_features', 50000)
    w_params.setdefault('min_df', 2)
    w_params.setdefault('max_df', default_max_df)
    w_params.setdefault('sublinear_tf', True)
    w_params.setdefault('norm', 'l2')
    w_params.setdefault('analyzer', 'word')
    w_params.setdefault('stop_words', get_dutch_stopwords_lemmatized())

    word_extractor_key = 'text_lemmatized' if use_pre_lemmatized else 'text'

    c_params = (char_tfidf_params or {}).copy()
    c_params.setdefault('analyzer', 'char')
    c_params.setdefault('ngram_range', (2, 5))
    c_params.setdefault('max_features', 50000)
    c_params.setdefault('min_df', 2)
    c_params.setdefault('max_df', default_max_df)
    c_params.setdefault('sublinear_tf', True)
    c_params.setdefault('norm', 'l2')

    transformers = [
        ('word_ngrams', Pipeline([
            ('extract', TextExtractor(key=word_extractor_key)),
            ('tfidf', TfidfVectorizer(**w_params))
        ])),
        ('char_ngrams', Pipeline([
            ('extract', TextExtractor(key='text')),
            ('tfidf', TfidfVectorizer(**c_params))
        ]))
    ]

    if sty_config.get('use_stylometrics', True):
        sty_weight = sty_config.get('sty_weight', 1.0)
        transformers.append(
            ('stylometrics', Pipeline([
                ('extractor', StylometricExtractor(n_jobs=1, granularity=granularity)),
                ('scaler', StandardScaler()),
                ('subspace_norm', Normalizer(norm='l2')),
                ('weight', StylometricScaler(weight=sty_weight))
            ]))
        )

    return Pipeline([
        ('union', FeatureUnion(transformers)),
        ('normalizer', Normalizer(norm='l2'))
    ])


def clear_optuna_cache(cache_dir="./.optuna_temp_cache"):
    if os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir)
            print(f"-> Cleared temporary Optuna trial feature cache at '{cache_dir}'")
        except Exception as e:
            print(f"Warning: Could not clear temporary Optuna cache: {e}")


def get_cached_split_features(
    X_train_raw,
    X_val_raw,
    word_params,
    char_params,
    sty_params=None,
    cache_dir='./.optuna_temp_cache',
    use_pre_lemmatized=True,
    granularity='full',
):
    os.makedirs(cache_dir, exist_ok=True)

    def get_data_hash(X_raw):
        serialized = f"{len(X_raw)}_" + "".join(
            str(x.get('_id', x.get('doc_id', x.get('text', '')[:20]))) if isinstance(x, dict) else str(x)[:20] for x in X_raw
        )
        return hashlib.md5(serialized.encode('utf-8')).hexdigest()
    
    train_hash = get_data_hash(X_train_raw)
    val_hash = get_data_hash(X_val_raw)

    has_tr_lemmatized = len(X_train_raw) > 0 and isinstance(X_train_raw[0], dict) and 'text_lemmatized' in X_train_raw[0]
    has_va_lemmatized = len(X_val_raw) == 0 or (isinstance(X_val_raw[0], dict) and 'text_lemmatized' in X_val_raw[0])
    actual_use_pre_lemmatized = use_pre_lemmatized and has_tr_lemmatized and has_va_lemmatized

    default_max_df = 1.0 if granularity == 'sentence' else 0.95
    w_params = word_params.copy() if word_params else {}
    extractor = TextExtractor(key='text_lemmatized') if actual_use_pre_lemmatized else TextExtractor(key='text')
    w_params.setdefault('analyzer', 'word')
    w_params.setdefault('sublinear_tf', True)
    w_params.setdefault('max_df', default_max_df)
    w_params.setdefault('norm', 'l2')
    w_params.setdefault('stop_words', get_dutch_stopwords_lemmatized())

    vectorizer_word = TfidfVectorizer(**w_params)
    X_tr_word = vectorizer_word.fit_transform(extractor.transform(X_train_raw))
    X_va_word = vectorizer_word.transform(extractor.transform(X_val_raw))

    c_params = char_params.copy() if char_params else {}
    c_params.setdefault('analyzer', 'char')
    c_params.setdefault('sublinear_tf', True)
    c_params.setdefault('max_df', default_max_df)
    c_params.setdefault('norm', 'l2')

    vectorizer_char = TfidfVectorizer(**c_params)
    extractor_char = TextExtractor(key='text')

    X_tr_char = vectorizer_char.fit_transform(extractor_char.transform(X_train_raw))
    X_va_char = vectorizer_char.transform(extractor_char.transform(X_val_raw))

    feature_blocks_tr = [X_tr_word, X_tr_char]
    feature_blocks_va = [X_va_word, X_va_char]

    sty_config = sty_params or {'use_stylometrics': True, 'sty_weight': 1.0}
    if sty_config.get('use_stylometrics', True):
        # FIX: Include granularity in cache key to prevent full vs sentence cache collisions
        train_sty_raw_path = os.path.join(cache_dir, f"tr_sty_raw_{granularity}_{train_hash}.joblib")
        val_sty_raw_path = os.path.join(cache_dir, f"val_sty_raw_{granularity}_tr_{train_hash}_val_{val_hash}.joblib")

        if os.path.exists(train_sty_raw_path) and os.path.exists(val_sty_raw_path):
            raw_tr = joblib.load(train_sty_raw_path)
            raw_va = joblib.load(val_sty_raw_path)
        else:
            extractor_sty = StylometricExtractor(n_jobs=1, granularity=granularity)
            raw_tr = extractor_sty.transform(X_train_raw)
            raw_va = extractor_sty.transform(X_val_raw)
            joblib.dump(raw_tr, train_sty_raw_path)
            joblib.dump(raw_va, val_sty_raw_path)

        scaler = StandardScaler()
        tr_scaled = scaler.fit_transform(raw_tr)
        va_scaled = scaler.transform(raw_va)

        sty_normalizer = Normalizer(norm='l2')
        tr_unit = sty_normalizer.fit_transform(tr_scaled)
        va_unit = sty_normalizer.transform(va_scaled)

        sty_weight = sty_config.get('sty_weight', 1.0)
        X_tr_sty = csr_matrix(tr_unit * sty_weight)
        X_va_sty = csr_matrix(va_unit * sty_weight)

        feature_blocks_tr.append(X_tr_sty)
        feature_blocks_va.append(X_va_sty)

    X_tr_combined = hstack(feature_blocks_tr).tocsr()
    X_va_combined = hstack(feature_blocks_va).tocsr()

    global_normalizer = Normalizer(norm='l2')
    X_tr_combined = global_normalizer.fit_transform(X_tr_combined)
    X_va_combined = global_normalizer.transform(X_va_combined)

    return X_tr_combined, X_va_combined