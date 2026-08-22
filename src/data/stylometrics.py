# src/data/stylometrics.py
import re
import string
import unicodedata
import warnings
from collections import Counter
from typing import Any, Dict, List, Optional
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
import numpy as np

warnings.filterwarnings('ignore', category=MarkupResemblesLocatorWarning)

DUTCH_TRANSITIONS = {
    'echter', 'bovendien', 'daarnaast', 'desalniettemin', 'kortom',
    'tevens', 'daardoor', 'derhalve', 'bijgevolg', 'namelijk'
}

NON_PRINTABLE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')
RE_MD_IMG = re.compile(r'!\[(.*?)\]\(.*?\)')
RE_MD_LINK = re.compile(r'\[(.*?)\]\(.*?\)')
RE_MD_BOLD = re.compile(r'(\*\*|__)(.*?)\1')
RE_MD_ITALIC = re.compile(r'(\*|_)(.*?)\1')
RE_MD_STRIKE = re.compile(r'(~~)(.*?)\1')
RE_MD_CODE = re.compile(r'(`)(.*?)\1')
RE_MD_HEADER = re.compile(r'^\s*[#> interrogation]+\s+', flags=re.MULTILINE)
RE_MD_HR = re.compile(r'^\s*[-*_]{3,}\s*$', flags=re.MULTILINE)
RE_WORDS = re.compile(r'\w+')
RE_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

def is_none_or_nan(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, (list, tuple, np.ndarray)):
        return len(val) == 0
    if isinstance(val, (float, np.floating)):
        return np.isnan(val)
    try:
        import pandas as pd
        return bool(pd.isna(val))
    except (ValueError, TypeError):
        return False

def strip_markdown(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ''
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
        return ''
    try:
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text(separator=' ')
    except Exception:
        pass
    return strip_markdown(text)

def normalize_text(text: Any) -> str:
    if is_none_or_nan(text) or isinstance(text, (list, tuple, np.ndarray)):
        return ''
    text_str = clean_html_markdown(str(text))
    text_str = unicodedata.normalize('NFKC', text_str)
    text_str = NON_PRINTABLE_RE.sub('', text_str)
    text_str = text_str.replace('“', '"').replace('”', '"').replace('’', "'").replace('‘', "'")
    text_str = text_str.replace('—', '-').replace('–', '-')
    return ' '.join(text_str.split()).strip()

def calculate_ttr(words: List[str]) -> float:
    return len(set(words)) / len(words) if words else 0.0

def calculate_hapax_ratio(words: List[str]) -> float:
    if not words:
        return 0.0
    counts = Counter(words)
    return sum(1 for _, c in counts.items() if c == 1) / len(words)

def extract_stylometric_features(
    text: str,
    sentences: Optional[List[str]] = None,
    granularity: str = 'full',
    raw_text: Optional[str] = None
) -> np.ndarray:
    is_sent = ('sentence' in granularity or granularity == 'single')
    num_features = 8 if is_sent else 11
    words = RE_WORDS.findall(text.lower())
    clean_chars = len(text)
    raw_str = raw_text if raw_text is not None else text
    raw_total_chars = max(len(raw_str), 1)

    if not words or clean_chars == 0:
        return np.zeros(num_features, dtype=np.float64)

    word_lengths = [len(w) for w in words]
    mean_word_len = float(np.mean(word_lengths))
    var_word_len = float(np.var(word_lengths))
    ttr = calculate_ttr(words)
    hapax_ratio = calculate_hapax_ratio(words)
    transition_count = sum(1 for w in words if w in DUTCH_TRANSITIONS)
    transition_ratio = transition_count / len(words)
    spaces_count = raw_str.count(' ')
    double_spaces = raw_str.count('  ')
    space_ratio = spaces_count / raw_total_chars
    double_space_ratio = double_spaces / raw_total_chars
    punc_count = sum(1 for c in text if c in string.punctuation)
    punc_ratio = punc_count / max(clean_chars, 1)

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

    if is_sent:
        return np.array(word_char_features, dtype=np.float64)

    sents = sentences or [text]
    sent_lengths = [len(RE_WORDS.findall(s)) for s in sents if len(RE_WORDS.findall(s)) > 0]
    if not sent_lengths or len(sent_lengths) <= 1:
        mean_sent_len = float(len(words))
        var_sent_len = 0.0
        burstiness = 0.0
    else:
        mean_sent_len = float(np.mean(sent_lengths))
        var_sent_len = float(np.var(sent_lengths))
        std_sent_len = float(np.std(sent_lengths))
        burstiness = (
            (std_sent_len - mean_sent_len) / (std_sent_len + mean_sent_len)
            if std_sent_len + mean_sent_len > 0
            else 0.0
        )

    sentence_features = [
        np.log1p(mean_sent_len),
        np.log1p(var_sent_len),
        burstiness
    ]
    return np.array(sentence_features + word_char_features, dtype=np.float64)