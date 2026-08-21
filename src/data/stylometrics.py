# src/data/stylometrics.py

import re
import string
import unicodedata
from collections import Counter
from typing import Dict, List, Optional
import numpy as np
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
import warnings

warnings.filterwarnings('ignore', category=MarkupResemblesLocatorWarning)
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
        return ""
    text = clean_html_markdown(text)
    text = unicodedata.normalize('NFKC', text)
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    text = text.replace('“', '"').replace('”', '"').replace('’', "'").replace('‘', "'")
    text = text.replace('—', '-').replace('–', '-')
    return " ".join(text.split())


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
    """
    Extracts stylometric representation:
    - Sentence scope: 8 word- and character-level features.
    - Full scope: 11 features (3 structural + 8 word/character features).
    """
    words = RE_WORDS.findall(text.lower())
    clean_chars = len(text)
    raw_str = raw_text if raw_text is not None else text
    raw_total_chars = max(len(raw_str), 1)

    num_features = 8 if granularity == 'sentence' else 11
    if not words or clean_chars == 0:
        return np.zeros(num_features, dtype=np.float64)

    # 1. Word length distribution
    word_lengths = [len(w) for w in words]
    mean_word_len = float(np.mean(word_lengths))
    var_word_len = float(np.var(word_lengths))

    # 2. Vocabulary richness
    ttr = calculate_ttr(words)
    hapax_ratio = calculate_hapax_ratio(words)

    # 3. Discourse transition marker ratio
    transition_count = sum(1 for w in words if w in DUTCH_TRANSITIONS)
    transition_ratio = transition_count / len(words)

    # 4. Spacing and punctuation distributions (measured from uncollapsed text)
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

    if granularity == 'sentence':
        return np.array(word_char_features, dtype=np.float64)

    # 5. Multi-sentence structural features (Full Abstracts only)
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
        burstiness = (std_sent_len - mean_sent_len) / (std_sent_len + mean_sent_len) if (std_sent_len + mean_sent_len) > 0 else 0.0

    sentence_features = [
        np.log1p(mean_sent_len),
        np.log1p(var_sent_len),
        burstiness
    ]

    return np.array(sentence_features + word_char_features, dtype=np.float64)