# src/data/synth_data.py

import ast
import json
import random
import unicodedata
import zlib
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Failed generation markers and invalid strings
FAILED_GENERATION_VALUES: Set[str] = {"generation_failed", "failed_generation"}
FAILED_VALIDATION_VALUES: Set[str] = {"validation_failed", "failed_validation"}
INVALID_SENTENCE_VALUES: Set[str] = {
    "nan", "none", "null", ""
} | FAILED_GENERATION_VALUES | FAILED_VALIDATION_VALUES


def is_valid_sentence(text: Any) -> bool:
    """Checks whether a text string is valid, non-empty, and free of sentinel error keywords."""
    if text is None:
        return False
    if isinstance(text, (float, np.floating)) and np.isnan(text):
        return False
    if isinstance(text, (list, tuple, np.ndarray)) and len(text) == 0:
        return False

    clean_str = str(text).strip()
    if not clean_str:
        return False

    clean_lower = clean_str.lower()
    if any(clean_lower.startswith(v) for v in FAILED_GENERATION_VALUES | FAILED_VALIDATION_VALUES):
        return False
    if clean_lower in INVALID_SENTENCE_VALUES:
        return False

    return True

def normalize_sentence(text: Any) -> str:
    """Normalizes Unicode NFKC and strips whitespace."""
    if not is_valid_sentence(text):
        return ""
    text_str = str(text)
    text_str = unicodedata.normalize("NFKC", text_str)
    return " ".join(text_str.split()).strip()


def parse_and_clean_sentence_array(raw_val: Any) -> List[str]:
    """
    Safely parses raw values (numpy arrays, lists, JSON strings, stringified python arrays)
    and removes failed sentinel values.
    """
    if raw_val is None:
        return []
    if isinstance(raw_val, (float, np.floating)) and np.isnan(raw_val):
        return []

    parsed_list: List[Any] = []

    if isinstance(raw_val, (list, tuple, np.ndarray)):
        parsed_list = list(raw_val)
    elif isinstance(raw_val, str):
        val_str = raw_val.strip()
        if val_str.startswith("[") and val_str.endswith("]"):
            try:
                parsed = ast.literal_eval(val_str)
                parsed_list = list(parsed) if isinstance(parsed, (list, tuple, np.ndarray)) else [str(parsed)]
            except (ValueError, SyntaxError):
                try:
                    parsed = json.loads(val_str)
                    parsed_list = list(parsed) if isinstance(parsed, (list, tuple, np.ndarray)) else [str(parsed)]
                except Exception:
                    parsed_list = [val_str]
        elif val_str:
            parsed_list = [val_str]

    # Filter and clean sentences
    cleaned_sentences = [
        normalize_sentence(s)
        for s in parsed_list
        if is_valid_sentence(s)
    ]
    return [s for s in cleaned_sentences if s]


def extract_valid_single_models(
    row: pd.Series,
    min_sentences: int = 4
) -> Dict[str, List[str]]:
    """
    Extracts and validates sentence arrays for all model '_single' columns present in a row.
    Returns a dict mapping model_name -> list of clean sentences.
    """
    valid_model_sents: Dict[str, List[str]] = {}

    for col in row.index:
        if not col.endswith("_single") and not col.endswith("_sentence"):
            continue
        if pd.isna(row[col]):
            continue

        clean_sents = parse_and_clean_sentence_array(row[col])
        if len(clean_sents) >= min_sentences:
            suffix = "_single" if col.endswith("_single") else "_sentence"
            model_name = col.rsplit(suffix, 1)[0]
            valid_model_sents[model_name] = clean_sents

    return valid_model_sents


def mix_abstract_at_ratio(
    human_sents: List[str],
    available_models: Dict[str, List[str]],
    target_ratio: float,
    seed: int
) -> Tuple[List[str], List[int], List[str], float]:
    """
    Substitutes target_ratio % of human sentences with corresponding LLM sentences from available models.

    Returns:
        mixed_sentences (List[str]): Reconstituted sentence list.
        sentence_labels (List[int]): 0 for human, 1 for LLM.
        sentence_models (List[str]): Author/model name for each sentence.
        actual_ratio (float): Actual substituted ratio achieved.
    """
    # Align to minimum available sentence length across models to prevent out-of-bounds or repetition
    min_len = min(len(human_sents), min(len(sents) for sents in available_models.values()))
    if min_len < 4:
        return [], [], [], 0.0

    h_aligned = human_sents[:min_len]

    # Number of sentences to replace (k), bounded to guarantee true partial mixture
    k = int(round(target_ratio * min_len))
    k = max(1, min(min_len - 1, k))

    rng = random.Random(seed)
    replace_indices = set(rng.sample(range(min_len), k))
    model_names = sorted(list(available_models.keys()))

    mixed_sents: List[str] = []
    sentence_labels: List[int] = []
    sentence_models: List[str] = []

    for i in range(min_len):
        if i in replace_indices:
            chosen_model = rng.choice(model_names)
            llm_sent = available_models[chosen_model][i]
            mixed_sents.append(llm_sent)
            sentence_labels.append(1)
            sentence_models.append(chosen_model)
        else:
            mixed_sents.append(h_aligned[i])
            sentence_labels.append(0)
            sentence_models.append("human")

    actual_ratio = k / min_len
    return mixed_sents, sentence_labels, sentence_models, actual_ratio


def generate_synthetic_rows_for_doc(
    row: pd.Series,
    target_ratios: List[float] = [0.25, 0.50, 0.75],
    seed: int = 42,
    min_sentences: int = 4
) -> List[Dict[str, Any]]:
    doc_id = str(row.get("_id", row.get("doc_id", row.get("id", "unknown"))))
    split = row.get("split", "test")
    source = row.get("source", "unknown")
    keywords = row.get("keywords", None)
    year = row.get("year", None)

    raw_human_sents = row.get("abstract_sentence") or row.get("abstract_sentences") or []
    human_sents = parse_and_clean_sentence_array(raw_human_sents)
    if len(human_sents) < min_sentences:
        return []

    valid_models = extract_valid_single_models(row, min_sentences=min_sentences)
    if not valid_models:
        return []

    synthetic_rows = []
    for ratio in target_ratios:
        seed_str = f"{seed}_{doc_id}_{ratio}"
        pair_seed = zlib.crc32(seed_str.encode("utf-8"))

        mixed_sents, s_labels, s_models, actual_ratio = mix_abstract_at_ratio(
            human_sents=human_sents,
            available_models=valid_models,
            target_ratio=ratio,
            seed=pair_seed
        )

        if not mixed_sents:
            continue

        reconstituted_text = " ".join(mixed_sents)

        synthetic_rows.append({
            "_id": doc_id,
            "source": source,
            "keywords": keywords,
            "year": year,
            "split": split,
            "text": reconstituted_text,
            "label": 1,
            "llm_ratio": actual_ratio,
            "target_ratio": ratio,
            "model_name": "synthetic_multi",
            "scope": "full",
            "generation_type": "synthetic_partial",
            "sentences": mixed_sents,
            "sentence_labels": s_labels,
            "sentence_models": s_models,
            "num_sentences": len(mixed_sents)
        })

    return synthetic_rows