# src/data/preprocess_data.py

import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Project-relative pathing
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_PATH = PROJECT_ROOT / "data_static" / "raw" / "llm_added.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data_static" / "preprocessed"

# Import modularized synthetic data generator and sentence parser
from src.data.synth_data import (
    generate_synthetic_rows_for_doc,
    parse_and_clean_sentence_array,
    is_valid_sentence as synth_is_valid_sentence
)

NON_PRINTABLE_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
RE_MD_IMG = re.compile(r"!\[(.*?)\]\(.*?\)")
RE_MD_LINK = re.compile(r"\[(.*?)\]\(.*?\)")
RE_MD_BOLD = re.compile(r"(\*\*|__)(.*?)\1")
RE_MD_ITALIC = re.compile(r"(\*|_)(.*?)\1")
RE_MD_STRIKE = re.compile(r"(~~)(.*?)\1")
RE_MD_CODE = re.compile(r"(`)(.*?)\1")
RE_MD_HEADER = re.compile(r"^\s*[#>]+\s+", flags=re.MULTILINE)
RE_MD_HR = re.compile(r"^\s*[-*_]{3,}\s*$", flags=re.MULTILINE)

FAILED_GENERATION_VALUES: Set[str] = {"generation_failed", "failed_generation"}
FAILED_VALIDATION_VALUES: Set[str] = {"validation_failed", "failed_validation"}
INVALID_SENTENCE_VALUES: Set[str] = {
    "nan", "none", "null", ""
} | FAILED_GENERATION_VALUES | FAILED_VALIDATION_VALUES


def is_none_or_nan(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, (list, tuple, np.ndarray)):
        return len(val) == 0
    if isinstance(val, (float, np.floating)):
        return np.isnan(val)
    try:
        return bool(pd.isna(val))
    except (ValueError, TypeError):
        return False


def strip_markdown(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = RE_MD_IMG.sub(r"\1", text)
    text = RE_MD_LINK.sub(r"\1", text)
    text = RE_MD_BOLD.sub(r"\2", text)
    text = RE_MD_ITALIC.sub(r"\2", text)
    text = RE_MD_STRIKE.sub(r"\2", text)
    text = RE_MD_CODE.sub(r"\2", text)
    text = RE_MD_HEADER.sub("", text)
    text = RE_MD_HR.sub("", text)
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


def normalize_text(text: Any) -> str:
    if is_none_or_nan(text) or isinstance(text, (list, tuple, np.ndarray)):
        return ""
    text_str = clean_html_markdown(str(text))
    text_str = unicodedata.normalize("NFKC", text_str)
    text_str = NON_PRINTABLE_RE.sub("", text_str)
    text_str = text_str.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    text_str = text_str.replace("—", "-").replace("–", "-")
    return " ".join(text_str.split()).strip()


def check_and_count_validity(text: Any, counts: Optional[Dict[str, int]] = None) -> bool:
    """Validates text while updating filter diagnostic metrics."""
    if is_none_or_nan(text) or isinstance(text, (list, tuple, np.ndarray)):
        if counts is not None:
            counts["null_or_empty"] += 1
        return False

    clean_str = normalize_text(text)
    if not clean_str:
        if counts is not None:
            counts["null_or_empty"] += 1
        return False

    clean_lower = clean_str.lower()
    if any(clean_lower.startswith(v) for v in FAILED_GENERATION_VALUES):
        if counts is not None:
            counts["failed_generation"] += 1
        return False

    if any(clean_lower.startswith(v) for v in FAILED_VALIDATION_VALUES):
        if counts is not None:
            counts["failed_validation"] += 1
        return False

    if clean_lower in INVALID_SENTENCE_VALUES:
        if counts is not None:
            counts["other_invalid"] += 1
        return False

    return True


def create_id_splits(
    df: pd.DataFrame,
    id_col: Optional[str] = None,
    stratify_col: Optional[str] = "source",
    train_ratio: float = 0.70,
    dev_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Dict[Any, str]:
    """
    Creates Group-Isolated AND Source-Stratified splits across unique document IDs.
    """
    if id_col is None or id_col not in df.columns:
        id_col = next((c for c in ["_id", "doc_id", "id"] if c in df.columns), df.columns[0])

    if stratify_col and stratify_col in df.columns:
        doc_meta = df[[id_col, stratify_col]].drop_duplicates(subset=[id_col]).reset_index(drop=True)
        unique_ids = doc_meta[id_col].values
        raw_strata = doc_meta[stratify_col].fillna("unknown").astype(str).values
        
        # Merge rare stratum categories (< 3 samples) into 'other' to prevent StratifiedKFold failures
        val_counts = pd.Series(raw_strata).value_counts()
        rare_classes = set(val_counts[val_counts < 3].index)
        strata = np.array(["other" if s in rare_classes else s for s in raw_strata])
    else:
        unique_ids = df[id_col].unique()
        strata = np.zeros(len(unique_ids))

    # First split: Train vs (Dev + Test)
    temp_ratio = dev_ratio + test_ratio
    n_splits_1 = max(2, int(round(1.0 / temp_ratio)))
    skf1 = StratifiedKFold(n_splits=n_splits_1, shuffle=True, random_state=random_state)
    train_idx, temp_idx = next(skf1.split(unique_ids, strata))

    train_ids = unique_ids[train_idx]
    temp_ids = unique_ids[temp_idx]
    temp_strata = strata[temp_idx]

    # Second split: Dev vs Test
    rel_test_ratio = test_ratio / (dev_ratio + test_ratio)
    n_splits_2 = max(2, int(round(1.0 / rel_test_ratio)))
    skf2 = StratifiedKFold(n_splits=n_splits_2, shuffle=True, random_state=random_state)
    dev_sub_idx, test_sub_idx = next(skf2.split(temp_ids, temp_strata))

    dev_ids = temp_ids[dev_sub_idx]
    test_ids = temp_ids[test_sub_idx]

    split_map = {}
    for doc_id in train_ids:
        split_map[doc_id] = "train"
    for doc_id in dev_ids:
        split_map[doc_id] = "dev"
    for doc_id in test_ids:
        split_map[doc_id] = "test"

    return split_map


def transform_to_long_format(raw_df: pd.DataFrame) -> pd.DataFrame:
    id_col = next((c for c in ["_id", "doc_id", "id"] if c in raw_df.columns), "_id")
    print(f"Assigning source-stratified group splits on '{id_col}'...")
    split_map = create_id_splits(raw_df, id_col=id_col, stratify_col="source")
    raw_df["split"] = raw_df[id_col].map(split_map)

    filter_counts = {
        "failed_generation": 0,
        "failed_validation": 0,
        "null_or_empty": 0,
        "other_invalid": 0
    }

    rows: List[Dict[str, Any]] = []
    meta_cols = [id_col, "_id", "source", "keywords", "year", "split"]
    human_cols = {"abstract", "abstract_sentence", "abstract_sentences", "abstract_full"}

    print("Processing, filtering sentinel values, and normalizing text...")
    for idx, row in raw_df.iterrows():
        if (idx + 1) % 500 == 0 or (idx + 1) == len(raw_df):
            print(f" Processing row {idx + 1}/{len(raw_df)}...", end="\r")

        row_split = row["split"]
        meta = {col: row[col] for col in meta_cols if col in row}
        if "_id" not in meta and id_col in meta:
            meta["_id"] = meta[id_col]

        # --- A. Pure Human Text ---
        human_full_val = row.get("abstract") or row.get("abstract_full")
        if human_full_val is not None and check_and_count_validity(human_full_val, counts=filter_counts):
            rows.append({
                **meta,
                "text": normalize_text(human_full_val),
                "label": 0,
                "llm_ratio": 0.0,
                "model_name": "human",
                "scope": "full",
                "generation_type": "human_full"
            })

        human_sent_val = row.get("abstract_sentence") or row.get("abstract_sentences")
        if human_sent_val is not None:
            human_sents = parse_and_clean_sentence_array(human_sent_val)
            for h_sent in human_sents:
                rows.append({
                    **meta,
                    "text": h_sent,
                    "label": 0,
                    "llm_ratio": 0.0,
                    "model_name": "human",
                    "scope": "sentence",
                    "generation_type": "human_sentence"
                })

        # --- B. Model Rewrites ---
        for col in raw_df.columns:
            if col in meta_cols or col in human_cols:
                continue

            raw_val = row[col]
            if is_none_or_nan(raw_val):
                continue

            if "_" not in col:
                continue

            model_name, suffix = col.rsplit("_", 1)

            if suffix in ["single", "sentence"]:
                clean_sents = parse_and_clean_sentence_array(raw_val)
                for s_text in clean_sents:
                    rows.append({
                        **meta,
                        "text": s_text,
                        "label": 1,
                        "llm_ratio": 1.0,
                        "model_name": model_name,
                        "scope": "sentence",
                        "generation_type": "sentence_rewrite"
                    })
            elif suffix == "full":
                if check_and_count_validity(raw_val, counts=filter_counts):
                    rows.append({
                        **meta,
                        "text": normalize_text(raw_val),
                        "label": 1,
                        "llm_ratio": 1.0,
                        "model_name": model_name,
                        "scope": "full",
                        "generation_type": "full_rewrite"
                    })
            elif suffix in ["25", "50", "75"]:
                if row_split == "test":
                    if check_and_count_validity(raw_val, counts=filter_counts):
                        rows.append({
                            **meta,
                            "text": normalize_text(raw_val),
                            "label": 1,
                            "llm_ratio": float(suffix) / 100.0,
                            "model_name": model_name,
                            "scope": "full",
                            "generation_type": "prompt_partial"
                        })

        # --- C. Synthetic Partial Mixes ---
        if row_split == "test":
            synth_rows = generate_synthetic_rows_for_doc(
                row=row,
                target_ratios=[0.25, 0.50, 0.75],
                seed=42,
                min_sentences=4
            )
            rows.extend(synth_rows)

    print("\n\n" + "=" * 45)
    print("          FILTERED DATA SUMMARY          ")
    print("=" * 45)
    print(f"  Failed Generation Filtered : {filter_counts['failed_generation']:,}")
    print(f"  Failed Validation Filtered : {filter_counts['failed_validation']:,}")
    print(f"  Null / Empty Filtered      : {filter_counts['null_or_empty']:,}")
    print(f"  Other Invalid Keywords     : {filter_counts['other_invalid']:,}")
    print("=" * 45)
    print(f"  TOTAL FILTERED ITEMS       : {sum(filter_counts.values()):,}")
    print("=" * 45 + "\n")

    long_df = pd.DataFrame(rows)
    return long_df


def main():
    print(f"Reading raw parquet data from: {INPUT_PATH}")
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found at: {INPUT_PATH}")

    raw_df = pd.read_parquet(INPUT_PATH)
    long_df = transform_to_long_format(raw_df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    parquet_out = OUTPUT_DIR / "preprocessed_dataset.parquet"
    csv_out = OUTPUT_DIR / "preprocessed_dataset.csv"

    print(f"Saving preprocessed dataset to {parquet_out}...")
    long_df.to_parquet(parquet_out, index=False)
    long_df.to_csv(csv_out, index=False)
    print("=== Preprocessing Complete ===")


if __name__ == "__main__":
    main()