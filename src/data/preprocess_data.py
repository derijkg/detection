# detection/src/data/preprocess_data.py

import os
import re
import ast
import json
import zlib
import random
import unicodedata
from pathlib import Path
from typing import Any, List, Dict, Tuple, Set, Optional

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# Project-relative pathing
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_PATH = PROJECT_ROOT / "data_static" / "raw" / "llm_added.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data_static" / "preprocessed"

NON_PRINTABLE_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
INVALID_SENTENCE_VALUES: Set[str] = {"generation_failed", "validation_failed", "nan", "none", "null", ""}


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


def normalize_text(text: Any) -> str:
    if is_none_or_nan(text) or isinstance(text, (list, tuple, np.ndarray)):
        return ""
    text_str = str(text)
    text_str = unicodedata.normalize("NFKC", text_str)
    text_str = NON_PRINTABLE_RE.sub("", text_str)
    return text_str.strip()


def is_valid_sentence(text: Any) -> bool:
    if is_none_or_nan(text) or isinstance(text, (list, tuple, np.ndarray)):
        return False
    clean_str = normalize_text(text)
    if not clean_str or clean_str.lower() in INVALID_SENTENCE_VALUES:
        return False
    return True


def parse_and_clean_sentence_array(raw_val: Any) -> List[str]:
    if is_none_or_nan(raw_val):
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

    cleaned = []
    for item in parsed_list:
        if is_valid_sentence(item):
            cleaned.append(normalize_text(item))
    return cleaned


def create_id_splits(df: pd.DataFrame, id_col: str = '_id', train_ratio: float = 0.7, 
                     dev_ratio: float = 0.15, test_ratio: float = 0.15, random_state: int = 42) -> Dict[Any, str]:
    unique_ids = df[id_col].unique()
    gss1 = GroupShuffleSplit(n_splits=1, test_size=(dev_ratio + test_ratio), random_state=random_state)
    train_idx, temp_idx = next(gss1.split(unique_ids, groups=unique_ids))
    train_ids, temp_ids = unique_ids[train_idx], unique_ids[temp_idx]
    
    relative_test_ratio = test_ratio / (dev_ratio + test_ratio)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_test_ratio, random_state=random_state)
    dev_sub_idx, test_sub_idx = next(gss2.split(temp_ids, groups=temp_ids))
    
    dev_ids, test_ids = temp_ids[dev_sub_idx], temp_ids[test_sub_idx]
    
    split_map = {}
    for _id in train_ids: split_map[_id] = 'train'
    for _id in dev_ids: split_map[_id] = 'dev'
    for _id in test_ids: split_map[_id] = 'test'
    return split_map


def mix_abstract_at_ratio(human_sents: List[str], available_models: Dict[str, List[str]], 
                         target_ratio: float, seed: int) -> Tuple[str, float]:
    n_sentences = len(human_sents)
    k = max(1, min(n_sentences - 1, int(round(target_ratio * n_sentences))))
    rng = random.Random(seed)
    replace_indices = set(rng.sample(range(n_sentences), k))
    model_names = list(available_models.keys())

    mixed_sents: List[str] = []
    for i in range(n_sentences):
        if i in replace_indices:
            chosen_model = rng.choice(model_names)
            model_sents = available_models[chosen_model]
            llm_sent = model_sents[i] if i < len(model_sents) else model_sents[i % len(model_sents)]
            mixed_sents.append(llm_sent)
        else:
            mixed_sents.append(human_sents[i])

    actual_ratio = k / n_sentences if n_sentences > 0 else 0.0
    return " ".join(mixed_sents), actual_ratio


def generate_synthetic_rows_for_row(row: pd.Series, split_map: Dict[Any, str], 
                                     target_ratios: List[float] = [0.25, 0.50, 0.75], 
                                     seed: int = 42) -> List[Dict[str, Any]]:
    doc_id = row['_id']
    row_split = split_map.get(doc_id, 'train')

    if row_split != 'test':
        return []

    meta = {
        '_id': doc_id,
        'source': row.get('source', 'unknown'),
        'keywords': row.get('keywords', None),
        'year': row.get('year', None),
        'split': row_split
    }

    human_sents = parse_and_clean_sentence_array(row.get('abstract_sentence', []))
    if len(human_sents) < 3:
        return []

    valid_models: Dict[str, List[str]] = {}
    for col in row.index:
        if col.endswith('_single'):
            clean_sents = parse_and_clean_sentence_array(row[col])
            if clean_sents:
                model_name = col.rsplit('_single', 1)[0]
                valid_models[model_name] = clean_sents

    if not valid_models:
        return []

    synthetic_rows = []
    for ratio in target_ratios:
        seed_str = f"{seed}_{doc_id}_{ratio}"
        pair_seed = zlib.crc32(seed_str.encode("utf-8"))

        reconstituted_text, actual_ratio = mix_abstract_at_ratio(
            human_sents=human_sents,
            available_models=valid_models,
            target_ratio=ratio,
            seed=pair_seed
        )

        synthetic_rows.append({
            **meta,
            'text': reconstituted_text,
            'label': 1,
            'llm_ratio': actual_ratio,
            'model_name': 'synthetic_multi',
            'scope': 'full',
            'generation_type': 'synthetic_partial'
        })

    return synthetic_rows


def transform_to_long_format(raw_df: pd.DataFrame) -> pd.DataFrame:
    print("Assigning group-stratified splits on '_id'...")
    split_map = create_id_splits(raw_df, id_col='_id')
    raw_df['split'] = raw_df['_id'].map(split_map)
    
    rows: List[Dict[str, Any]] = []
    meta_cols = ['_id', 'source', 'keywords', 'year', 'split']
    
    print("Processing and early-normalizing dataset...")
    for idx, row in raw_df.iterrows():
        if (idx + 1) % 500 == 0 or (idx + 1) == len(raw_df):
            print(f" Processing row {idx + 1}/{len(raw_df)}...", end='\r')

        row_split = row['split']
        meta = {col: row[col] for col in meta_cols if col in row}
        
        # --- A. Pure Human Text ---
        if 'abstract' in row and is_valid_sentence(row['abstract']):
            rows.append({
                **meta, 'text': normalize_text(row['abstract']),
                'label': 0, 'llm_ratio': 0.0, 'model_name': 'human',
                'scope': 'full', 'generation_type': 'human_full'
            })

        if 'abstract_sentence' in row:
            human_sents = parse_and_clean_sentence_array(row['abstract_sentence'])
            for h_sent in human_sents:
                rows.append({
                    **meta, 'text': h_sent,
                    'label': 0, 'llm_ratio': 0.0, 'model_name': 'human',
                    'scope': 'single', 'generation_type': 'human_single'
                })

        # --- B. Model Rewrites ---
        for col in raw_df.columns:
            if col in meta_cols or col in ['abstract', 'abstract_sentence']:
                continue
            
            raw_val = row[col]
            if is_none_or_nan(raw_val):
                continue

            model_name, suffix = col.rsplit('_', 1)
            
            if suffix == 'single':
                clean_sents = parse_and_clean_sentence_array(raw_val)
                for s_text in clean_sents:
                    rows.append({
                        **meta, 'text': s_text,
                        'label': 1, 'llm_ratio': 1.0, 'model_name': model_name,
                        'scope': 'single', 'generation_type': 'single_rewrite'
                    })
            elif suffix == 'full':
                if is_valid_sentence(raw_val):
                    rows.append({
                        **meta, 'text': normalize_text(raw_val),
                        'label': 1, 'llm_ratio': 1.0, 'model_name': model_name,
                        'scope': 'full', 'generation_type': 'full_rewrite'
                    })
            elif suffix in ['25', '50', '75']:
                if row_split == 'test':
                    if is_valid_sentence(raw_val):
                        rows.append({
                            **meta, 'text': normalize_text(raw_val),
                            'label': 1, 'llm_ratio': float(suffix) / 100.0,
                            'model_name': model_name, 'scope': 'full',
                            'generation_type': 'prompt_partial'
                        })

        # --- C. Synthetic Partial Mixes ---
        if row_split == 'test':
            synth_rows = generate_synthetic_rows_for_row(row, split_map)
            rows.extend(synth_rows)

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