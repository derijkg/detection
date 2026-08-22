# src/data/preprocess_data.py
import ast
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.data.stylometrics import normalize_text, is_none_or_nan
from src.data.synth_data import (
    FAILED_GENERATION_VALUES,
    FAILED_VALIDATION_VALUES,
    INVALID_SENTENCE_VALUES,
    generate_synthetic_rows_for_doc,
    parse_and_clean_sentence_array,
    is_valid_sentence
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_PATH = PROJECT_ROOT / 'data_static' / 'raw' / 'llm_added.parquet'
OUTPUT_DIR = PROJECT_ROOT / 'data_static' / 'preprocessed'


def classify_and_count_item(val: Any, counts: Optional[Dict[str, int]] = None) -> bool:
    """
    Validates a cell value. If invalid, increments the appropriate counter category.
    Returns True if valid, False if it should be filtered.
    """
    if val is None:
        if counts is not None: counts['null_or_empty'] += 1
        return False
    if isinstance(val, (float, np.floating)) and np.isnan(val):
        if counts is not None: counts['null_or_empty'] += 1
        return False
    if isinstance(val, (list, tuple, np.ndarray)):
        if len(val) == 0:
            if counts is not None: counts['null_or_empty'] += 1
            return False
        return True
    if isinstance(val, str):
        s = val.strip()
        if len(s) < 3:
            if counts is not None: counts['null_or_empty'] += 1
            return False
        s_lower = s.lower()
        if any(s_lower.startswith(v) for v in FAILED_GENERATION_VALUES):
            if counts is not None: counts['failed_generation'] += 1
            return False
        if any(s_lower.startswith(v) for v in FAILED_VALIDATION_VALUES):
            if counts is not None: counts['failed_validation'] += 1
            return False
        if s_lower in INVALID_SENTENCE_VALUES:
            if counts is not None: counts['other_invalid'] += 1
            return False
        return True

    return True


def get_first_valid_content(row: Any, candidate_cols: List[str]) -> Any:
    """Safely retrieves the first non-empty column content from a list of candidate column names."""
    for col in candidate_cols:
        if col in row:
            val = row[col]
            if classify_and_count_item(val):
                return val
    return None


def extract_meaningful_keywords(val: Any) -> List[str]:
    """Inspects and extracts non-empty, clean keyword strings from arrays, lists, or serialized strings."""
    if val is None or is_none_or_nan(val):
        return []
    
    parsed_items: List[Any] = []
    if isinstance(val, (list, tuple, np.ndarray)):
        parsed_items = list(val)
    elif isinstance(val, str):
        s = val.strip()
        if s.startswith('[') and s.endswith(']'):
            try:
                parsed = ast.literal_eval(s)
                parsed_items = list(parsed) if isinstance(parsed, (list, tuple, np.ndarray)) else [str(parsed)]
            except Exception:
                try:
                    parsed = json.loads(s)
                    parsed_items = list(parsed) if isinstance(parsed, (list, tuple, np.ndarray)) else [str(parsed)]
                except Exception:
                    parsed_items = [s]
        elif s:
            parsed_items = [item.strip() for item in s.split(',') if item.strip()]

    clean_keywords = []
    for item in parsed_items:
        if item is not None and not is_none_or_nan(item):
            k_str = str(item).strip()
            if len(k_str) >= 2 and k_str.lower() not in INVALID_SENTENCE_VALUES:
                clean_keywords.append(k_str)

    return clean_keywords


def get_year_era_bin(year_val: Any) -> str:
    """Bins skewed publication years into balanced chronological eras."""
    if is_none_or_nan(year_val):
        return "year_unknown"
    try:
        y = int(float(year_val))
        if y < 2016:
            return "year_<2016"
        elif y <= 2018:
            return "year_2016-2018"
        elif y <= 2020:
            return "year_2019-2020"
        elif y <= 2022:
            return "year_2021-2022"
        else:
            return "year_2023+"
    except Exception:
        return "year_unknown"


def analyze_raw_parquet(df: pd.DataFrame, path: Path):
    """Prints a thorough architectural breakdown and raw sentinel audit before processing."""
    id_col = next((c for c in ['_id', 'doc_id', 'id'] if c in df.columns), df.columns[0])
    total_rows = len(df)
    unique_ids = df[id_col].nunique() if id_col in df.columns else total_rows
    mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

    meta_and_human = {'_id', 'id', 'doc_id', 'source', 'keywords', 'year', 'split',
                      'abstract', 'abstract_sentence', 'abstract_sentences', 'abstract_full'}
    gen_cols = [c for c in df.columns if c not in meta_and_human and '_' in c]
    models_detected = sorted(list({c.rsplit('_', 1)[0] for c in gen_cols}))

    print("\n" + "=" * 80)
    print("            [STAGE 0] INITIAL RAW PARQUET FILE ANALYSIS            ")
    print("=" * 80)
    print(f" File Location            : {path}")
    print(f" Total Rows / Documents   : {total_rows:,}")
    print(f" Unique Document ID Key   : '{id_col}' ({unique_ids:,} unique IDs)")
    print(f" Total Schema Columns     : {len(df.columns)}")
    print(f" Total Memory Footprint   : {mem_mb:.2f} MB")

    # 1. Metadata Columns Inspection
    print("\n--- METADATA & HUMAN BASELINE SUMMARY ---")
    if 'source' in df.columns:
        src_counts = df['source'].value_counts(dropna=False).to_dict()
        src_str = ", ".join([f"{k}: {v:,}" for k, v in src_counts.items()])
        print(f" * 'source': {len(src_counts)} categories -> ({src_str})")

    if 'year' in df.columns:
        valid_years = pd.to_numeric(df['year'], errors='coerce').dropna()
        if not valid_years.empty:
            print(f" * 'year': Range [{int(valid_years.min())} - {int(valid_years.max())}] | Median: {int(valid_years.median())} | Valid: {len(valid_years):,}/{total_rows:,}")
            era_counts = df['year'].apply(get_year_era_bin).value_counts().to_dict()
            era_str = ", ".join([f"{k}: {v:,}" for k, v in sorted(era_counts.items())])
            print(f"   -> Binned Eras: ({era_str})")

    if 'keywords' in df.columns:
        kw_counts = [len(extract_meaningful_keywords(x)) for x in df['keywords']]
        with_kw = sum(1 for c in kw_counts if c > 0)
        avg_kw = np.mean([c for c in kw_counts if c > 0]) if with_kw > 0 else 0
        print(f" * 'keywords' (Deep Inspection):")
        print(f"   - Documents with >= 1 non-empty keyword : {with_kw:,}/{total_rows:,} ({with_kw/total_rows*100:.1f}%)")
        print(f"   - Documents with 0 / empty keywords     : {total_rows - with_kw:,} ({(total_rows - with_kw)/total_rows*100:.1f}%)")
        print(f"   - Avg keywords per populated document   : {avg_kw:.1f}")

    human_full_col = next((c for c in ['abstract', 'abstract_full'] if c in df.columns), None)
    human_sent_col = next((c for c in ['abstract_sentence', 'abstract_sentences'] if c in df.columns), None)

    if human_full_col:
        valid_full = sum(classify_and_count_item(x) for x in df[human_full_col])
        print(f" * Human Full Abstracts ('{human_full_col}'): {valid_full:,}/{total_rows:,} valid ({valid_full/total_rows*100:.1f}%)")
    if human_sent_col:
        valid_sents = sum(classify_and_count_item(x) for x in df[human_sent_col])
        print(f" * Human Sentences ('{human_sent_col}'): {valid_sents:,}/{total_rows:,} valid ({valid_sents/total_rows*100:.1f}%)")

    # 2. Generator Availability Matrix
    print("\n--- GENERATOR MODEL COLUMNS & AVAILABILITY MATRIX ---")
    print(f"{'Generator Model':<20} | {'Full Abstract':<16} | {'Sentence List':<16} | {'Partials (25/50/75)':<20}")
    print("-" * 80)
    for model in models_detected:
        full_col = f"{model}_full"
        sent_col = next((f"{model}_{s}" for s in ['single', 'sentence'] if f"{model}_{s}" in df.columns), None)
        p25_col = f"{model}_25"
        p50_col = f"{model}_50"
        p75_col = f"{model}_75"

        n_full = sum(classify_and_count_item(x) for x in df[full_col]) if full_col in df.columns else 0
        n_sent = sum(classify_and_count_item(x) for x in df[sent_col]) if sent_col and sent_col in df.columns else 0
        
        n_p25 = sum(classify_and_count_item(x) for x in df[p25_col]) if p25_col in df.columns else 0
        n_p50 = sum(classify_and_count_item(x) for x in df[p50_col]) if p50_col in df.columns else 0
        n_p75 = sum(classify_and_count_item(x) for x in df[p75_col]) if p75_col in df.columns else 0

        full_str = f"{n_full:,} ({n_full/total_rows*100:.1f}%)" if full_col in df.columns else "[MISSING]"
        sent_str = f"{n_sent:,} ({n_sent/total_rows*100:.1f}%)" if sent_col else "[MISSING]"
        part_str = f"{n_p25:,} / {n_p50:,} / {n_p75:,}" if p25_col in df.columns else "[MISSING]"

        print(f"{model:<20} | {full_str:<16} | {sent_str:<16} | {part_str:<20}")

    # 3. Raw Sentinel Audit
    print("\n--- RAW PARQUET SENTINEL & FAILURE AUDIT ---")
    raw_sentinel_counts = {'failed_generation': 0, 'failed_validation': 0, 'null_or_empty': 0, 'other_invalid': 0}
    for col in gen_cols:
        for val in df[col]:
            classify_and_count_item(val, counts=raw_sentinel_counts)
    print(f" * 'failed_generation' tokens detected : {raw_sentinel_counts['failed_generation']:,}")
    print(f" * 'failed_validation' tokens detected : {raw_sentinel_counts['failed_validation']:,}")
    print(f" * Null / None / Empty cells detected  : {raw_sentinel_counts['null_or_empty']:,}")
    print(f" * Other invalid keywords detected     : {raw_sentinel_counts['other_invalid']:,}")
    print("=" * 80 + "\n")


def create_id_splits(
    df: pd.DataFrame,
    id_col: Optional[str] = None,
    train_ratio: float = 0.7,
    dev_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Dict[Any, str]:
    """
    Creates disjoint document-level splits stratified by Source + Year Era + Generator Signature.
    """
    if id_col is None or id_col not in df.columns:
        id_col = next((c for c in ['_id', 'doc_id', 'id'] if c in df.columns), df.columns[0])

    meta_and_human = {'_id', 'id', 'doc_id', 'source', 'keywords', 'year', 'split',
                      'abstract', 'abstract_sentence', 'abstract_sentences', 'abstract_full'}
    gen_cols = [c for c in df.columns if c not in meta_and_human and '_' in c]

    doc_records = []
    for doc_id, group in df.groupby(id_col):
        first_row = group.iloc[0]
        src = str(first_row.get('source', 'unknown'))
        year_era = get_year_era_bin(first_row.get('year'))
        
        available_models = sorted(list({
            c.rsplit('_', 1)[0] for c in gen_cols
            if classify_and_count_item(first_row.get(c))
        }))
        gen_sig = "+".join(available_models) if available_models else "none"
        
        composite_stratum = f"{src}___{year_era}___{gen_sig}"
        doc_records.append({'doc_id': doc_id, 'stratum': composite_stratum})

    doc_meta = pd.DataFrame(doc_records)
    unique_ids = doc_meta['doc_id'].values
    raw_strata = doc_meta['stratum'].values

    # Collapse rare strata (< 3 occurrences)
    val_counts = pd.Series(raw_strata).value_counts()
    rare_classes = set(val_counts[val_counts < 3].index)
    strata = np.array(['other' if s in rare_classes else s for s in raw_strata])

    # Stage 1: Split Train (70%) vs Temp (30%)
    temp_ratio = dev_ratio + test_ratio
    n_splits_1 = max(2, int(round(1.0 / temp_ratio)))
    skf1 = StratifiedKFold(n_splits=n_splits_1, shuffle=True, random_state=random_state)
    train_idx, temp_idx = next(skf1.split(unique_ids, strata))
    train_ids = unique_ids[train_idx]
    temp_ids = unique_ids[temp_idx]
    temp_strata = strata[temp_idx]

    # Stage 2: Split Temp into Dev (15%) and Test (15%)
    rel_test_ratio = test_ratio / (dev_ratio + test_ratio)
    n_splits_2 = max(2, int(round(1.0 / rel_test_ratio)))

    temp_val_counts = pd.Series(temp_strata).value_counts()
    temp_rare = set(temp_val_counts[temp_val_counts < n_splits_2].index)
    if temp_rare:
        temp_strata = np.array(['other' if s in temp_rare else s for s in temp_strata])
        if pd.Series(temp_strata).value_counts().min() < n_splits_2:
            temp_strata = np.zeros(len(temp_ids))

    skf2 = StratifiedKFold(n_splits=n_splits_2, shuffle=True, random_state=random_state)
    dev_sub_idx, test_sub_idx = next(skf2.split(temp_ids, temp_strata))
    dev_ids = temp_ids[dev_sub_idx]
    test_ids = temp_ids[test_sub_idx]

    split_map = {}
    for d in train_ids:
        split_map[d] = 'train'
    for d in dev_ids:
        split_map[d] = 'dev'
    for d in test_ids:
        split_map[d] = 'test'

    return split_map


def transform_to_long_format(raw_df: pd.DataFrame) -> pd.DataFrame:
    id_col = next((c for c in ['_id', 'doc_id', 'id'] if c in raw_df.columns), '_id')
    print(f"[1/3] Assigning composite (Source + Era + Generator) stratified group splits on '{id_col}'...")
    split_map = create_id_splits(raw_df, id_col=id_col)
    raw_df['split'] = raw_df[id_col].map(split_map)

    filter_counts = {'failed_generation': 0, 'failed_validation': 0, 'null_or_empty': 0, 'other_invalid': 0}
    rows: List[Dict[str, Any]] = []
    meta_cols = [id_col, '_id', 'source', 'keywords', 'year', 'split']
    human_cols = {'abstract', 'abstract_sentence', 'abstract_sentences', 'abstract_full'}

    print("[2/3] Processing, filtering sentinel values, and normalizing text...")
    for idx, row in raw_df.iterrows():
        if (idx + 1) % 500 == 0 or idx + 1 == len(raw_df):
            print(f" Processing row {idx + 1:,}/{len(raw_df):,}...", end='\r')

        row_split = row['split']
        meta = {col: row[col] for col in meta_cols if col in row}
        if '_id' not in meta and id_col in meta:
            meta['_id'] = meta[id_col]

        # 1. Pure Human Full Abstract (Min 25 characters)
        human_full_val = get_first_valid_content(row, ['abstract', 'abstract_full'])
        if human_full_val is not None:
            clean_human_full = normalize_text(human_full_val)
            if len(clean_human_full) >= 25:
                rows.append({
                    **meta,
                    'text': clean_human_full,
                    'label': 0,
                    'llm_ratio': 0.0,
                    'model_name': 'human',
                    'scope': 'full',
                    'generation_type': 'human_full'
                })

        # 2. Pure Human Sentences (Min 3 characters)
        human_sent_val = get_first_valid_content(row, ['abstract_sentence', 'abstract_sentences'])
        if human_sent_val is not None:
            human_sents = parse_and_clean_sentence_array(human_sent_val)
            for h_sent in human_sents:
                if len(h_sent) >= 3:
                    rows.append({
                        **meta,
                        'text': h_sent,
                        'label': 0,
                        'llm_ratio': 0.0,
                        'model_name': 'human',
                        'scope': 'sentence',
                        'generation_type': 'human_sentence'
                    })

        # 3. Model Generations (Full, Sentences, Prompt Partials)
        for col in raw_df.columns:
            if col in meta_cols or col in human_cols or '_' not in col:
                continue
            
            raw_val = row[col]
            model_name, suffix = col.rsplit('_', 1)

            if not classify_and_count_item(raw_val, counts=filter_counts):
                continue

            if suffix in ['single', 'sentence']:
                clean_sents = parse_and_clean_sentence_array(raw_val)
                for s_text in clean_sents:
                    if len(s_text) >= 3:
                        rows.append({
                            **meta,
                            'text': s_text,
                            'label': 1,
                            'llm_ratio': 1.0,
                            'model_name': model_name,
                            'scope': 'sentence',
                            'generation_type': 'sentence_rewrite'
                        })
            elif suffix == 'full':
                norm_full = normalize_text(raw_val)
                if len(norm_full) >= 25:
                    rows.append({
                        **meta,
                        'text': norm_full,
                        'label': 1,
                        'llm_ratio': 1.0,
                        'model_name': model_name,
                        'scope': 'full',
                        'generation_type': 'full_rewrite'
                    })
            elif suffix in ['25', '50', '75']:
                if row_split == 'test':
                    norm_partial = normalize_text(raw_val)
                    if len(norm_partial) >= 25:
                        rows.append({
                            **meta,
                            'text': norm_partial,
                            'label': 1,
                            'llm_ratio': float(suffix) / 100.0,
                            'model_name': model_name,
                            'scope': 'full',
                            'generation_type': 'prompt_partial'
                        })

        # 4. Synthetic Multi-Model Mixtures (Test Split Only)
        if row_split == 'test':
            synth_rows = generate_synthetic_rows_for_doc(
                row=row,
                target_ratios=[0.25, 0.5, 0.75],
                seed=42,
                min_sentences=4
            )
            rows.extend(synth_rows)

    print("\n\n" + "=" * 50)
    print("             FILTERED DATA SUMMARY            ")
    print("=" * 50)
    print(f"  Failed Generation Filtered : {filter_counts['failed_generation']:,}")
    print(f"  Failed Validation Filtered : {filter_counts['failed_validation']:,}")
    print(f"  Null / Empty Filtered      : {filter_counts['null_or_empty']:,}")
    print(f"  Other Invalid Keywords     : {filter_counts['other_invalid']:,}")
    print("=" * 50)
    print(f"  TOTAL FILTERED ITEMS       : {sum(filter_counts.values()):,}")
    print("=" * 50 + "\n")

    long_df = pd.DataFrame(rows)

    # Strict Deduplication of identical text per document, model, scope, and generation_type
    dedup_subset = ['_id', 'model_name', 'scope', 'text']
    before_dedup = len(long_df)
    long_df = long_df.drop_duplicates(subset=dedup_subset).reset_index(drop=True)
    dedup_removed = before_dedup - len(long_df)
    if dedup_removed > 0:
        print(f"[Deduplication] Removed {dedup_removed:,} exact duplicate text rows.")

    return long_df


def verify_preprocessed_dataset(df: pd.DataFrame):
    """Automated test suite verifying 7 strict scientific and data integrity invariants."""
    print("=" * 70)
    print("       RUNNING PREPROCESSED DATASET VERIFICATION SUITE       ")
    print("=" * 70)

    # 1. Test Zero Document ID Leakage Across Splits
    id_col = '_id' if '_id' in df.columns else 'doc_id'
    train_ids = set(df[df['split'] == 'train'][id_col].unique())
    dev_ids = set(df[df['split'] == 'dev'][id_col].unique())
    test_ids = set(df[df['split'] == 'test'][id_col].unique())

    overlap_tr_dv = train_ids.intersection(dev_ids)
    overlap_tr_ts = train_ids.intersection(test_ids)
    overlap_dv_ts = dev_ids.intersection(test_ids)

    assert len(overlap_tr_dv) == 0, f"[FAIL] Leakage detected: {len(overlap_tr_dv)} docs in Train and Dev!"
    assert len(overlap_tr_ts) == 0, f"[FAIL] Leakage detected: {len(overlap_tr_ts)} docs in Train and Test!"
    assert len(overlap_dv_ts) == 0, f"[FAIL] Leakage detected: {len(overlap_dv_ts)} docs in Dev and Test!"
    print(" [PASS] Test 1: Zero Document Leakage Across All Splits (Disjoint IDs).")

    # 2. Test Synthetic Abstracts Strict Test-Set Isolation
    synth_docs = df[df['model_name'] == 'synthetic_multi']
    synth_ids = set(synth_docs[id_col].unique())
    assert synth_ids.issubset(test_ids), "[FAIL] Synthetic abstracts were derived from non-Test documents!"
    assert (synth_docs['split'] == 'test').all(), "[FAIL] Synthetic abstracts found outside of 'test' split!"
    print(f" [PASS] Test 2: Synthetic Multi-Model Abstracts Strictly Derived from Test Split ({len(synth_docs):,} rows).")

    # 3. Test Zero Nulls, NaNs, or Empty Strings in Text
    assert df['text'].isna().sum() == 0, "[FAIL] NaN values found in 'text' column!"
    assert (df['text'].str.strip() == '').sum() == 0, "[FAIL] Empty text strings found in dataset!"
    
    clean_lower = df['text'].str.lower()
    for sentinel in FAILED_GENERATION_VALUES | FAILED_VALIDATION_VALUES:
        bad_count = clean_lower.str.startswith(sentinel).sum()
        assert bad_count == 0, f"[FAIL] Found {bad_count} unhandled sentinel values starting with '{sentinel}'!"
    print(" [PASS] Test 3: Zero Nulls, NaNs, Empty Strings, or Failure Sentinels in Final Dataset.")

    # 4. Test Binary Label & Ratio Consistency
    human_invalids = df[(df['label'] == 0) & ((df['llm_ratio'] != 0.0) | (df['model_name'] != 'human'))]
    assert len(human_invalids) == 0, f"[FAIL] {len(human_invalids)} human rows have inconsistent labels/ratios!"

    ai_invalids = df[(df['label'] == 1) & ((df['llm_ratio'] <= 0.0) | (df['model_name'] == 'human'))]
    assert len(ai_invalids) == 0, f"[FAIL] {len(ai_invalids)} AI rows have inconsistent labels/ratios!"
    print(" [PASS] Test 4: Binary Label and LLM Ratio Invariants Hold (100% Consistent).")

    # 5. Test Split Document Ratios (Target ~70/15/15)
    total_docs = len(train_ids) + len(dev_ids) + len(test_ids)
    pct_train = len(train_ids) / total_docs * 100
    pct_dev = len(dev_ids) / total_docs * 100
    pct_test = len(test_ids) / total_docs * 100

    print(f" [PASS] Test 5: Document Split Ratios -> Train: {pct_train:.1f}% | Dev: {pct_dev:.1f}% | Test: {pct_test:.1f}%")

    # 6. Generator Distribution Across Splits Table
    print("\n" + "-" * 70)
    print("     GENERATOR MODEL STRATIFICATION BREAKDOWN ACROSS SPLITS")
    print("-" * 70)
    ai_df = df[df['label'] == 1]
    gen_table = pd.crosstab(ai_df['model_name'], ai_df['split'], normalize='index') * 100
    gen_counts = pd.crosstab(ai_df['model_name'], ai_df['split'])
    
    summary_gen = pd.concat([gen_counts, gen_table.round(1)], axis=1, keys=['Sample Count', 'Split %'])
    print(summary_gen.to_string())
    print("-" * 70)

    # 7. Sample Sizes Summary Table
    print("\n" + "-" * 70)
    print("                 FINAL DATASET SAMPLE SIZES")
    print("-" * 70)
    breakdown = df.groupby(['split', 'scope', 'label']).size().unstack(fill_value=0)
    breakdown.columns = ['Human (0)', 'AI (1)']
    breakdown['Total Rows'] = breakdown['Human (0)'] + breakdown['AI (1)']
    breakdown['% AI'] = (breakdown['AI (1)'] / breakdown['Total Rows'] * 100).round(1).astype(str) + '%'
    print(breakdown.to_string())
    print("=" * 70 + "\n")


def main():
    print(f"Reading raw parquet data from: {INPUT_PATH}")
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found at: {INPUT_PATH}")
    raw_df = pd.read_parquet(INPUT_PATH)

    # Stage 0: Initial Raw Parquet Diagnostic Printout & Sentinel Audit
    analyze_raw_parquet(raw_df, INPUT_PATH)

    # Transform to long format with composite stratification & sentinel tracking
    long_df = transform_to_long_format(raw_df)

    # Execute automated verification suite before saving
    verify_preprocessed_dataset(long_df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    parquet_out = OUTPUT_DIR / 'preprocessed_dataset.parquet'
    csv_out = OUTPUT_DIR / 'preprocessed_dataset.csv'

    print(f"[3/3] Saving preprocessed dataset to {parquet_out}...")
    long_df.to_parquet(parquet_out, index=False)
    long_df.to_csv(csv_out, index=False)
    print("=== [SUCCESS] Preprocessing & Validation Complete ===")


if __name__ == '__main__':
    main()