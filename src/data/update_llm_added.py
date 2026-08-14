'''
TODO
verwijs naar andere repo /home/gderijck/internship/data/gold/checkpoint_rewrites.jsonl
check
'''

import argparse
import ast
import json
from pathlib import Path
import re
from collections import Counter
import numpy as np
import pandas as pd
import unicodedata
from bs4 import BeautifulSoup

# Define paths relative to this script
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_TABLE = BASE_DIR / 'data' / 'gold' / 'merged_publications.parquet'
CHECKPOINT_FILE = BASE_DIR / 'data' / 'gold' / 'checkpoint_rewrites.jsonl'
OUTPUT_PARQUET = BASE_DIR / 'data' / 'gold' / 'llm_added.parquet'
OUTPUT_CSV = BASE_DIR / 'data' / 'gold' / 'llm_added.csv'


def strip_markdown(text: str) -> str:
    """Strips standard Markdown formatting using regular expressions."""
    if not isinstance(text, str):
        return ""
    # Remove Markdown images: ![alt](url) -> keeps alt text
    text = re.sub(r'!\[(.*?)\]\(.*?\)', r'\1', text)
    # Remove Markdown links: [text](url) -> keeps text
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    # Remove emphasis/bold/strikethrough/inline code: **, __, *, _, ~~, `
    text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', text)
    text = re.sub(r'(\*|_)(.*?)\1', r'\2', text)
    text = re.sub(r'(~~)(.*?)\1', r'\2', text)
    text = re.sub(r'(`)(.*?)\1', r'\2', text)
    # Remove blockquotes and headers symbols
    text = re.sub(r'^\s*[#>]+\s+', '', text, flags=re.MULTILINE)
    # Remove horizontal rules
    text = re.sub(r'^\s*[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    return text


def clean_html_markdown(text: str) -> str:
    """Strips HTML tags and Markdown syntax from the text."""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # 1. Strip HTML using BeautifulSoup
    try:
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text(separator=" ")
    except Exception:
        pass
    
    # 2. Strip Markdown
    text = strip_markdown(text)
    return text


def normalize_text(text: str) -> str:
    """
    Applies HTML/Markdown stripping, NFKC normalization, standardizes smart 
    punctuation (quotes/dashes), and collapses duplicate whitespaces.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Strip HTML and Markdown
    text = clean_html_markdown(text)
    
    # 2. NFKC normalization (standardizes ligatures, accents, and symbol variants)
    text = unicodedata.normalize('NFKC', text)
    
    # 3. Convert curly/smart punctuation and non-standard dashes to standard ASCII
    text = text.replace('“', '"').replace('”', '"').replace('’', "'").replace('‘', "'")
    text = text.replace('—', '-').replace('–', '-')
    
    # 4. Collapse multiple whitespaces and convert carriage returns/tabs into standard spaces
    text = " ".join(text.split())
    
    return text


def clean_and_normalize_value(val):
    """Recursively cleans and normalizes strings, lists of strings, series, or arrays."""
    if isinstance(val, str):
        return normalize_text(val)
    elif isinstance(val, list):
        return [clean_and_normalize_value(v) for v in val]
    elif isinstance(val, np.ndarray):
        return np.array([clean_and_normalize_value(v) for v in val], dtype=object)
    elif isinstance(val, pd.Series):
        return val.apply(clean_and_normalize_value)
    return val


def format_list_for_csv(val) -> str:
    """Converts list-like structures into a single string joined by '|' for safe CSV writing."""
    if isinstance(val, (list, np.ndarray, pd.Series)):
        cleaned_elements = []
        for x in val:
            if x is None or (isinstance(x, float) and np.isnan(x)):
                cleaned_elements.append("")
            else:
                # Replace existing inner pipe characters with a space to prevent delimiter collisions
                cleaned_elements.append(str(x).replace('|', ' '))
        return "|".join(cleaned_elements)
    return val


def safe_literal_eval(val):
    """Safely converts string representations of lists/dicts back to Python objects."""
    if not isinstance(val, str):
        return val
    
    val_stripped = val.strip()
    if (val_stripped.startswith('[') and val_stripped.endswith(']')) or \
       (val_stripped.startswith('{') and val_stripped.endswith('}')):
        try:
            return ast.literal_eval(val_stripped)
        except (ValueError, SyntaxError):
            try:
                return json.loads(val_stripped)
            except json.JSONDecodeError:
                return val
    return val


def is_null_or_empty(val) -> bool:
    """Helper to check if a value is null/empty, safe for numpy arrays, lists, and scalars."""
    if val is None:
        return True
    if isinstance(val, (float, np.floating)) and np.isnan(val):
        return True
    if isinstance(val, (list, np.ndarray, pd.Series, dict)):
        return len(val) == 0
    return pd.isna(val) or val == ""


def is_already_applied(row: dict, task: dict) -> bool:
    """Checks if the rewrite in the task has already been applied to the row."""
    t_type = task["type"]
    model = task["model"]
    
    if t_type == "sentence":
        sent_idx = task["sent_idx"]
        col_name = f'{model}_single'
        if col_name not in row:
            return False
        
        val_list = row[col_name]
        if isinstance(val_list, str):
            val_list = safe_literal_eval(val_list)
            
        if not isinstance(val_list, (list, np.ndarray, pd.Series)):
            return False
            
        if sent_idx >= len(val_list):
            return False
            
        val = val_list[sent_idx]
        return not is_null_or_empty(val)
        
    elif t_type == "percentage":
        pct = task["percentage"]
        col_name = f"{model}_{pct}"
        if col_name not in row:
            return False
        val = row[col_name]
        return not is_null_or_empty(val)
        
    elif t_type == "full_abstract":
        col_name = f"{model}_full"
        if col_name not in row:
            return False
        val = row[col_name]
        return not is_null_or_empty(val)
        
    return False


def apply_rewrite_to_row(row: dict, task: dict, rewritten: str):
    """Applies the rewrite to the record dictionary safely."""
    if not row:
        return
    t_type = task["type"]
    model = task["model"]
    
    if t_type == "sentence":
        sent_idx = task["sent_idx"]
        
        abstract_sentence = row.get('abstract_sentence')
        if isinstance(abstract_sentence, str):
            abstract_sentence = safe_literal_eval(abstract_sentence)
        if not isinstance(abstract_sentence, (list, np.ndarray, pd.Series)):
            abstract_sentence = []
            
        num_sentences = len(abstract_sentence)
        col_name = f'{model}_single'
        
        val = row.get(col_name)
        if isinstance(val, str):
            val = safe_literal_eval(val)
            
        if not isinstance(val, (list, np.ndarray, pd.Series)):
            if hasattr(val, 'tolist'):
                val = val.tolist()
            else:
                val = [None] * num_sentences
        else:
            val = list(val)
        
        if len(val) < num_sentences:
            val = val + [None] * (num_sentences - len(val))
        elif len(val) > num_sentences:
            val = val[:num_sentences]
            
        if 0 <= sent_idx < num_sentences:
            val[sent_idx] = rewritten
            
        row[col_name] = val
        
    elif t_type == "percentage":
        pct = task["percentage"]
        row[f"{model}_{pct}"] = rewritten
        
    elif t_type == "full_abstract":
        row[f"{model}_full"] = rewritten


def clean_id(val) -> str:
    """Standardizes row IDs to string format for reliable mapping."""
    if pd.isna(val) or val is None:
        return ""
    if isinstance(val, float):
        if val.is_integer():
            return str(int(val))
        return str(val).strip()
    return str(val).strip()


def main():
    parser = argparse.ArgumentParser(description="Update dataset with checkpoint rewrites.")
    parser.add_argument("--input", type=Path, default=INPUT_TABLE, help="Path to base input table.")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_FILE, help="Path to checkpoints jsonl file.")
    parser.add_argument("--output-parquet", type=Path, default=OUTPUT_PARQUET, help="Path to output Parquet file.")
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV, help="Path to output CSV file.")
    args = parser.parse_args()

    # 1. Load the most current version of the dataset
    already_in_output = False
    if args.output_parquet.exists():
        print(f"Loading existing Parquet dataset: {args.output_parquet}")
        df = pd.read_parquet(args.output_parquet)
        already_in_output = True
    elif args.output_csv.exists():
        print(f"Loading existing CSV dataset: {args.output_csv}")
        df = pd.read_csv(args.output_csv)
        already_in_output = True
    else:
        print(f"No existing output found. Initializing from base table: {args.input}")
        df = pd.read_parquet(args.input)

    initial_row_count = len(df)

    # 2. Parse any serialized list structures in existing columns
    list_cols = ['abstract_sentence', 'keywords']
    for col in df.columns:
        if col.endswith('_single') or col in list_cols:
            df[col] = df[col].apply(safe_literal_eval)

    # 3. Clean and normalize all existing text and sentence columns in the loaded dataset
    print("Normalizing existing text and sentences in the loaded dataset...")
    cols_to_normalize = []
    for col in df.columns:
        if (col.endswith('_single') or 
                col.endswith('_full') or 
                col == 'abstract_sentence' or 
                any(col.endswith(f'_{p}') for p in [25, 50, 70, 75])):
            cols_to_normalize.append(col)
            
    for col in cols_to_normalize:
        df[col] = df[col].apply(clean_and_normalize_value)

    # 4. Read the checkpoint task records
    tasks = []
    if args.checkpoint.exists():
        print(f"Reading checkpoint file: {args.checkpoint}")
        with open(args.checkpoint, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        tasks.append(json.loads(line))
                    except Exception as e:
                        print(f"Skipping malformed JSON line: {line}. Error: {e}")
    else:
        print(f"Checkpoint file not found: {args.checkpoint}")
        return

    # Convert dataframe to records for quicker modifications
    records = df.to_dict('records')
    record_map = {clean_id(r.get('_id')): r for r in records if r.get('_id') is not None}

    # Track distinct models in the workspace to construct missing columns later
    models = set()
    for col in df.columns:
        if col.endswith('_single'):
            models.add(col[:-7])
        elif col.endswith('_full'):
            models.add(col[:-5])
        elif any(col.endswith(f'_{p}') for p in [25, 50, 70, 75]):
            parts = col.split('_')
            models.add('_'.join(parts[:-1]))

    # Add any models introduced by checkpoints
    for task in tasks:
        if 'model' in task:
            models.add(task['model'])

    # 5. Perform updates and track metrics
    applied_stats = Counter()         # Key: (task_type, model)
    already_applied_stats = Counter() # Key: (task_type, model)
    not_found_stats = Counter()       # Key: (task_type, model)

    for task in tasks:
        task_id = clean_id(task.get("id"))
        t_type = task.get("type", "unknown")
        model = task.get("model", "unknown")
        
        if not task_id or task_id not in record_map:
            not_found_stats[(t_type, model)] += 1
            continue
            
        row = record_map[task_id]
        rewritten = task.get("rewritten")
        
        # Clean and normalize incoming new rewrites before insertion
        if isinstance(rewritten, str):
            rewritten = clean_and_normalize_value(rewritten)
            task["rewritten"] = rewritten
        
        # Skip if the rewrite target cell is already populated
        if is_already_applied(row, task):
            already_applied_stats[(t_type, model)] += 1
            continue
            
        apply_rewrite_to_row(row, task, rewritten)
        applied_stats[(t_type, model)] += 1

    # Print compilation analysis report
    total_applied = sum(applied_stats.values())
    total_already_applied = sum(already_applied_stats.values())
    total_not_found = sum(not_found_stats.values())

    print("\n" + "=" * 75)
    print("                      DATASET COMPILED UPDATE REPORT                      ")
    print("=" * 75)
    print(f"Base Configuration Status:")
    print(f"  - Source file loaded:          {'llm_added (checkpoint)' if already_in_output else 'base merged_publications'}")
    print(f"  - Record rows already present: {initial_row_count:,}")
    print(f"  - Unique row ID mappings:      {len(record_map):,}")
    print(f"  - Checkpoint tasks evaluated:  {len(tasks):,}")
    print("-" * 75)
    print(f"Processing Execution Stats:")
    print(f"  - New updates applied:         {total_applied:,}")
    print(f"  - Already applied (skipped):   {total_already_applied:,}")
    print(f"  - IDs not matched (skipped):   {total_not_found:,}")
    print("-" * 75)

    if total_applied > 0:
        print("New Updates Applied (by Task Type & Model Config):")
        
        by_type = Counter()
        by_model = Counter()
        for (t_type, model), count in applied_stats.items():
            by_type[t_type] += count
            by_model[model] += count
            
        print("  By Task Type:")
        for t_type, count in sorted(by_type.items()):
            print(f"    * {t_type:<15}: {count:,} updates")
            
        print("\n  By Model Engine:")
        for model, count in sorted(by_model.items()):
            print(f"    * {model:<25}: {count:,} updates")
            
        print("\n  Detailed Grouping Matrix (Type | Model):")
        for (t_type, model), count in sorted(applied_stats.items()):
            print(f"    * {t_type:<15} | {model:<25} -> {count:,} edits")
    else:
        print("No new updates were processed or written during this execution.")
    print("=" * 75 + "\n")

    # 6. Convert records back to DataFrame
    df_updated = pd.DataFrame(records)

    # 7. Initialize missing model columns explicitly (single, full, percentages)
    # Corrected layout: Always map list columns to verify every single record 
    # gets initialized as a clean list of correct length.
    for model in sorted(models):
        col_single = f"{model}_single"
        df_updated[col_single] = df_updated.apply(
            lambda r: r.get(col_single) if isinstance(r.get(col_single), (list, np.ndarray, pd.Series))
            else ([None] * len(r['abstract_sentence']) if isinstance(r.get('abstract_sentence'), (list, np.ndarray, pd.Series)) else []),
            axis=1
        )
            
        col_full = f"{model}_full"
        if col_full not in df_updated.columns:
            df_updated[col_full] = None
            
        for pct in [25, 50, 70, 75]:
            col_pct = f"{model}_{pct}"
            if col_pct not in df_updated.columns:
                df_updated[col_pct] = None

    # Check and remove {model}_70 columns if they are empty
    cols_70 = [col for col in df_updated.columns if col.endswith('_70')]
    for col in cols_70:
        is_empty = df_updated[col].apply(is_null_or_empty).all()
        print(f"Found column: {col}. Checking if empty: {is_empty}")
        if is_empty:
            df_updated.drop(columns=[col], inplace=True)
            print(f"Removed empty column: {col}")

    # Re-order columns so original order is preserved and new model columns are sorted at the end
    original_cols = [c for c in df.columns if c in df_updated.columns]
    new_cols = [c for c in df_updated.columns if c not in df.columns]
    df_updated = df_updated[original_cols + sorted(new_cols)]

    # 8. Write back to Parquet and CSV
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving updated dataset to {args.output_parquet}...")
    df_updated.to_parquet(args.output_parquet, index=False)
    
    # Create a separate copy for CSV export to keep nested lists in the Parquet file clean
    print(f"Saving updated dataset to {args.output_csv}...")
    df_csv = df_updated.copy()
    
    # Join list elements with '|' in columns containing lists/arrays
    for col in df_csv.columns:
        if col.endswith('_single') or col in ['abstract_sentence', 'keywords']:
            df_csv[col] = df_csv[col].apply(format_list_for_csv)
            
    df_csv.to_csv(args.output_csv, index=False)
    print("Done.")


if __name__ == "__main__":
    main()