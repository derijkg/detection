# detection/src/data/synth_data.py

import ast
import json
import random
import zlib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Set

# String values to strictly reject during parsing
INVALID_SENTENCE_VALUES: Set[str] = {
    "generation_failed",
    "validation_failed",
    "nan",
    "none",
    "null",
    "",
}


def is_valid_sentence(text: Any) -> bool:
    """
    Checks whether a sentence is valid and free of failure markers, NaNs, or empty strings.
    """
    if text is None:
        return False
    if isinstance(text, (float, np.floating)) and np.isnan(text):
        return False
    
    clean_str = str(text).strip()
    if not clean_str:
        return False
    
    if clean_str.lower() in INVALID_SENTENCE_VALUES:
        return False
    
    return True


def parse_and_clean_sentence_array(raw_val: Any) -> List[str]:
    """
    Safely parses raw values (numpy arrays, python lists, stringified arrays)
    and strips out invalid/failed values.
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

    # Filter out invalid, failed, or NaN strings
    cleaned_sentences = [
        str(s).strip() 
        for s in parsed_list 
        if is_valid_sentence(s)
    ]

    return cleaned_sentences


def extract_valid_single_models(
    row: pd.Series, 
    model_columns: Optional[List[str]] = None
) -> Dict[str, List[str]]:
    """
    Extracts and validates sentence arrays for all model '_single' columns present in a row.
    Returns a dict mapping model_name -> list of clean sentences.
    """
    valid_model_sents: Dict[str, List[str]] = {}

    # Find candidate single-sentence columns if not explicitly provided
    if model_columns is None:
        model_columns = [col for col in row.index if col.endswith("_single")]

    for col in model_columns:
        if col not in row or pd.isna(row[col]):
            continue

        clean_sents = parse_and_clean_sentence_array(row[col])
        if clean_sents:
            # Extract model name from column name (e.g. 'qwen3.6:27b_single' -> 'qwen3.6:27b')
            model_name = col.rsplit("_single", 1)[0]
            valid_model_sents[model_name] = clean_sents

    return valid_model_sents


def mix_abstract_at_ratio(
    human_sents: List[str],
    available_models: Dict[str, List[str]],
    target_ratio: float,
    seed: int
) -> Tuple[List[str], List[int], List[str], float]:
    """
    Substitutes target_ratio % of human sentences with corresponding LLM sentences from random models.

    Returns:
        mixed_sentences (List[str]): Reconstituted sentence list.
        sentence_labels (List[int]): 0 for human, 1 for LLM.
        sentence_models (List[str]): Author/model name for each sentence.
        actual_ratio (float): Actual substituted ratio achieved.
    """
    n_sentences = len(human_sents)
    
    # Calculate number of sentences to replace (k)
    k = int(round(target_ratio * n_sentences))
    # Clamp k between 1 and n_sentences - 1 to guarantee a mix if n_sentences >= 2
    k = max(1, min(n_sentences - 1, k))

    # Set deterministic RNG per doc and ratio
    rng = random.Random(seed)

    # Pick indices for replacement
    replace_indices = set(rng.sample(range(n_sentences), k))

    mixed_sents: List[str] = []
    sentence_labels: List[int] = []
    sentence_models: List[str] = []

    model_names = list(available_models.keys())

    for i in range(n_sentences):
        if i in replace_indices:
            # Pick a random model that has a valid rewrite
            chosen_model = rng.choice(model_names)
            model_sents = available_models[chosen_model]
            
            # Use corresponding index if available, else fallback safely via modulo
            llm_sent = model_sents[i] if i < len(model_sents) else model_sents[i % len(model_sents)]
            
            mixed_sents.append(llm_sent)
            sentence_labels.append(1)
            sentence_models.append(chosen_model)
        else:
            mixed_sents.append(human_sents[i])
            sentence_labels.append(0)
            sentence_models.append("human")

    actual_ratio = k / n_sentences if n_sentences > 0 else 0.0
    return mixed_sents, sentence_labels, sentence_models, actual_ratio


def generate_synthetic_mixes(
    df: pd.DataFrame,
    target_ratios: List[float] = [0.25, 0.50, 0.75],
    seed: int = 42,
    min_sentences: int = 3
) -> pd.DataFrame:
    """
    Generates synthetic mixed-authorship documents at target ratios (25%, 50%, 75%)
    for all abstracts in the DataFrame.

    Returns a long-format DataFrame ready for model training and benchmark evaluation.
    """
    synthetic_records: List[Dict[str, Any]] = []
    print(f"Generating synthetic mixed abstracts for {len(df)} rows...")

    for idx, row in df.iterrows():
        # Get abstract identifier
        doc_id = str(row["_id"]) if "_id" in row and pd.notna(row["_id"]) else f"doc_{idx}"
        source = str(row.get("source", "unknown"))
        keywords = row.get("keywords", None)
        year = row.get("year", None)

        # 1. Parse and clean human abstract sentences
        human_sents = parse_and_clean_sentence_array(row.get("abstract_sentence", []))
        if len(human_sents) < min_sentences:
            continue

        # 2. Extract and clean available single-sentence LLM rewrites
        valid_models = extract_valid_single_models(row)
        if not valid_models:
            continue

        # 3. Generate 25%, 50%, 75% synthetic reconstitutions
        for ratio in target_ratios:
            # Deterministic seed per abstract and target ratio
            seed_str = f"{seed}_{doc_id}_{ratio}"
            pair_seed = zlib.crc32(seed_str.encode("utf-8"))

            mixed_sents, labels, sentence_models, actual_ratio = mix_abstract_at_ratio(
                human_sents=human_sents,
                available_models=valid_models,
                target_ratio=ratio,
                seed=pair_seed
            )

            reconstituted_text = " ".join(mixed_sents)

            synthetic_records.append({
                "_id": doc_id,
                "source": source,
                "keywords": keywords,
                "year": year,
                "text": reconstituted_text,
                "label": 1,  # Mixed document
                "llm_ratio": actual_ratio,
                "target_ratio": ratio,
                "model_name": "synthetic_multi",
                "scope": "full",
                "generation_type": "synthetic_partial",
                "sentences": mixed_sents,
                "sentence_labels": labels,
                "sentence_models": sentence_models,
                "num_sentences": len(mixed_sents)
            })

    synth_df = pd.DataFrame(synthetic_records)
    print(f"-> Successfully generated {len(synth_df)} synthetic mixed document records.")
    return synth_df


if __name__ == "__main__":
    # Example usage / smoke test
    sample_raw_df = pd.DataFrame([
        {
            "_id": "abs_001",
            "source": "arxiv",
            "keywords": ["AI", "Detection"],
            "year": 2024,
            "abstract_sentence": np.array([
                "This paper presents a new detector.",
                "We evaluate it on multiple datasets.",
                "Our results demonstrate high accuracy.",
                "GENERATION_FAILED",  # Will be cleanly filtered out
                "Finally, we discuss future work."
            ]),
            "qwen3.6:27b_single": np.array([
                "This study introduces a novel detection model.",
                "We test the framework across diverse benchmarks.",
                "Our findings show exceptional performance.",
                "VALIDATION_FAILED",  # Will be cleanly filtered out
                "In conclusion, potential future directions are explored."
            ]),
            "gemma4:e4b_single": np.array([
                "Here, we propose an improved detector method.",
                "Evaluations were conducted on various datasets.",
                "The outcomes highlight strong predictive accuracy.",
                "None",  # Will be cleanly filtered out
                "Lastly, future research avenues are considered."
            ])
        }
    ])

    synthetic_df = generate_synthetic_mixes(sample_raw_df)
    print("\nSample Output Row:")
    print(synthetic_df[['_id', 'target_ratio', 'llm_ratio', 'text']].to_string())