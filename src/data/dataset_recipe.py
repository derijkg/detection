# src/data/dataset_recipe.py

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from src.data.data_loader import DataFilter, DetectionDataManager
from src.data.stylometrics import normalize_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_ROOT = PROJECT_ROOT / "data_static" / "preprocessed" / "recipe_cache"

# Regex for sentence splitting consistent with preprocessing
RE_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')


class DataTransforms:
    """Library of reusable in-memory virtual transformations."""

    @staticmethod
    def sentence_tokenize_and_combine_with_sentences(
        df_full: pd.DataFrame, 
        df_sentence: pd.DataFrame,
        seed: int = 42
    ) -> pd.DataFrame:
        """
        1. Keeps ALL original human sentences (label == 0) and AI sentences from `sentence` scope.
        2. Sentence-tokenizes ONLY the AI full-abstract rewrites (label == 1) from `full` scope.
        3. Preserves document `_id` for group-stratified cross-validation and tuning.
        """
        extracted_ai_records = []
        id_col = next((c for c in ["_id", "doc_id", "id"] if c in df_full.columns), "_id")

        # Filter ONLY AI rewrites from full abstracts (label == 1)
        ai_full_df = df_full[df_full["label"] == 1]

        for _, row in ai_full_df.iterrows():
            doc_id = row.get(id_col, "doc_unknown")
            gen_model = row.get("generator_model", row.get("model_name", "unknown"))
            llm_ratio = row.get("llm_ratio", 1.0)
            split_val = row.get("split", "train")
            raw_text = str(row.get("text", "")).strip()

            if not raw_text:
                continue

            # Split AI full abstract into individual sentences
            sents = [s.strip() for s in RE_SENT_SPLIT.split(raw_text) if len(s.strip()) > 5]
            if not sents:
                sents = [raw_text]

            for idx, s in enumerate(sents):
                norm_s = normalize_text(s)
                extracted_ai_records.append({
                    "_id": doc_id,                        # Parent document ID for grouping
                    "sentence_id": f"{doc_id}_ai_full_s{idx}",
                    "text": norm_s,
                    "normalized_text": norm_s,
                    "label": 1,                           # AI class
                    "scope": "sentence_augmented",
                    "generator_model": gen_model,
                    "model_name": gen_model,
                    "llm_ratio": llm_ratio,
                    "split": split_val,
                    "source": row.get("source", "unknown"),
                    "year": row.get("year", 2024),
                })

        df_ai_extracted = pd.DataFrame(extracted_ai_records)

        # Merge base sentence dataset (containing genuine human + original AI sentences) with newly extracted AI sentences
        if not df_sentence.empty:
            if "normalized_text" not in df_sentence.columns and "text" in df_sentence.columns:
                df_sentence["normalized_text"] = df_sentence["text"].apply(normalize_text)
            
            combined_df = pd.concat([df_sentence, df_ai_extracted], ignore_index=True)
        else:
            combined_df = df_ai_extracted

        # Deduplicate identical sentences under the same document ID
        dedup_cols = ["_id", "text", "label"] if "_id" in combined_df.columns else ["text", "label"]
        combined_df = combined_df.drop_duplicates(subset=dedup_cols).reset_index(drop=True)

        return combined_df


@dataclass
class DatasetRecipe:
    name: str
    splits: List[str] = field(default_factory=lambda: ["train"])
    include_standard_sentences: bool = True
    include_full_abstract_sentences: bool = True
    sample_size: int = -1
    seed: int = 42

    def compute_fingerprint(self) -> str:
        """Deterministic fingerprint for caching to eliminate duplicate work."""
        payload = {
            "name": self.name,
            "splits": sorted(self.splits),
            "include_standard_sentences": self.include_standard_sentences,
            "include_full_abstract_sentences": self.include_full_abstract_sentences,
            "sample_size": self.sample_size,
            "seed": self.seed,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


class RecipeDataBuilder:
    """Builds and caches dataset recipes safely without duplicate disk I/O."""

    def __init__(self, manager: Optional[DetectionDataManager] = None, use_cache: bool = True):
        self.manager = manager or DetectionDataManager()
        self.use_cache = use_cache

    def build(self, recipe: Union[Dict[str, Any], DatasetRecipe]) -> pd.DataFrame:
        if isinstance(recipe, dict):
            recipe = DatasetRecipe(**recipe)

        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        cache_path = CACHE_ROOT / f"recipe_{recipe.name}_{recipe.compute_fingerprint()}.parquet"

        # Load from cache if already built
        if self.use_cache and cache_path.exists():
            return pd.read_parquet(cache_path)

        # 1. Fetch preprocessed data from DetectionDataManager
        df_sentence_parts = []
        df_full_parts = []

        for sp in recipe.splits:
            if recipe.include_standard_sentences:
                part_sent = self.manager.filter_dataframe(
                    DataFilter(splits=[sp], scopes=["sentence"]), 
                    sample_size=-1, 
                    seed=recipe.seed
                )
                if not part_sent.empty:
                    df_sentence_parts.append(part_sent)

            if recipe.include_full_abstract_sentences:
                part_full = self.manager.filter_dataframe(
                    DataFilter(splits=[sp], scopes=["full"]), 
                    sample_size=-1, 
                    seed=recipe.seed
                )
                if not part_full.empty:
                    df_full_parts.append(part_full)

        df_sentence = pd.concat(df_sentence_parts, ignore_index=True) if df_sentence_parts else pd.DataFrame()
        df_full = pd.concat(df_full_parts, ignore_index=True) if df_full_parts else pd.DataFrame()

        # 2. Apply Sentence Tokenization and Combination
        if recipe.include_full_abstract_sentences:
            final_df = DataTransforms.sentence_tokenize_and_combine_with_sentences(
                df_full=df_full,
                df_sentence=df_sentence,
                seed=recipe.seed
            )
        else:
            final_df = df_sentence

        # 3. Optional Stratified Subsampling (by document ID)
        if 0 < recipe.sample_size < len(final_df):
            id_col = next((c for c in ["_id", "doc_id", "id"] if c in final_df.columns), None)
            if id_col:
                u_ids = final_df[id_col].unique()
                avg_sents_per_doc = len(final_df) / max(len(u_ids), 1)
                n_docs = max(1, int(recipe.sample_size / avg_sents_per_doc))
                
                rng = np.random.default_rng(recipe.seed)
                selected_ids = rng.choice(u_ids, size=min(n_docs, len(u_ids)), replace=False)
                final_df = final_df[final_df[id_col].isin(selected_ids)].copy().reset_index(drop=True)
            else:
                final_df = final_df.sample(n=recipe.sample_size, random_state=recipe.seed).reset_index(drop=True)

        # 4. Save to cache
        if self.use_cache and len(final_df) > 0:
            final_df.to_parquet(cache_path, index=False)

        return final_df