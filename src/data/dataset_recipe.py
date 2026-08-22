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

RE_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')


class DataTransforms:
    """Library of reusable in-memory virtual transformations."""

    @staticmethod
    def extract_ai_sentences_from_full_abstracts(df_full: pd.DataFrame) -> pd.DataFrame:
        """
        Slices full AI abstracts (label == 1) into individual sentence records.
        """
        extracted_ai_records = []
        id_col = next((c for c in ["_id", "doc_id", "id"] if c in df_full.columns), "_id")

        ai_full_df = df_full[df_full["label"] == 1]

        for _, row in ai_full_df.iterrows():
            doc_id = row.get(id_col, "doc_unknown")
            gen_model = row.get("generator_model", row.get("model_name", "unknown"))
            llm_ratio = row.get("llm_ratio", 1.0)
            split_val = row.get("split", "train") if pd.notna(row.get('split')) else 'train'
            raw_text = str(row.get("text", "")).strip()

            if not raw_text:
                continue

            sents = [s.strip() for s in RE_SENT_SPLIT.split(raw_text) if len(s.strip()) > 5]
            if not sents:
                sents = [raw_text]

            for idx, s in enumerate(sents):
                norm_s = normalize_text(s)
                extracted_ai_records.append({
                    "_id": doc_id,
                    "sentence_id": f"{doc_id}_ai_s{idx}",
                    "text": norm_s,
                    "normalized_text": norm_s,
                    "label": 1,
                    "scope": "sentence_augmented",
                    "scope_type": "sentence",
                    "is_sentence": 1,
                    "generator_model": gen_model,
                    "model_name": gen_model,
                    "llm_ratio": llm_ratio,
                    "split": split_val,
                    "source": row.get("source", "unknown"),
                    "year": row.get("year", 2024),
                })

        return pd.DataFrame(extracted_ai_records)


@dataclass
class DatasetRecipe:
    name: str
    splits: List[str] = field(default_factory=lambda: ["train"])
    include_full_abstracts: bool = False          # Intact full abstracts (is_sentence=0)
    include_standard_sentences: bool = True       # Standard sentences (is_sentence=1)
    include_full_abstract_sentences: bool = False # Slice AI abstracts into sentences
    sample_size: int = -1
    seed: int = 42

    def compute_fingerprint(self) -> str:
        payload = {
            "name": self.name,
            "splits": sorted(self.splits),
            "include_full_abstracts": self.include_full_abstracts,
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

        if self.use_cache and cache_path.exists():
            return pd.read_parquet(cache_path)

        data_blocks: List[pd.DataFrame] = []

        for sp in recipe.splits:
            # 1. Fetch Intact Full Abstracts (Document Scale)
            if recipe.include_full_abstracts or recipe.include_full_abstract_sentences:
                df_full = self.manager.filter_dataframe(
                    DataFilter(splits=[sp], scopes=["full"]), 
                    sample_size=-1, 
                    seed=recipe.seed
                )
            else:
                df_full = pd.DataFrame()

            if recipe.include_full_abstracts and not df_full.empty:
                df_full_intact = df_full.copy()
                df_full_intact["is_sentence"] = 0
                df_full_intact["scope_type"] = "full"
                if "normalized_text" not in df_full_intact.columns and "text" in df_full_intact.columns:
                    df_full_intact["normalized_text"] = df_full_intact["text"].apply(normalize_text)
                data_blocks.append(df_full_intact)

            # 2. Fetch Standard Sentences (Sentence Scale)
            if recipe.include_standard_sentences:
                df_sent = self.manager.filter_dataframe(
                    DataFilter(splits=[sp], scopes=["sentence"]), 
                    sample_size=-1, 
                    seed=recipe.seed
                )
                if not df_sent.empty:
                    df_sent_clean = df_sent.copy()
                    df_sent_clean["is_sentence"] = 1
                    df_sent_clean["scope_type"] = "sentence"
                    if "normalized_text" not in df_sent_clean.columns and "text" in df_sent_clean.columns:
                        df_sent_clean["normalized_text"] = df_sent_clean["text"].apply(normalize_text)
                    data_blocks.append(df_sent_clean)

            # 3. Optional: Extract Sentences from Full AI Abstracts
            if recipe.include_full_abstract_sentences and not df_full.empty:
                df_ai_sents = DataTransforms.extract_ai_sentences_from_full_abstracts(df_full)
                if not df_ai_sents.empty:
                    data_blocks.append(df_ai_sents)

        if not data_blocks:
            raise ValueError(f"Recipe '{recipe.name}' produced an empty dataset.")

        final_df = pd.concat(data_blocks, ignore_index=True)

        # Deduplicate identical strings under the same document ID and scale
        dedup_cols = ["_id", "text", "label", "is_sentence"] if "_id" in final_df.columns else ["text", "label", "is_sentence"]
        final_df = final_df.drop_duplicates(subset=dedup_cols).reset_index(drop=True)

        # Stratified Subsampling by Document ID (keeps document + sentences of the same doc together)
        if 0 < recipe.sample_size < len(final_df):
            id_col = next((c for c in ["_id", "doc_id", "id"] if c in final_df.columns), None)
            if id_col:
                u_ids = final_df[id_col].unique()
                avg_rows_per_doc = len(final_df) / max(len(u_ids), 1)
                n_docs = max(1, int(recipe.sample_size / avg_rows_per_doc))
                
                rng = np.random.default_rng(recipe.seed)
                selected_ids = rng.choice(u_ids, size=min(n_docs, len(u_ids)), replace=False)
                final_df = final_df[final_df[id_col].isin(selected_ids)].copy().reset_index(drop=True)
            else:
                final_df = final_df.sample(n=recipe.sample_size, random_state=recipe.seed).reset_index(drop=True)

        if self.use_cache and len(final_df) > 0:
            final_df.to_parquet(cache_path, index=False)

        return final_df