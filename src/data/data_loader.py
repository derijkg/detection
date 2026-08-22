# src/data/data_loader.py

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data_static" / "preprocessed" / "preprocessed_dataset.parquet"
DEFAULT_FEATURES_DIR = PROJECT_ROOT / "data_static" / "model_features"

# PyTorch import safeguard
try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    TorchDataset = object

# Hugging Face Datasets import safeguard
try:
    from datasets import Dataset as HFDataset, DatasetDict, load_from_disk
    HAS_HF = True
except ImportError:
    HAS_HF = False

# Hugging Face Transformers import safeguard
try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

def group_stratified_sample(
    df: pd.DataFrame, 
    sample_size: int, 
    seed: int = 42, 
    id_candidates: Tuple[str, ...] = ("_id", "doc_id", "id")
) -> pd.DataFrame:
    """
    Subsamples rows by grouping unique document IDs together to preserve group integrity.
    """
    if sample_size <= 0 or len(df) <= sample_size:
        return df

    id_col = next((c for c in id_candidates if c in df.columns), None)
    if id_col:
        unique_ids = df[id_col].unique()
        avg_rows = len(df) / max(len(unique_ids), 1)
        target_groups = max(1, int(sample_size / avg_rows))
        
        if target_groups < len(unique_ids):
            rng = np.random.default_rng(seed)
            sampled_ids = rng.choice(unique_ids, size=target_groups, replace=False)
            return df[df[id_col].isin(sampled_ids)].copy().reset_index(drop=True)
            
    return df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

@dataclass
class TextSample:
    id: str
    text: str
    label: int
    llm_ratio: float
    model_name: str
    scope: str
    generation_type: str
    split: str
    source: Optional[str] = None
    keywords: Optional[Any] = None
    year: Optional[int] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TextSample":
        year_val = d.get("year")
        parsed_year = None
        if year_val is not None and not pd.isna(year_val):
            try:
                parsed_year = int(float(year_val))
            except (ValueError, TypeError):
                parsed_year = None

        return cls(
            id=str(d.get("_id", d.get("id", d.get("doc_id", "")))),
            text=str(d.get("text", "")),
            label=int(d.get("label", 0)),
            llm_ratio=float(d.get("llm_ratio", 0.0)),
            model_name=str(d.get("model_name", "unknown")),
            scope=str(d.get("scope", "full")),
            generation_type=str(d.get("generation_type", "unknown")),
            split=str(d.get("split", "train")),
            source=d.get("source"),
            keywords=d.get("keywords"),
            year=parsed_year
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DataFilter:
    splits: Optional[List[str]] = None
    scopes: Optional[List[str]] = None
    generation_types: Optional[List[str]] = None
    model_names: Optional[List[str]] = None
    llm_ratios: Optional[List[float]] = None
    labels: Optional[List[int]] = None
    test_suite: Optional[str] = None  # 'standard', 'prompt_partial', 'synthetic_multi', 'all'


class DetectionDataManager:
    def __init__(
        self, 
        data_path: Optional[Union[str, Path]] = None, 
        features_dir: Optional[Union[str, Path]] = None
    ):
        self.data_path = Path(data_path) if data_path else DEFAULT_DATA_PATH
        self.features_dir = Path(features_dir) if features_dir else DEFAULT_FEATURES_DIR

        if not self.data_path.exists():
            raise FileNotFoundError(f"Preprocessed dataset not found at: {self.data_path}")
        
        self._df = pd.read_parquet(self.data_path)

    @property
    def raw_dataframe(self) -> pd.DataFrame:
        return self._df

    def filter_dataframe(
        self, 
        filter_config: Optional[DataFilter] = None, 
        sample_size: int = -1, 
        seed: int = 42, 
        **kwargs
    ) -> pd.DataFrame:
        if filter_config is None:
            filter_config = DataFilter(**kwargs)
        else:
            for k, v in kwargs.items():
                if hasattr(filter_config, k) and v is not None:
                    setattr(filter_config, k, v)

        df = self._df.copy()

        # 1. Splits
        if filter_config.splits:
            requested_splits = set(filter_config.splits)
            if 'dev' in requested_splits or 'val' in requested_splits:
                requested_splits.update(['dev', 'val'])
            df = df[df['split'].isin(requested_splits)]
            
        # 2. Scopes
        if filter_config.scopes:
            normalized_scopes = []
            for s in filter_config.scopes:
                if s in ['sentence', 'single']:
                    normalized_scopes.extend(['sentence', 'single'])
                else:
                    normalized_scopes.append(s)
            df = df[df['scope'].isin(normalized_scopes)]

        # 3. Predefined Test Suite Presets
        if filter_config.test_suite:
            mode = filter_config.test_suite.lower()
            if mode == 'standard':
                df = df[df['generation_type'].isin(['human_full', 'human_sentence', 'full_rewrite', 'sentence_rewrite'])]
            elif mode == 'prompt_partial':
                df = df[df['generation_type'].isin(['human_full', 'human_sentence', 'full_rewrite', 'sentence_rewrite', 'prompt_partial'])]
            elif mode == 'synthetic_multi':
                df = df[df['generation_type'].isin(['human_full', 'human_sentence', 'full_rewrite', 'sentence_rewrite', 'synthetic_partial'])]

        # 4. Explicit generation types & models
        if filter_config.generation_types:
            df = df[df['generation_type'].isin(filter_config.generation_types)]
        if filter_config.model_names:
            df = df[df['model_name'].isin(filter_config.model_names) | (df['model_name'] == 'human')]
        if filter_config.llm_ratios:
            ratios_arr = np.array(filter_config.llm_ratios)
            df = df[df['llm_ratio'].apply(lambda r: any(np.isclose(r, ratios_arr)))]
        if filter_config.labels:
            df = df[df['label'].isin(filter_config.labels)]

        # 5. Group-Stratified Subsampling on _id
        if sample_size > 0 and len(df) > sample_size:
            df = group_stratified_sample(df, sample_size=sample_size, seed=seed)

        return df

    def get_benchmark_test_suites(self, scope: str = "full") -> Dict[str, Any]:
        test_base = self.filter_dataframe(DataFilter(splits=['test'], scopes=[scope]))
        
        suites = {
            "standard": self.filter_dataframe(DataFilter(splits=['test'], scopes=[scope], test_suite='standard')),
            "prompt_partial": self.filter_dataframe(DataFilter(splits=['test'], scopes=[scope], test_suite='prompt_partial')),
            "synthetic_multi": self.filter_dataframe(DataFilter(splits=['test'], scopes=[scope], test_suite='synthetic_multi')),
            "all": test_base,
            "per_generator": {}
        }

        # Slices for individual generator LLMs
        models_in_test = [m for m in test_base['model_name'].unique() if m not in ['human', 'synthetic_multi']]
        for m in models_in_test:
            suites["per_generator"][m] = test_base[test_base['model_name'].isin(['human', m])].copy()

        return suites

    def get_samples(self, filter_config: Optional[DataFilter] = None, **kwargs) -> List[TextSample]:
        filtered_df = self.filter_dataframe(filter_config, **kwargs)
        return [TextSample.from_dict(row) for row in filtered_df.to_dict(orient="records")]

    def get_sklearn_data(self, filter_config: Optional[DataFilter] = None, **kwargs) -> Tuple[List[str], List[int]]:
        filtered_df = self.filter_dataframe(filter_config, **kwargs)
        texts = filtered_df['text'].fillna("").astype(str).tolist()
        labels = filtered_df['label'].astype(int).tolist()
        return texts, labels

    def get_hf_dataset(self, filter_config: Optional[DataFilter] = None, **kwargs):
        if not HAS_HF:
            raise ImportError("Hugging Face `datasets` library required.")
        
        if filter_config is None:
            filter_config = DataFilter(**kwargs)
        else:
            for k, v in kwargs.items():
                if hasattr(filter_config, k) and v is not None:
                    setattr(filter_config, k, v)

        filtered_df = self.filter_dataframe(filter_config)
        
        if filter_config.splits is None:
            dataset_dict = {}
            for split_name in filtered_df['split'].unique():
                split_df = filtered_df[filtered_df['split'] == split_name]
                dataset_dict[split_name] = HFDataset.from_pandas(split_df, preserve_index=False)
            return DatasetDict(dataset_dict)
        else:
            return HFDataset.from_pandas(filtered_df, preserve_index=False)