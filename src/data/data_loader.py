# detection/src/data/data_loader.py

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np

# PyTorch import safeguard
try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    TorchDataset = object  # Fallback type

# Hugging Face Datasets import safeguard
try:
    from datasets import Dataset as HFDataset, DatasetDict
    HAS_HF = True
except ImportError:
    HAS_HF = False


# =============================================================================
# 1. DATACLASS STRUCTURES
# =============================================================================

@dataclass
class TextSample:
    """Dataclass representing a single text sample in the benchmark."""
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
        """Factory method to construct a TextSample from a DataFrame row dict."""
        return cls(
            id=str(d.get("_id", d.get("id", ""))),
            text=str(d.get("text", "")),
            label=int(d.get("label", 0)),
            llm_ratio=float(d.get("llm_ratio", 0.0)),
            model_name=str(d.get("model_name", "unknown")),
            scope=str(d.get("scope", "full")),
            generation_type=str(d.get("generation_type", "unknown")),
            split=str(d.get("split", "train")),
            source=d.get("source"),
            keywords=d.get("keywords"),
            year=d.get("year")
        )

    def to_dict(self) -> Dict[str, Any]:
        """Converts sample back to a standard Python dictionary."""
        return asdict(self)


@dataclass
class DataFilter:
    """Dataclass to specify filtering criteria for dataset slicing."""
    splits: Optional[List[str]] = None            # e.g., ['train'], ['dev'], ['test']
    scopes: Optional[List[str]] = None            # e.g., ['full'], ['single']
    generation_types: Optional[List[str]] = None  # e.g., ['human_full', 'full_rewrite', 'synthetic_partial']
    model_names: Optional[List[str]] = None       # e.g., ['qwen3.6:27b', 'gemma4:e4b', 'human']
    llm_ratios: Optional[List[float]] = None      # e.g., [0.0, 1.0] or [0.25, 0.50, 0.75]
    labels: Optional[List[int]] = None            # e.g., [0, 1]


# =============================================================================
# 2. PYTORCH DATASET WRAPPER
# =============================================================================

if HAS_TORCH:
    class PyTorchDetectionDataset(TorchDataset):
        """PyTorch Dataset wrapper for TextSample dataclass instances."""
        def __init__(
            self, 
            samples: List[TextSample], 
            tokenizer: Optional[Any] = None, 
            max_length: int = 512
        ):
            self.samples = samples
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int) -> Dict[str, Any]:
            sample = self.samples[idx]
            item = {
                "id": sample.id,
                "text": sample.text,
                "label": torch.tensor(sample.label, dtype=torch.long),
                "llm_ratio": torch.tensor(sample.llm_ratio, dtype=torch.float),
                "model_name": sample.model_name,
                "scope": sample.scope,
                "generation_type": sample.generation_type,
                "split": sample.split
            }

            if self.tokenizer is not None:
                encoded = self.tokenizer(
                    sample.text,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                item["input_ids"] = encoded["input_ids"].squeeze(0)
                item["attention_mask"] = encoded["attention_mask"].squeeze(0)
                if "token_type_ids" in encoded:
                    item["token_type_ids"] = encoded["token_type_ids"].squeeze(0)

            return item


# =============================================================================
# 3. BENCHMARK DATA MANAGER & LOADERS
# =============================================================================

class DetectionDataManager:
    """
    Manager class for the preprocessed detection benchmark dataset.
    Loads data and produces scikit-learn tuples, PyTorch Datasets, 
    Hugging Face DatasetDicts, or lists of typed TextSample objects.
    """
    DEFAULT_PATH = Path("/home/gderijck/detection/data/preprocessed/preprocessed_dataset.parquet")

    def __init__(self, data_path: Optional[Union[str, Path]] = None):
        self.data_path = Path(data_path) if data_path else self.DEFAULT_PATH
        if not self.data_path.exists():
            raise FileNotFoundError(f"Preprocessed dataset not found at: {self.data_path}")
        
        # Read dataset into memory
        self._df = pd.read_parquet(self.data_path)

    @property
    def raw_dataframe(self) -> pd.DataFrame:
        """Returns the full raw pandas DataFrame."""
        return self._df

    def summary(self) -> None:
        """Prints a quick breakdown of available splits, scopes, and generation types."""
        print("=== Benchmark Dataset Summary ===")
        print(f"Total Samples: {len(self._df)}")
        print("\nSamples per Split:")
        print(self._df['split'].value_counts())
        print("\nSamples per Generation Type:")
        print(self._df['generation_type'].value_counts())
        print("\nSamples per Scope:")
        print(self._df['scope'].value_counts())

    def filter_dataframe(self, filter_config: Optional[DataFilter] = None, sample_size: int = -1, seed: int = 42, **kwargs) -> pd.DataFrame:
        if filter_config is None:
            filter_config = DataFilter(**kwargs)
        else:
            for k, v in kwargs.items():
                if hasattr(filter_config, k) and v is not None:
                    setattr(filter_config, k, v)

        df = self._df.copy()

        if filter_config.splits:
            df = df[df['split'].isin(filter_config.splits)]
            
        # --- Normalize 'sentence' and 'single' as automatic synonyms ---
        if filter_config.scopes:
            normalized_scopes = []
            for s in filter_config.scopes:
                if s in ['sentence', 'single']:
                    normalized_scopes.extend(['sentence', 'single'])
                else:
                    normalized_scopes.append(s)
            df = df[df['scope'].isin(normalized_scopes)]

        if filter_config.generation_types:
            df = df[df['generation_type'].isin(filter_config.generation_types)]
        if filter_config.model_names:
            df = df[df['model_name'].isin(filter_config.model_names)]
        if filter_config.llm_ratios:
            ratios_arr = np.array(filter_config.llm_ratios)
            df = df[df['llm_ratio'].apply(lambda r: any(np.isclose(r, ratios_arr)))]
        if filter_config.labels:
            df = df[df['label'].isin(filter_config.labels)]

        # Group-Stratified Subsampling on _id
        if sample_size > 0 and len(df) > sample_size:
            id_col = '_id' if '_id' in df.columns else ('doc_id' if 'doc_id' in df.columns else 'id')
            if id_col in df.columns:
                unique_ids = df[id_col].unique()
                avg_rows = len(df) / len(unique_ids)
                target_groups = max(1, int(sample_size / avg_rows))
                
                if target_groups < len(unique_ids):
                    rng = np.random.default_rng(seed)
                    sampled_ids = rng.choice(unique_ids, size=target_groups, replace=False)
                    df = df[df[id_col].isin(sampled_ids)].copy().reset_index(drop=True)
            else:
                df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

        return df

    def get_samples(self, filter_config: Optional[DataFilter] = None, **kwargs) -> List[TextSample]:
        """
        Retrieves filtered data as a list of typed TextSample dataclass objects.
        """
        filtered_df = self.filter_dataframe(filter_config, **kwargs)
        return [TextSample.from_dict(row) for row in filtered_df.to_dict(orient="records")]

    def get_sklearn_data(self, filter_config: Optional[DataFilter] = None, **kwargs) -> Tuple[List[str], List[int]]:
        """
        Extracts (X_text, y_labels) tuple suitable for scikit-learn classifiers (e.g. SVM).
        """
        filtered_df = self.filter_dataframe(filter_config, **kwargs)
        X = filtered_df['text'].tolist()
        y = filtered_df['label'].tolist()
        return X, y

    def get_pytorch_dataset(
        self, 
        filter_config: Optional[DataFilter] = None, 
        tokenizer: Optional[Any] = None, 
        max_length: int = 512,
        **kwargs
    ):
        """
        Returns a PyTorch Dataset instance over the filtered samples.
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for `get_pytorch_dataset()`. Please install `torch`.")
        samples = self.get_samples(filter_config, **kwargs)
        return PyTorchDetectionDataset(samples, tokenizer=tokenizer, max_length=max_length)

    def get_hf_dataset(self, filter_config: Optional[DataFilter] = None, **kwargs):
        """
        Returns a Hugging Face Dataset or DatasetDict.
        """
        if not HAS_HF:
            raise ImportError("Hugging Face `datasets` library is required for `get_hf_dataset()`.")
        
        filtered_df = self.filter_dataframe(filter_config, **kwargs)
        
        # If no explicit split was passed in filter, return DatasetDict grouped by split
        if filter_config is None or filter_config.splits is None:
            dataset_dict = {}
            for split_name in filtered_df['split'].unique():
                split_df = filtered_df[filtered_df['split'] == split_name]
                dataset_dict[split_name] = HFDataset.from_pandas(split_df, preserve_index=False)
            return DatasetDict(dataset_dict)
        else:
            return HFDataset.from_pandas(filtered_df, preserve_index=False)