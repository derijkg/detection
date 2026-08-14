# src/data/data_loader.py

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np

# Calculate project root dynamically (~/detection)
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
        return asdict(self)


@dataclass
class DataFilter:
    splits: Optional[List[str]] = None
    scopes: Optional[List[str]] = None
    generation_types: Optional[List[str]] = None
    model_names: Optional[List[str]] = None
    llm_ratios: Optional[List[float]] = None
    labels: Optional[List[int]] = None


if HAS_TORCH:
    class PyTorchDetectionDataset(TorchDataset):
        def __init__(self, samples: List[TextSample], tokenizer: Optional[Any] = None, max_length: int = 256):
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
                    sample.text, padding="max_length", truncation=True, 
                    max_length=self.max_length, return_tensors="pt"
                )
                item["input_ids"] = encoded["input_ids"].squeeze(0)
                item["attention_mask"] = encoded["attention_mask"].squeeze(0)
            return item


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

    def filter_dataframe(self, filter_config: Optional[DataFilter] = None, sample_size: int = -1, seed: int = 42, **kwargs) -> pd.DataFrame:
        if filter_config is None:
            filter_config = DataFilter(**kwargs)
        else:
            for k, v in kwargs.items():
                if hasattr(filter_config, k) and v is not None:
                    setattr(filter_config, k, v)

        df = self._df.copy()

        if filter_config.splits:
            requested_splits = set(filter_config.splits)
            if 'dev' in requested_splits or 'val' in requested_splits:
                requested_splits.update(['dev', 'val'])
            df = df[df['split'].isin(requested_splits)]
            
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
        filtered_df = self.filter_dataframe(filter_config, **kwargs)
        return [TextSample.from_dict(row) for row in filtered_df.to_dict(orient="records")]

    def get_sklearn_data(self, filter_config: Optional[DataFilter] = None, **kwargs) -> Tuple[List[str], List[int]]:
        filtered_df = self.filter_dataframe(filter_config, **kwargs)
        return filtered_df['text'].tolist(), filtered_df['label'].tolist()

    def get_hf_dataset(self, filter_config: Optional[DataFilter] = None, **kwargs):
        if not HAS_HF:
            raise ImportError("Hugging Face `datasets` library required.")
        filtered_df = self.filter_dataframe(filter_config, **kwargs)
        
        if filter_config is None or filter_config.splits is None:
            dataset_dict = {}
            for split_name in filtered_df['split'].unique():
                split_df = filtered_df[filtered_df['split'] == split_name]
                dataset_dict[split_name] = HFDataset.from_pandas(split_df, preserve_index=False)
            return DatasetDict(dataset_dict)
        else:
            return HFDataset.from_pandas(filtered_df, preserve_index=False)

    def get_tokenized_dataset(
        self,
        scope: str,
        split: str,
        tokenizer: Optional[Union[str, Any]] = "microsoft/mdeberta-v3-base",
        max_length: int = 512,
        padding: Union[bool, str] = "max_length",
        model_prefix: str = "deberta",
        force_reprocess: bool = False,
        return_format: Optional[str] = "torch",
    ):
        if not HAS_HF:
            raise ImportError("Hugging Face `datasets` library required.")

        cache_dir = self.features_dir / f"{model_prefix}_{scope}" / f"{split}_tokenized"

        # Load from disk if present
        if cache_dir.exists() and any(cache_dir.iterdir()) and not force_reprocess:
            print(f"[Cache Hit] Loading pretokenized dataset: {cache_dir}")
            ds = load_from_disk(str(cache_dir))
            if return_format and HAS_TORCH:
                ds.set_format(type=return_format)
            return ds

        print(f"[Cache Miss] Pretokenizing split [{scope} | {split}]...")

        # Get filtered dataset
        split_query = [split]
        if split in ["dev", "val"]:
            split_query = ["dev", "val"]

        hf_ds = self.get_hf_dataset(DataFilter(splits=split_query, scopes=[scope]))
        if len(hf_ds) == 0:
            raise ValueError(f"No samples found for scope='{scope}' and split='{split}'.")

        # Resolve tokenizer
        if isinstance(tokenizer, str):
            if not HAS_TRANSFORMERS:
                raise ImportError("`transformers` library required to load tokenizer by name.")
            tokenizer_obj = AutoTokenizer.from_pretrained(tokenizer)
        elif tokenizer is not None:
            tokenizer_obj = tokenizer
        else:
            raise ValueError("Must provide tokenizer instance or name.")

        # Map tokenization
        def _tokenize_fn(batch):
            return tokenizer_obj(
                batch["text"],
                padding=padding,
                truncation=True,
                max_length=max_length
            )

        tokenized_ds = hf_ds.map(
            _tokenize_fn,
            batched=True,
            desc=f"Tokenizing {model_prefix}_{scope}/{split}_tokenized"
        )

        # Save pretokenized cache to disk
        cache_dir.mkdir(parents=True, exist_ok=True)
        tokenized_ds.save_to_disk(str(cache_dir))
        print(f"[Saved Cache] Tokenized dataset saved to: {cache_dir}")

        if return_format and HAS_TORCH:
            tokenized_ds.set_format(type=return_format)

        return tokenized_ds

    def build_all_tokenized_caches(
        self,
        scopes: List[str] = ["full", "sentence"],
        splits: List[str] = ["train", "dev", "test"],
        tokenizer: Union[str, Any] = "microsoft/mdeberta-v3-base",
        max_length: int = 512,
        model_prefix: str = "deberta",
        force_reprocess: bool = False
    ) -> Dict[str, Any]:
        tokenized_dict = {}
        for scope in scopes:
            for split in splits:
                try:
                    key = f"{scope}_{split}"
                    tokenized_dict[key] = self.get_tokenized_dataset(
                        scope=scope,
                        split=split,
                        tokenizer=tokenizer,
                        max_length=max_length,
                        model_prefix=model_prefix,
                        force_reprocess=force_reprocess
                    )
                except ValueError as e:
                    print(f"Skipping ({scope}, {split}): {e}")
        return tokenized_dict