# detection/src/models/deberta.py

import os
# Silence Hugging Face Windows Symlink Warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import inspect
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)

from src.models.base_model import BaseDetector
from src.training.metrics import hf_compute_metrics

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FEATURE_CACHE_DIR = PROJECT_ROOT / "data_static" / "model_features"


def build_safe_training_args(output_dir: str, kwargs_dict: Dict[str, Any]) -> TrainingArguments:
    """Dynamically filters training arguments to guarantee compatibility with installed transformers version."""
    valid_keys = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    eval_key = "eval_strategy" if "eval_strategy" in valid_keys else "evaluation_strategy"
    
    # Enable BF16 safely on supported GPUs (RTX 30xx/40xx, A100, etc.)
    bf16_supported = torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()

    args_payload = {
        "output_dir": output_dir,
        eval_key: "epoch",
        "save_strategy": "epoch",
        "learning_rate": kwargs_dict.get("learning_rate", 1.5e-5),
        "per_device_train_batch_size": kwargs_dict.get("per_device_train_batch_size", 16),
        "per_device_eval_batch_size": kwargs_dict.get("per_device_eval_batch_size", 16),
        "num_train_epochs": kwargs_dict.get("num_train_epochs", 3),
        "weight_decay": kwargs_dict.get("weight_decay", 0.05),
        "warmup_ratio": kwargs_dict.get("warmup_ratio", 0.1),
        "label_smoothing_factor": kwargs_dict.get("label_smoothing_factor", 0.05),
        "load_best_model_at_end": True,
        "metric_for_best_model": "roc_auc",
        "greater_is_better": True,
        "fp16": kwargs_dict.get("fp16", False),  # Disabled FP16 by default to fix DeBERTa v3 gradient unscaling bug
        "bf16": kwargs_dict.get("bf16", bf16_supported),
        "logging_steps": 50,
        "report_to": "none"
    }

    filtered_payload = {k: v for k, v in args_payload.items() if k in valid_keys}
    return TrainingArguments(**filtered_payload)


class DeBERTaDetector(BaseDetector):
    def __init__(self, pretrained_model_name: str = "microsoft/mdeberta-v3-base", max_length: int = 256, granularity: str = "full", **kwargs):
        self.pretrained_model_name = pretrained_model_name
        self.max_length = max_length
        self.granularity = granularity
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name, use_fast=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name, use_fast=False)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.pretrained_model_name,
            num_labels=2,
            id2label={0: "Human", 1: "LLM"},
            label2id={"Human": 0, "LLM": 1},
            use_safetensors=True
        )
        self.trainer: Optional[Trainer] = None

    def _prepare_hf_dataset(self, dataset_or_data: Any, split_name: str = "") -> Any:
        """Loads cached tokenized dataset from disk or computes and saves it if missing."""
        from datasets import Dataset as HFDataset

        if dataset_or_data is None:
            return None

        # Determine input size for sample-size-aware caching
        expected_len = None
        if hasattr(dataset_or_data, '__len__'):
            expected_len = len(dataset_or_data)
        elif isinstance(dataset_or_data, tuple):
            expected_len = len(dataset_or_data[0])

        # Check disk cache if split_name is provided ('train' or 'dev')
        if split_name:
            cache_suffix = f"_{expected_len}" if expected_len else ""
            cache_folder = FEATURE_CACHE_DIR / f"deberta_{self.granularity}" / f"{split_name}{cache_suffix}_tokenized"
            if cache_folder.exists():
                print(f"-> [TOKENIZER CACHE HIT] Loading tokenized '{split_name}' dataset from: {cache_folder}")
                return HFDataset.load_from_disk(str(cache_folder))

        # Convert raw inputs to HF Dataset
        if hasattr(dataset_or_data, 'to_pandas'):
            df = dataset_or_data.to_pandas()
            ds = HFDataset.from_pandas(df, preserve_index=False)
        elif isinstance(dataset_or_data, tuple):
            texts, labels = dataset_or_data
            ds = HFDataset.from_dict({"text": texts, "label": labels})
        elif isinstance(dataset_or_data, list) and len(dataset_or_data) > 0 and hasattr(dataset_or_data[0], 'text'):
            ds = HFDataset.from_dict({
                "text": [s.text for s in dataset_or_data],
                "label": [s.label for s in dataset_or_data]
            })
        elif isinstance(dataset_or_data, HFDataset):
            ds = dataset_or_data
        else:
            ds = HFDataset.from_dict(dataset_or_data)

        # Standardize label column
        if 'label' not in ds.column_names and 'labels' in ds.column_names:
            ds = ds.rename_column('labels', 'label')

        # Tokenize if input_ids missing
        if "input_ids" not in ds.column_names:
            print(f"-> [TOKENIZER CACHE MISS] Tokenizing '{split_name}' dataset for scope '{self.granularity}'...")
            def tokenize_fn(examples):
                return self.tokenizer(
                    examples["text"],
                    truncation=True,
                    max_length=self.max_length,
                    padding=False
                )
            ds = ds.map(tokenize_fn, batched=True, desc="Tokenizing dataset")

        # Save tokenized dataset to disk cache
        if split_name:
            cache_suffix = f"_{len(ds)}"
            cache_folder = FEATURE_CACHE_DIR / f"deberta_{self.granularity}" / f"{split_name}{cache_suffix}_tokenized"
            cache_folder.parent.mkdir(parents=True, exist_ok=True)
            ds.save_to_disk(str(cache_folder))
            print(f"-> Saved tokenized dataset to: {cache_folder}")

        return ds

    def train(self, train_ds: Any, val_ds: Any, training_args_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Loads/tokenizes datasets from disk cache and runs Hugging Face Trainer."""
        output_dir = training_args_dict.get("output_dir", "outputs/checkpoints/deberta")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Tokenize train & val sets with disk caching
        tokenized_train = self._prepare_hf_dataset(train_ds, split_name="train")
        tokenized_val = self._prepare_hf_dataset(val_ds, split_name="dev")

        training_args = build_safe_training_args(output_dir, training_args_dict)

        callbacks = [EarlyStoppingCallback(early_stopping_patience=training_args_dict.get("early_stopping_patience", 2))]

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            processing_class=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            compute_metrics=hf_compute_metrics,
            callbacks=callbacks
        )

        self.trainer.train()
        eval_metrics = self.trainer.evaluate()
        return eval_metrics

    def predict_proba(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.model.to(self.device)
        self.model.eval()

        all_probs = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)
                all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs, axis=0)

    def save(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        if self.trainer:
            self.trainer.save_model(output_dir)
        else:
            self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"[MODEL SAVED] DeBERTa model saved to '{output_dir}'")

    def load(self, input_dir: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(input_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(input_dir)
        self.model.to(self.device)
        print(f"[MODEL LOADED] DeBERTa model loaded from '{input_dir}'")