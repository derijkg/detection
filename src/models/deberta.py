import os
import torch
import torch.nn.functional as F
import numpy as np
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


class DeBERTaDetector(BaseDetector):
    """DeBERTa sequence classification wrapper implementing BaseDetector."""

    def __init__(self, pretrained_model_name: str = "microsoft/mdeberta-v3-base", max_length: int = 256, **kwargs):
        self.pretrained_model_name = pretrained_model_name
        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.pretrained_model_name,
            num_labels=2,
            id2label={0: "Human", 1: "LLM"},
            label2id={"Human": 0, "LLM": 1},
            use_safetensors=True
        )
        self.trainer: Optional[Trainer] = None

    def train(self, train_ds: Any, val_ds: Any, training_args_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Runs Hugging Face Trainer with provided hyperparameters."""
        output_dir = training_args_dict.get("output_dir", "outputs/checkpoints/deberta")

        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=training_args_dict.get("learning_rate", 1.5e-5),
            per_device_train_batch_size=training_args_dict.get("per_device_train_batch_size", 16),
            per_device_eval_batch_size=training_args_dict.get("per_device_eval_batch_size", 16),
            num_train_epochs=training_args_dict.get("num_train_epochs", 3),
            weight_decay=training_args_dict.get("weight_decay", 0.05),
            warmup_ratio=training_args_dict.get("warmup_ratio", 0.1),
            label_smoothing_factor=training_args_dict.get("label_smoothing_factor", 0.05),
            load_best_model_at_end=True,
            metric_for_best_model="roc_auc",
            greater_is_better=True,
            fp16=torch.cuda.is_available(),
            logging_steps=50,
            report_to="none"
        )

        callbacks = [EarlyStoppingCallback(early_stopping_patience=training_args_dict.get("early_stopping_patience", 2))]

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            compute_metrics=hf_compute_metrics,
            callbacks=callbacks
        )

        self.trainer.train()
        eval_metrics = self.trainer.evaluate()
        return eval_metrics

    def predict_proba(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Computes probability matrix [P(Human), P(LLM)] for input text batch."""
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
        """Saves weights and tokenizer."""
        os.makedirs(output_dir, exist_ok=True)
        if self.trainer:
            self.trainer.save_model(output_dir)
        else:
            self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"[MODEL SAVED] DeBERTa model saved to '{output_dir}'")

    def load(self, input_dir: str) -> None:
        """Loads weights and tokenizer from disk."""
        self.tokenizer = AutoTokenizer.from_pretrained(input_dir, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(input_dir)
        self.model.to(self.device)
        print(f"[MODEL LOADED] DeBERTa model loaded from '{input_dir}'")