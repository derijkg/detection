# src/models/deberta.py

import gc
import json
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    DebertaV2Model,
    DebertaV2PreTrainedModel,
    EarlyStoppingCallback,
    TrainingArguments,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from src.models.base import BaseDetector


class MultiSampleDropoutHead(nn.Module):
    """
    Multi-Pooling ([CLS] + Mean + Max) fused representation with Multi-Sample Dropout.
    """
    def __init__(
        self, 
        hidden_size: int = 768, 
        num_labels: int = 2, 
        drop_rates: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5)
    ):
        super().__init__()
        self.dense = nn.Linear(hidden_size * 3, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in drop_rates])
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        cls_rep = last_hidden_state[:, 0, :]

        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeds = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        mean_rep = sum_embeds / sum_mask

        min_val = torch.finfo(last_hidden_state.dtype).min
        masked_hidden = last_hidden_state.masked_fill(~attention_mask.unsqueeze(-1).bool(), min_val)
        max_rep = torch.max(masked_hidden, dim=1)[0]

        fused = torch.cat([cls_rep, mean_rep, max_rep], dim=-1)
        features = F.gelu(self.layer_norm(self.dense(fused)))

        if self.training:
            logits_list = [self.out_proj(d(features)) for d in self.dropouts]
            return torch.mean(torch.stack(logits_list, dim=0), dim=0)
        return self.out_proj(features)


class CustomMDeBERTaForDetection(DebertaV2PreTrainedModel):
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = getattr(config, "num_labels", 2)
        self.deberta = DebertaV2Model(config)
        self.classifier = MultiSampleDropoutHead(hidden_size=config.hidden_size, num_labels=self.num_labels)
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self, 
        input_ids: Optional[torch.Tensor] = None, 
        attention_mask: Optional[torch.Tensor] = None, 
        token_type_ids: Optional[torch.Tensor] = None, 
        labels: Optional[torch.Tensor] = None, 
        **kwargs
    ) -> SequenceClassifierOutput:
        if attention_mask is None and input_ids is not None:
            attention_mask = torch.ones_like(input_ids)

        outputs = self.deberta(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids,
            return_dict=True
        )
        logits = self.classifier(outputs.last_hidden_state, attention_mask)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MDeBERTaDetector(BaseDetector):
    """
    Polymorphic BaseDetector for fine-tuned mDeBERTa-v3 sequence classification.
    """
    def __init__(
        self,
        model_path: str = "microsoft/mdeberta-v3-base",
        scope: str = "full",
        max_length: Optional[int] = None,
        batch_size: int = 32,
        seed: int = 42,
        device: Optional[str] = None,
        log_dir: Optional[Union[str, Path]] = None,
        **kwargs
    ):
        super().__init__(model_name="mdeberta", scope=scope, seed=seed, log_dir=log_dir)
        self.model_path = str(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length or (128 if scope == "sentence" else 384)
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
        config = AutoConfig.from_pretrained(self.model_path)
        config.num_labels = 2
        self.model = CustomMDeBERTaForDetection.from_pretrained(self.model_path, config=config)
        self.model.eval().to(self.device)

    def fit(
        self, 
        train_data: Union[pd.DataFrame, List[Dict[str, Any]]], 
        y_train: Optional[np.ndarray] = None,
        dev_data: Optional[pd.DataFrame] = None,
        epochs: int = 3,
        lr: float = 2.5e-5,
        llrd_decay: float = 0.90,
        lambda_neg: float = 2.0,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        target_fpr: float = 0.01,
        use_pauc_loss: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
        tune: bool = False,
        n_trials: int = 10,
        tuning_sample_size: int = 4000,
        val_sample_size: int = 2000,
        batch_size: Optional[int] = None,
        gradient_accumulation_steps: Optional[int] = None,
        **kwargs
    ) -> "MDeBERTaDetector":
        from src.training.trainer_deberta import (
            CVaRTrackingCallback,
            ImbalancedLowFPRTrainer,
            compute_deberta_metrics,
            compute_stratified_sample_weights
        )
        from src.training.tune_deberta import DebertaOptunaTuner, DEBERTA_SEARCH_SPACES, get_or_create_cached_hf_dataset
        from src.visualization.latex_tables import export_deberta_hyperparameters_table

        df_train = pd.DataFrame(train_data)
        if "label" not in df_train.columns and y_train is not None:
            df_train["label"] = y_train

        out_path = Path(output_dir or f"./output/deberta_{self.scope}")
        out_path.mkdir(parents=True, exist_ok=True)
        scratch_dir = out_path / "checkpoints_tmp"
        params_file = out_path / "best_params.json"

        if tune and dev_data is not None:
            self.logger.info(f"Running Optuna tuning for mDeBERTa [{self.scope.upper()}]...")
            best_params, tune_sz = DebertaOptunaTuner.run(
                train_df=df_train,
                dev_df=dev_data,
                scope=self.scope,
                n_trials=n_trials,
                tuning_sample_size=tuning_sample_size,
                val_sample_size=val_sample_size,
                target_fpr=target_fpr,
                seed=self.seed,
                model_name=self.model_path,
                output_dir=out_path
            )
            lr = float(best_params.get("learning_rate", lr))
            llrd_decay = float(best_params.get("llrd_decay", llrd_decay))
            lambda_neg = float(best_params.get("lambda_neg", lambda_neg))
            weight_decay = float(best_params.get("weight_decay", weight_decay))
            warmup_ratio = float(best_params.get("warmup_ratio", warmup_ratio))

            latex_dir = out_path / "latex_tables"
            latex_dir.mkdir(parents=True, exist_ok=True)
            export_deberta_hyperparameters_table(
                best_params=best_params,
                search_spaces=DEBERTA_SEARCH_SPACES,
                scope=self.scope,
                output_path=latex_dir / f"table_hyperparams_deberta_{self.scope}.tex",
                tuning_sample_size=tune_sz,
                final_sample_size=len(df_train),
                n_trials=n_trials
            )
        elif params_file.exists():
            self.logger.info(f"Loaded existing best hyperparameters from: {params_file}")
            best_params = json.loads(params_file.read_text(encoding="utf-8"))
            lr = float(best_params.get("learning_rate", lr))
            llrd_decay = float(best_params.get("llrd_decay", llrd_decay))
            lambda_neg = float(best_params.get("lambda_neg", lambda_neg))
            weight_decay = float(best_params.get("weight_decay", weight_decay))
            warmup_ratio = float(best_params.get("warmup_ratio", warmup_ratio))

        # 2. Dataset Tokenization & Caching
        train_ds = get_or_create_cached_hf_dataset(df_train, self.tokenizer, max_len=self.max_length, cache_tag=f"train_{self.scope}")
        dev_ds = get_or_create_cached_hf_dataset(dev_data, self.tokenizer, max_len=self.max_length, cache_tag=f"dev_{self.scope}") if dev_data is not None else None


        train_bs = batch_size or (32 if self.max_length <= 128 else (16 if self.max_length <= 256 else 8))
        grad_accum = gradient_accumulation_steps or (1 if self.max_length <= 128 else (2 if self.max_length <= 256 else 4))
        use_grad_ckpt = (self.max_length > 256)
        has_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

        effective_batch_size = train_bs * grad_accum
        steps_per_epoch = max(1, len(train_ds) // effective_batch_size)

        if dev_ds is not None:
            eval_steps = max(50, min(500, steps_per_epoch // 4))
            eval_strategy = "steps"
            save_strategy = "steps"
            early_stopping_patience = 5
        else:
            eval_steps = None
            eval_strategy = "no"
            save_strategy = "no"
            early_stopping_patience = None

        training_args = TrainingArguments(
            output_dir=str(scratch_dir),
            eval_strategy=eval_strategy,
            save_strategy=save_strategy,
            eval_steps=eval_steps,
            save_steps=eval_steps,
            save_total_limit=2,
            learning_rate=lr,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            adam_epsilon=1e-6,
            max_grad_norm=1.0,
            per_device_train_batch_size=train_bs,
            per_device_eval_batch_size=64 if self.max_length <= 128 else 16,
            gradient_accumulation_steps=grad_accum,
            gradient_checkpointing=use_grad_ckpt,
            gradient_checkpointing_kwargs={"use_reentrant": False} if use_grad_ckpt else None,
            bf16=has_bf16,
            fp16=(not has_bf16 and torch.cuda.is_available()),
            num_train_epochs=epochs,
            load_best_model_at_end=(dev_ds is not None),
            metric_for_best_model="pauc_1fpr" if dev_ds is not None else None,
            greater_is_better=True,
            report_to="none",
            logging_steps=25,
        )

        cvar_cb = CVaRTrackingCallback(output_dir=out_path)
        sample_weights = compute_stratified_sample_weights(df_train)

        callbacks = [cvar_cb]
        if dev_ds is not None and early_stopping_patience is not None:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

        trainer = ImbalancedLowFPRTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=dev_ds,
            sample_weights=sample_weights,
            processing_class=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            compute_metrics=compute_deberta_metrics if dev_ds is not None else None,
            callbacks=callbacks,
            use_pauc_loss=use_pauc_loss,
            target_fpr=target_fpr,
            lambda_neg=lambda_neg,
            llrd_decay=llrd_decay,
        )

        self.logger.info(f"Training mDeBERTa-v3 detector [Scope: {self.scope.upper()}]...")
        trainer.train()
        self.model.eval().to(self.device)

        if scratch_dir.exists():
            shutil.rmtree(scratch_dir, ignore_errors=True)

        return self

    def predict_proba(self, texts: Union[List[str], List[Dict[str, Any]], pd.DataFrame, np.ndarray]) -> np.ndarray:
        if isinstance(texts, pd.DataFrame):
            raw_texts = texts["text"].astype(str).tolist()
        elif isinstance(texts, (list, np.ndarray)) and len(texts) > 0 and isinstance(texts[0], str):
            raw_texts = [str(t) for t in texts]
        elif isinstance(texts, list) and len(texts) > 0 and isinstance(texts[0], dict):
            raw_texts = [str(r.get("text", "")) for r in texts]
        else:
            raw_texts = [str(t) for t in texts]

        if not raw_texts:
            return np.array([], dtype=np.float32)

        all_probs = []
        is_cuda = "cuda" in str(self.device)
        amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

        with torch.inference_mode():
            for i in range(0, len(raw_texts), self.batch_size):
                batch = raw_texts[i : i + self.batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                ).to(self.device)

                with torch.autocast(device_type="cuda" if is_cuda else "cpu", dtype=amp_dtype, enabled=is_cuda):
                    outputs = self.model(**encoded)
                    batch_probs = F.softmax(outputs.logits.detach().float(), dim=-1)[:, 1].cpu().numpy()

                all_probs.append(batch_probs)

        return np.concatenate(all_probs, axis=0) if all_probs else np.array([], dtype=np.float32)

    def save(self, path: Union[str, Path]):
        save_p = Path(path)
        if save_p.name != "best_model" and not (save_p / "config.json").exists():
            target_p = save_p / "best_model" if not save_p.name.endswith(".bin") else save_p
        else:
            target_p = save_p

        target_p.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(target_p))
        self.tokenizer.save_pretrained(str(target_p))
        meta = {"calibrated_threshold": self.calibrated_threshold, "scope": self.scope}
        (target_p / "detector_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self.logger.info(f"Saved mDeBERTa model to: {target_p}")

    @classmethod
    def load(cls, path: Union[str, Path], scope: str = "full", device: Optional[str] = None, **kwargs) -> "MDeBERTaDetector":
        load_p = Path(path)
        if (load_p / "best_model").exists():
            target_p = load_p / "best_model"
        elif (load_p / "checkpoint").exists():
            target_p = load_p / "checkpoint"
        else:
            target_p = load_p

        detector = cls(model_path=str(target_p), scope=scope, device=device, **kwargs)
        meta_file = target_p / "detector_meta.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            detector.calibrated_threshold = float(meta.get("calibrated_threshold", 0.5))
        return detector