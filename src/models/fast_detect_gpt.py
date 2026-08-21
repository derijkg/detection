# src/models/fast_detect_gpt.py

import gc
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models.base import BaseDetector


def load_causal_tokenizer(model_name: str, cache_dir: Optional[str] = None) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        padding_side="left",
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_model(model_name: str, device: str, cache_dir: Optional[str] = None) -> AutoModelForCausalLM:
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    if device != "cpu":
        model = model.to(device)
    model.eval()
    return model


def get_sampling_discrepancy_analytic(logits_ref: torch.Tensor, logits_score: torch.Tensor, labels: torch.Tensor) -> float:
    if logits_ref.size(-1) != logits_score.size(-1):
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    min_seq_len = min(logits_ref.size(1), logits_score.size(1), labels.size(1))
    if min_seq_len == 0:
        return 0.0

    logits_ref = logits_ref[:, :min_seq_len, :]
    logits_score = logits_score[:, :min_seq_len, :]
    labels = labels[:, :min_seq_len]

    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)
    
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    
    var_ref = torch.clamp((probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref), min=0.0)
    var_sum = var_ref.sum(dim=-1).clamp(min=1e-9).sqrt()
    
    discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_sum
    val = float(discrepancy.mean().item())
    return val if np.isfinite(val) else 0.0


def compute_batch_sampling_discrepancy(
    logits_ref: torch.Tensor,
    logits_score: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor
) -> torch.Tensor:
    if logits_ref.size(-1) != logits_score.size(-1):
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    min_seq_len = min(logits_ref.size(1), logits_score.size(1), labels.size(1))
    if min_seq_len == 0:
        return torch.zeros(logits_score.size(0), device=logits_score.device)

    logits_ref = logits_ref[:, :min_seq_len, :]
    logits_score = logits_score[:, :min_seq_len, :]
    labels = labels[:, :min_seq_len]

    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)

    log_likelihood = lprobs_score.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = torch.clamp((probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref), min=0.0)

    # Valid causal transition requires BOTH the context token and the predicted token to be non-pad
    context_mask = attention_mask[:, :min_seq_len].float()
    target_mask = attention_mask[:, 1:min_seq_len + 1].float()
    valid_mask = context_mask * target_mask

    masked_ll = (log_likelihood * valid_mask).sum(dim=-1)
    masked_mean = (mean_ref * valid_mask).sum(dim=-1)
    masked_var = (var_ref * valid_mask).sum(dim=-1)

    var_sum = torch.clamp(masked_var, min=1e-9).sqrt()
    discrepancies = (masked_ll - masked_mean) / var_sum
    return discrepancies


class FastDetectGPTDetector(BaseDetector):
    def __init__(
        self,
        scoring_model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        sampling_model_name: str = "Qwen/Qwen2.5-3B",
        scope: str = "full",
        seed: int = 42,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        max_length: Optional[int] = None,
        batch_size: int = 8,
        log_dir: Optional[Union[str, Path]] = None,
        **kwargs
    ):
        super().__init__(model_name="fdgpt", scope=scope, seed=seed, log_dir=log_dir)
        self.scoring_model_name = scoring_model_name
        self.sampling_model_name = sampling_model_name
        self.max_length = max_length or (128 if scope == "sentence" else 384)
        self.batch_size = batch_size
        self.cache_dir = cache_dir

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.is_same_model = (scoring_model_name == sampling_model_name)

        if torch.cuda.is_available() and torch.cuda.device_count() >= 2 and not self.is_same_model:
            self.scoring_device = "cuda:0"
            self.sampling_device = "cuda:1"
        else:
            self.scoring_device = self.device
            self.sampling_device = self.device

        self.logger.info(f"Loading Scoring Model: {scoring_model_name} on {self.scoring_device}")
        self.scoring_tokenizer = load_causal_tokenizer(scoring_model_name, cache_dir)
        self.scoring_model = load_causal_model(scoring_model_name, self.scoring_device, cache_dir)

        if self.is_same_model:
            self.sampling_tokenizer = self.scoring_tokenizer
            self.sampling_model = self.scoring_model
        else:
            self.logger.info(f"Loading Sampling Model: {sampling_model_name} on {self.sampling_device}")
            self.sampling_tokenizer = load_causal_tokenizer(sampling_model_name, cache_dir)
            self.sampling_model = load_causal_model(sampling_model_name, self.sampling_device, cache_dir)

        self.mu0: float = 0.0
        self.sigma0: float = 1.0
        self.mu1: float = 1.0
        self.sigma1: float = 1.0
        self.is_calibrated: bool = False


    def _compute_discrepancy_batch(self, batch_texts: List[str]) -> np.ndarray:
        """Helper to run batched forward passes and return raw discrepancy scores."""
        tok_kwargs = {
            "return_tensors": "pt",
            "padding": True,
            "truncation": (self.max_length is not None),
            "max_length": self.max_length,
            "return_token_type_ids": False
        }
        inputs_score = self.scoring_tokenizer(batch_texts, **tok_kwargs).to(self.scoring_device)
        labels = inputs_score.input_ids[:, 1:]

        if labels.shape[1] == 0:
            return np.zeros(len(batch_texts), dtype=np.float32)

        with torch.inference_mode():
            logits_score = self.scoring_model(**inputs_score).logits[:, :-1]
            if self.is_same_model:
                logits_ref = logits_score
            else:
                inputs_ref = self.sampling_tokenizer(batch_texts, **tok_kwargs).to(self.sampling_device)
                logits_ref = self.sampling_model(**inputs_ref).logits[:, :-1].to(self.scoring_device)

            discrepancies = compute_batch_sampling_discrepancy(
                logits_ref=logits_ref,
                logits_score=logits_score,
                labels=labels,
                attention_mask=inputs_score.attention_mask
            )
        return discrepancies.detach().cpu().numpy()

    
    def calculate_prob(self, raw_score: float) -> float:
        if not np.isfinite(raw_score):
            return 0.5

        if not self.is_calibrated:
            return float(1.0 / (1.0 + np.exp(-raw_score)))

        s0 = max(self.sigma0, 1e-4)
        s1 = max(self.sigma1, 1e-4)

        log_pdf0 = -np.log(s0) - 0.5 * ((raw_score - self.mu0) / s0) ** 2
        log_pdf1 = -np.log(s1) - 0.5 * ((raw_score - self.mu1) / s1) ** 2
        log_odds = np.clip(log_pdf1 - log_pdf0, -50.0, 50.0)

        return float(1.0 / (1.0 + np.exp(-log_odds)))


    def fit(
        self, 
        train_data: Union[pd.DataFrame, List[Dict[str, Any]]], 
        y_train: Optional[np.ndarray] = None, 
        batch_size: Optional[int] = None,
        **kwargs
    ) -> "FastDetectGPTDetector":
        df = pd.DataFrame(train_data)
        if "label" not in df.columns and y_train is not None:
            df["label"] = y_train

        bs = batch_size or self.batch_size
        self.logger.info(f"Batched Gaussian calibration on {len(df)} samples (Batch Size: {bs})...")

        for label_val, name in [(0, "Human"), (1, "AI")]:
            sub_df = df[df["label"] == label_val]
            texts = sub_df["text"].dropna().astype(str).tolist()
            scores = []

            for i in range(0, len(texts), bs):
                batch = texts[i : i + bs]
                try:
                    b_scores = self._compute_discrepancy_batch(batch)
                    scores.extend(b_scores[np.isfinite(b_scores)].tolist())
                except Exception as e:
                    self.logger.warning(f"Batch failed during calibration, falling back to individual scoring: {e}")
                    for t in batch:
                        s = self.compute_discrepancy(t)
                        if s is not None and np.isfinite(s):
                            scores.append(s)

            if label_val == 0:
                self.mu0 = float(np.mean(scores)) if scores else 0.0
                self.sigma0 = max(float(np.std(scores, ddof=1)) if len(scores) > 1 else 1.0, 1e-4)
            else:
                self.mu1 = float(np.mean(scores)) if scores else 1.0
                self.sigma1 = max(float(np.std(scores, ddof=1)) if len(scores) > 1 else 1.0, 1e-4)

        self.is_calibrated = True
        self.logger.info(f"Fitted Human Distribution: mu0 = {self.mu0:.4f}, sigma0 = {self.sigma0:.4f}")
        self.logger.info(f"Fitted AI Distribution:    mu1 = {self.mu1:.4f}, sigma1 = {self.sigma1:.4f}")
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

        if len(raw_texts) == 0:
            return np.array([], dtype=np.float32)

        all_probs = []
        batch_size = max(1, self.batch_size)

        for i in range(0, len(raw_texts), batch_size):
            batch_texts = raw_texts[i:i + batch_size]
            tok_kwargs = {
                "return_tensors": "pt",
                "padding": True,
                "truncation": (self.max_length is not None),
                "max_length": self.max_length,
                "return_token_type_ids": False
            }

            try:
                inputs_score = self.scoring_tokenizer(batch_texts, **tok_kwargs).to(self.scoring_device)
                labels = inputs_score.input_ids[:, 1:]

                if labels.shape[1] == 0:
                    all_probs.extend([0.5] * len(batch_texts))
                    continue

                with torch.inference_mode():
                    logits_score = self.scoring_model(**inputs_score).logits[:, :-1]
                    if self.is_same_model:
                        logits_ref = logits_score
                    else:
                        inputs_ref = self.sampling_tokenizer(batch_texts, **tok_kwargs).to(self.sampling_device)
                        logits_ref = self.sampling_model(**inputs_ref).logits[:, :-1].to(self.scoring_device)

                    discrepancies = compute_batch_sampling_discrepancy(
                        logits_ref=logits_ref,
                        logits_score=logits_score,
                        labels=labels,
                        attention_mask=inputs_score.attention_mask
                    )

                for d in discrepancies.detach().cpu().numpy():
                    all_probs.append(self.calculate_prob(float(d)))

            except Exception:
                for t in batch_texts:
                    try:
                        s = self.compute_discrepancy(t)
                        all_probs.append(self.calculate_prob(s) if s is not None else 0.5)
                    except Exception:
                        all_probs.append(0.5)

            if i % (batch_size * 10) == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        return np.array(all_probs, dtype=np.float32)

    def save(self, path: Union[str, Path]):
        save_p = Path(path)
        if save_p.is_dir() or not save_p.name.endswith(".json"):
            save_p = save_p / "model_calibration.json"
        save_p.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "scoring_model_name": self.scoring_model_name,
            "sampling_model_name": self.sampling_model_name,
            "scope": self.scope,
            "max_length": self.max_length,
            "distribution_params": {
                "mu0": self.mu0, "sigma0": self.sigma0,
                "mu1": self.mu1, "sigma1": self.sigma1,
            },
            "calibrated_threshold": self.calibrated_threshold,
            "is_calibrated": self.is_calibrated
        }
        with open(save_p, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4)
        self.logger.info(f"Saved FastDetectGPT calibration metadata to: {save_p}")

    @classmethod
    def load(cls, path: Union[str, Path], device: Optional[str] = None, **kwargs) -> "FastDetectGPTDetector":
        load_p = Path(path)
        if load_p.is_dir():
            load_p = load_p / "model_calibration.json"
        if not load_p.exists():
            raise FileNotFoundError(f"Calibration file not found at: {load_p}")
        meta = json.loads(load_p.read_text(encoding="utf-8"))

        detector = cls(
            scoring_model_name=meta["scoring_model_name"],
            sampling_model_name=meta["sampling_model_name"],
            scope=meta.get("scope", "full"),
            max_length=meta.get("max_length", None),
            device=device,
            **kwargs
        )
        detector.mu0 = float(meta["distribution_params"]["mu0"])
        detector.sigma0 = float(meta["distribution_params"]["sigma0"])
        detector.mu1 = float(meta["distribution_params"]["mu1"])
        detector.sigma1 = float(meta["distribution_params"]["sigma1"])
        detector.calibrated_threshold = float(meta.get("calibrated_threshold", 0.5))
        detector.is_calibrated = meta.get("is_calibrated", True)
        return detector