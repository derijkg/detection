# src/training/trainer_deberta.py

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.special import softmax
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import Trainer, TrainerCallback


def compute_stratified_sample_weights(df: pd.DataFrame) -> torch.Tensor:
    model_col = next((c for c in ["model_name", "generator_model", "generator"] if c in df.columns), None)
    if model_col:
        group_keys = df["label"].astype(str) + "___" + df[model_col].astype(str)
    else:
        group_keys = df["label"].astype(str)

    group_counts = group_keys.value_counts().to_dict()
    raw_weights = group_keys.map(lambda k: 1.0 / max(group_counts[k], 1)).values.astype(np.float64)

    neg_mask = (df["label"] == 0).values
    pos_mask = (df["label"] == 1).values

    if neg_mask.sum() > 0 and pos_mask.sum() > 0:
        raw_weights[neg_mask] = (raw_weights[neg_mask] / raw_weights[neg_mask].sum()) * 0.5
        raw_weights[pos_mask] = (raw_weights[pos_mask] / raw_weights[pos_mask].sum()) * 0.5
    elif neg_mask.sum() > 0:
        raw_weights[neg_mask] = raw_weights[neg_mask] / raw_weights[neg_mask].sum()
    elif pos_mask.sum() > 0:
        raw_weights[pos_mask] = raw_weights[pos_mask] / raw_weights[pos_mask].sum()

    return torch.tensor(raw_weights, dtype=torch.float32)


class RockafellarUryasevCVaRLoss(nn.Module):
    """
    Rockafellar-Uryasev (2000) Conditional Value-at-Risk (CVaR) loss for low-FPR optimization.
    Penalizes upper alpha-tail of human (negative) sample cross-entropy losses.
    """
    def __init__(self, alpha: float = 0.01, lambda_neg: float = 2.0, initial_eta: float = 0.693, temp: float = 0.1):
        super().__init__()
        self.alpha = float(max(alpha, 1e-4))
        self.lambda_neg = float(lambda_neg)
        self.temp = float(max(temp, 1e-3))
        self.eta = nn.Parameter(torch.tensor(initial_eta, dtype=torch.float32))
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        losses = self.ce(logits, targets)
        pos_mask = (targets == 1)
        neg_mask = (targets == 0)

        n_pos = pos_mask.sum()
        pos_loss = losses[pos_mask].mean() if n_pos > 0 else torch.tensor(0.0, device=logits.device)

        n_neg = neg_mask.sum()
        if n_neg > 0:
            neg_losses = losses[neg_mask]
            eta_dev = torch.clamp(self.eta.to(logits.device), min=0.0)
            diff = torch.clamp((neg_losses - eta_dev) / self.temp, -50.0, 50.0)
            smooth_excess = self.temp * F.softplus(diff)
            cvar_neg_loss = eta_dev + (1.0 / self.alpha) * smooth_excess.mean()
        else:
            cvar_neg_loss = torch.tensor(0.0, device=logits.device)

        if n_pos > 0 and n_neg > 0:
            return (pos_loss + self.lambda_neg * cvar_neg_loss) / (1.0 + self.lambda_neg)
        elif n_neg > 0:
            return cvar_neg_loss
        else:
            return pos_loss


class CVaRTrackingCallback(TrainerCallback):
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.records: List[Dict] = []

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs is None:
            return
        
        step = state.global_step
        epoch = state.epoch

        eta_val = None
        unwrapped = getattr(model, "module", model)
        if hasattr(unwrapped, "custom_loss_fn") and hasattr(unwrapped.custom_loss_fn, "eta"):
            eta_val = float(unwrapped.custom_loss_fn.eta.detach().cpu().item())

        entry = {
            "step": int(step),
            "epoch": round(float(epoch), 4) if epoch is not None else None,
            "eta": eta_val,
            "train_loss": logs.get("loss"),
            "learning_rate": logs.get("learning_rate"),
            "eval_loss": logs.get("eval_loss"),
            "eval_pauc_1fpr": logs.get("eval_pauc_1fpr"),
            "eval_tpr_at_1fpr": logs.get("eval_tpr_at_1fpr"),
        }
        self.records.append(entry)

        df = pd.DataFrame(self.records)
        df.to_csv(self.output_dir / "cvar_history.csv", index=False)


class ImbalancedLowFPRTrainer(Trainer):
    def __init__(
        self, 
        *args, 
        sample_weights: Optional[torch.Tensor] = None, 
        use_pauc_loss: bool = True, 
        target_fpr: float = 0.01, 
        lambda_neg: float = 2.0, 
        llrd_decay: float = 0.90, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.sample_weights = sample_weights
        self.use_pauc_loss = use_pauc_loss
        self.llrd_decay = llrd_decay
        if self.use_pauc_loss:
            self.custom_loss_fn = RockafellarUryasevCVaRLoss(alpha=target_fpr, lambda_neg=lambda_neg)
            self.model.custom_loss_fn = self.custom_loss_fn

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Training requires a train_dataset.")

        if self.sample_weights is not None:
            sampler = WeightedRandomSampler(
                weights=self.sample_weights,
                num_samples=len(self.sample_weights),
                replacement=True
            )
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                sampler=sampler,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
        return super().get_train_dataloader()

    def create_optimizer(self):
        if self.optimizer is None:
            base_lr = self.args.learning_rate
            weight_decay = self.args.weight_decay
            no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
            num_layers = getattr(self.model.config, "num_hidden_layers", 12)

            loss_param_ids = set()
            if self.use_pauc_loss and hasattr(self, "custom_loss_fn"):
                loss_param_ids = {id(p) for p in self.custom_loss_fn.parameters()}

            param_groups: Dict[Tuple[float, float], List[torch.nn.Parameter]] = {}

            for name, param in self.model.named_parameters():
                if not param.requires_grad or id(param) in loss_param_ids:
                    continue

                wd = 0.0 if any(nd in name for nd in no_decay) else weight_decay

                if "classifier" in name:
                    lr = base_lr * 1.5
                elif "encoder.layer." in name:
                    layer_idx = int(name.split("encoder.layer.")[1].split(".")[0])
                    lr = base_lr * (self.llrd_decay ** (num_layers - 1 - layer_idx))
                elif "embeddings" in name:
                    lr = base_lr * (self.llrd_decay ** num_layers)
                else:
                    lr = base_lr

                key = (lr, wd)
                param_groups.setdefault(key, []).append(param)

            grouped_parameters = [
                {"params": params, "lr": lr, "weight_decay": wd}
                for (lr, wd), params in param_groups.items()
                if len(params) > 0
            ]

            if self.use_pauc_loss and hasattr(self, "custom_loss_fn"):
                self.custom_loss_fn.to(self.args.device)
                cvar_params = [p for p in self.custom_loss_fn.parameters() if p.requires_grad]
                if cvar_params:
                    grouped_parameters.append({
                        "params": cvar_params,
                        "lr": base_lr * 5.0,
                        "weight_decay": 0.0,
                    })

            opt_cls, opt_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = opt_cls(grouped_parameters, **opt_kwargs)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.use_pauc_loss and labels is not None:
            self.custom_loss_fn.to(logits.device)
            loss = self.custom_loss_fn(logits, labels)
        else:
            loss = outputs.loss

        return (loss, outputs) if return_outputs else loss


def compute_deberta_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    probs = softmax(logits, axis=-1)[:, 1]
    probs = np.nan_to_num(probs, nan=0.5)

    if len(np.unique(labels)) < 2:
        return {"pauc_1fpr": 0.5, "tpr_at_1fpr": 0.0, "roc_auc": 0.5, "accuracy": 0.0, "f1": 0.0}

    fpr, tpr, _ = roc_curve(labels, probs)
    interp_fn = interp1d(fpr, tpr, bounds_error=False, fill_value=(0.0, 1.0))
    tpr_at_1fpr = float(interp_fn(0.01))

    try:
        pauc_1fpr = float(roc_auc_score(labels, probs, max_fpr=0.01))
        overall_auc = float(roc_auc_score(labels, probs))
    except Exception:
        pauc_1fpr, overall_auc = 0.5, 0.5

    acc = float(accuracy_score(labels, preds))
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)

    return {
        "pauc_1fpr": pauc_1fpr,
        "tpr_at_1fpr": tpr_at_1fpr,
        "roc_auc": overall_auc,
        "accuracy": acc,
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
    }