#!/usr/bin/env python3
"""
High-Speed, Imbalance-Aware mDeBERTa-v3 Detection Pipeline
- Complete Prediction Export (Test & Validation CSVs with Logits, Probs, Error Types, Metadata)
- Unrestricted Test Set Evaluation (Never truncated by presets)
- Auto-Generated LaTeX Summary Table for Papers
- Dynamic Memory Management & Checkpointing (Fixes Full-Abstract OOM)
- Multi-Sample Dropout & Multi-Pooling Head ([CLS] + Mean + Max)
- Stratified 50:50 Generator Batch Sampler
- Continuous CVaR-DRO Loss with Smooth Surrogate
"""

import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import WeightedRandomSampler
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    DebertaV2Model,
    DebertaV2PreTrainedModel,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.modeling_outputs import SequenceClassifierOutput

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("mDeBERTa-Detection")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.data.data_loader import DetectionDataManager
except ImportError:
    class DetectionDataManager:
        def __init__(self, data_path=None): pass
        def filter_dataframe(self, scopes, splits, sample_size=-1, seed=42):
            return pd.DataFrame({
                "text": ["Human sentence example.", "AI generated sentence text sample."] * 50,
                "label": [0, 1] * 50,
                "model_name": ["human", "qwen3.5:4b"] * 50
            })


# ---------------------------------------------------------
# 1. DIAGNOSTIC DATA INSPECTION
# ---------------------------------------------------------
def inspect_and_print_dataset(train_df: Optional[pd.DataFrame], val_df: pd.DataFrame, test_dfs: Dict[str, pd.DataFrame], max_len: int):
    print("\n" + "=" * 80)
    print("                      DATASET DIAGNOSTIC INSPECTION REPORT")
    print("=" * 80)

    all_dfs = {}
    if train_df is not None:
        all_dfs["Train"] = train_df
    all_dfs["Validation"] = val_df
    for k, v in test_dfs.items():
        all_dfs[f"Test ({k})"] = v

    summary_rows = []
    for split_name, df in all_dfs.items():
        total = len(df)
        if total == 0:
            continue
        n_human = int((df["label"] == 0).sum())
        n_ai = int((df["label"] == 1).sum())
        ratio_str = f"1 : {n_ai / max(1, n_human):.2f}"
        
        word_counts = df["text"].astype(str).apply(lambda x: len(x.split()))
        p50 = int(np.percentile(word_counts, 50))
        p95 = int(np.percentile(word_counts, 95))
        p99 = int(np.percentile(word_counts, 99))
        max_words = int(np.max(word_counts))

        summary_rows.append({
            "Split": split_name,
            "Total": total,
            "Human (0)": n_human,
            "AI (1)": n_ai,
            "Imbalance (H:AI)": ratio_str,
            "Med Words": p50,
            "P95 Words": p95,
            "P99 Words": p99,
            "Max Words": max_words,
        })

    sum_df = pd.DataFrame(summary_rows)
    print(sum_df.to_string(index=False))
    print("-" * 80)

    model_col = next((c for c in ["model_name", "generator_model", "generator"] if c in val_df.columns), None)
    if model_col:
        print("\n--- Generator Breakdown per Split ---")
        for split_name, df in all_dfs.items():
            print(f"\n[{split_name}] Generator Distribution:")
            gen_counts = df.groupby(["label", model_col]).size().unstack(fill_value=0)
            print(gen_counts.to_string())

    print("-" * 80)
    print(f"Active max_length setting: {max_len} tokens")
    max_train_words = summary_rows[0]["Max Words"] if summary_rows else 0
    if max_train_words > max_len:
        print(f"[NOTE] Long sequences (> {max_len} tokens) will be truncated cleanly.")
    else:
        print("[NOTE] max_length accommodates the full text distribution.")
    print("=" * 80 + "\n")


# ---------------------------------------------------------
# 2. STRATIFIED MULTI-GENERATOR BATCH SAMPLER
# ---------------------------------------------------------
def compute_stratified_sample_weights(df: pd.DataFrame) -> torch.Tensor:
    model_col = next((c for c in ["model_name", "generator_model", "generator"] if c in df.columns), None)
    if model_col:
        group_keys = df["label"].astype(str) + "___" + df[model_col].astype(str)
    else:
        group_keys = df["label"].astype(str)

    group_counts = group_keys.value_counts().to_dict()
    raw_weights = group_keys.map(lambda k: 1.0 / group_counts[k]).values.astype(np.float64)

    neg_mask = (df["label"] == 0).values
    pos_mask = (df["label"] == 1).values

    if neg_mask.sum() > 0:
        raw_weights[neg_mask] = (raw_weights[neg_mask] / raw_weights[neg_mask].sum()) * 0.5
    if pos_mask.sum() > 0:
        raw_weights[pos_mask] = (raw_weights[pos_mask] / raw_weights[pos_mask].sum()) * 0.5

    return torch.tensor(raw_weights, dtype=torch.float32)


# ---------------------------------------------------------
# 3. MEMORY-EFFICIENT MODEL ARCHITECTURE
# ---------------------------------------------------------
class MultiSampleDropoutHead(nn.Module):
    def __init__(self, hidden_size: int = 768, num_labels: int = 2, drop_rates=(0.1, 0.2, 0.3, 0.4, 0.5)):
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

        mask_bool = attention_mask.unsqueeze(-1).bool()
        masked_hidden = torch.where(mask_bool, last_hidden_state, torch.tensor(-1e4, device=last_hidden_state.device, dtype=last_hidden_state.dtype))
        max_rep = torch.max(masked_hidden, dim=1)[0]

        fused = torch.cat([cls_rep, mean_rep, max_rep], dim=-1)
        features = F.gelu(self.layer_norm(self.dense(fused)))

        logits_list = [self.out_proj(d(features)) for d in self.dropouts]
        return torch.mean(torch.stack(logits_list, dim=0), dim=0)


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

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
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


# ---------------------------------------------------------
# 4. CVaR-DRO LOSS & LLRD TRAINER
# ---------------------------------------------------------
class RockafellarUryasevCVaRLoss(nn.Module):
    def __init__(self, alpha: float = 0.01, lambda_neg: float = 2.0, initial_eta: float = 1.0, temp: float = 0.1):
        super().__init__()
        self.alpha = float(alpha)
        self.lambda_neg = float(lambda_neg)
        self.temp = float(temp)
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
            diff = neg_losses - self.eta
            smooth_excess = self.temp * F.softplus(diff / self.temp)
            cvar_neg_loss = self.eta + (1.0 / self.alpha) * smooth_excess.mean()
        else:
            cvar_neg_loss = torch.tensor(0.0, device=logits.device)

        return pos_loss + self.lambda_neg * cvar_neg_loss


class ImbalancedLowFPRTrainer(Trainer):
    def __init__(self, *args, sample_weights: Optional[torch.Tensor] = None, use_pauc_loss: bool = True, target_fpr: float = 0.01, lambda_neg: float = 2.0, llrd_decay: float = 0.90, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_weights = sample_weights
        self.use_pauc_loss = use_pauc_loss
        self.llrd_decay = llrd_decay
        if self.use_pauc_loss:
            self.custom_loss_fn = RockafellarUryasevCVaRLoss(alpha=target_fpr, lambda_neg=lambda_neg)
            self.custom_loss_fn.to(self.args.device)

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Training requires a train_dataset.")

        if self.sample_weights is not None:
            sampler = WeightedRandomSampler(
                weights=self.sample_weights,
                num_samples=len(self.sample_weights),
                replacement=True
            )
            return torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.args.train_batch_size,
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
            num_layers = 12

            grouped_parameters = []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
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

                grouped_parameters.append({"params": [param], "weight_decay": wd, "lr": lr})

            if self.use_pauc_loss:
                grouped_parameters.append({
                    "params": list(self.custom_loss_fn.parameters()),
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
        loss = self.custom_loss_fn(logits, labels) if (self.use_pauc_loss and labels is not None) else outputs.loss
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------
# 5. DATA PREPARATION & METRICS
# ---------------------------------------------------------
def prepare_hf_dataset(df: pd.DataFrame, tokenizer: AutoTokenizer, max_len: int) -> Dataset:
    df_clean = df.copy()
    if "label" in df_clean.columns:
        df_clean["labels"] = df_clean["label"].astype(int)

    ds = Dataset.from_pandas(df_clean, preserve_index=False)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_len)

    ds = ds.map(tokenize_fn, batched=True)
    keep_cols = ["input_ids", "attention_mask", "token_type_ids", "labels"]
    remove_cols = [c for c in ds.column_names if c not in keep_cols]
    if remove_cols:
        ds = ds.remove_columns(remove_cols)
    return ds


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    probs = softmax(logits, axis=-1)[:, 1]

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


def calibrate_threshold_at_fpr(val_labels: np.ndarray, val_probs: np.ndarray, max_fpr: float = 0.01, buffer_std_ratio: float = 0.0) -> Tuple[float, float, float]:
    val_probs = np.asarray(val_probs, dtype=np.float64)
    val_labels = np.asarray(val_labels, dtype=np.int32)
    neg_scores = val_probs[val_labels == 0]
    pos_scores = val_probs[val_labels == 1]

    if len(neg_scores) == 0:
        return 0.5, 0.0, 0.0

    raw_tau = float(np.quantile(neg_scores, 1.0 - max_fpr))
    buffer = buffer_std_ratio * float(np.std(neg_scores))
    calibrated_tau = float(np.clip(raw_tau + buffer, 0.0, 1.0))

    calibrated_fpr = float(np.mean(neg_scores >= calibrated_tau))
    calibrated_tpr = float(np.mean(pos_scores >= calibrated_tau)) if len(pos_scores) > 0 else 0.0

    logger.info(f"CALIBRATION (@ FPR<={max_fpr*100:.1f}%): tau={calibrated_tau:.6f} | Dev FPR={calibrated_fpr*100:.2f}% | Dev TPR={calibrated_tpr*100:.2f}%")
    return calibrated_tau, calibrated_fpr, calibrated_tpr


# ---------------------------------------------------------
# 6. FAST INFERENCE, EXPORT & LATEX GENERATION
# ---------------------------------------------------------
def run_fast_inference(model: nn.Module, dataset: Dataset, tokenizer: AutoTokenizer, batch_size: int = 32, device: str = "cuda") -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    model.to(device)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=(device == "cuda"),
        shuffle=False,
    )

    all_logits = []
    use_amp = (device == "cuda")
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    with torch.inference_mode():
        for batch in dataloader:
            inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                outputs = model(**inputs)
            all_logits.append(outputs.logits.detach().float().cpu())

    logits_tensor = torch.cat(all_logits, dim=0)
    probs_tensor = F.softmax(logits_tensor, dim=-1)

    del dataloader, all_logits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return logits_tensor.numpy(), probs_tensor[:, 1].numpy()


def save_split_predictions_csv(df_split: pd.DataFrame, logits: np.ndarray, probs_llm: np.ndarray, calibrated_tau: float, save_dir: str, split_name: str) -> str:
    """
    Saves every sample with all original metadata, raw logits, calibrated probabilities,
    threshold decisions, and TP/TN/FP/FN diagnostic tags.
    """
    os.makedirs(save_dir, exist_ok=True)
    df_export = df_split.copy().reset_index(drop=True)

    preds_calibrated = (probs_llm >= calibrated_tau).astype(int)
    preds_standard = (probs_llm >= 0.5).astype(int)
    labels = df_export["label"].values.astype(int)

    # 1. Logits and Probabilities
    df_export["logit_human"] = np.round(logits[:, 0], 6)
    df_export["logit_ai"] = np.round(logits[:, 1], 6)
    df_export["prob_human"] = np.round(1.0 - probs_llm, 6)
    df_export["prob_ai"] = np.round(probs_llm, 6)

    # 2. Binary Predictions
    df_export["calibrated_tau"] = round(float(calibrated_tau), 6)
    df_export["pred_calibrated_tau"] = preds_calibrated
    df_export["pred_standard_05"] = preds_standard
    df_export["is_correct_calibrated"] = (preds_calibrated == labels).astype(int)
    df_export["is_correct_standard"] = (preds_standard == labels).astype(int)

    # 3. Diagnostic Error Categorization
    error_types = []
    for y_true, y_pred in zip(labels, preds_calibrated):
        if y_true == 1 and y_pred == 1:
            error_types.append("TP")
        elif y_true == 0 and y_pred == 0:
            error_types.append("TN")
        elif y_true == 0 and y_pred == 1:
            error_types.append("FP")
        else:
            error_types.append("FN")
    df_export["error_type_calibrated"] = error_types

    # 4. Text Length Metrics
    if "text" in df_export.columns:
        df_export["word_count"] = df_export["text"].astype(str).apply(lambda x: len(x.split()))
        df_export["char_count"] = df_export["text"].astype(str).apply(len)

    csv_path = os.path.join(save_dir, f"predictions_{split_name}.csv")
    df_export.to_csv(csv_path, index=False)
    logger.info(f"Successfully exported FULL predictions ({len(df_export)} samples) to: {csv_path}")
    return csv_path


def evaluate_paper_results(df_test: pd.DataFrame, logits: np.ndarray, probs_llm: np.ndarray, preds_calibrated: np.ndarray, save_dir: str, eval_name: str, held_out_llm: Optional[str], calibrated_tau: float) -> Dict[str, Any]:
    os.makedirs(save_dir, exist_ok=True)
    labels = df_test["label"].values

    # 1. Save Full Predictions to CSV
    save_split_predictions_csv(df_test, logits, probs_llm, calibrated_tau, save_dir, eval_name)

    # 2. Compute Evaluation Metrics
    fpr, tpr, _ = roc_curve(labels, probs_llm)
    interp_fn = interp1d(fpr, tpr, bounds_error=False, fill_value=(0.0, 1.0))
    tpr_at_1fpr = float(interp_fn(0.01))

    cm = confusion_matrix(labels, preds_calibrated)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    empirical_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        pauc_1fpr = float(roc_auc_score(labels, probs_llm, max_fpr=0.01))
    except Exception:
        pauc_1fpr = 0.5

    overall_metrics = {
        "Evaluation Set": eval_name,
        "Calibrated tau": round(float(calibrated_tau), 6),
        "Held-Out LLM": held_out_llm if held_out_llm else "None",
        "Total Test Samples": int(len(labels)),
        "ROC-AUC": round(float(auc(fpr, tpr)), 4),
        "PR-AUC": round(float(average_precision_score(labels, probs_llm)), 4),
        "Partial AUROC @ 1% FPR": round(float(pauc_1fpr), 4),
        "TPR @ 1% FPR": round(float(tpr_at_1fpr), 4),
        "Empirical Test FPR": round(float(empirical_fpr), 4),
        "Accuracy": round(float(accuracy_score(labels, preds_calibrated)), 4),
        "F1-Score": round(float(precision_recall_fscore_support(labels, preds_calibrated, average="binary", zero_division=0)[2]), 4),
        "Specificity": round(float(specificity), 4),
    }

    logger.info(f"RESULTS [{eval_name.upper()}]: ROC-AUC={overall_metrics['ROC-AUC']} | TPR@1%FPR={overall_metrics['TPR @ 1% FPR']} | Test-FPR={overall_metrics['Empirical Test FPR']}")
    with open(os.path.join(save_dir, f"evaluation_summary_{eval_name}.json"), "w") as f:
        json.dump({"overall_metrics": overall_metrics}, f, indent=4)
    return overall_metrics


def generate_latex_table(all_metrics: List[Dict[str, Any]], save_dir: str) -> str:
    latex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{LLM Detection Performance and Low-FPR Operating Results.}",
        r"\label{tab:llm_detection_results}",
        r"\begin{tabular}{llcccccccccc}",
        r"\toprule",
        r"\textbf{Split / Scope} & \textbf{Held-Out LLM} & \textbf{Samples} & \textbf{AUROC} & \textbf{PR-AUC} & \textbf{pAUC@1\%} & \textbf{TPR@1\%} & \textbf{FPR (emp)} & \textbf{Acc} & \textbf{F1} & \textbf{Spec.} & $\boldsymbol{\tau}$ \\",
        r"\midrule",
    ]

    for m in all_metrics:
        eval_set_str = str(m["Evaluation Set"]).replace("_", r"\_")
        held_out_str = str(m["Held-Out LLM"]).replace("_", r"\_")
        total_samples_str = f"{m['Total Test Samples']:,}"
        roc_auc_str = f"{m['ROC-AUC']:.4f}"
        pr_auc_str = f"{m['PR-AUC']:.4f}"
        pauc_str = f"{m['Partial AUROC @ 1% FPR']:.4f}"
        tpr_str = f"{m['TPR @ 1% FPR'] * 100:.2f}" + r"\%"
        fpr_str = f"{m['Empirical Test FPR'] * 100:.2f}" + r"\%"
        acc_str = f"{m['Accuracy']:.4f}"
        f1_str = f"{m['F1-Score']:.4f}"
        spec_str = f"{m['Specificity']:.4f}"
        tau_str = f"{m['Calibrated tau']:.4f}"

        row = f"{eval_set_str} & {held_out_str} & {total_samples_str} & {roc_auc_str} & {pr_auc_str} & {pauc_str} & {tpr_str} & {fpr_str} & {acc_str} & {f1_str} & {spec_str} & {tau_str} \\\\"
        latex_lines.append(row)

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    full_table = "\n".join(latex_lines)
    tex_path = os.path.join(save_dir, "evaluation_table.tex")
    with open(tex_path, "w") as f:
        f.write(full_table)

    print("\n" + "=" * 80)
    print("                      GENERATED PUBLICATION LATEX TABLE")
    print("=" * 80)
    print(full_table)
    print("=" * 80)
    logger.info(f"LaTeX summary table saved to: {tex_path}\n")
    return full_table


# ---------------------------------------------------------
# 7. MAIN PIPELINE
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="High-Speed Imbalance-Aware mDeBERTa Detection Pipeline")
    parser.add_argument("--scope", type=str, choices=["sentence", "full", "both"], default="sentence")
    parser.add_argument("--preset", type=str, choices=["debug", "fast", "standard", "full"], default="standard")
    parser.add_argument("--max_length", type=int, default=None, help="Sequence max length (default: 128 for sentence, 384 for full/both)")
    parser.add_argument("--train_sample_size", type=int, default=None)
    parser.add_argument("--val_sample_size", type=int, default=None)
    parser.add_argument("--test_sample_size", type=int, default=None)
    parser.add_argument("--leave_out_llm", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--learning_rate", type=float, default=2.5e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--model_name", type=str, default="microsoft/mdeberta-v3-base")
    parser.add_argument("--target_fpr", type=float, default=0.01)
    parser.add_argument("--buffer_std_ratio", type=float, default=0.0, help="Calibration safety buffer (0.0 recommended)")
    parser.add_argument("--lambda_neg", type=float, default=2.0)
    parser.add_argument("--llrd_decay", type=float, default=0.90)
    parser.add_argument("--output_dir", type=str, default="./outputs_imbalanced")
    parser.add_argument("--use_pauc_loss", action="store_true", help="Enable CVaR DRO pAUC Loss")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_only", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.max_length is not None:
        max_len = args.max_length
    else:
        max_len = 128 if args.scope == "sentence" else 384

    # Dynamic Batch Sizing based on Sequence Length (Prevents Full-Abstract OOM)
    if max_len <= 128:
        train_bs = 32
        eval_bs = 64
        grad_accum = 1
        use_grad_checkpointing = False
    elif max_len <= 256:
        train_bs = 16
        eval_bs = 32
        grad_accum = 2
        use_grad_checkpointing = False
    else:  # Full abstracts (384-512 tokens)
        train_bs = 8
        eval_bs = 16
        grad_accum = 4
        use_grad_checkpointing = True

    # NOTE: Presets only constrain Train/Val budgets. Test set is NEVER truncated (-1).
    preset_configs = {
        "debug":    {"train": 1000,   "val": 500,  "test": -1},
        "fast":     {"train": 40000,  "val": 6000, "test": -1},
        "standard": {"train": 100000, "val": 10000, "test": -1},
        "full":     {"train": -1,     "val": -1,   "test": -1},
    }
    cfg = preset_configs[args.preset]
    train_sz = args.train_sample_size if args.train_sample_size is not None else cfg["train"]
    val_sz = args.val_sample_size if args.val_sample_size is not None else cfg["val"]
    test_sz = args.test_sample_size if args.test_sample_size is not None else cfg["test"]

    active_output_dir = os.path.join(args.output_dir, f"deberta_{args.scope}_{args.preset}")
    os.makedirs(active_output_dir, exist_ok=True)
    model_save_path = os.path.join(active_output_dir, f"best_model_{args.scope}")

    manager = DetectionDataManager(data_path=args.data_path)

    # 1. Split Extraction
    if args.scope in ["sentence", "full"]:
        train_df = manager.filter_dataframe(scopes=[args.scope], splits=["train"], sample_size=train_sz, seed=args.seed) if not args.test_only else None
        val_df = manager.filter_dataframe(scopes=[args.scope], splits=["val"], sample_size=val_sz, seed=args.seed)
        test_df = manager.filter_dataframe(scopes=[args.scope], splits=["test"], sample_size=test_sz, seed=args.seed)
    else:
        train_df = pd.concat([
            manager.filter_dataframe(scopes=["sentence"], splits=["train"], sample_size=train_sz // 2 if train_sz > 0 else -1, seed=args.seed),
            manager.filter_dataframe(scopes=["full"], splits=["train"], sample_size=train_sz // 2 if train_sz > 0 else -1, seed=args.seed)
        ]).sample(frac=1.0, random_state=args.seed).reset_index(drop=True) if not args.test_only else None

        val_df = pd.concat([
            manager.filter_dataframe(scopes=["sentence"], splits=["val"], sample_size=val_sz // 2 if val_sz > 0 else -1, seed=args.seed),
            manager.filter_dataframe(scopes=["full"], splits=["val"], sample_size=val_sz // 2 if val_sz > 0 else -1, seed=args.seed)
        ]).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

        test_df = pd.concat([
            manager.filter_dataframe(scopes=["sentence"], splits=["test"], sample_size=test_sz // 2 if test_sz > 0 else -1, seed=args.seed),
            manager.filter_dataframe(scopes=["full"], splits=["test"], sample_size=test_sz // 2 if test_sz > 0 else -1, seed=args.seed)
        ]).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # 2. Leave-One-Out Generator Slicing
    test_dfs = {}
    model_col = next((col for col in ["model_name", "generator_model", "generator"] if col in val_df.columns), None)

    if args.leave_out_llm and model_col:
        pat = args.leave_out_llm.lower().strip()
        if train_df is not None:
            train_df = train_df[~train_df[model_col].astype(str).str.lower().str.contains(pat)].reset_index(drop=True)
        val_df = val_df[~val_df[model_col].astype(str).str.lower().str.contains(pat)].reset_index(drop=True)

        test_dfs["test_seen"] = test_df[~test_df[model_col].astype(str).str.lower().str.contains(pat)].reset_index(drop=True)
        test_dfs["test_unseen_heldout"] = test_df[test_df[model_col].astype(str).str.lower().str.contains(pat)].reset_index(drop=True)
    else:
        test_dfs["test_standard"] = test_df

    # 3. Diagnostic Report
    inspect_and_print_dataset(train_df, val_df, test_dfs, max_len=max_len)

    # 4. Tokenization & Dataset Pruning
    tok_source = model_save_path if args.test_only else args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tok_source, use_fast=False)

    val_ds = prepare_hf_dataset(val_df, tokenizer, max_len=max_len)

    if not args.test_only:
        train_ds = prepare_hf_dataset(train_df, tokenizer, max_len=max_len)
        sample_weights = compute_stratified_sample_weights(train_df)

        config = AutoConfig.from_pretrained(args.model_name)
        config.num_labels = 2
        model = CustomMDeBERTaForDetection.from_pretrained(args.model_name, config=config)

        has_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

        training_args = TrainingArguments(
            output_dir=model_save_path,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            learning_rate=args.learning_rate,
            warmup_ratio=0.1,
            adam_epsilon=1e-6,
            max_grad_norm=1.0,
            per_device_train_batch_size=train_bs,
            per_device_eval_batch_size=eval_bs,
            gradient_accumulation_steps=grad_accum,
            gradient_checkpointing=use_grad_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False} if use_grad_checkpointing else None,
            bf16=has_bf16,
            fp16=(not has_bf16 and torch.cuda.is_available()),
            num_train_epochs=args.epochs,
            load_best_model_at_end=True,
            metric_for_best_model="pauc_1fpr",
            greater_is_better=True,
            report_to="none",
            logging_steps=50,
            optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
            dataloader_num_workers=min(4, os.cpu_count() or 1),
            dataloader_pin_memory=torch.cuda.is_available(),
        )

        trainer = ImbalancedLowFPRTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            sample_weights=sample_weights,
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
            use_pauc_loss=args.use_pauc_loss,
            target_fpr=args.target_fpr,
            lambda_neg=args.lambda_neg,
            llrd_decay=args.llrd_decay,
        )

        logger.info(f"Training started on {len(train_df)} samples (Batch size: {train_bs} x Accum {grad_accum}, Checkpointing: {use_grad_checkpointing})...")
        trainer.train()
        trainer.save_model(model_save_path)
        tokenizer.save_pretrained(model_save_path)
    else:
        logger.info(f"Loading detector from {model_save_path}...")
        model = CustomMDeBERTaForDetection.from_pretrained(model_save_path)

    # 5. Threshold Calibration & Validation Prediction Export
    val_logits, val_probs = run_fast_inference(model, val_ds, tokenizer, batch_size=eval_bs, device=device)
    calibrated_tau, _, _ = calibrate_threshold_at_fpr(
        val_df["label"].values, val_probs, max_fpr=args.target_fpr, buffer_std_ratio=args.buffer_std_ratio
    )
    # Save validation predictions CSV
    save_split_predictions_csv(val_df, val_logits, val_probs, calibrated_tau, active_output_dir, split_name="validation")

    # 6. Test Evaluation & Full Prediction Export
    all_evaluated_metrics = []
    for eval_key, sub_df in test_dfs.items():
        if sub_df.empty:
            continue
        sub_ds = prepare_hf_dataset(sub_df, tokenizer, max_len=max_len)
        test_logits, test_probs = run_fast_inference(model, sub_ds, tokenizer, batch_size=eval_bs, device=device)
        test_preds_calibrated = (test_probs >= calibrated_tau).astype(int)

        metrics = evaluate_paper_results(
            sub_df,
            test_logits,
            test_probs,
            test_preds_calibrated,
            active_output_dir,
            eval_name=eval_key,
            held_out_llm=args.leave_out_llm,
            calibrated_tau=calibrated_tau,
        )
        all_evaluated_metrics.append(metrics)

    # 7. Generate and Print LaTeX Table
    if all_evaluated_metrics:
        generate_latex_table(all_evaluated_metrics, save_dir=active_output_dir)


if __name__ == "__main__":
    main()