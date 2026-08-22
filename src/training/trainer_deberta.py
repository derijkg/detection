import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, EarlyStoppingCallback, PreTrainedTokenizerBase, Trainer, TrainerCallback, TrainingArguments
try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

def compute_stratified_sample_weights(df: pd.DataFrame) -> torch.Tensor:
    model_col = next((c for c in ['model_name', 'generator_model', 'generator'] if c in df.columns), None)
    if model_col:
        group_keys = df['label'].astype(str) + '___' + df[model_col].astype(str)
    else:
        group_keys = df['label'].astype(str)
    group_counts = group_keys.value_counts().to_dict()
    raw_weights = group_keys.map(lambda k: 1.0 / max(group_counts[k], 1)).values.astype(np.float64)
    neg_mask = (df['label'] == 0).values
    pos_mask = (df['label'] == 1).values
    if neg_mask.sum() > 0 and pos_mask.sum() > 0:
        raw_weights[neg_mask] = raw_weights[neg_mask] / raw_weights[neg_mask].sum() * 0.5
        raw_weights[pos_mask] = raw_weights[pos_mask] / raw_weights[pos_mask].sum() * 0.5
    elif neg_mask.sum() > 0:
        raw_weights[neg_mask] = raw_weights[neg_mask] / raw_weights[neg_mask].sum()
    elif pos_mask.sum() > 0:
        raw_weights[pos_mask] = raw_weights[pos_mask] / raw_weights[pos_mask].sum()
    return torch.tensor(raw_weights, dtype=torch.float32)

class RockafellarUryasevCVaRLoss(nn.Module):

    def __init__(self, alpha: float=0.05, lambda_neg: float=2.0, initial_eta: float=0.6931, temp: float=0.1):
        super().__init__()
        self.alpha = float(max(alpha, 0.001))
        self.lambda_neg = float(lambda_neg)
        self.temp = float(max(temp, 0.01))
        self.eta = nn.Parameter(torch.tensor(float(initial_eta), dtype=torch.float32))
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        losses = self.ce(logits, targets)
        pos_mask = targets == 1
        neg_mask = targets == 0
        n_pos = pos_mask.sum()
        n_neg = neg_mask.sum()
        pos_loss = losses[pos_mask].mean() if n_pos > 0 else 0.0 * logits.sum()
        eta_val = self.eta.to(logits.device)
        if n_neg > 0:
            neg_losses = losses[neg_mask]
            diff = torch.clamp((neg_losses - eta_val) / self.temp, -30.0, 30.0)
            smooth_excess = self.temp * F.softplus(diff)
            cvar_neg_loss = eta_val + 1.0 / self.alpha * smooth_excess.mean()
        else:
            cvar_neg_loss = 0.0 * eta_val
        if n_pos > 0 and n_neg > 0:
            return (pos_loss + self.lambda_neg * cvar_neg_loss) / (1.0 + self.lambda_neg)
        elif n_neg > 0:
            return cvar_neg_loss
        else:
            return pos_loss + 0.0 * eta_val

class MultiScaleRockafellarUryasevCVaRLoss(nn.Module):

    def __init__(self, alpha: float=0.05, lambda_neg: float=2.0, w_doc: float=1.0, w_sent: float=1.0, temp: float=0.1):
        super().__init__()
        self.alpha = float(max(alpha, 0.001))
        self.lambda_neg = float(lambda_neg)
        self.w_doc = float(w_doc)
        self.w_sent = float(w_sent)
        self.temp = float(max(temp, 0.01))
        self.eta_doc = nn.Parameter(torch.tensor(0.6931, dtype=torch.float32))
        self.eta_sent = nn.Parameter(torch.tensor(0.6931, dtype=torch.float32))
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, is_sentence_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        losses = self.ce(logits, targets)
        pos_mask = targets == 1
        neg_mask = targets == 0
        n_pos = pos_mask.sum()
        pos_loss = losses[pos_mask].mean() if n_pos > 0 else 0.0 * logits.sum()
        if is_sentence_mask is None:
            is_sentence_mask = torch.zeros(logits.size(0), dtype=torch.bool, device=logits.device)
        else:
            is_sentence_mask = is_sentence_mask.bool().to(logits.device)
        doc_neg_mask = neg_mask & ~is_sentence_mask
        sent_neg_mask = neg_mask & is_sentence_mask
        eta_d = self.eta_doc.to(logits.device)
        eta_s = self.eta_sent.to(logits.device)
        if doc_neg_mask.sum() > 0:
            diff_d = torch.clamp((losses[doc_neg_mask] - eta_d) / self.temp, -30.0, 30.0)
            cvar_doc = eta_d + 1.0 / self.alpha * (self.temp * F.softplus(diff_d)).mean()
        else:
            cvar_doc = 0.0 * eta_d
        if sent_neg_mask.sum() > 0:
            diff_s = torch.clamp((losses[sent_neg_mask] - eta_s) / self.temp, -30.0, 30.0)
            cvar_sent = eta_s + 1.0 / self.alpha * (self.temp * F.softplus(diff_s)).mean()
        else:
            cvar_sent = 0.0 * eta_s
        total_w = 0.0
        cvar_total = 0.0 * (eta_d + eta_s)
        if doc_neg_mask.sum() > 0:
            cvar_total = cvar_total + self.w_doc * cvar_doc
            total_w += self.w_doc
        if sent_neg_mask.sum() > 0:
            cvar_total = cvar_total + self.w_sent * cvar_sent
            total_w += self.w_sent
        cvar_neg_loss = cvar_total / max(total_w, 0.0001) if total_w > 0 else 0.0 * (eta_d + eta_s)
        if n_pos > 0 and total_w > 0:
            return (pos_loss + self.lambda_neg * cvar_neg_loss) / (1.0 + self.lambda_neg)
        elif total_w > 0:
            return cvar_neg_loss
        else:
            return pos_loss + 0.0 * (eta_d + eta_s)

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
        (eta_val, eta_doc, eta_sent) = (None, None, None)
        unwrapped = getattr(model, 'module', model)
        if hasattr(unwrapped, 'custom_loss_fn'):
            fn = unwrapped.custom_loss_fn
            if hasattr(fn, 'eta'):
                eta_val = float(fn.eta.detach().cpu().item())
            if hasattr(fn, 'eta_doc'):
                eta_doc = float(fn.eta_doc.detach().cpu().item())
            if hasattr(fn, 'eta_sent'):
                eta_sent = float(fn.eta_sent.detach().cpu().item())
        entry = {
            'step': int(step),
            'epoch': round(float(epoch), 4) if epoch is not None else None,
            'eta': eta_val,
            'eta_doc': eta_doc,
            'eta_sent': eta_sent,
            'train_loss': logs.get('loss'),
            'learning_rate': logs.get('learning_rate'),
            'eval_loss': logs.get('eval_loss'),
            'eval_pauc_1fpr': logs.get('eval_pauc_1fpr'),
            'eval_tpr_at_1fpr': logs.get('eval_tpr_at_1fpr')
        }
        self.records.append(entry)
        df = pd.DataFrame(self.records)
        df.to_csv(self.output_dir / 'cvar_history.csv', index=False)

class OptunaStepPruningCallback(TrainerCallback):

    def __init__(self, trial: Any, metric_name: str='eval_pauc_1fpr'):
        self.trial = trial
        self.metric_name = metric_name

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or self.trial is None:
            return
        val = metrics.get(self.metric_name)
        if val is not None:
            self.trial.report(val, step=state.global_step)
            if self.trial.should_prune():
                control.should_training_stop = True
                if HAS_OPTUNA:
                    raise optuna.TrialPruned(f'Pruned at step {state.global_step} with {self.metric_name}={val:.4f}')

class ImbalancedLowFPRTrainer(Trainer):

    def __init__(self, *args, sample_weights: Optional[torch.Tensor]=None, use_pauc_loss: bool=True, target_fpr: float=0.05, lambda_neg: float=2.0, llrd_decay: float=0.9, is_multi_scale: bool=False, w_doc: float=1.0, w_sent: float=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_weights = sample_weights
        self.use_pauc_loss = use_pauc_loss
        self.llrd_decay = llrd_decay
        self.is_multi_scale = is_multi_scale
        if self.use_pauc_loss:
            if self.is_multi_scale:
                self.custom_loss_fn = MultiScaleRockafellarUryasevCVaRLoss(alpha=target_fpr, lambda_neg=lambda_neg, w_doc=w_doc, w_sent=w_sent)
            else:
                self.custom_loss_fn = RockafellarUryasevCVaRLoss(alpha=target_fpr, lambda_neg=lambda_neg)
            self.model.custom_loss_fn = self.custom_loss_fn

    def create_optimizer(self):
        if self.optimizer is None:
            base_lr = self.args.learning_rate
            weight_decay = self.args.weight_decay
            no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
            num_layers = getattr(self.model.config, 'num_hidden_layers', 12)
            loss_param_ids = set()
            if self.use_pauc_loss and hasattr(self, 'custom_loss_fn'):
                self.custom_loss_fn.to(self.args.device)
                loss_param_ids = {id(p) for p in self.custom_loss_fn.parameters()}
            param_groups: Dict[Tuple[float, float], List[torch.nn.Parameter]] = {}
            for (name, param) in self.model.named_parameters():
                if not param.requires_grad or id(param) in loss_param_ids:
                    continue
                wd = 0.0 if any((nd in name for nd in no_decay)) else weight_decay
                if 'classifier' in name:
                    lr = base_lr * 1.5
                elif 'encoder.layer.' in name:
                    layer_idx = int(name.split('encoder.layer.')[1].split('.')[0])
                    lr = base_lr * self.llrd_decay ** (num_layers - 1 - layer_idx)
                elif 'embeddings' in name:
                    lr = base_lr * self.llrd_decay ** num_layers
                else:
                    lr = base_lr
                key = (lr, wd)
                param_groups.setdefault(key, []).append(param)
            grouped_parameters = [{'params': params, 'lr': lr, 'weight_decay': wd} for ((lr, wd), params) in param_groups.items() if len(params) > 0]
            if self.use_pauc_loss and hasattr(self, 'custom_loss_fn'):
                cvar_params = [p for p in self.custom_loss_fn.parameters() if p.requires_grad]
                if cvar_params:
                    grouped_parameters.append({'params': cvar_params, 'lr': base_lr, 'weight_decay': 0.0})
            (opt_cls, opt_kwargs) = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = opt_cls(grouped_parameters, **opt_kwargs)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get('labels')
        is_sentence = inputs.get('is_sentence', None)
        clean_inputs = {k: v for (k, v) in inputs.items() if k not in ['is_sentence', 'scope_type']}
        clean_inputs['defer_loss'] = self.use_pauc_loss
        outputs = model(**clean_inputs)
        logits = outputs.get('logits')
        if self.use_pauc_loss and labels is not None:
            self.custom_loss_fn.to(logits.device)
            if self.is_multi_scale:
                if is_sentence is None and 'input_ids' in inputs:
                    tok = getattr(self, 'processing_class', getattr(self, 'tokenizer', None))
                    pad_id = tok.pad_token_id if (tok is not None and tok.pad_token_id is not None) else 0
                    seq_lens = (inputs['input_ids'] != pad_id).sum(dim=-1)
                    is_sentence = (seq_lens <= 128).long()
                loss = self.custom_loss_fn(logits, labels, is_sentence_mask=is_sentence)
            else:
                loss = self.custom_loss_fn(logits, labels)
        else:
            loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

def compute_deberta_metrics(eval_pred) -> Dict[str, float]:
    (logits, labels) = eval_pred
    preds = np.argmax(logits, axis=-1)
    probs = softmax(logits, axis=-1)[:, 1]
    probs = np.nan_to_num(probs, nan=0.5)
    if len(np.unique(labels)) < 2:
        return {'pauc_1fpr': 0.5, 'tpr_at_1fpr': 0.0, 'roc_auc': 0.5, 'accuracy': 0.0, 'f1': 0.0}
    (fpr, tpr, _) = roc_curve(labels, probs)
    (unique_fpr, rev_indices) = np.unique(fpr, return_inverse=True)
    max_tpr = np.maximum.reduceat(tpr, np.r_[0, np.where(np.diff(rev_indices))[0] + 1])
    max_tpr_accum = np.maximum.accumulate(max_tpr)
    tpr_at_1fpr = float(np.interp(0.01, unique_fpr, max_tpr_accum, left=0.0, right=float(max_tpr_accum[-1])))
    try:
        pauc_1fpr = float(roc_auc_score(labels, probs, max_fpr=0.01))
        overall_auc = float(roc_auc_score(labels, probs))
    except Exception:
        (pauc_1fpr, overall_auc) = (0.5, 0.5)
    acc = float(accuracy_score(labels, preds))
    (prec, rec, f1, _) = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    return {'pauc_1fpr': pauc_1fpr, 'tpr_at_1fpr': tpr_at_1fpr, 'roc_auc': overall_auc, 'accuracy': acc, 'f1': float(f1), 'precision': float(prec), 'recall': float(rec)}

def build_deberta_trainer(model: nn.Module, tokenizer: PreTrainedTokenizerBase, train_dataset: Any, eval_dataset: Optional[Any], output_dir: Union[str, Path], max_length: int, sample_weights: Optional[torch.Tensor]=None, epochs: int=4, learning_rate: float=2.5e-05, llrd_decay: float=0.9, lambda_neg: float=2.0, w_doc: float=1.0, w_sent: float=1.0, weight_decay: float=0.01, warmup_ratio: float=0.1, target_fpr: float=0.05, batch_size: Optional[int]=None, gradient_accumulation_steps: Optional[int]=None, is_multi_scale: bool=False, trial: Optional[Any]=None, is_tuning: bool=False) -> ImbalancedLowFPRTrainer:
    out_path = Path(output_dir)
    scratch_dir = out_path if is_tuning else out_path / 'checkpoints_tmp'
    scratch_dir.mkdir(parents=True, exist_ok=True)
    train_bs = batch_size or (32 if max_length <= 128 else 16 if max_length <= 256 else 8)
    grad_accum = gradient_accumulation_steps or (1 if max_length <= 128 else 2 if max_length <= 256 else 4)
    eval_bs = 64 if max_length <= 128 else 16
    use_grad_ckpt = max_length > 256 or is_multi_scale
    has_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    effective_bs = train_bs * grad_accum
    steps_per_epoch = max(1, len(train_dataset) // effective_bs)
    eval_steps = max(1, min(500, steps_per_epoch // 2 if steps_per_epoch <= 100 else steps_per_epoch // 4))
    callbacks = []
    if eval_dataset is not None:
        eval_strategy = 'steps'
        if is_tuning:
            save_strategy = 'no'
            save_steps = None
            load_best_at_end = False
            if trial is not None:
                callbacks.append(OptunaStepPruningCallback(trial=trial, metric_name='eval_pauc_1fpr'))
        else:
            save_strategy = 'steps'
            save_steps = eval_steps
            load_best_at_end = True
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=5))
            callbacks.append(CVaRTrackingCallback(output_dir=out_path))
    else:
        eval_strategy = 'no'
        save_strategy = 'no'
        save_steps = None
        eval_steps = None
        load_best_at_end = False

    targs_kwargs = {
        'output_dir': str(scratch_dir),
        'save_total_limit': 2 if not is_tuning else None,
        'learning_rate': learning_rate,
        'warmup_ratio': warmup_ratio,
        'weight_decay': weight_decay,
        'adam_epsilon': 1e-06,
        'max_grad_norm': 1.0,
        'per_device_train_batch_size': train_bs,
        'per_device_eval_batch_size': eval_bs,
        'gradient_accumulation_steps': grad_accum,
        'gradient_checkpointing': use_grad_ckpt,
        'gradient_checkpointing_kwargs': {'use_reentrant': False} if use_grad_ckpt else None,
        'bf16': has_bf16,
        'fp16': not has_bf16 and torch.cuda.is_available(),
        'num_train_epochs': epochs,
        'load_best_model_at_end': load_best_at_end,
        'metric_for_best_model': 'pauc_1fpr' if load_best_at_end else None,
        'greater_is_better': True,
        'report_to': 'none',
        'logging_steps': 25,
        'group_by_length': True
    }
    if eval_dataset is not None:
        targs_kwargs['eval_steps'] = eval_steps
        targs_kwargs['save_steps'] = save_steps
        targs_kwargs['eval_strategy'] = eval_strategy
        targs_kwargs['save_strategy'] = save_strategy
    else:
        targs_kwargs['eval_strategy'] = 'no'
        targs_kwargs['save_strategy'] = 'no'

    try:
        training_args = TrainingArguments(**targs_kwargs)
    except TypeError:
        if 'eval_strategy' in targs_kwargs:
            targs_kwargs['evaluation_strategy'] = targs_kwargs.pop('eval_strategy')
        training_args = TrainingArguments(**targs_kwargs)

    trainer_kwargs = {
        'model': model,
        'args': training_args,
        'train_dataset': train_dataset,
        'eval_dataset': eval_dataset,
        'sample_weights': sample_weights,
        'data_collator': DataCollatorWithPadding(tokenizer=tokenizer),
        'compute_metrics': compute_deberta_metrics if eval_dataset is not None else None,
        'callbacks': callbacks,
        'use_pauc_loss': True,
        'target_fpr': target_fpr,
        'lambda_neg': lambda_neg,
        'llrd_decay': llrd_decay,
        'is_multi_scale': is_multi_scale,
        'w_doc': w_doc,
        'w_sent': w_sent
    }
    try:
        trainer = ImbalancedLowFPRTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = ImbalancedLowFPRTrainer(tokenizer=tokenizer, **trainer_kwargs)
    return trainer