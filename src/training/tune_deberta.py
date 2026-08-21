# src/training/tune_deberta.py

import gc
import json
from pathlib import Path
import shutil
from typing import Any, Dict, Optional, Tuple, Union
from datasets import Dataset, load_from_disk
import numpy as np
import optuna
from optuna.trial import Trial
import pandas as pd
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    set_seed,
)

from src.data.data_loader import DataFilter, DetectionDataManager
from src.models.deberta import CustomMDeBERTaForDetection
from src.training.trainer_deberta import (
    ImbalancedLowFPRTrainer,
    compute_deberta_metrics,
    compute_stratified_sample_weights,
)
from src.utils.optuna_utils import TqdmOptunaCallback

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = PROJECT_ROOT / "data_static" / "preprocessed" / "deberta_cache"

DEBERTA_SEARCH_SPACES = {
    "learning_rate": r"$[1.0\times 10^{-5}, 5.0\times 10^{-5}]$ (Log-uniform)",
    "llrd_decay": "$[0.80, 0.95]$",
    "lambda_neg": "$[1.0, 4.0]$ (CVaR tail penalty)",
    "weight_decay": r"$[1.0\times 10^{-3}, 1.0\times 10^{-1}]$ (Log-uniform)",
    "warmup_ratio": "$[0.05, 0.15]$",
}

# Strong baseline defaults used to warm-start Optuna (Trial 0)
DEFAULT_DEBERTA_PRIORS = {
    "learning_rate": 1.8272e-05,
    "llrd_decay": 0.9426,
    "lambda_neg": 3.1960,
    "weight_decay": 0.0158,
    "warmup_ratio": 0.0656,
}


def get_or_create_cached_hf_dataset(
    df: pd.DataFrame, 
    tokenizer: AutoTokenizer, 
    max_len: int, 
    cache_tag: str
) -> Dataset:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{cache_tag}_{len(df)}_{max_len}"

    if cache_path.exists():
        print(f"Loaded cached tokenized dataset from: {cache_path}")
        return load_from_disk(str(cache_path))

    df_clean = df.copy()
    if "label" in df_clean.columns:
        df_clean["labels"] = df_clean["label"].astype(int)
    if "text" in df_clean.columns:
        df_clean["text"] = df_clean["text"].fillna("").astype(str)

    ds = Dataset.from_pandas(df_clean, preserve_index=False)

    def tok_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_len)

    ds = ds.map(tok_fn, batched=True, desc=f"Tokenizing [{cache_tag}]")
    keep_cols = ["input_ids", "attention_mask", "token_type_ids", "labels"]
    remove_cols = [c for c in ds.column_names if c not in keep_cols]
    if remove_cols:
        ds = ds.remove_columns(remove_cols)

    ds.save_to_disk(str(cache_path))
    return ds


class DebertaOptunaTuner:
    """
    Optuna optimization engine for mDeBERTa-v3 with CVaR loss at low FPR regime.
    Persists trials to SQLite and immediately saves best parameters to disk.
    """
    @classmethod
    def run(
        cls,
        manager: Optional[DetectionDataManager] = None,
        train_df: Optional[pd.DataFrame] = None,
        dev_df: Optional[pd.DataFrame] = None,
        scope: str = "full",
        n_trials: int = 10,
        tuning_sample_size: int = 4000,
        val_sample_size: int = 2000,
        model_name: str = "microsoft/mdeberta-v3-base",
        target_fpr: float = 0.01,
        seed: int = 42,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Tuple[Dict[str, Any], int]:

        set_seed(seed)
        max_len = 128 if scope == "sentence" else 384
        batch_size = 32 if scope == "sentence" else 8
        grad_accum = 1 if scope == "sentence" else 4

        out_path = Path(output_dir or f"./output/deberta_{scope}")
        out_path.mkdir(parents=True, exist_ok=True)
        params_file = out_path / "best_params.json"
        db_path = out_path / "optuna_study.db"
        storage_url = f"sqlite:///{db_path.resolve()}"

        print(f"\n==================================================================")
        print(f"   RUNNING mDeBERTa OPTUNA TUNING [{scope.upper()}] ({n_trials} Trials)   ")
        print(f"   Database Storage : {storage_url}")
        print(f"   Live Params File : {params_file}")
        print(f"==================================================================")

        if train_df is not None:
            df_train = train_df.copy()
            if 0 < tuning_sample_size < len(df_train):
                id_col = '_id' if '_id' in df_train.columns else 'id'
                if id_col in df_train.columns:
                    u_ids = df_train[id_col].unique()
                    target_groups = max(1, int(tuning_sample_size / (len(df_train) / len(u_ids))))
                    rng = np.random.default_rng(seed)
                    s_ids = rng.choice(u_ids, size=min(target_groups, len(u_ids)), replace=False)
                    df_train = df_train[df_train[id_col].isin(s_ids)].copy().reset_index(drop=True)
                else:
                    df_train = df_train.sample(n=tuning_sample_size, random_state=seed).reset_index(drop=True)
        elif manager is not None:
            df_train = manager.filter_dataframe(DataFilter(splits=["train"], scopes=[scope]), sample_size=tuning_sample_size, seed=seed)
        else:
            raise ValueError("Either `train_df` or `manager` must be provided.")

        if dev_df is not None:
            df_val = dev_df.copy()
            if 0 < val_sample_size < len(df_val):
                df_val = df_val.sample(n=val_sample_size, random_state=seed).reset_index(drop=True)
        elif manager is not None:
            df_val = manager.filter_dataframe(DataFilter(splits=["dev"], scopes=[scope]), sample_size=val_sample_size, seed=seed)
        else:
            raise ValueError("Either `dev_df` or `manager` must be provided.")

        actual_tune_size = len(df_train)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

        train_ds = get_or_create_cached_hf_dataset(
            df_train, tokenizer, max_len=max_len, cache_tag=f"tune_train_{scope}"
        )
        val_ds = get_or_create_cached_hf_dataset(
            df_val, tokenizer, max_len=max_len, cache_tag=f"tune_val_{scope}"
        )
        
        sample_weights = compute_stratified_sample_weights(df_train)

        # Persistent SQLite study
        study = optuna.create_study(
            study_name=f"mdeberta_{scope}",
            storage=storage_url,
            load_if_exists=True,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=0),
        )

        # Enqueue baseline parameters to warm-start Trial 0
        initial_params = DEFAULT_DEBERTA_PRIORS.copy()
        if params_file.exists():
            try:
                saved = json.loads(params_file.read_text(encoding="utf-8"))
                clean_saved = {k: v for k, v in saved.items() if not k.startswith("_")}
                if clean_saved:
                    initial_params.update(clean_saved)
            except Exception:
                pass

        try:
            study.enqueue_trial(initial_params, skip_if_exists=True)
        except TypeError:
            study.enqueue_trial(initial_params)

        def objective(trial: Trial) -> float:
            lr = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
            llrd = trial.suggest_float("llrd_decay", 0.80, 0.95)
            lambda_neg = trial.suggest_float("lambda_neg", 1.0, 4.0)
            wd = trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True)
            warmup = trial.suggest_float("warmup_ratio", 0.05, 0.15)

            tmp_dir = Path(f"./.tmp_optuna_deberta_{scope}_t{trial.number}")
            tmp_dir.mkdir(parents=True, exist_ok=True)

            config = AutoConfig.from_pretrained(model_name)
            config.num_labels = 2
            model = CustomMDeBERTaForDetection.from_pretrained(model_name, config=config)

            training_args = TrainingArguments(
                output_dir=str(tmp_dir),
                eval_strategy="epoch",
                save_strategy="no",
                learning_rate=lr,
                warmup_ratio=warmup,
                weight_decay=wd,
                max_grad_norm=1.0,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size * 2,
                gradient_accumulation_steps=grad_accum,
                gradient_checkpointing=(scope == "full"),
                gradient_checkpointing_kwargs=({"use_reentrant": False} if scope == "full" else None),
                bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
                num_train_epochs=1,
                report_to="none",
                logging_steps=25,
            )

            trainer = ImbalancedLowFPRTrainer(
                model=model,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                sample_weights=sample_weights,
                processing_class=tokenizer,
                data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                compute_metrics=compute_deberta_metrics,
                use_pauc_loss=True,
                target_fpr=target_fpr,
                lambda_neg=lambda_neg,
                llrd_decay=llrd,
            )

            try:
                trainer.train()
                eval_res = trainer.evaluate()
                score = eval_res.get("eval_pauc_1fpr", 0.5)
            finally:
                del model
                del trainer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir, ignore_errors=True)

            trial.report(score, step=1)
            if trial.should_prune():
                raise optuna.TrialPruned()

            return float(score)

        with TqdmOptunaCallback(n_trials=n_trials, desc=f"Tuning DeBERTa [{scope.upper()}]", save_path=params_file) as opt_cb:
            study.optimize(objective, n_trials=n_trials, callbacks=[opt_cb])

        # Strip internal metadata before returning
        best_clean_params = {k: v for k, v in study.best_params.items() if not k.startswith("_")}
        print(f"\n-> Best mDeBERTa Parameters ({scope}): {best_clean_params} | pAUC={study.best_value:.4f}")
        return best_clean_params, actual_tune_size