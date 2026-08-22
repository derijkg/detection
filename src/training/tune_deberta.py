import gc
import hashlib
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
from transformers import AutoConfig, AutoTokenizer, set_seed
from src.data.data_loader import DetectionDataManager, group_stratified_sample
from src.models.deberta import CustomMDeBERTaForDetection
from src.training.trainer_deberta import build_deberta_trainer, compute_stratified_sample_weights
from src.utils.optuna_utils import TqdmOptunaCallback

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = PROJECT_ROOT / 'data_static' / 'preprocessed' / 'deberta_cache'

DEBERTA_SEARCH_SPACES = {
    'learning_rate': '$[1.0\\times 10^{-5}, 5.0\\times 10^{-5}]$ (Log-uniform)',
    'llrd_decay': '$[0.85, 0.95]$',
    'weight_decay': '$[1.0\\times 10^{-3}, 1.0\\times 10^{-1}]$ (Log-uniform)',
    'warmup_ratio': '$[0.05, 0.15]$'
}

DEFAULT_DEBERTA_PRIORS = {
    'learning_rate': 2.5e-05,
    'llrd_decay': 0.9,
    'weight_decay': 0.01,
    'warmup_ratio': 0.1
}

def generate_df_content_hash(df: pd.DataFrame) -> str:
    hasher = hashlib.md5()
    sample_texts = df['text'].dropna().astype(str).values[:500]
    for t in sample_texts:
        hasher.update(t.encode('utf-8', errors='ignore'))
    if 'label' in df.columns:
        hasher.update(df['label'].values[:500].tobytes())
    return hasher.hexdigest()[:10]

def get_or_create_cached_hf_dataset(df: pd.DataFrame, tokenizer: AutoTokenizer, max_len: int, cache_tag: str) -> Dataset:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    content_hash = generate_df_content_hash(df)
    cache_path = CACHE_DIR / f'{cache_tag}_{len(df)}_{content_hash}_{max_len}'
    if cache_path.exists():
        return load_from_disk(str(cache_path))
    df_clean = df.copy()
    if 'label' in df_clean.columns:
        df_clean['labels'] = df_clean['label'].astype(int)
    if 'text' in df_clean.columns:
        df_clean['text'] = df_clean['text'].fillna('').astype(str)
    if 'is_sentence' in df_clean.columns:
        df_clean['is_sentence'] = df_clean['is_sentence'].fillna(0).astype(int)
    ds = Dataset.from_pandas(df_clean, preserve_index=False)

    def tok_fn(batch):
        return tokenizer(batch['text'], truncation=True, max_length=max_len)

    ds = ds.map(tok_fn, batched=True, desc=f'Tokenizing [{cache_tag}]')
    keep_cols = ['input_ids', 'attention_mask', 'token_type_ids', 'labels', 'is_sentence']
    remove_cols = [c for c in ds.column_names if c not in keep_cols]
    if remove_cols:
        ds = ds.remove_columns(remove_cols)
    ds.save_to_disk(str(cache_path))
    return ds

class DebertaOptunaTuner:

    @classmethod
    def run(
        cls,
        manager: Optional[DetectionDataManager] = None,
        train_df: Optional[pd.DataFrame] = None,
        dev_df: Optional[pd.DataFrame] = None,
        scope: str = 'full',
        n_trials: int = 15,
        epochs: int = 4,
        tuning_sample_size: int = 12000,
        val_sample_size: int = -1,
        model_name: str = 'microsoft/mdeberta-v3-base',
        target_fpr: float = 0.05,
        seed: int = 42,
        output_dir: Optional[Union[str, Path]] = None
    ) -> Tuple[Dict[str, Any], int]:
        set_seed(seed)
        is_multi_scale = scope in ['mixed', 'multi_scale', 'combined']
        max_len = 128 if scope == 'sentence' else 256 if is_multi_scale else 384
        out_path = Path(output_dir or f'./output/deberta_{scope}')
        out_path.mkdir(parents=True, exist_ok=True)
        params_file = out_path / 'best_params.json'
        db_path = out_path / 'optuna_study.db'
        storage_url = f'sqlite:///{db_path.resolve()}'

        if train_df is not None:
            df_train = group_stratified_sample(train_df, sample_size=tuning_sample_size, seed=seed)
        elif manager is not None:
            from src.data.data_loader import DataFilter
            df_train = manager.filter_dataframe(DataFilter(splits=['train'], scopes=[scope]), sample_size=tuning_sample_size, seed=seed)
        else:
            raise ValueError('Either `train_df` or `manager` must be provided.')

        if dev_df is not None:
            df_val = group_stratified_sample(dev_df, sample_size=val_sample_size, seed=seed) if val_sample_size > 0 else dev_df.copy()
        elif manager is not None:
            from src.data.data_loader import DataFilter
            df_val = manager.filter_dataframe(DataFilter(splits=['dev'], scopes=[scope]), sample_size=val_sample_size, seed=seed)
        else:
            raise ValueError('Either `dev_df` or `manager` must be provided.')

        actual_tune_size = len(df_train)
        actual_val_size = len(df_val)
        print(f'\n==================================================================')
        print(f'   RUNNING mDeBERTa OPTUNA TUNING [{scope.upper()}] ({n_trials} Trials)   ')
        print(f'   Train Tuning Size: {actual_tune_size:,} | Validation Size: {actual_val_size:,}')
        print(f'   Max Epochs/Trial : {epochs} (Step-Evaluated & Pruned with Patience)')
        print(f'   Database Storage : {storage_url}')
        print(f'   Live Params File : {params_file}')
        print(f'==================================================================')

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        train_ds = get_or_create_cached_hf_dataset(df_train, tokenizer, max_len=max_len, cache_tag=f'tune_train_{scope}')
        val_ds = get_or_create_cached_hf_dataset(df_val, tokenizer, max_len=max_len, cache_tag=f'tune_val_{scope}')
        sample_weights = compute_stratified_sample_weights(df_train)

        base_pruner = optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=2, interval_steps=1)
        patient_pruner = optuna.pruners.PatientPruner(wrapped_pruner=base_pruner, patience=2)

        storage = optuna.storages.RDBStorage(url=storage_url, engine_kwargs={'connect_args': {'timeout': 60}})
        study = optuna.create_study(
            study_name=f'mdeberta_{scope}',
            storage=storage,
            load_if_exists=True,
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=seed),
            pruner=patient_pruner
        )

        initial_params = DEFAULT_DEBERTA_PRIORS.copy()
        if params_file.exists():
            try:
                saved = json.loads(params_file.read_text(encoding='utf-8'))
                clean_saved = {k: v for (k, v) in saved.items() if not k.startswith('_') and k in DEBERTA_SEARCH_SPACES}
                if clean_saved:
                    initial_params.update(clean_saved)
            except Exception:
                pass
        try:
            study.enqueue_trial(initial_params, skip_if_exists=True)
        except TypeError:
            study.enqueue_trial(initial_params)

        def objective(trial: Trial) -> float:
            lr = trial.suggest_float('learning_rate', 1e-05, 5e-05, log=True)
            llrd = trial.suggest_float('llrd_decay', 0.85, 0.95)
            wd = trial.suggest_float('weight_decay', 0.001, 0.1, log=True)
            warmup = trial.suggest_float('warmup_ratio', 0.05, 0.15)
            lambda_neg_val = 2.0
            w_sent_val = 1.0
            w_doc_val = 1.0
            tmp_dir = Path(f'./.tmp_optuna_deberta_{scope}_t{trial.number}')
            tmp_dir.mkdir(parents=True, exist_ok=True)
            config = AutoConfig.from_pretrained(model_name)
            config.num_labels = 2
            model = CustomMDeBERTaForDetection.from_pretrained(model_name, config=config)
            trainer = build_deberta_trainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                output_dir=tmp_dir,
                max_length=max_len,
                sample_weights=sample_weights,
                epochs=epochs,
                learning_rate=lr,
                llrd_decay=llrd,
                lambda_neg=lambda_neg_val,
                w_doc=w_doc_val,
                w_sent=w_sent_val,
                weight_decay=wd,
                warmup_ratio=warmup,
                target_fpr=target_fpr,
                is_multi_scale=is_multi_scale,
                trial=trial,
                is_tuning=True
            )
            try:
                trainer.train()
                eval_res = trainer.evaluate()
                score = float(eval_res.get('eval_pauc_1fpr', 0.5))
            except optuna.TrialPruned:
                raise
            finally:
                del model
                del trainer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir, ignore_errors=True)
            return score

        with TqdmOptunaCallback(n_trials=n_trials, desc=f'Tuning DeBERTa [{scope.upper()}]', save_path=params_file) as opt_cb:
            study.optimize(objective, n_trials=n_trials, callbacks=[opt_cb])

        try:
            best_clean_params = {k: v for (k, v) in study.best_params.items() if not k.startswith('_')}
            best_val = study.best_value
        except Exception:
            best_clean_params = DEFAULT_DEBERTA_PRIORS.copy()
            best_val = 0.5
        print(f'\n-> Best mDeBERTa Parameters ({scope}): {best_clean_params} | pAUC={best_val:.4f}')
        return (best_clean_params, actual_tune_size)