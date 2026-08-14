# train.py

# TODO !!!!! 'FAILED_GENERATION' / 'FAILED_VALIDATION' GUARD!!!!!


import sys
import os
import argparse
import random
import re
import zlib
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from features import prepare_classification_dataset, pre_lemmatize_dataset
from models.robbert import train_transformer
from models.svm import train_svm
from evaluation import run_full_evaluation

from bs4 import MarkupResemblesLocatorWarning
import warnings

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

DEFAULT_MODELS = ['qwen3.5:4b', 'qwen3.6:27b', 'gemma4:e4b', 'gemma4:26b']


def generate_setup_name(args) -> str:
    parts = ["svm"]
    parts.append(f"gran-{args.granularity}")
    parts.append(f"tune-{args.tuning}")
    
    samp = args.tuning_sample_size
    samp_str = f"{samp // 1000}k" if isinstance(samp, int) and samp >= 1000 and samp % 1000 == 0 else str(samp)
    parts.append(f"samp-{samp_str}")
    
    parts.append(f"kernel-{args.kernel}")
    parts.append(f"score-{args.score}")
    if args.score in ['pauc', 'set_fp']:
        parts.append(f"maxfpr-{args.max_fpr}")
    
    if args.llm_ratio != -1:
        parts.append(f"ratio-{args.llm_ratio}")
        
    default_sources = {'UG', 'SB', 'HBO'}
    if set(args.source) != default_sources:
        src_str = "-".join(sorted(args.source))
        parts.append(f"src-{src_str}")

    return "_".join(parts)


def parse_sample_size(value):
    try:
        if '.' in value:
            val = float(value)
            if not (0.0 < val <= 1.0):
                raise argparse.ArgumentTypeError(
                    f"Float sample size must be between 0.0 and 1.0, got {val}"
                )
            return val
        val = int(value)
        if val <= 0:
            raise argparse.ArgumentTypeError(
                f"Integer sample size must be positive, got {val}"
            )
        return val
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Sample size must be an int or float, but got '{value}'"
        )


def main():
    parser = argparse.ArgumentParser(description="LLM Detection Training & Evaluation Orchestrator")

    # Core Controls
    parser.add_argument('--data_path', type=str, default='/home/gderijck/internship/data/gold/llm_added.parquet', help="Path to raw parquet or csv file")
    parser.add_argument('--classifier', type=str, choices=['svm', 'bert', 'both'], default='svm', help="Model type to train")
    parser.add_argument('--granularity', type=str, choices=['full', 'sentence', 'both'], default='full', help="Train on full abstracts, split sentences, or both")

    # Dataset Filtering Parameters
    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS, help="LLM models to include in classification task")
    parser.add_argument('--source', nargs='+', default=['UG', 'SB', 'HBO'], help="Sources to isolate (default all: UG, SB, HBO)")
    parser.add_argument('--llm_ratio', type=int, default=-1, help="Number of LLM rewrites to sample per human record (-1 for all)")
    parser.add_argument('--reset_study', action='store_true', help="Clear and restart the Optuna hyperparameter study database")

    # Hyperparameter Tuning Controls
    parser.add_argument('--trials', type=int, default=50, help="Number of trials to execute during tuning")
    parser.add_argument('--tuning', type=str, choices=['model', '2stage', 'merged'], default='2stage', help="Tuning strategy")
    parser.add_argument('--tuning_sample_size', type=parse_sample_size, default=3000, help="Subsample size for tuning")
    parser.add_argument('--trials_stage1', type=int, default=30, help="Stage 1 trials (TF-IDF)")
    parser.add_argument('--trials_stage2', type=int, default=30, help="Stage 2 trials (SVM)")

    parser.add_argument('--study_name', type=str, default=None, help="Name for the Optuna study entry and saved .pkl file.")
    parser.add_argument('--kernel', type=str, choices=['linear', 'rbf', 'sigmoid', 'all'], default='all', help="SVM kernel choice")
    parser.add_argument('--transformer_name', type=str, default='pdelobelle/robbert-2023-dutch-base', help="HuggingFace model string")
    parser.add_argument(
        '--score', 
        type=str, 
        choices=['f1', 'precision', 'f0.5', 'roc_auc', 'pr_auc', 'set_fp', 'mcc', 'pauc'], 
        default='pauc', 
        help="Metric to optimize during Optuna tuning (default: pauc)"
    )
    parser.add_argument(
        '--max_fpr', 
        type=float, 
        default=0.01, 
        help="Maximum False Positive Rate limit for pAUC metric or set_fp thresholding (default: 0.01)"
    )
    parser.add_argument('--n_jobs_optuna', type=int, default=1, help="Number of parallel jobs for Optuna tuning")

    # --- Refined Model Loading & Evaluation Arguments ---
    parser.add_argument('--load_model', type=str, default=None, help="Path to existing model .pkl. Automatically skips training and evaluates.")
    parser.add_argument('--eval_only', action='store_true', help="Skip training phase and run evaluation.")
    parser.add_argument('--eval_mode', type=str, choices=['both', 'standard', 'synth'], default='both', help="Evaluation target: 'both' (default), 'standard', or 'synth'")
    parser.add_argument('--mixed_ratios', nargs='+', type=float, default=[0.25, 0.50, 0.75], help="LLM substitution ratios for synthetic data testing")

    parser.add_argument('--calibrate', action=argparse.BooleanOptionalAction, default=None, 
                    help="Enable/disable probability calibration (defaults to True if score=='set_fp' or classifier=='both')")
    
    args = parser.parse_args()

    is_eval_only = args.eval_only or (args.load_model is not None)

    print(f"Loading data from: {args.data_path}")
    if args.data_path.endswith('.csv'):
        raw_df = pd.read_csv(args.data_path)
    elif args.data_path.endswith('.parquet') or args.data_path.endswith('.pq'):
        raw_df = pd.read_parquet(args.data_path)
    else:
        raise ValueError("Unsupported data format (Must be CSV or Parquet)")

    if args.source:
        if 'source' in raw_df.columns:
            raw_df = raw_df[raw_df['source'].isin(args.source)].copy()
            print(f"Early filtered dataset to sources: {args.source} ({len(raw_df)} rows remaining)")
        else:
            print("Warning: 'source' column not found in dataset. Skipping early filtering.")

    stratify_col = raw_df['source'] if 'source' in raw_df.columns else None

    # Strict isolation: 80% train / 20% test (1 row per abstract _id = zero leakage)
    train_raw_df, test_raw_df = train_test_split(
        raw_df,
        test_size=0.20,
        random_state=42,
        stratify=stratify_col
    )

    models_dir = 'trained_models'
    os.makedirs(models_dir, exist_ok=True)

    if args.study_name:
        clean_name = args.study_name[:-4] if args.study_name.endswith('.pkl') else args.study_name
        study_name_clean = f"{clean_name}_{args.granularity}" if args.granularity not in clean_name else clean_name
    else:
        study_name_clean = generate_setup_name(args)

    if args.load_model:
        save_path = args.load_model
    else:
        save_path = os.path.join(models_dir, f"{study_name_clean}.pkl")

    # Shape and Lemmatize Test Split
    test_df = prepare_classification_dataset(
        test_raw_df, selected_models=args.models, granularity=args.granularity, source_filter=None, llm_ratio=-1
    )
    test_df = pre_lemmatize_dataset(test_df, text_column='text')

    if args.calibrate is not None:
        do_calibrate = args.calibrate
    else:
        do_calibrate = True

    print(f"-> Probability Calibration: {do_calibrate}")

    if not is_eval_only:
        train_df = prepare_classification_dataset(
            train_raw_df, selected_models=args.models, granularity=args.granularity, source_filter=None, llm_ratio=args.llm_ratio
        )
        train_df = pre_lemmatize_dataset(train_df, text_column='text')

        print(f"\nDataset compiled successfully:")
        print(f"-> Combined Train Split: {len(train_df)} rows")
        print(f"-> Test Split:           {len(test_df)} rows\n")

        if args.classifier in ['svm', 'both']:
            print(f"\n[Experiment Identity]")
            print(f"-> Optuna Study Name: {study_name_clean}")
            print(f"-> Model Save Path:   {save_path}\n")

        train_svm(
            train_df=train_df,
            test_df=test_df,
            test_raw_df=test_raw_df,
            c_val=1.0,
            kernel=args.kernel,
            save_path=save_path,
            granularity=args.granularity,
            run_optuna=True,
            reset_study=args.reset_study,
            trials=args.trials,
            trials_stage1=args.trials_stage1,
            trials_stage2=args.trials_stage2,
            tuning_strategy=args.tuning,
            tuning_sample_size=args.tuning_sample_size,
            score_metric=args.score,
            max_fpr=args.max_fpr,
            study_name=study_name_clean,
            n_jobs_optuna=args.n_jobs_optuna,
            eval_mode=args.eval_mode,
            mixed_ratios=args.mixed_ratios,
            selected_models=args.models,
            calibrate=do_calibrate
        )

        if args.classifier in ['bert', 'both']:
            train_transformer(
                train_df=train_df,
                val_df=None,
                test_df=test_df,
                model_name=args.transformer_name,
                epochs=3,
                batch_size=8,
                lr=2e-5,
                save_path=f"./transformer_{args.granularity}_model",
                run_optuna=True
            )
    else:
        print(f"\n[INFO] Skipping training. Loading model from: {save_path}")
        try:
            loaded_model = joblib.load(save_path)
            clf = loaded_model.named_steps.get('classifier', loaded_model)
            metadata = getattr(loaded_model, 'metadata', {
                'study_name': study_name_clean,
                'save_path': save_path,
                'granularity': getattr(loaded_model, 'granularity', args.granularity),
                'calibrated_threshold': getattr(loaded_model, 'optimal_threshold', 0.5),
            })

            run_full_evaluation(
                model_pipeline=loaded_model,
                test_raw_df=test_raw_df,
                test_df=test_df,
                metadata=metadata,
                selected_models=args.models,
                mixed_ratios=args.mixed_ratios,
                eval_mode=args.eval_mode,
                experiments_dir="experiments"
            )
        except Exception as e:
            print(f"Error loading model from '{save_path}': {e}")


if __name__ == "__main__":
    main()