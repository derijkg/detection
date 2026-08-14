# detection/scripts/run.py

import argparse
import sys
import json
from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer

# Standard path insertion to find 'src' modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data.data_loader import DetectionDataManager, DataFilter
from src.training.metrics import evaluate_predictions
from src.models.deberta import prepare_hf_datasets, run_deberta_tuning, train_deberta_model, predict_deberta


def main():
    parser = argparse.ArgumentParser(description="Unified Hyperparameter Tuning and Training Script")
    parser.add_argument("--model_type", type=str, choices=["deberta", "svm"], default="deberta", help="Model family to run")
    parser.add_argument("--data_path", type=str, default=None, help="Path to parquet dataset (optional)")
    parser.add_argument("--deberta_name", type=str, default="microsoft/mdeberta-v3-base", help="HuggingFace model identifier")
    parser.add_argument("--n_trials", type=int, default=10, help="Optuna hyperparameter tuning trials")
    parser.add_argument("--sample_size", type=int, default=-1, help="Sample size for training data (-1 for full)")
    parser.add_argument("--tune_sample_size", type=int, default=1000, help="Sample size per class for Optuna tuning")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save artifacts")

    args = parser.parse_args()

    # Output Directory Setup
    if args.output_dir is None:
        args.output_dir = str(PROJECT_ROOT / "outputs" / args.model_type)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Data Manager and Load Splits
    print(f"\n[DATA MANAGER] Loading dataset from: {args.data_path or 'DEFAULT'}")
    dm = DetectionDataManager(data_path=args.data_path)

    print("[DATA MANAGER] Filtering Train/Val/Test splits...")
    train_df = dm.filter_dataframe(filter_config=DataFilter(splits=["train"]), sample_size=args.sample_size, seed=args.seed)
    val_df = dm.filter_dataframe(filter_config=DataFilter(splits=["val"]), sample_size=args.sample_size, seed=args.seed)
    test_df = dm.filter_dataframe(filter_config=DataFilter(splits=["test"]), sample_size=-1, seed=args.seed)

    print(f"Data Loaded -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # ---------------------------------------------------------
    # PIPELINE: DEBERTA
    # ---------------------------------------------------------
    if args.model_type == "deberta":
        tokenizer = AutoTokenizer.from_pretrained(args.deberta_name, use_fast=False)

        # Prepare HF datasets
        train_ds, val_ds, test_ds = prepare_hf_datasets(train_df, val_df, test_df, tokenizer)

        # Subsample dataset for hyperparameter tuning if requested
        if args.tune_sample_size > 0 and len(train_df) > args.tune_sample_size:
            tune_train_df = dm.filter_dataframe(filter_config=DataFilter(splits=["train"]), sample_size=args.tune_sample_size, seed=args.seed)
            tune_val_df = dm.filter_dataframe(filter_config=DataFilter(splits=["val"]), sample_size=args.tune_sample_size // 2, seed=args.seed)
            optuna_train_ds, optuna_val_ds, _ = prepare_hf_datasets(tune_train_df, tune_val_df, test_df, tokenizer)
        else:
            optuna_train_ds, optuna_val_ds = train_ds, val_ds

        # Step A: Optuna Hyperparameter Tuning
        best_params = run_deberta_tuning(
            optuna_train_ds, 
            optuna_val_ds, 
            tokenizer, 
            model_name=args.deberta_name, 
            n_trials=args.n_trials
        )

        with open(output_path / "best_hyperparameters.json", "w") as f:
            json.dump(best_params, f, indent=4)

        # Step B: Train Final Model on Full Training Set
        print(f"\n[DEBERTA TRAINING] Training final model with best parameters...")
        model, trainer = train_deberta_model(
            train_ds=train_ds,
            val_ds=val_ds,
            tokenizer=tokenizer,
            hyperparams=best_params,
            model_name=args.deberta_name,
            output_dir=str(output_path / "model_weights")
        )

        # Step C: Evaluate on Test Set & Save Predictions
        print("\n[DEBERTA EVALUATION] Running test inference...")
        test_pred_df = predict_deberta(test_df, model, tokenizer)
        test_pred_df.to_parquet(output_path / "test_predictions.parquet", index=False)

        metrics, per_model_df = evaluate_predictions(test_pred_df, output_dir=str(output_path))

        print("\n" + "="*60)
        print(" FINAL TEST METRICS (DEBERTA) ")
        print("="*60)
        for k, v in metrics.items():
            print(f"  {k:<18}: {v}")
        print("="*60 + "\n")

    # ---------------------------------------------------------
    # PIPELINE: SVM (Ready for integration)
    # ---------------------------------------------------------
    elif args.model_type == "svm":
        print("\n[SVM PIPELINE] Ready to be attached to run.py!")


if __name__ == "__main__":
    main()