# detection/scripts/run_experiments.py

import os
import sys
from pathlib import Path

# Add project root (~/detection) to Python path FIRST
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import shutil
import argparse
import pandas as pd
import numpy as np
from typing import List, Dict, Any

from src.utils.config import Config
from src.utils.seed import set_seed
from src.utils.logger import setup_logger
from src.models.factory import ModelFactory
from src.training.metrics import compute_classification_metrics
from src.utils.plotting import plot_4panel_evaluation, plot_partial_sensitivity
from src.training.tuner import run_tuning
from src.data.data_loader import DetectionDataManager

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
UNIFIED_METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"

def run_experiment(model_name: str, scope: str, tune: bool, seed: int, logger) -> Dict[str, Any]:
    """Runs tuning (optional), training, and evaluation for a (model_name, scope) pair."""
    config_path = Config.resolve_config_path(model_name, scope)
    logger.info(f"Using config file: '{config_path}'")
    
    cfg = Config.from_yaml(str(config_path))
    cfg.model.granularity = scope
    
    checkpoint_dir = f"outputs/checkpoints/{model_name}_{scope}"
    metrics_dir = f"outputs/metrics/{model_name}_{scope}"
    
    # 1. Optuna Tuning (If requested)
    if tune:
        logger.info(f"Running Optuna hyperparameter search for '{model_name}' ({scope})...")
        tuned_file = run_tuning(model_name, scope, seed=seed)
    else:
        tuned_file = f"outputs/metrics/{model_name}_{scope}_optuna/best_hyperparameters.json"

    # 2. Setup Training Arguments
    training_dict = vars(cfg.training).copy()
    training_dict["output_dir"] = checkpoint_dir

    if os.path.exists(tuned_file):
        logger.info(f"Loading tuned parameters from '{tuned_file}'...")
        with open(tuned_file, "r", encoding="utf-8") as f:
            tuned_data = json.load(f)
            training_dict.update(tuned_data.get("best_params", {}))

    # 3. Load Train & Dev Data
    data_mgr = DetectionDataManager()
    train_size = getattr(cfg.training, "train_sample_size", -1)

    if model_name.lower() in ["deberta", "mdeberta", "mdeberta-v3"]:
        train_df = data_mgr.filter_dataframe(splits=["train"], scopes=[scope], sample_size=train_size, seed=seed)
        val_df = data_mgr.filter_dataframe(splits=["dev"], scopes=[scope], seed=seed)
        
        from datasets import Dataset as HFDataset
        train_ds = HFDataset.from_pandas(train_df, preserve_index=False)
        val_ds = HFDataset.from_pandas(val_df, preserve_index=False)
    else:
        train_df = data_mgr.filter_dataframe(splits=["train"], scopes=[scope], sample_size=train_size, seed=seed)
        val_df = data_mgr.filter_dataframe(splits=["dev"], scopes=[scope], seed=seed)
        
        train_ds = (train_df['text'].tolist(), train_df['label'].values)
        val_ds = (val_df['text'].tolist(), val_df['label'].values)

    # 4. Train Model
    detector = ModelFactory.create(
        model_name, 
        max_length=getattr(cfg.model, "max_length", 256),
        granularity=scope,
        calibrate=getattr(cfg.model, "calibrate", True)
    )
    detector.train(train_ds, val_ds, training_dict)
    detector.save(checkpoint_dir)

    # Save copy of best hyperparameters inside checkpoint directory
    if os.path.exists(tuned_file):
        os.makedirs(checkpoint_dir, exist_ok=True)
        shutil.copy(tuned_file, os.path.join(checkpoint_dir, "best_hyperparameters.json"))

    # 5. Evaluate on Test Set
    test_df = data_mgr.filter_dataframe(splits=['test'], scopes=[scope])
    probs = detector.predict_proba(test_df['text'].tolist(), batch_size=cfg.eval.batch_size)
    probs_llm = probs[:, 1]

    labels = test_df['label'].values
    overall_metrics = compute_classification_metrics(labels, probs_llm)

    # Export Predictions and Artifacts
    test_df_out = test_df.copy()
    test_df_out['prob_llm'] = probs_llm
    test_df_out['pred'] = (probs_llm >= 0.5).astype(int)
    
    os.makedirs(metrics_dir, exist_ok=True)
    test_df_out.to_csv(os.path.join(metrics_dir, f"{model_name}_{scope}_predictions.csv"), index=False)

    plot_4panel_evaluation(labels, probs_llm, test_df['generation_type'].values, f"{model_name}_{scope}", metrics_dir)
    plot_partial_sensitivity(test_df, probs_llm, metrics_dir)

    record = {
        "model": model_name,
        "scope": scope,
        "test_samples": len(test_df),
        "roc_auc": round(overall_metrics.get("ROC-AUC", 0), 4),
        "pr_auc": round(overall_metrics.get("PR-AUC (AP)", 0), 4),
        "accuracy": round(overall_metrics.get("Accuracy", 0), 4),
        "f1_score": round(overall_metrics.get("F1-Score", 0), 4),
        "specificity": round(overall_metrics.get("Specificity", 0), 4),
        "tpr_at_1fpr": round(overall_metrics.get("TPR @ 1% FPR", 0), 4)
    }

    logger.info(f"Finished Run ({model_name} - {scope}) | ROC-AUC: {record['roc_auc']} | F1: {record['f1_score']}")
    return record


def compile_unified_reports(results: List[Dict[str, Any]], logger) -> None:
    UNIFIED_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results)

    csv_path = UNIFIED_METRICS_DIR / "unified_benchmark_results.csv"
    results_df.to_csv(csv_path, index=False)

    tex_path = UNIFIED_METRICS_DIR / "unified_paper_table.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{Unified Performance Comparison across Detector Architectures and Scopes.}\n")
        f.write("\\begin{tabular}{llcccccc}\n\\toprule\n")
        f.write("\\textbf{Detector} & \\textbf{Scope} & \\textbf{ROC-AUC} & \\textbf{PR-AUC} & \\textbf{Accuracy} & \\textbf{F1-Score} & \\textbf{Spec.} & \\textbf{TPR @ 1\\% FPR} \\\\\n\\midrule\n")
        for _, row in results_df.iterrows():
            f.write(
                f"{row['model'].upper()} & {row['scope']} & \\textbf{{{row['roc_auc']:.4f}}} & "
                f"{row['pr_auc']:.4f} & {row['accuracy']:.4f} & {row['f1_score']:.4f} & "
                f"{row['specificity']:.4f} & {row['tpr_at_1fpr']:.4f} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    logger.info(f"Unified CSV saved to: '{csv_path}'")
    logger.info(f"Unified LaTeX paper table saved to: '{tex_path}'")


def main():
    parser = argparse.ArgumentParser(description="Unified Experiment Orchestration Benchmark")
    parser.add_argument("--models", nargs="+", default=["svm", "deberta"], help="Models to run")
    parser.add_argument("--scopes", nargs="+", default=["full", "sentence"], help="Scopes ('full', 'sentence')")
    parser.add_argument("--tune", action="store_true", help="Run Optuna tuning before training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    logger = setup_logger(name="experiments", log_file="benchmark_orchestrator.log")

    all_results = []
    for model_name in args.models:
        for scope in args.scopes:
            try:
                res = run_experiment(model_name, scope, tune=args.tune, seed=args.seed, logger=logger)
                all_results.append(res)
            except Exception as e:
                logger.error(f"Failed run for Model='{model_name}' Scope='{scope}': {e}", exc_info=True)

    if all_results:
        compile_unified_reports(all_results, logger)


if __name__ == "__main__":
    main()