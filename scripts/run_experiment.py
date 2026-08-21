#!/usr/bin/env python3
# scripts/run_experiment.py

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data.data_loader import DetectionDataManager, DataFilter
from src.evaluation.benchmark import BenchmarkOrchestrator
from src.evaluation.error_analysis import export_top_error_cases
from src.models.registry import (
    MODEL_METADATA,
    get_detector_class,
    get_canonical_directory_name,
)
from src.utils.seed import set_seed
from src.visualization.latex_tables import export_multi_model_comparison_table
from src.visualization.plots import (
    plot_cvar_trajectory,
    plot_feature_importance,
    plot_zoomed_roc_curves,
)


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def main():
    parser = argparse.ArgumentParser(description="Modular Unified Thesis Detection Benchmark Runner")
    parser.add_argument("--config", type=str, default="configs/default_experiment.yaml", help="Path to YAML config")
    parser.add_argument("--model", type=str, default="all", choices=["svm", "mdeberta", "fdgpt", "stat_trajectory", "all"])
    parser.add_argument("--scopes", nargs="+", default=None, choices=["full", "sentence"])
    parser.add_argument("--preset", type=str, choices=["debug", "fast", "standard", "full"], default=None)
    parser.add_argument("--tune", action="store_true", default=None, help="Force hyperparameter tuning")
    parser.add_argument("--n_trials", type=int, default=None, help="Global override for tuning trials")
    parser.add_argument("--train_sample_size", type=int, default=None)
    parser.add_argument("--dev_sample_size", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # 1. Load YAML Configuration
    cfg_file = Path(args.config)
    cfg = load_yaml_config(cfg_file) if cfg_file.exists() else {}

    # 2. Resolve Global Parameters
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    output_dir = Path(args.output_dir if args.output_dir is not None else cfg.get("output_dir", "output"))
    target_fpr = cfg.get("target_fpr", 0.01)
    scopes = args.scopes if args.scopes is not None else ["full", "sentence"]

    set_seed(seed)
    manager = DetectionDataManager()

    # Preset handling
    preset_configs = {
        "debug":    {"train": 1000,   "dev": 500},
        "fast":     {"train": 40000,  "dev": 4000},
        "standard": {"train": 100000, "dev": 10000},
        "full":     {"train": -1,     "dev": -1},
    }
    if args.preset:
        train_sz = preset_configs[args.preset]["train"]
        dev_sz = preset_configs[args.preset]["dev"]
    else:
        train_sz = args.train_sample_size or cfg.get("train_sample_size", 100000)
        dev_sz = args.dev_sample_size or cfg.get("dev_sample_size", 10000)

    models_to_run = list(MODEL_METADATA.keys()) if args.model == "all" else [args.model]

    for scope in scopes:
        print(f"\n=======================================================")
        print(f"   EXECUTING BENCHMARK: Scope [{scope.upper()}]")
        print(f"=======================================================")

        df_dev = manager.filter_dataframe(DataFilter(splits=["dev"], scopes=[scope]), sample_size=dev_sz, seed=seed)
        df_train = manager.filter_dataframe(DataFilter(splits=["train"], scopes=[scope]), sample_size=train_sz, seed=seed)
        test_suites = manager.get_benchmark_test_suites(scope=scope)

        for model_key in models_to_run:
            canonical_dir = get_canonical_directory_name(model_key)
            out_dir = output_dir / f"{canonical_dir}_{scope}"
            out_dir.mkdir(parents=True, exist_ok=True)

            # Retrieve granular model & scope config (supports both 'mdeberta' and 'deberta', 'stat' and 'stat_trajectory')
            models_dict = cfg.get("models", {})
            model_cfg = models_dict.get(model_key) or models_dict.get(canonical_dir, {})
            model_scope_cfg = model_cfg.get(scope, {})

            # Merge with CLI overrides
            do_tune = args.tune if args.tune is not None else model_scope_cfg.get("tune", False)
            n_trials = args.n_trials if args.n_trials is not None else model_scope_cfg.get("n_trials", 10)
            tuning_sz = model_scope_cfg.get("tuning_sample_size", 15000 if canonical_dir == "svm" else 4000)
            val_sz = model_scope_cfg.get("val_sample_size", -1)
            lr = model_scope_cfg.get("learning_rate", 2.5e-5)
            epochs = model_scope_cfg.get("epochs", 3)
            max_len = model_scope_cfg.get("max_length", (128 if scope == "sentence" else 384))
            batch_sz = model_scope_cfg.get("batch_size", None)
            grad_accum = model_scope_cfg.get("gradient_accumulation_steps", None)

            print(f"\n>>> Running Detector: {model_key.upper()} ({scope})")
            print(f"    Config -> Tune: {do_tune} | Trials: {n_trials} | Tune Sz: {tuning_sz} | Final Epochs: {epochs}")

            model_cls = get_detector_class(model_key)
            detector = model_cls(scope=scope, seed=seed, log_dir=out_dir, max_length=max_len)

            # Fit detector with granular settings
            detector.fit(
                train_data=df_train,
                dev_data=df_dev,
                epochs=epochs,
                lr=lr,
                target_fpr=target_fpr,
                output_dir=out_dir,
                tune=do_tune,
                n_trials=n_trials,
                tuning_sample_size=tuning_sz,
                val_sample_size=val_sz,
                batch_size=batch_sz,
                gradient_accumulation_steps=grad_accum,
            )
            detector.save(out_dir)

            # Evaluation
            BenchmarkOrchestrator.run_full_benchmark(
                model_pipeline=detector,
                dev_df=df_dev,
                test_suites=test_suites,
                model_name=canonical_dir,
                scope=scope,
                output_dir=out_dir,
                max_fpr=target_fpr
            )

            # Plots
            if canonical_dir == "svm":
                feat_df = detector.extract_feature_importances()
                if not feat_df.empty:
                    feat_csv = out_dir / "feature_importance.csv"
                    feat_df.to_csv(feat_csv, index=False)
                    plot_feature_importance(feat_csv, scope=scope, output_path=out_dir / "plot_feature_importance.png")

            if canonical_dir == "deberta":
                cvar_csv = out_dir / "cvar_history.csv"
                if cvar_csv.exists():
                    plot_cvar_trajectory(cvar_csv, scope=scope, output_path=out_dir / "plot_cvar_trajectory.png")

        # Comparisons
        comp_dir = output_dir / "comparisons"
        error_dir = output_dir / "error_analysis"
        comp_dir.mkdir(parents=True, exist_ok=True)
        error_dir.mkdir(parents=True, exist_ok=True)

        summary_jsons = [
            output_dir / f"{get_canonical_directory_name(m)}_{scope}" / "evaluation_summary.json"
            for m in MODEL_METADATA.keys()
        ]
        export_multi_model_comparison_table(summary_jsons, scope=scope, output_path=comp_dir / f"table_comparison_{scope}.tex")

    print(f"\n=============================================================")
    print(f"   [ALL RUNS COMPLETE] Results exported to: {output_dir}")
    print(f"=============================================================")


if __name__ == "__main__":
    main()