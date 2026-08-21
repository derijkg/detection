#!/usr/bin/env python3
# scripts/run_experiment.py

import argparse
from pathlib import Path
import sys

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


def main():
    parser = argparse.ArgumentParser(description="Unified Thesis Detection Benchmark Runner")
    parser.add_argument("--model", type=str, default="all", choices=["svm", "mdeberta", "fdgpt", "stat_trajectory", "all"])
    parser.add_argument("--scopes", nargs="+", default=["full", "sentence"], choices=["full", "sentence"])
    parser.add_argument("--preset", type=str, choices=["debug", "fast", "standard", "full"], default="standard")
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter tuning prior to training")
    parser.add_argument("--n_trials", type=int, default=10, help="Number of Optuna tuning trials")
    parser.add_argument("--train_sample_size", type=int, default=None)
    parser.add_argument("--dev_sample_size", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=2.5e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--target_fpr", type=float, default=0.01)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    manager = DetectionDataManager()

    preset_configs = {
        "debug":    {"train": 1000,   "dev": 500},
        "fast":     {"train": 40000,  "dev": 4000},
        "standard": {"train": 100000, "dev": 10000},
        "full":     {"train": -1,     "dev": -1},
    }
    cfg = preset_configs[args.preset]
    train_sz = args.train_sample_size if args.train_sample_size is not None else cfg["train"]
    dev_sz = args.dev_sample_size if args.dev_sample_size is not None else cfg["dev"]

    models_to_run = list(MODEL_METADATA.keys()) if args.model == "all" else [args.model]

    for scope in args.scopes:
        print(f"\n=======================================================")
        print(f"   EXECUTING BENCHMARK: Scope [{scope.upper()}]")
        print(f"=======================================================")

        df_dev = manager.filter_dataframe(DataFilter(splits=["dev"], scopes=[scope]), sample_size=dev_sz, seed=args.seed)
        df_train = manager.filter_dataframe(DataFilter(splits=["train"], scopes=[scope]), sample_size=train_sz, seed=args.seed)
        test_suites = manager.get_benchmark_test_suites(scope=scope)

        for model_key in models_to_run:
            canonical_dir = get_canonical_directory_name(model_key)
            out_dir = Path(args.output_dir) / f"{canonical_dir}_{scope}"
            out_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n>>> Running Detector: {model_key.upper()} ({scope})")
            model_cls = get_detector_class(model_key)
            detector = model_cls(scope=scope, seed=args.seed, log_dir=out_dir)

            # Fit model with tuning flags connected
            detector.fit(
                train_data=df_train,
                dev_data=df_dev,
                epochs=args.epochs,
                lr=args.learning_rate,
                target_fpr=args.target_fpr,
                output_dir=out_dir,
                tune=args.tune,
                n_trials=args.n_trials
            )
            # Save directly to model root directory (creates model.joblib or model_calibration.json)
            detector.save(out_dir)

            # Run evaluation suite
            BenchmarkOrchestrator.run_full_benchmark(
                model_pipeline=detector,
                dev_df=df_dev,
                test_suites=test_suites,
                model_name=canonical_dir,
                scope=scope,
                output_dir=out_dir,
                max_fpr=args.target_fpr
            )

            # Visualizations & diagnostic outputs
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

        # Cross-model comparisons & error reports
        comp_dir = Path(args.output_dir) / "comparisons"
        error_dir = Path(args.output_dir) / "error_analysis"
        comp_dir.mkdir(parents=True, exist_ok=True)
        error_dir.mkdir(parents=True, exist_ok=True)

        summary_jsons = [
            Path(args.output_dir) / f"{get_canonical_directory_name(m)}_{scope}" / "evaluation_summary.json"
            for m in MODEL_METADATA.keys()
        ]
        export_multi_model_comparison_table(summary_jsons, scope=scope, output_path=comp_dir / f"table_comparison_{scope}.tex")

        pred_map = {}
        for m in MODEL_METADATA.keys():
            c_dir = get_canonical_directory_name(m)
            pred_file = Path(args.output_dir) / f"{c_dir}_{scope}" / "predictions_test_standard.csv"
            if pred_file.exists():
                pred_map[c_dir] = pred_file
                export_top_error_cases(pred_file, error_dir, scope=scope, model_name=c_dir, top_k=10)

        if pred_map:
            plot_zoomed_roc_curves(
                prediction_csvs=pred_map,
                scope=scope,
                output_path=comp_dir / f"plot_zoomed_roc_{scope}.png",
                target_fpr=args.target_fpr
            )

    print(f"\n=============================================================")
    print(f"   [ALL RUNS COMPLETE] Results exported to: {args.output_dir}")
    print(f"=============================================================")


if __name__ == "__main__":
    main()