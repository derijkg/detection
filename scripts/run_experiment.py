# scripts/run_experiment.py

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data.data_loader import DataFilter, DetectionDataManager
from src.data.dataset_recipe import DatasetRecipe, RecipeDataBuilder
from src.evaluation.benchmark import BenchmarkOrchestrator
from src.models.registry import (
    MODEL_METADATA,
    get_canonical_directory_name,
    get_detector_class,
)
from src.utils.seed import set_seed
from src.visualization.latex_tables import export_multi_model_comparison_table
from src.visualization.plots import plot_cvar_trajectory, plot_feature_importance


def load_yaml(path: Path) -> Dict[str, Any]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def run_single_experiment(exp: Dict[str, Any], global_cfg: Dict[str, Any], builder: RecipeDataBuilder, manager: DetectionDataManager):
    exp_name = exp.get("name", exp.get("id", f"{exp['model']}_{exp.get('scope', 'custom')}"))
    out_root = Path(global_cfg.get("output_dir", "output"))
    out_dir = out_root / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model_key = exp["model"]
    canonical_name = get_canonical_directory_name(model_key)
    scope = exp.get("scope", "sentence")
    seed = exp.get("seed", global_cfg.get("seed", 42))
    target_fpr = global_cfg.get("target_fpr", 0.01)

    print("\n" + "=" * 80)
    print(f"   STARTING EXPERIMENT: {exp_name}")
    print(f"   Model: {model_key.upper()} | Output: {out_dir}")
    print("=" * 80)

    # 1. Resolve Data Recipes
    recipes_dict = global_cfg.get("data_recipes", {})
    tr_recipe = recipes_dict.get(exp["train_recipe"], exp["train_recipe"])
    dv_recipe = recipes_dict.get(exp["dev_recipe"], exp["dev_recipe"])

    print(f"\n[1/4] Building Training Data ({exp['train_recipe']})...")
    df_train = builder.build(tr_recipe)
    print(f" -> Train Dataset Loaded: {len(df_train):,} rows")

    print(f"\n[2/4] Building Dev/Validation Data ({exp['dev_recipe']})...")
    df_dev = builder.build(dv_recipe)
    print(f" -> Dev Dataset Loaded: {len(df_dev):,} rows")

    # 2. Instantiate Model
    max_len = exp.get("max_length", (128 if scope == "sentence" else 384))
    batch_sz = exp.get("batch_size", None)
    grad_accum = exp.get("gradient_accumulation_steps", None)
    lr = float(exp.get("learning_rate", 3.0e-5 if scope == "sentence" else 2.0e-5))
    epochs = int(exp.get("epochs", 4))
    do_tune = bool(exp.get("tune", False))
    n_trials = int(exp.get("n_trials", 15))
    tuning_sz = int(exp.get("tuning_sample_size", 12000))
    val_sz = int(exp.get("val_sample_size", -1))

    model_cls = get_detector_class(model_key)
    detector = model_cls(scope=scope, seed=seed, log_dir=out_dir, max_length=max_len)

    # 3. Fit Model (Tuning + Final Training)
    print(f"\n[3/4] Fitting Model Pipeline (Tune: {do_tune})...")
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

    # 4. Multi-Suite Testing & Benchmark Evaluation
    eval_benchmarks = exp.get("eval_benchmarks", [scope])
    for eval_scope in eval_benchmarks:
        print(f"\n[4/4] Running Benchmark Evaluation on Scope: [{eval_scope.upper()}]")
        test_suites = manager.get_benchmark_test_suites(scope=eval_scope)

        suite_dir = out_dir if len(eval_benchmarks) == 1 else (out_dir / f"eval_{eval_scope}")
        suite_dir.mkdir(parents=True, exist_ok=True)

        # Ensure threshold calibration uses dev data matching the evaluated target scope
        if eval_scope == scope:
            eval_dev_df = df_dev
        else:
            eval_dev_filter = DataFilter(splits=["dev"], scopes=[eval_scope])
            eval_dev_df = manager.filter_dataframe(eval_dev_filter, sample_size=-1, seed=seed)

        BenchmarkOrchestrator.run_full_benchmark(
            model_pipeline=detector,
            dev_df=eval_dev_df,
            test_suites=test_suites,
            model_name=exp_name,
            scope=eval_scope,
            output_dir=suite_dir,
            max_fpr=target_fpr
        )

    # Export Diagnostic Plots
    if canonical_name == "svm":
        feat_df = detector.extract_feature_importances()
        if not feat_df.empty:
            feat_csv = out_dir / "feature_importance.csv"
            feat_df.to_csv(feat_csv, index=False)
            plot_feature_importance(feat_csv, scope=scope, output_path=out_dir / "plot_feature_importance.png")

    if canonical_name == "deberta":
        cvar_csv = out_dir / "cvar_history.csv"
        if cvar_csv.exists():
            plot_cvar_trajectory(cvar_csv, scope=scope, output_path=out_dir / "plot_cvar_trajectory.png")

    print(f"\n[DONE] Experiment '{exp_name}' complete! All outputs saved to: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Custom Experiment Runner")
    parser.add_argument("--config", type=str, default="configs/custom_experiments.yaml", help="Path to config YAML")
    parser.add_argument("--experiment", type=str, default=None, help="Name of a single experiment to run")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path)
    seed = cfg.get("seed", 42)
    set_seed(seed)

    manager = DetectionDataManager()
    builder = RecipeDataBuilder(manager=manager, use_cache=True)

    if "experiments" in cfg:
        all_exps = cfg["experiments"]
        if args.experiment:
            all_exps = [e for e in all_exps if e.get("name") == args.experiment or e.get("id") == args.experiment]
            if not all_exps:
                print(f"[!] Error: Experiment '{args.experiment}' not found in {args.config}")
                sys.exit(1)

        for exp in all_exps:
            run_single_experiment(exp, global_cfg=cfg, builder=builder, manager=manager)


if __name__ == "__main__":
    main()