# scripts/run_experiment.py
import argparse
from pathlib import Path
import shutil
import sys
import traceback

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data.data_loader import DataFilter, DetectionDataManager
from src.data.dataset_recipe import RecipeDataBuilder
from src.evaluation.benchmark import BenchmarkOrchestrator
from src.models.registry import get_canonical_directory_name, get_detector_class
from src.utils.config import ExperimentConfig, GlobalExperimentConfig
from src.utils.manifest import RunTracker
from src.utils.seed import set_seed
from src.visualization.plots import plot_cvar_trajectory, plot_feature_importance

def run_single_experiment(
    exp: ExperimentConfig,
    global_cfg: GlobalExperimentConfig,
    builder: RecipeDataBuilder,
    manager: DetectionDataManager,
    tracker: RunTracker
):
    model_key = exp.model
    canonical_name = get_canonical_directory_name(model_key)
    scope = exp.scope
    seed = exp.seed
    target_fpr = exp.target_fpr

    tr_recipe = global_cfg.data_recipes[exp.train_recipe]
    dv_recipe = global_cfg.data_recipes[exp.dev_recipe]

    ctx = tracker.start_run(
        exp_name=exp.name,
        model_name=model_key,
        scope=scope,
        config=exp.raw_dict,
        train_recipe_meta=tr_recipe.__dict__,
        dev_recipe_meta=dv_recipe.__dict__
    )

    print('\n' + '=' * 80)
    print(f'   STARTING RUN: {ctx.run_id}')
    print(f'   Model: {model_key.upper()} | Scope: {scope.upper()}')
    print(f'   Artifact Directory: {ctx.run_dir}')
    print('=' * 80)

    try:
        print(f'\n[1/4] Building Training Data ({exp.train_recipe})...')
        df_train = builder.build(tr_recipe)
        print(f' -> Train Dataset Loaded: {len(df_train):,} rows')

        print(f'\n[2/4] Building Dev/Validation Data ({exp.dev_recipe})...')
        df_dev = builder.build(dv_recipe)
        print(f' -> Dev Dataset Loaded: {len(df_dev):,} rows')

        model_cls = get_detector_class(model_key)
        detector = model_cls.from_config(exp, log_dir=ctx.run_dir)

        print(f'\n[3/4] Fitting Model Pipeline (Tune: {exp.tuning.enabled})...')
        detector.fit(train_data=df_train, dev_data=df_dev, config=exp, output_dir=ctx.model_dir)
        detector.save(ctx.model_dir)

        primary_summary = {}
        for eval_scope in exp.eval_benchmarks:
            print(f'\n[4/4] Running Benchmark Evaluation on Scope: [{eval_scope.upper()}]')
            test_suites = manager.get_benchmark_test_suites(scope=eval_scope)
            suite_dir = ctx.metrics_dir if len(exp.eval_benchmarks) == 1 else ctx.metrics_dir / f'eval_{eval_scope}'
            suite_dir.mkdir(parents=True, exist_ok=True)

            if eval_scope == scope:
                eval_dev_df = df_dev
            else:
                eval_dev_filter = DataFilter(splits=['dev'], scopes=[eval_scope])
                eval_dev_df = manager.filter_dataframe(eval_dev_filter, sample_size=-1, seed=seed)

            summary = BenchmarkOrchestrator.run_full_benchmark(
                model_pipeline=detector,
                dev_df=eval_dev_df,
                test_suites=test_suites,
                model_name=exp.name,
                scope=eval_scope,
                output_dir=suite_dir,
                max_fpr=target_fpr
            )

            for pred_file in suite_dir.glob('predictions_*.csv'):
                dest_file = ctx.predictions_dir / (
                    f'{eval_scope}_{pred_file.name}' if len(exp.eval_benchmarks) > 1 else pred_file.name
                )
                shutil.copy(pred_file, dest_file)

            if eval_scope == scope or not primary_summary:
                primary_summary = summary

        if canonical_name == 'svm':
            feat_df = detector.extract_feature_importances()
            if not feat_df.empty:
                feat_csv = ctx.metrics_dir / 'feature_importance.csv'
                feat_df.to_csv(feat_csv, index=False)
                plot_feature_importance(
                    feature_csv=feat_csv,
                    scope=scope,
                    output_path=ctx.plots_dir / 'plot_feature_importance.png'
                )

        if canonical_name == 'deberta':
            cvar_csv = ctx.model_dir / 'cvar_history.csv'
            if cvar_csv.exists():
                plot_cvar_trajectory(
                    cvar_csv=cvar_csv,
                    scope=scope,
                    output_path=ctx.plots_dir / 'plot_cvar_trajectory.png'
                )

        ctx.record_to_manifest(summary_dict=primary_summary, status='COMPLETED')
        print(f"\n[DONE] Run '{ctx.run_id}' complete! Logged to {ctx.manifest_path}")

    except Exception as e:
        err_msg = traceback.format_exc()
        print(f"\n[ERROR] Run '{ctx.run_id}' failed with exception:\n{err_msg}")
        ctx.record_to_manifest(status='FAILED', error_msg=str(e))
        raise e

def main():
    parser = argparse.ArgumentParser(description='Unified Experiment Runner')
    parser.add_argument('--config', type=str, default='configs/experiments.yaml', help='Path to unified config YAML')
    parser.add_argument('--experiment', type=str, default=None, help='Name of a single experiment to run (runs all if not specified)')
    args = parser.parse_args()

    global_cfg = GlobalExperimentConfig.load(args.config)
    set_seed(global_cfg.seed)

    tracker = RunTracker(output_root=global_cfg.output_dir)
    manager = DetectionDataManager()
    builder = RecipeDataBuilder(manager=manager, use_cache=True)

    exps_to_run = global_cfg.experiments
    if args.experiment:
        exps_to_run = [e for e in exps_to_run if e.name == args.experiment]
        if not exps_to_run:
            print(f"[!] Error: Experiment '{args.experiment}' not found in {args.config}")
            sys.exit(1)

    for exp in exps_to_run:
        run_single_experiment(
            exp=exp,
            global_cfg=global_cfg,
            builder=builder,
            manager=manager,
            tracker=tracker
        )

if __name__ == '__main__':
    main()