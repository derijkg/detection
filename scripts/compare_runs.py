# scripts/compare_runs.py
import argparse
from pathlib import Path
import sys
from typing import Dict, List, Optional
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.evaluation.benchmark import test_saved_model
from src.evaluation.error_analysis import export_top_error_cases
from src.models.registry import normalize_model_name
from src.utils.manifest import RunTracker
from src.visualization.latex_tables import (
    export_multi_model_comparison_table,
    export_multi_model_robustness_table
)
from src.visualization.plots import (
    plot_feature_importance,
    plot_rewrite_sensitivity,
    plot_zoomed_roc_curves
)

def get_target_runs(
    tracker: RunTracker,
    scope: str,
    explicit_run_ids: Optional[List[str]] = None,
    metric: str = 'overall_pauc'
) -> List[Dict]:
    df = tracker.load_manifest()
    if df.empty:
        print('[!] Manifest is empty. Run some experiments first.')
        return []

    if explicit_run_ids:
        df_valid = df[(df['status'] == 'COMPLETED') & (df['run_id'].isin(explicit_run_ids))].copy()
    else:
        valid_scopes = [scope, 'mixed', 'multi_scale', 'combined', 'all']
        df_valid = df[(df['status'] == 'COMPLETED') & (df['scope'].isin(valid_scopes))].copy()

    if df_valid.empty:
        print(f"[!] No completed runs found in manifest matching scope='{scope}'.")
        return []

    df_valid['canonical_model'] = df_valid['model'].apply(normalize_model_name)
    df_valid = df_valid.sort_values(by=metric, ascending=False)
    df_selected = df_valid.groupby('canonical_model', as_index=False).first()
    return df_selected.to_dict(orient='records')

def run_comparison_for_scope(
    tracker: RunTracker,
    scope: str,
    output_dir: Path,
    explicit_run_ids: Optional[List[str]] = None,
    target_fpr: float = 0.01
):
    print(f"\n{'=' * 75}")
    print(f'   GENERATING COMPARATIVE REPORT FOR SCOPE: [{scope.upper()}]')
    print(f"{'=' * 75}")

    selected_runs = get_target_runs(tracker, scope=scope, explicit_run_ids=explicit_run_ids)
    if not selected_runs:
        return

    comp_dir = output_dir / 'comparisons' / scope
    error_dir = output_dir / 'error_analysis' / scope
    comp_dir.mkdir(parents=True, exist_ok=True)
    error_dir.mkdir(parents=True, exist_ok=True)

    summary_json_paths = []
    pred_csv_map = {}

    print('\n[1/3] Selected Models for Comparison:')
    for run in selected_runs:
        r_id = run['run_id']
        m_name = run['model']
        canonical_m = normalize_model_name(m_name)
        r_dir = Path(run['run_dir'])

        summary_candidates = [
            r_dir / 'metrics' / f'eval_{scope}' / 'evaluation_summary.json',
            r_dir / 'metrics' / 'evaluation_summary.json',
            r_dir / f'eval_{scope}' / 'evaluation_summary.json',
            r_dir / 'evaluation_summary.json'
        ]
        summary_path = next((p for p in summary_candidates if p.exists()), None)

        pred_candidates = [
            r_dir / 'predictions' / f'{scope}_predictions_test_standard.csv',
            r_dir / 'predictions' / 'predictions_test_standard.csv',
            r_dir / f'{scope}_predictions_test_standard.csv',
            r_dir / 'predictions_test_standard.csv'
        ]
        pred_path = next((p for p in pred_candidates if p.exists()), None)

        if summary_path is not None:
            summary_json_paths.append(summary_path)
            score_val = run.get('overall_pauc', 'N/A')
            tpr_val = run.get('tpr_at_1fpr', 0.0)
            tpr_str = f'{tpr_val * 100:.2f}%' if isinstance(tpr_val, (int, float)) else 'N/A'
            print(f'  • {canonical_m.upper():<16} | pAUC: {score_val} | TPR@1%FPR: {tpr_str} | Run: {r_id}')

        if pred_path is not None:
            pred_csv_map[canonical_m] = pred_path
            export_top_error_cases(
                predictions_csv=pred_path,
                output_dir=error_dir,
                scope=scope,
                model_name=canonical_m,
                top_k=10
            )

    print(f'\n[2/3] Exporting LaTeX Tables to: {comp_dir}')
    if summary_json_paths:
        tex_main = comp_dir / f'table_comparison_{scope}.tex'
        export_multi_model_comparison_table(
            summary_json_paths=summary_json_paths,
            scope=scope,
            output_path=tex_main
        )
        print(f'  -> Exported Performance Comparison Table: {tex_main.name}')

        tex_rob = comp_dir / f'table_comparison_robustness_{scope}.tex'
        export_multi_model_robustness_table(
            summary_json_paths=summary_json_paths,
            scope=scope,
            output_path=tex_rob
        )
        print(f'  -> Exported Rewrite Robustness Table:     {tex_rob.name}')

    print(f'\n[3/3] Exporting Diagnostic Visualizations to: {comp_dir}')
    if pred_csv_map:
        roc_plot_path = comp_dir / f'plot_zoomed_roc_{scope}.png'
        plot_zoomed_roc_curves(
            prediction_csvs=pred_csv_map,
            scope=scope,
            output_path=roc_plot_path,
            max_fpr=0.05,
            target_fpr=target_fpr
        )

    if summary_json_paths:
        rob_plot_path = comp_dir / f'plot_rewrite_sensitivity_{scope}.png'
        plot_rewrite_sensitivity(
            summary_json_paths=summary_json_paths,
            scope=scope,
            output_path=rob_plot_path
        )

def main():
    parser = argparse.ArgumentParser(description='Multi-Model Comparison & Evaluation CLI')
    parser.add_argument('--scope', type=str, default='all', choices=['full', 'sentence', 'all'], help='Scope to evaluate/compare')
    parser.add_argument('--runs', nargs='+', default=None, help='Explicit list of Run IDs to compare')
    parser.add_argument('--output_dir', type=str, default='output', help='Root output directory')
    parser.add_argument('--target_fpr', type=float, default=0.01, help='Operational regime FPR ceiling')
    parser.add_argument('--test_checkpoint', type=str, default=None, help='Path to a standalone saved checkpoint')
    parser.add_argument('--model_type', type=str, default=None, choices=['svm', 'mdeberta', 'fdgpt', 'stat_trajectory'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    tracker = RunTracker(output_root=out_root)

    if args.test_checkpoint:
        if not args.model_type:
            print('[!] Error: --model_type is required when testing a standalone checkpoint.')
            sys.exit(1)
        test_scope = 'full' if args.scope == 'all' else args.scope
        print(f'\n[Testing Checkpoint] {args.test_checkpoint} ({args.model_type.upper()}, {test_scope})')
        test_saved_model(
            model_type=args.model_type,
            model_path=args.test_checkpoint,
            scope=test_scope,
            output_dir=out_root / f'eval_{args.model_type}_{test_scope}',
            max_fpr=args.target_fpr,
            seed=args.seed
        )
        return

    scopes = ['full', 'sentence'] if args.scope == 'all' else [args.scope]
    for s in scopes:
        run_comparison_for_scope(
            tracker=tracker,
            scope=s,
            output_dir=out_root,
            explicit_run_ids=args.runs,
            target_fpr=args.target_fpr
        )

    print(f"\n[DONE] Comparative tables and plots ready in: {out_root / 'comparisons'}/\n")

if __name__ == '__main__':
    main()