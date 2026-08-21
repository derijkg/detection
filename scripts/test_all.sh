#!/usr/bin/env bash
# scripts/test_all.sh
# Universal test runner: Discovers all trained model checkpoints in output/
# and runs leak-free Dev calibration, multi-suite benchmarking, prediction exports,
# zoomed ROC plots, error analysis, and comparative LaTeX tables.

set -e

OUTPUT_DIR="output"
TARGET_FPR=0.01
SEED=42
DEVICE="cuda"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        --target_fpr) TARGET_FPR="$2"; shift ;;
        --device) DEVICE="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        -h|--help)
            echo "Usage: ./scripts/test_all.sh [--output_dir output] [--target_fpr 0.01] [--device cuda|cpu] [--seed 42]"
            exit 0
            ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "================================================================="
echo "   THESIS UNIFIED BENCHMARK & EVALUATION TEST SUITE             "
echo "================================================================="
echo " Target Output Directory : ${OUTPUT_DIR}"
echo " Operational Regime      : FPR <= ${TARGET_FPR} (1%)"
echo " Evaluation Device       : ${DEVICE}"
echo " Random Seed             : ${SEED}"
echo "================================================================="

python - <<EOF
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent if "__file__" in locals() else Path(".").resolve()
sys.path.append(str(PROJECT_ROOT))

from src.evaluation.benchmark import test_saved_model
from src.visualization.latex_tables import export_multi_model_comparison_table
from src.visualization.plots import plot_zoomed_roc_curves, plot_feature_importance
from src.evaluation.error_analysis import export_top_error_cases

output_root = Path("${OUTPUT_DIR}")
scopes = ["full", "sentence"]
target_fpr = float("${TARGET_FPR}")
seed = int("${SEED}")

models_to_check = [
    {"name": "svm",             "path_pattern": "svm_{scope}/model.joblib"},
    {"name": "fdgpt",           "path_pattern": "fdgpt_{scope}/model_calibration.json"},
    {"name": "stat_trajectory", "path_pattern": "stat_{scope}/model.joblib"},
    {"name": "mdeberta",        "path_pattern": "deberta_{scope}/best_model"}
]

evaluated_models = {s: [] for s in scopes}

print("\n>>> [Step 1/3] Scanning and Evaluating Saved Checkpoints...")

for scope in scopes:
    print(f"\n==================== Scope: {scope.upper()} ====================")
    for m in models_to_check:
        model_name = m["name"]
        rel_path = m["path_pattern"].format(scope=scope)
        model_path = output_root / rel_path
        model_out_dir = output_root / f"{model_name if model_name != 'mdeberta' else 'deberta'}_{scope}"

        if model_path.exists():
            print(f"\n[FOUND] Evaluating {model_name.upper()} ({scope}) from: {model_path}")
            try:
                test_saved_model(
                    model_type=model_name,
                    model_path=model_path,
                    scope=scope,
                    output_dir=model_out_dir,
                    max_fpr=target_fpr,
                    seed=seed
                )
                evaluated_models[scope].append(model_name)
            except Exception as e:
                print(f"[ERROR] Failed evaluating {model_name} ({scope}): {e}")
        else:
            print(f"[SKIP] No checkpoint found for {model_name.upper()} at: {model_path}")

# 2. Multi-Model Comparative LaTeX Tables
print("\n>>> [Step 2/3] Generating Comparative Multi-Model LaTeX Tables...")
comp_dir = output_root / "comparisons"
comp_dir.mkdir(parents=True, exist_ok=True)

for scope in scopes:
    tex_path = comp_dir / f"table_comparison_{scope}.tex"
    candidate_jsons = [
        output_root / f"svm_{scope}" / "evaluation_summary.json",
        output_root / f"fdgpt_{scope}" / "evaluation_summary.json",
        output_root / f"stat_{scope}" / "evaluation_summary.json",
        output_root / f"deberta_{scope}" / "evaluation_summary.json",
    ]
    existing_jsons = [p for p in candidate_jsons if p.exists()]
    if existing_jsons:
        export_multi_model_comparison_table(
            summary_json_paths=existing_jsons,
            scope=scope,
            output_path=tex_path
        )
        print(f" -> Exported Multi-Model Table ({scope}): {tex_path}")

# 3. High-Resolution Plots & Qualitative Error Analysis
print("\n>>> [Step 3/3] Generating Zoomed Low-FPR ROC Plots & Error Dumps...")
error_dir = output_root / "error_analysis"
error_dir.mkdir(parents=True, exist_ok=True)

for scope in scopes:
    pred_map = {}
    svm_preds = output_root / f"svm_{scope}" / "predictions_test_standard.csv"
    deb_preds = output_root / f"deberta_{scope}" / "predictions_test_standard.csv"
    fd_preds = output_root / f"fdgpt_{scope}" / "predictions_test_standard.csv"
    stat_preds = output_root / f"stat_{scope}" / "predictions_test_standard.csv"

    if svm_preds.exists(): pred_map["svm"] = svm_preds
    if deb_preds.exists(): pred_map["mdeberta"] = deb_preds
    if fd_preds.exists(): pred_map["fdgpt"] = fd_preds
    if stat_preds.exists(): pred_map["stat_trajectory"] = stat_preds

    if pred_map:
        roc_plot_path = comp_dir / f"plot_zoomed_roc_{scope}.png"
        plot_zoomed_roc_curves(
            prediction_csvs=pred_map,
            scope=scope,
            output_path=roc_plot_path,
            max_fpr=0.05,
            target_fpr=target_fpr
        )

    # SVM Feature Importance Bar Chart
    svm_feat_csv = output_root / f"svm_{scope}" / "feature_importance.csv"
    if svm_feat_csv.exists():
        feat_plot_path = output_root / f"svm_{scope}" / "plot_feature_importance.png"
        plot_feature_importance(
            feature_csv=svm_feat_csv,
            scope=scope,
            output_path=feat_plot_path,
            top_n=15
        )

    # Qualitative Top False Positive / False Negative Dumps
    for m_key, p_file in [("svm", svm_preds), ("mdeberta", deb_preds), ("stat_trajectory", stat_preds)]:
        if p_file.exists():
            export_top_error_cases(
                predictions_csv=p_file,
                output_dir=error_dir,
                scope=scope,
                model_name=m_key,
                top_k=10
            )

print("\n=================================================================")
print("   [TEST COMPLETE] All benchmarks, tables, and plots exported!")
print(f"   Check your results inside: {output_root}/")
print("=================================================================")
EOF