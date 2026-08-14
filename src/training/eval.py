# detection/src/training/eval.py

import os
import json
import argparse
import numpy as np
import pandas as pd

from src.utils.config import Config
from src.utils.seed import set_seed
from src.utils.logger import setup_logger
from src.models.factory import ModelFactory
from src.training.metrics import compute_classification_metrics
from src.utils.plotting import plot_4panel_evaluation, plot_partial_sensitivity
from src.utils.paper_reporter import generate_results_latex_table
from src.data.data_loader import DetectionDataManager


def evaluate_and_generate_paper_artifacts(
    test_df: pd.DataFrame, 
    probs: np.ndarray, 
    save_dir: str, 
    model_name: str, 
    logger
):
    """Generates 300 DPI plots, LaTeX tables, and JSON evaluation summaries for publication."""
    os.makedirs(save_dir, exist_ok=True)
    labels = test_df['label'].values
    probs_llm = probs[:, 1]
    gen_types = test_df['generation_type'].values

    # 1. Overall Metrics
    overall_metrics = compute_classification_metrics(labels, probs_llm)

    # 2. Per-Generator Model Subgroup Metrics
    test_df_eval = test_df.copy()
    test_df_eval['prob_llm'] = probs_llm
    test_df_eval['pred'] = (probs_llm >= 0.5).astype(int)

    per_model_results = []
    human_df = test_df_eval[test_df_eval['label'] == 0]

    for gen_model in test_df_eval['model_name'].unique():
        if gen_model == "human":
            continue
        
        llm_sub_df = test_df_eval[test_df_eval['model_name'] == gen_model]
        combined_sub = pd.concat([human_df, llm_sub_df])
        
        sub_metrics = compute_classification_metrics(combined_sub['label'].values, combined_sub['prob_llm'].values)
        per_model_results.append({
            "Generator Model": gen_model,
            "Samples": len(llm_sub_df),
            "ROC-AUC": round(sub_metrics["ROC-AUC"], 4),
            "Accuracy": round(sub_metrics["Accuracy"], 4),
            "F1-Score": round(sub_metrics["F1-Score"], 4)
        })

    # 3. Generate 4-Panel Evaluation Plot (300 DPI)
    fig_4panel_path = plot_4panel_evaluation(labels, probs_llm, gen_types, model_name, save_dir)
    logger.info(f"4-Panel Diagnostic Figure saved to: {fig_4panel_path}")

    # 4. Generate Sensitivity Curve for Partials
    fig_sens_path = plot_partial_sensitivity(test_df, probs_llm, save_dir)
    if fig_sens_path:
        logger.info(f"Partial Sensitivity Curve saved to: {fig_sens_path}")

    # 5. Export LaTeX Table
    latex_path = os.path.join(save_dir, f"{model_name}_paper_metrics_table.tex")
    generate_results_latex_table(overall_metrics, per_model_results, model_name, latex_path)
    logger.info(f"LaTeX results table exported to '{latex_path}'")

    # 6. Export JSON Summary
    summary_path = os.path.join(save_dir, f"{model_name}_paper_evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"overall_metrics": overall_metrics, "per_model_metrics": per_model_results}, f, indent=4)
    logger.info(f"JSON evaluation summary exported to '{summary_path}'")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Detector Model")
    parser.add_argument("--config", type=str, default="configs/models/deberta.yaml", help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)

    cfg = Config.from_yaml(args.config)
    logger = setup_logger(name="eval", log_file=f"eval_{cfg.model.name}.log")

    scope = getattr(cfg.model, "granularity", "full")
    logger.info(f"Loading test split for model '{cfg.model.name}' (Scope: {scope})...")
    
    data_mgr = DetectionDataManager()
    test_df = data_mgr.filter_dataframe(splits=['test'], scopes=[scope])

    logger.info(f"Loading checkpoint from '{cfg.training.output_dir}'...")
    detector = ModelFactory.create(
        cfg.model.name, 
        granularity=scope,
        calibrate=getattr(cfg.model, "calibrate", True)
    )
    detector.load(cfg.training.output_dir)

    logger.info("Running inference on test set...")
    probs = detector.predict_proba(test_df['text'].tolist(), batch_size=cfg.eval.batch_size)

    test_df['prob_llm'] = probs[:, 1]
    csv_out = os.path.join(cfg.eval.save_dir, f"{cfg.model.name}_test_predictions.csv")
    os.makedirs(cfg.eval.save_dir, exist_ok=True)
    test_df.to_csv(csv_out, index=False)
    
    logger.info(f"Test predictions exported to '{csv_out}'")
    evaluate_and_generate_paper_artifacts(test_df, probs, cfg.eval.save_dir, cfg.model.name, logger)


if __name__ == "__main__":
    main()