# scripts/generate_data_report.py

import os
from pathlib import Path
from src.data.data_loader import DetectionDataManager
from src.utils.plotting import plot_dataset_overview
from src.utils.paper_reporter import generate_dataset_latex_table
from src.utils.logger import setup_logger

OUTPUT_DIR = Path("/home/gderijck/detection/outputs/metrics/data_summary")


def main():
    logger = setup_logger(name="data_report", log_file="data_report.log")
    logger.info("Loading preprocessed dataset for EDA analysis...")

    data_mgr = DetectionDataManager()
    full_df = data_mgr.raw_dataframe

    logger.info(f"Loaded {len(full_df):,} preprocessed samples.")

    # 1. Generate 300 DPI Overview Figures
    logger.info("Generating publication EDA plots...")
    fig_path = plot_dataset_overview(full_df, str(OUTPUT_DIR))
    logger.info(f"Overview figure saved to: {fig_path}")

    # 2. Generate LaTeX Table
    logger.info("Generating dataset summary LaTeX table...")
    latex_path = os.path.join(OUTPUT_DIR, "dataset_overview_table.tex")
    generate_dataset_latex_table(full_df, latex_path)
    logger.info(f"LaTeX table saved to: {latex_path}")

    print("\n=== Dataset Paper Artifacts Ready ===")
    print(f"Figures : {fig_path}")
    print(f"LaTeX   : {latex_path}\n")


if __name__ == "__main__":
    main()