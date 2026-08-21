# src/evaluation/error_analysis.py

from pathlib import Path
from typing import Union
import pandas as pd


def export_top_error_cases(
    predictions_csv: Union[str, Path],
    output_dir: Union[str, Path],
    scope: str,
    model_name: str,
    top_k: int = 10
):
    """
    Exports top False Positives and False Negatives ranked by confidence for qualitative analysis.
    """
    csv_p = Path(predictions_csv)
    if not csv_p.exists():
        return

    df = pd.read_csv(csv_p)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. False Positives (Human text with highest AI probability)
    fps = df[(df["label"] == 0) & (df["predicted_label"] == 1)].sort_values(by="predicted_prob", ascending=False).head(top_k)

    # 2. False Negatives (AI text with lowest AI probability)
    fns = df[(df["label"] == 1) & (df["predicted_label"] == 0)].sort_values(by="predicted_prob", ascending=True).head(top_k)

    fp_path = out_dir / f"errors_false_positives_{model_name}_{scope}.csv"
    fn_path = out_dir / f"errors_false_negatives_{model_name}_{scope}.csv"

    cols = [c for c in ["text", "predicted_prob", "threshold_used", "generator_model", "model_name", "word_count"] if c in df.columns]

    fps[cols].to_csv(fp_path, index=False)
    fns[cols].to_csv(fn_path, index=False)

    print(f" -> [Error Analysis] Saved Top {top_k} False Positives to: {fp_path}")
    print(f" -> [Error Analysis] Saved Top {top_k} False Negatives to: {fn_path}")