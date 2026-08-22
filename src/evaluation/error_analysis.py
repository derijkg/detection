"""
src/evaluation/error_analysis.py
Extracts top-k false positives (human text falsely flagged as AI) and
false negatives (AI text falsely accepted as human) with full diagnostic attributes.
"""

from pathlib import Path
from typing import List, Union
import pandas as pd


def export_top_error_cases(
    predictions_csv: Union[str, Path],
    output_dir: Union[str, Path],
    scope: str,
    model_name: str,
    top_k: int = 10
):
    csv_p = Path(predictions_csv)
    if not csv_p.exists():
        return

    df = pd.read_csv(csv_p)
    if df.empty or 'label' not in df.columns or 'predicted_label' not in df.columns:
        return

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Top false positives: True Human (0), Predicted AI (1), highest predicted probability
    fps = df[(df['label'] == 0) & (df['predicted_label'] == 1)].sort_values(
        by='predicted_prob', ascending=False
    ).head(top_k)

    # Top false negatives: True AI (1), Predicted Human (0), lowest predicted probability
    fns = df[(df['label'] == 1) & (df['predicted_label'] == 0)].sort_values(
        by='predicted_prob', ascending=True
    ).head(top_k)

    fp_path = out_dir / f"errors_false_positives_{model_name}_{scope}.csv"
    fn_path = out_dir / f"errors_false_negatives_{model_name}_{scope}.csv"

    pref_cols = ['_id', 'text', 'predicted_prob', 'threshold_used', 'generator_model', 'model_name', 'word_count', 'char_count']
    available_cols = [c for c in pref_cols if c in df.columns]
    if not available_cols:
        available_cols = list(df.columns)

    fps[available_cols].to_csv(fp_path, index=False)
    fns[available_cols].to_csv(fn_path, index=False)

    print(f" -> [Error Analysis] Saved Top {len(fps)} False Positives to: {fp_path}")
    print(f" -> [Error Analysis] Saved Top {len(fns)} False Negatives to: {fn_path}")