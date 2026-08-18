import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def get_substitution_bucket(ratio: float) -> str:
    """Bins continuous or discrete substitution ratios into 25%, 50%, 75%, or 100% buckets."""
    if np.isclose(ratio, 0.0):
        return "0%"
    elif np.isclose(ratio, 1.0) or ratio >= 0.90:
        return "100% (Full Rewrite)"
    elif ratio <= 0.375:
        return "25% Substitution"
    elif ratio <= 0.625:
        return "50% Substitution"
    else:
        return "75% Substitution"


def evaluate_mdeberta_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Computes binary classification metrics across overall splits,

    bucketed substitution ratios (25%, 50%, 75%, 100%), and generator models.
    """
    # Filter test split if split column exists
    if "split" in df.columns:
        df_test = df[df["split"] == "test"].copy()
    else:
        df_test = df.copy()

    # Ensure correct numeric types
    df_test["label"] = df_test["label"].astype(int)
    df_test["prob_llm"] = df_test["prob_llm"].astype(float)
    df_test["pred"] = df_test["pred"].astype(int) #CHANGED FROM PRED_LLM for other changed pred back to pred_llm
    df_test["llm_ratio"] = df_test["llm_ratio"].astype(float)

    # Normalize llm_ratio if expressed as percentage (0-100) instead of fraction (0.0-1.0)
    if df_test["llm_ratio"].max() > 1.0:
        df_test["llm_ratio"] = df_test["llm_ratio"] / 100.0

    # Assign substitution ratio bucket
    df_test["sub_bucket"] = df_test["llm_ratio"].apply(get_substitution_bucket)

    # Separate Human (label 0) and LLM (label 1) subsets
    df_human = df_test[df_test["label"] == 0].copy()
    df_llm = df_test[df_test["label"] == 1].copy()

    results = []

    # -------------------------------------------------------------
    # 1. Overall System Performance
    # -------------------------------------------------------------
    results.append({
        "Category / Subset": "Overall Test Set",
        "Group": "Overall System Performance",
        "N": f"{len(df_test):,}",
        "Mean P(LLM)": df_test["prob_llm"].mean(),
        "ROC-AUC": roc_auc_score(df_test["label"], df_test["prob_llm"]),
        "Accuracy": accuracy_score(df_test["label"], df_test["pred"]),
        "F1-Score": f1_score(
            df_test["label"], df_test["pred"], zero_division=0
        ),
        "Precision": precision_score(
            df_test["label"], df_test["pred"], zero_division=0
        ),
        "Recall": recall_score(
            df_test["label"], df_test["pred"], zero_division=0
        ),
    })

    # Full Clean Abstracts (0% Human / 100% LLM Full Rewrite)
    df_clean = df_test[
        df_test["sub_bucket"].isin(["0%", "100% (Full Rewrite)"])
    ].copy()
    if len(df_clean) > 0:
        results.append({
            "Category / Subset": "Full Clean Abstracts (0% / 100%)",
            "Group": "Overall System Performance",
            "N": f"{len(df_clean):,}",
            "Mean P(LLM)": df_clean["prob_llm"].mean(),
            "ROC-AUC": roc_auc_score(df_clean["label"], df_clean["prob_llm"]),
            "Accuracy": accuracy_score(
                df_clean["label"], df_clean["pred"]
            ),
            "F1-Score": f1_score(
                df_clean["label"], df_clean["pred"], zero_division=0
            ),
            "Precision": precision_score(
                df_clean["label"], df_clean["pred"], zero_division=0
            ),
            "Recall": recall_score(
                df_clean["label"], df_clean["pred"], zero_division=0
            ),
        })

    # -------------------------------------------------------------
    # 2. Human Baseline
    # -------------------------------------------------------------
    tnr_spec = accuracy_score(df_human["label"], df_human["pred"])
    fpr = (df_human["pred"] == 1).mean()
    results.append({
        "Category / Subset": "Human Text",
        "Group": "Human Baseline",
        "N": f"{len(df_human):,}",
        "Mean P(LLM)": df_human["prob_llm"].mean(),
        "ROC-AUC": None,
        "Accuracy": tnr_spec,  # TNR (Specificity)
        "F1-Score": None,
        "Precision": None,
        "Recall": fpr,  # FPR
    })

    # -------------------------------------------------------------
    # 3. Bucketed Substitution Ratios (25%, 50%, 75%, 100%)
    # -------------------------------------------------------------
    ordered_buckets = [
        "25% Substitution",
        "50% Substitution",
        "75% Substitution",
        "100% (Full Rewrite)",
    ]

    for bucket_name in ordered_buckets:
        df_sub = df_llm[df_llm["sub_bucket"] == bucket_name]
        if len(df_sub) == 0:
            continue

        # Combine bucketed LLM subset with Human baseline for classification evaluation
        df_eval = pd.concat([df_human, df_sub], ignore_index=True)

        results.append({
            "Category / Subset": bucket_name,
            "Group": "Bucketed Substitution Ratios",
            "N": f"{len(df_sub):,}",
            "Mean P(LLM)": df_sub["prob_llm"].mean(),
            "ROC-AUC": roc_auc_score(df_eval["label"], df_eval["prob_llm"]),
            "Accuracy": accuracy_score(df_eval["label"], df_eval["pred"]),
            "F1-Score": f1_score(
                df_eval["label"], df_eval["pred"], zero_division=0
            ),
            "Precision": precision_score(
                df_eval["label"], df_eval["pred"], zero_division=0
            ),
            "Recall": recall_score(
                df_eval["label"], df_eval["pred"], zero_division=0
            ),
        })

    # -------------------------------------------------------------
    # 4. Breakdown by Generator Model
    # -------------------------------------------------------------
    models = sorted(df_llm["model_name"].unique())
    for model in models:
        if model.lower() == "human":
            continue

        df_mod = df_llm[df_llm["model_name"] == model]
        if len(df_mod) == 0:
            continue

        # Pair model subset with Human baseline
        df_eval = pd.concat([df_human, df_mod], ignore_index=True)

        results.append({
            "Category / Subset": model,
            "Group": "Breakdown by Generator Model",
            "N": f"{len(df_mod):,}",
            "Mean P(LLM)": df_mod["prob_llm"].mean(),
            "ROC-AUC": roc_auc_score(df_eval["label"], df_eval["prob_llm"]),
            "Accuracy": accuracy_score(df_eval["label"], df_eval["pred"]),
            "F1-Score": f1_score(
                df_eval["label"], df_eval["pred"], zero_division=0
            ),
            "Precision": precision_score(
                df_eval["label"], df_eval["pred"], zero_division=0
            ),
            "Recall": recall_score(
                df_eval["label"], df_eval["pred"], zero_division=0
            ),
        })

    return pd.DataFrame(results)


def generate_latex_table(
    df_res: pd.DataFrame,
    model_name: str = "mDeBERTa-v3",
    highlight_best_generator: bool = True,
) -> str:
    """Formats the metrics dataframe into LaTeX table code using booktabs."""
    latex = []
    latex.append(r"\begin{table*}[htbp]")
    latex.append(r"\centering")
    latex.append(
        rf"\caption{{{model_name} detection performance for full abstracts. Baseline metrics compare human text against bucketed substitution ratios and individual LLM generator models.}}"
    )
    latex.append(r"\label{tab:mdeberta_abstract_performance}")
    latex.append(r"\small")
    latex.append(r"\begin{tabular}{l r r r r r r r}")
    latex.append(r"\toprule")
    latex.append(
        r"\textbf{Category / Subset} & \textbf{Samples ($N$)} & \textbf{Mean $P(\text{LLM})$} & \textbf{ROC-AUC} & \textbf{Accuracy} & \textbf{F1-Score} & \textbf{Precision} & \textbf{Recall} \\"
    )
    latex.append(r"\midrule")

    groups = df_res["Group"].unique()

    for grp in groups:
        latex.append(rf"\multicolumn{{8}}{{l}}{{\textbf{{{grp}}}}} \\")

        grp_df = df_res[df_res["Group"] == grp]

        # Best metrics for generator breakdown
        best_vals = {}
        if highlight_best_generator and grp == "Breakdown by Generator Model":
            for col in [
                "ROC-AUC",
                "Accuracy",
                "F1-Score",
                "Precision",
                "Recall",
            ]:
                valid_scores = grp_df[col].dropna()
                if len(valid_scores) > 0:
                    best_vals[col] = valid_scores.max()

        for _, row in grp_df.iterrows():
            subset_raw = str(row["Category / Subset"])
            # Escape special LaTeX characters
            subset_tex = subset_raw.replace("%", r"\%").replace("_", r"\_")
            subset_tex = rf"\quad {subset_tex}"

            n_str = str(row["N"])

            def format_metric(col_name):
                val = row[col_name]
                if val is None or pd.isna(val):
                    return "--"
                formatted = f"{val:.4f}"
                if col_name in best_vals and np.isclose(val, best_vals[col_name]):
                    formatted = rf"\textbf{{{formatted}}}"
                return formatted

            p_llm = format_metric("Mean P(LLM)")
            auc = format_metric("ROC-AUC")
            acc = format_metric("Accuracy")
            f1 = format_metric("F1-Score")
            prec = format_metric("Precision")
            rec = format_metric("Recall")

            if subset_raw == "Human Text":
                acc += r"$^{\dagger}$"
                rec += r"$^{\dagger}$"

            latex.append(
                f"{subset_tex} & {n_str} & {p_llm} & {auc} & {acc} & {f1} & {prec} & {rec} \\\\"
            )

        latex.append(r"\midrule")

    if latex[-1] == r"\midrule":
        latex.pop()

    latex.append(r"\bottomrule")
    latex.append(r"\multicolumn{8}{p{0.95\linewidth}}{\footnotesize ")
    latex.append(
        r"$^{\dagger}$ \textit{Note: For Human Text, Recall represents the False Positive Rate (FPR), and Accuracy represents Specificity (True Negative Rate, TNR). Evaluation metrics for substitution buckets and generator models pair the respective LLM subset with the Human baseline samples.}"
    )
    latex.append(r"} \\")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table*}")

    return "\n".join(latex)


# -------------------------------------------------------------
# Script Entry Point
# -------------------------------------------------------------
if __name__ == "__main__":
    csv_file_path = r"/home/gderijck/detection/outputs/deberta_no_tune/deberta_full/mdeberta_predictions_full.csv"

    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: File '{csv_file_path}' not found.")
        exit(1)

    # Calculate metrics dataframe
    df_metrics = evaluate_mdeberta_predictions(df)

    # Print terminal dataframe view
    print("\nEVALUATION METRICS TABLE:")
    print(df_metrics.to_string(index=False))

    # Generate LaTeX table code
    latex_output = generate_latex_table(
        df_metrics, model_name="mDeBERTa-v3", highlight_best_generator=True
    )

    print("\n" + "=" * 80)
    print("LATEX CODE:")
    print("=" * 80)
    print(latex_output)

    with open("mdeberta_abstract_table.tex", "w") as f:
        f.write(latex_output)
    print("\n Saved LaTeX table code to 'mdeberta_abstract_table.tex'")