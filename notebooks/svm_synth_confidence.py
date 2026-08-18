import pandas as pd

# File paths
PREPROCESSED_PATH = "/home/gderijck/detection/data_static/preprocessed/preprocessed_dataset.csv"
TEST_PRED_PATH = "/home/gderijck/detection/outputs/svm/full/test_predictions.csv"

# ---------------------------------------------------------
# 1. Load Train Set (Full Scope)
# ---------------------------------------------------------
print("Loading preprocessed train data...")
df_all = pd.read_csv(
    PREPROCESSED_PATH,
    usecols=["_id", "split", "scope", "label", "generation_type", "model_name"]
)

# Filter for train split and full scope
df_train = df_all[(df_all["split"] == "train") & (df_all["scope"] == "full")].copy()
df_train["_id"] = df_train["_id"].astype(str)

# Group by _id to capture all labels associated with each train ID
train_summary = (
    df_train.groupby("_id")
    .agg(
        train_labels=("label", lambda s: sorted(list(s.unique()))),
        train_gen_types=("generation_type", lambda s: list(s.unique()))
    )
    .reset_index()
)

# Convert labels list to a readable tag
train_summary["train_label_str"] = train_summary["train_labels"].apply(
    lambda lbls: "Human (0)" if lbls == [0] else ("Synthetic (1)" if lbls == [1] else "Both (0 & 1)")
)

print(f"Total Unique Train IDs (full scope): {len(train_summary):,}")


# ---------------------------------------------------------
# 2. Load Synthetic Test Predictions
# ---------------------------------------------------------
print("Loading test predictions...")
df_test = pd.read_csv(TEST_PRED_PATH)
df_test["_id"] = df_test["_id"].astype(str)

# Filter for synthetic test samples (label == 1 or model_name != 'human')
df_test_synth = df_test[df_test["label"] == 1].copy()
print(f"Total Synthetic Test Samples: {len(df_test_synth):,}")
print(f"Total Unique Synthetic Test IDs: {df_test_synth['_id'].nunique():,}")


# ---------------------------------------------------------
# 3. Check for Shared IDs (Data Leakage)
# ---------------------------------------------------------
train_id_set = set(train_summary["_id"])
test_synth_id_set = set(df_test_synth["_id"])
shared_ids = test_synth_id_set.intersection(train_id_set)

print("\n" + "=" * 65)
print(f"SHARED IDs DETECTED: {len(shared_ids):,} / {len(test_synth_id_set):,} test synthetic IDs")
print("=" * 65)

if not shared_ids:
    print("No leakage found! All synthetic test _ids are novel to the test set.")
else:
    # Merge test synthetic samples with their training counterpart info
    leaked_samples = df_test_synth[df_test_synth["_id"].isin(shared_ids)].merge(
        train_summary[["_id", "train_label_str", "train_labels"]],
        on="_id",
        how="left"
    )

    print(f"Total Affected Leaked Rows in Test: {len(leaked_samples):,}\n")

    # ---------------------------------------------------------
    # 4. Check Train Label vs. Test prob_llm
    # ---------------------------------------------------------
    print("=== SUMMARY: Train Label vs. Test Confidence (prob_llm) ===")
    summary_table = (
        leaked_samples.groupby(["train_label_str", "llm_ratio"])
        .agg(
            count=("_id", "count"),
            mean_prob_llm=("prob_llm", "mean"),
            median_prob_llm=("prob_llm", "median"),
            min_prob_llm=("prob_llm", "min"),
            max_prob_llm=("prob_llm", "max"),
            detection_rate_tpr=("pred_llm", "mean")
        )
        .reset_index()
    )

    summary_table["detection_rate_tpr"] = (summary_table["detection_rate_tpr"] * 100).round(2).astype(str) + "%"
    summary_table["mean_prob_llm"] = summary_table["mean_prob_llm"].round(4)
    summary_table["median_prob_llm"] = summary_table["median_prob_llm"].round(4)
    print(summary_table.to_string(index=False))

    # ---------------------------------------------------------
    # 5. Inspect Sample Rows (First 20 Leaked Instances)
    # ---------------------------------------------------------
    print("\n=== SAMPLE INSPECTION OF LEAKED ROWS ===")
    display_cols = [
        "_id",
        "train_label_str",
        "model_name",
        "generation_type",
        "llm_ratio",
        "prob_llm",
        "pred_llm",
        "error_type"
    ]
    print(leaked_samples[display_cols].head(20).to_string(index=False))

    # Save complete leaked audit to CSV for inspection
    output_path = "/home/gderijck/detection/outputs/svm/full/leaked_shared_ids_analysis.csv"
    leaked_samples[display_cols].to_csv(output_path, index=False)
    print(f"\nFull leaked samples saved to: {output_path}")