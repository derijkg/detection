import os
import ast
import gc
import sys
import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import nltk
import optuna
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)

# Suppress Optuna's standard verbose logging per trial
optuna.logging.set_verbosity(optuna.logging.WARNING)
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    roc_curve, 
    auc, 
    precision_recall_curve, 
    average_precision_score, 
    confusion_matrix, 
    classification_report,
    accuracy_score, 
    precision_recall_fscore_support
)



def evaluate_paper_results(df_test, model, tokenizer, save_dir="./paper_results", device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Generates publication-ready metrics, LaTeX tables, and 300 DPI figures 
    for research paper inclusion.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.to(device)
    model.eval()

    print("\n" + "="*70)
    print(" GENERATING PUBLICATION-READY EVALUATION REPORT ")
    print("="*70)

    # 1. Run Inference on Test Set
    texts = df_test['text'].tolist()
    labels = df_test['is_llm'].values
    batch_size = 32
    
    all_logits = []
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    logits_arr = np.concatenate(all_logits, axis=0)
    probs_arr = np.concatenate(all_probs, axis=0)
    probs_llm = probs_arr[:, 1]
    preds = np.argmax(logits_arr, axis=-1)

    # 2. Compute Overall Metrics
    fpr, tpr, _ = roc_curve(labels, probs_llm)
    roc_auc_val = auc(fpr, tpr)
    
    precision_curve, recall_curve, _ = precision_recall_curve(labels, probs_llm)
    pr_auc_val = average_precision_score(labels, probs_llm)

    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Calculate TPR at Low False Positive Rates (1% FPR and 5% FPR)
    tpr_at_1fpr = tpr[np.where(fpr <= 0.01)[0][-1]] if len(np.where(fpr <= 0.01)[0]) > 0 else 0.0
    tpr_at_5fpr = tpr[np.where(fpr <= 0.05)[0][-1]] if len(np.where(fpr <= 0.05)[0]) > 0 else 0.0

    overall_metrics = {
        "Total Test Samples": len(labels),
        "ROC-AUC": round(roc_auc_val, 4),
        "PR-AUC (AP)": round(pr_auc_val, 4),
        "Accuracy": round(acc, 4),
        "F1-Score": round(f1, 4),
        "Precision": round(prec, 4),
        "Recall (Sensitivity)": round(rec, 4),
        "Specificity": round(specificity, 4),
        "TPR @ 1% FPR": round(tpr_at_1fpr, 4),
        "TPR @ 5% FPR": round(tpr_at_5fpr, 4)
    }

    # Print Summary to Console
    print("\n--- OVERALL TEST SET PERFORMANCE ---")
    for k, v in overall_metrics.items():
        print(f"  {k:<22}: {v}")

    # 3. Compute Per-LLM Subgroup Performance
    df_test_eval = df_test.copy()
    df_test_eval['prob_llm'] = probs_llm
    df_test_eval['pred'] = preds

    per_model_results = []
    human_df = df_test_eval[df_test_eval['is_llm'] == 0]

    for model_name in df_test_eval['generator_model'].unique():
        if model_name == "Human":
            continue
        
        llm_sub_df = df_test_eval[df_test_eval['generator_model'] == model_name]
        combined_sub = pd.concat([human_df, llm_sub_df])
        
        sub_labels = combined_sub['is_llm'].values
        sub_probs = combined_sub['prob_llm'].values
        sub_preds = combined_sub['pred'].values

        sub_auc = roc_auc_score(sub_labels, sub_probs)
        sub_acc = accuracy_score(sub_labels, sub_preds)
        sub_prec, sub_rec, sub_f1, _ = precision_recall_fscore_support(sub_labels, sub_preds, average='binary', zero_division=0)

        per_model_results.append({
            "Generator Model": model_name,
            "LLM Samples": len(llm_sub_df),
            "ROC-AUC": round(sub_auc, 4),
            "Accuracy": round(sub_acc, 4),
            "F1-Score": round(sub_f1, 4),
            "Precision": round(sub_prec, 4),
            "Recall": round(sub_rec, 4)
        })

    per_model_df = pd.DataFrame(per_model_results)
    print("\n--- PER-LLM GENERATOR BREAKDOWN ---")
    print(per_model_df.to_string(index=False))

    # 4. Generate Publication Plots (4-Panel Figure, 300 DPI)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    plt.subplots_adjust(wspace=0.25, hspace=0.3)

    # Panel A: ROC Curve
    axes[0, 0].plot(fpr, tpr, color='#2b5c8f', lw=2, label=f'mDeBERTa-v3 (AUC = {roc_auc_val:.4f})')
    axes[0, 0].plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5, label='Random Guess')
    axes[0, 0].set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    axes[0, 0].set_ylabel('True Positive Rate (Sensitivity)', fontsize=11)
    axes[0, 0].set_title('(A) Receiver Operating Characteristic (ROC)', fontsize=12, fontweight='bold')
    axes[0, 0].legend(loc='lower right', fontsize=10)
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)

    # Panel B: Precision-Recall Curve
    axes[0, 1].plot(recall_curve, precision_curve, color='#d95f02', lw=2, label=f'mDeBERTa-v3 (AP = {pr_auc_val:.4f})')
    axes[0, 1].set_xlabel('Recall', fontsize=11)
    axes[0, 1].set_ylabel('Precision', fontsize=11)
    axes[0, 1].set_title('(B) Precision-Recall Curve', fontsize=12, fontweight='bold')
    axes[0, 1].legend(loc='lower left', fontsize=10)
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)

    # Panel C: Normalized Confusion Matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    im = axes[1, 0].imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[1, 0].set_title('(C) Normalized Confusion Matrix', fontsize=12, fontweight='bold')
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    classes = ['Human', 'LLM']
    tick_marks = np.arange(len(classes))
    axes[1, 0].set_xticks(tick_marks)
    axes[1, 0].set_xticklabels(classes, fontsize=10)
    axes[1, 0].set_yticks(tick_marks)
    axes[1, 0].set_yticklabels(classes, fontsize=10)
    axes[1, 0].set_ylabel('True Label', fontsize=11)
    axes[1, 0].set_xlabel('Predicted Label', fontsize=11)

    # Annotate Confusion Matrix Counts & Percentages
    for i in range(2):
        for j in range(2):
            axes[1, 0].text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)",
                            ha="center", va="center",
                            color="white" if cm_norm[i, j] > 0.5 else "black",
                            fontsize=11, fontweight='bold')

    # Panel D: Probability Density Distribution
    human_probs = probs_llm[labels == 0]
    llm_probs = probs_llm[labels == 1]
    
    axes[1, 1].hist(human_probs, bins=25, alpha=0.6, color='#1b9e77', label='Human Text', density=True)
    axes[1, 1].hist(llm_probs, bins=25, alpha=0.6, color='#7570b3', label='LLM Text', density=True)
    axes[1, 1].set_xlabel('Predicted Probability $P(\\text{LLM})$', fontsize=11)
    axes[1, 1].set_ylabel('Density', fontsize=11)
    axes[1, 1].set_title('(D) Output Probability Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].legend(loc='upper center', fontsize=10)
    axes[1, 1].grid(True, linestyle='--', alpha=0.5)

    plot_path = os.path.join(save_dir, "paper_evaluation_plots.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"\n[FIGURE SAVED] Publication plot figure saved to: '{plot_path}'")
    plt.show()

    # 5. Export LaTeX Table for Direct Paper Inclusion
    latex_table_path = os.path.join(save_dir, "paper_metrics_table.tex")
    with open(latex_table_path, "w") as f:
        f.write("% Auto-generated LaTeX table for paper inclusion\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Performance Evaluation of mDeBERTa-v3 AI Text Detector on Held-out Test Set.}\n")
        f.write("\\label{tab:mdeberta_results}\n")
        f.write("\\begin{tabular}{lccccc}\n")
        f.write("\\hline\n")
        f.write("\\textbf{Evaluation Group} & \\textbf{ROC-AUC} & \\textbf{PR-AUC} & \\textbf{Accuracy} & \\textbf{F1-Score} & \\textbf{TPR @ 1\\% FPR} \\\\\n")
        f.write("\\hline\n")
        f.write(f"Overall Test Set & {roc_auc_val:.4f} & {pr_auc_val:.4f} & {acc:.4f} & {f1:.4f} & {tpr_at_1fpr:.4f} \\\\\n")
        f.write("\\hline\n")
        for row in per_model_results:
            f.write(f"vs. {row['Generator Model']} & {row['ROC-AUC']:.4f} & -- & {row['Accuracy']:.4f} & {row['F1-Score']:.4f} & -- \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"[LATEX SAVED] Copy-pasteable LaTeX table saved to: '{latex_table_path}'")

    # 6. Save Structured JSON Summary
    summary_path = os.path.join(save_dir, "paper_evaluation_summary.json")
    json_data = {
        "overall_metrics": overall_metrics,
        "per_model_metrics": per_model_results
    }
    with open(summary_path, "w") as f:
        json.dump(json_data, f, indent=4)
    print(f"[JSON SAVED] Detailed evaluation summary saved to: '{summary_path}'\n")

    return overall_metrics, per_model_df

# ---------------------------------------------------------
# 1. DIVERSE DOCUMENT (_id) SAMPLING ALGORITHM
# ---------------------------------------------------------
def select_diverse_sentences(candidates, target_count, seed=42):
    """
    Selects `target_count` sentences from candidates while MAXIMIZING `doc_id` (_id) diversity.
    Iterates round-robin over distinct doc_ids, taking 1 sentence per doc_id per round.
    """
    if not candidates or target_count <= 0 or len(candidates) <= target_count:
        return candidates

    # Group candidates by doc_id
    doc_groups = {}
    for item in candidates:
        d_id = item['doc_id']
        if d_id not in doc_groups:
            doc_groups[d_id] = []
        doc_groups[d_id].append(item)

    rng = np.random.default_rng(seed)
    unique_doc_ids = list(doc_groups.keys())
    rng.shuffle(unique_doc_ids)

    # Shuffle sentences within each doc_id group
    for d_id in unique_doc_ids:
        rng.shuffle(doc_groups[d_id])

    selected = []
    round_num = 0
    while len(selected) < target_count:
        added_in_this_round = 0
        for d_id in unique_doc_ids:
            if round_num < len(doc_groups[d_id]):
                selected.append(doc_groups[d_id][round_num])
                added_in_this_round += 1
                if len(selected) == target_count:
                    break
        if added_in_this_round == 0:
            break  # Candidates exhausted
        round_num += 1

    return selected

# ---------------------------------------------------------
# 2. PARQUET COLUMN RESOLVER & VALIDATOR
# ---------------------------------------------------------
def resolve_llm_columns(parquet_path, requested_cols):
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Could not find Parquet file at: {parquet_path}")

    df_schema = pd.read_parquet(parquet_path)
    all_cols = list(df_schema.columns)
    single_full_cols = [c for c in all_cols if '_single' in c or '_full' in c]

    resolved_cols = []
    unmatched = []

    if isinstance(requested_cols, str):
        requested_cols = [requested_cols]

    for req in requested_cols:
        req_clean = req.strip().lower()
        
        # Flexible substring match against _single / _full columns
        matches = [c for c in single_full_cols if req_clean in c.lower()]
        
        # Fallback substring match across all columns in dataset
        if not matches:
            matches = [c for c in all_cols if req_clean in c.lower()]

        if matches:
            resolved_cols.extend(matches)
        else:
            unmatched.append(req)

    resolved_cols = list(dict.fromkeys(resolved_cols))

    if unmatched or not resolved_cols:
        print("\n" + "!" * 70)
        print(f"ERROR: Could not resolve LLM column(s) or model name(s): {unmatched}")
        print("\nAvailable '_single' and '_full' columns in your Parquet file:")
        if single_full_cols:
            for col in sorted(single_full_cols):
                print(f"  - {col}")
        else:
            print("  (No columns containing '_single' or '_full' were found in dataset)")
        print("!" * 70 + "\n")
        sys.exit(1)

    print(f"Successfully resolved LLM column(s): {resolved_cols}")
    return resolved_cols

# ---------------------------------------------------------
# 3. DATA LOADERS WITH DIVERSITY SAMPLING
# ---------------------------------------------------------
def is_valid_sentence(s, min_words=10):
    if s is None or pd.isna(s):
        return False
    s_str = str(s).strip()
    invalid_flags = {'FAILED_GENERATION', 'FAILED_VALIDATION', 'NONE', 'NAN', 'NA', 'NULL', '<NA>', ''}
    if s_str.upper() in invalid_flags:
        return False
    if len(s_str.split()) < min_words:
        return False
    return True


def parse_sentence_list(val):
    if isinstance(val, (list, np.ndarray)):
        return list(val)
    if isinstance(val, str):
        val_str = val.strip()
        if val_str.startswith('[') and val_str.endswith(']'):
            try:
                parsed = ast.literal_eval(val_str)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
        return [val_str]
    return [val] if val is not None else []


def load_abstracts_dataset(
    parquet_path="/home/gderijck/internship/data/gold/llm_added.parquet", 
    llm_columns=None, 
    samples_per_class=2000, 
    min_words=10, 
    seed=42
):
    print(f"\nLoading Human and LLM sentences from Parquet: {parquet_path}")
    
    llm_columns = resolve_llm_columns(parquet_path, llm_columns)

    df_parquet = pd.read_parquet(parquet_path)
    human_candidates = []
    llm_candidates = []

    for idx, row in df_parquet.iterrows():
        doc_id = row['_id'] if '_id' in row else (row['id'] if 'id' in row else f'doc_{idx}')

        h_sents = parse_sentence_list(row.get("abstract_sentence"))
        valid_h = [str(s).strip() for s in h_sents if is_valid_sentence(s, min_words=min_words)]
        for s in valid_h:
            human_candidates.append({
                "text": s,
                "doc_id": doc_id,
                "is_llm": 0,
                "generator_model": "Human"
            })

        for col in llm_columns:
            if col not in row.index:
                continue

            val = row[col]
            if val is None or (np.isscalar(val) and pd.isna(val)):
                continue

            parsed_items = parse_sentence_list(val)

            if col.endswith("_single") or "_single" in col:
                llm_sents = parsed_items
            else:
                llm_sents = []
                for item in parsed_items:
                    if item is not None and not (np.isscalar(item) and pd.isna(item)):
                        llm_sents.extend(nltk.sent_tokenize(str(item)))

            valid_llm = [str(s).strip() for s in llm_sents if is_valid_sentence(s, min_words=min_words)]

            for s in valid_llm:
                llm_candidates.append({
                    "text": s,
                    "doc_id": doc_id,
                    "is_llm": 1,
                    "generator_model": col
                })

    # Apply Document Diversity Selection for Full Training Set
    target_count = samples_per_class if samples_per_class > 0 else max(len(human_candidates), len(llm_candidates))
    selected_human = select_diverse_sentences(human_candidates, target_count, seed=seed)
    selected_llm = select_diverse_sentences(llm_candidates, target_count, seed=seed)

    final_records = selected_human + selected_llm
    rng = np.random.default_rng(seed)
    rng.shuffle(final_records)

    df = pd.DataFrame(final_records)
    
    unique_human_docs = len(set(x['doc_id'] for x in selected_human))
    unique_llm_docs = len(set(x['doc_id'] for x in selected_llm))
    print(f"Extracted {len(df)} total sentences ({sum(df['is_llm']==0)} Human across {unique_human_docs} unique '_id's, {sum(df['is_llm']==1)} LLM across {unique_llm_docs} unique '_id's)")
    return df

# ---------------------------------------------------------
# 4. GROUP-STRATIFIED 3-WAY SPLITTING
# ---------------------------------------------------------
def prepare_train_val_test_splits(df, seed=42):
    sgkf_test = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    train_val_idx, test_idx = next(sgkf_test.split(df, y=df['is_llm'], groups=df['doc_id']))

    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    sgkf_val = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
    train_idx, val_idx = next(sgkf_val.split(train_val_df, y=train_val_df['is_llm'], groups=train_val_df['doc_id']))

    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    print("\nDataset Split Summary (Group-Stratified by '_id' / 'doc_id'):")
    print(f"  - Train: {len(train_df)} samples ({train_df['is_llm'].sum()} LLM, {len(train_df)-train_df['is_llm'].sum()} Human)")
    print(f"  - Val  : {len(val_df)} samples ({val_df['is_llm'].sum()} LLM, {len(val_df)-val_df['is_llm'].sum()} Human)")
    print(f"  - Test : {len(test_df)} samples ({test_df['is_llm'].sum()} LLM, {len(test_df)-test_df['is_llm'].sum()} Human)")

    return train_df, val_df, test_df


def prepare_and_cache_datasets(
    df, 
    tokenizer, 
    cache_dir="./cached_tokenized_dataset", 
    seed=42, 
    force_retokenize=False
):
    train_path = os.path.join(cache_dir, "train")
    val_path = os.path.join(cache_dir, "val")
    test_path = os.path.join(cache_dir, "test")

    if not force_retokenize and os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        print(f"\n[CACHE MATCH] Loading pre-tokenized Train/Val/Test datasets from '{cache_dir}'...")
        train_ds = Dataset.load_from_disk(train_path)
        val_ds = Dataset.load_from_disk(val_path)
        test_ds = Dataset.load_from_disk(test_path)
        
        train_df, val_df, test_df = prepare_train_val_test_splits(df, seed=seed)
        return train_ds, val_ds, test_ds, train_df, val_df, test_df

    print(f"\n[CACHE MISS] Tokenizing dataset and creating splits at '{cache_dir}'...")
    train_df, val_df, test_df = prepare_train_val_test_splits(df, seed=seed)
    
    train_ds = Dataset.from_pandas(train_df[['text', 'is_llm', 'doc_id']].rename(columns={'is_llm': 'label'}))
    val_ds = Dataset.from_pandas(val_df[['text', 'is_llm', 'doc_id']].rename(columns={'is_llm': 'label'}))
    test_ds = Dataset.from_pandas(test_df[['text', 'is_llm', 'doc_id']].rename(columns={'is_llm': 'label'}))

    def tokenize_fn(examples):
        return tokenizer(examples['text'], truncation=True, max_length=256)

    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)
    test_ds = test_ds.map(tokenize_fn, batched=True)

    os.makedirs(cache_dir, exist_ok=True)
    train_ds.save_to_disk(train_path)
    val_ds.save_to_disk(val_path)
    test_ds.save_to_disk(test_path)
    print(f"Tokenized datasets successfully stored at '{cache_dir}'.")

    return train_ds, val_ds, test_ds, train_df, val_df, test_df

# ---------------------------------------------------------
# 5. DIVERSE OPTUNA SUBSAMPLING
# ---------------------------------------------------------
def get_optuna_subsampled_datasets(train_df, val_df, train_ds, val_ds, sample_per_class, tokenizer, seed=42):
    if sample_per_class is None or sample_per_class <= 0:
        print("\n[OPTUNA] Using FULL training and validation sets for hyperparameter tuning.")
        return train_ds, val_ds

    total_train_human = (train_df['is_llm'] == 0).sum()
    total_val_human = (val_df['is_llm'] == 0).sum()
    available_human = total_train_human + total_val_human

    if sample_per_class >= available_human:
        print(f"\n[OPTUNA] Requested sample size ({sample_per_class}/class) >= total available ({available_human}/class). Using full dataset.")
        return train_ds, val_ds

    train_ratio = len(train_df) / (len(train_df) + len(val_df))
    target_train_per_class = int(sample_per_class * train_ratio)
    target_val_per_class = sample_per_class - target_train_per_class

    def sample_group_diverse(df, target_per_class):
        sampled_records = []
        for class_val in [0, 1]:
            sub_df = df[df['is_llm'] == class_val]
            candidates = sub_df.to_dict('records')
            selected = select_diverse_sentences(candidates, target_per_class, seed=seed)
            sampled_records.extend(selected)
        return pd.DataFrame(sampled_records)

    optuna_train_df = sample_group_diverse(train_df, target_train_per_class)
    optuna_val_df = sample_group_diverse(val_df, target_val_per_class)

    print(f"\n[OPTUNA DIVERSE SUBSAMPLING ACTIVE]")
    print(f"Target sample size: {sample_per_class} per class ({sample_per_class * 2} total)")
    print(f"  - Optuna Train Set: {len(optuna_train_df)} samples ({optuna_train_df['is_llm'].sum()} LLM, {len(optuna_train_df)-optuna_train_df['is_llm'].sum()} Human)")
    print(f"  - Optuna Val Set  : {len(optuna_val_df)} samples ({optuna_val_df['is_llm'].sum()} LLM, {len(optuna_val_df)-optuna_val_df['is_llm'].sum()} Human)")

    optuna_train_ds = Dataset.from_pandas(optuna_train_df[['text', 'is_llm', 'doc_id']].rename(columns={'is_llm': 'label'}))
    optuna_val_ds = Dataset.from_pandas(optuna_val_df[['text', 'is_llm', 'doc_id']].rename(columns={'is_llm': 'label'}))

    def tokenize_fn(examples):
        return tokenizer(examples['text'], truncation=True, max_length=256)

    optuna_train_ds = optuna_train_ds.map(tokenize_fn, batched=True)
    optuna_val_ds = optuna_val_ds.map(tokenize_fn, batched=True)

    return optuna_train_ds, optuna_val_ds

# ---------------------------------------------------------
# 6. METRICS EVALUATION
# ---------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.5

    return {
        'roc_auc': auc,
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# ---------------------------------------------------------
# 7. OPTUNA OBJECTIVE FUNCTION
# ---------------------------------------------------------
def optuna_objective(trial, optuna_train_ds, optuna_val_ds, tokenizer, model_name):
    print(f"\n--- Starting Trial #{trial.number} ---")

    # 1. Cap epochs to 2-4 (DeBERTa rarely needs >3 epochs)
    num_train_epochs = trial.suggest_int("num_train_epochs", 2, 4)

    # 2. Slower, more stable learning rate range
    learning_rate = trial.suggest_float("learning_rate", 8e-6, 3e-5, log=True)

    # 3. Prefer larger batch size (16 or 32) for smoother gradients
    per_device_train_batch_size = trial.suggest_categorical("per_device_train_batch_size", [8, 16])

    # 4. Enforce STRONGER Weight Decay (lower bound raised to 0.01)
    weight_decay = trial.suggest_float("weight_decay", 1e-2, 1e-1, log=True)

    # 5. Slightly higher label smoothing to prevent overconfident probabilities
    label_smoothing_factor = trial.suggest_float("label_smoothing_factor", 0.03, 0.15)

    warmup_ratio = trial.suggest_float("warmup_ratio", 0.05, 0.2)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2,
        id2label={0: "Human", 1: "LLM"},
        label2id={"Human": 0, "LLM": 1},
        use_safetensors=True
    )

    training_args = TrainingArguments(
        output_dir=f"./optuna_trials/trial_{trial.number}",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=16,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        label_smoothing_factor=label_smoothing_factor,
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none",
        disable_tqdm=True
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=optuna_train_ds,
        eval_dataset=optuna_val_ds,
        processing_class=tokenizer,  # Updated to avoid FutureWarning
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )

    trainer.train()

    eval_metrics = trainer.evaluate()
    val_auc = eval_metrics["eval_roc_auc"]

    print(f"--> [Trial #{trial.number} Finished] Validation ROC-AUC Score: {val_auc:.4f}")

    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return val_auc

# ---------------------------------------------------------
# 8. SAVE BEST HYPERPARAMETERS TO JSON
# ---------------------------------------------------------
def save_best_hyperparameters(best_trial, json_path="best_hyperparameters.json"):
    data = {
        "best_trial_number": best_trial.number,
        "best_val_roc_auc": best_trial.value,
        "best_hyperparameters": best_trial.params
    }
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"\n[PARAMS SAVED] Best hyperparameter configuration saved to '{json_path}'")

# ---------------------------------------------------------
# 9. LOCAL PLOTTING FUNCTIONS
# ---------------------------------------------------------
def plot_optuna_progress(study, save_path="optuna_study_progress.png"):
    trial_numbers = [t.number for t in study.trials if t.value is not None]
    trial_values = [t.value for t in study.trials if t.value is not None]
    
    best_so_far = []
    current_best = -1.0
    for val in trial_values:
        if val > current_best:
            current_best = val
        best_so_far.append(current_best)

    plt.figure(figsize=(10, 5))
    plt.plot(trial_numbers, trial_values, marker='o', linestyle='--', color='#2b5c8f', alpha=0.7, label='Trial ROC-AUC')
    plt.plot(trial_numbers, best_so_far, marker='s', linestyle='-', color='#d95f02', linewidth=2.5, label='Best ROC-AUC So Far')
    
    plt.xlabel('Optuna Trial Number', fontsize=12)
    plt.ylabel('Validation ROC-AUC Score', fontsize=12)
    plt.title('Optuna Hyperparameter Optimization Progress', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[PLOT SAVED] Optuna study progress saved locally to '{save_path}'")
    plt.show()


def plot_training_curves(log_history, save_path="final_training_curves.png"):
    train_epochs, train_losses = [], []
    eval_epochs, eval_losses, eval_aucs = [], [], []

    for log in log_history:
        if 'epoch' in log:
            ep = log['epoch']
            if 'loss' in log:
                train_epochs.append(ep)
                train_losses.append(log['loss'])
            if 'eval_loss' in log:
                eval_epochs.append(ep)
                eval_losses.append(log['eval_loss'])
            if 'eval_roc_auc' in log:
                eval_aucs.append(log['eval_roc_auc'])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if train_losses and train_epochs:
        axes[0].plot(train_epochs, train_losses, label='Train Loss', marker='o', color='#1b9e77')
    if eval_losses and eval_epochs:
        axes[0].plot(eval_epochs, eval_losses, label='Val Loss', marker='s', color='#d95f02')
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].set_title('Training & Validation Loss', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)

    if eval_aucs and eval_epochs:
        axes[1].plot(eval_epochs, eval_aucs, label='Val ROC-AUC', marker='^', color='#7570b3', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('ROC-AUC Score', fontsize=11)
    axes[1].set_title('Validation ROC-AUC Score Progression', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[PLOT SAVED] Final training curves saved locally to '{save_path}'")
    plt.show()

# ---------------------------------------------------------
# 10. TUNING RUNNER
# ---------------------------------------------------------
def run_hyperparameter_tuning(
    df, 
    model_name="microsoft/mdeberta-v3-base", 
    n_trials=10, 
    optuna_sample_per_class=500, 
    seed=42
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    train_ds, val_ds, test_ds, train_df, val_df, test_df = prepare_and_cache_datasets(
        df, 
        tokenizer, 
        cache_dir="./cached_tokenized_dataset", 
        seed=seed
    )

    optuna_train_ds, optuna_val_ds = get_optuna_subsampled_datasets(
        train_df, val_df, train_ds, val_ds, optuna_sample_per_class, tokenizer, seed=seed
    )

    study = optuna.create_study(direction="maximize", study_name="mdeberta_auc_optimization")
    
    print(f"\nStarting Optuna Hyperparameter Search ({n_trials} Trials, Target Metric: ROC-AUC)...")
    study.optimize(
        lambda trial: optuna_objective(trial, optuna_train_ds, optuna_val_ds, tokenizer, model_name), 
        n_trials=n_trials
    )

    best_trial = study.best_trial
    print("\n" + "="*60)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION COMPLETE")
    print(f"Best Trial Number          : #{best_trial.number}")
    print(f"Best Validation ROC-AUC    : {best_trial.value:.4f}")
    print("Best Hyperparameters Found :")
    for key, val in best_trial.params.items():
        if isinstance(val, float):
            print(f"  - {key}: {val:.6f}")
        else:
            print(f"  - {key}: {val}")
    print("="*60 + "\n")

    # SAVE BEST HYPERPARAMETERS TO JSON
    save_best_hyperparameters(best_trial, json_path="best_hyperparameters.json")

    plot_optuna_progress(study, save_path="optuna_study_progress.png")

    return best_trial.params, train_ds, val_ds, test_ds, tokenizer, test_df

# ---------------------------------------------------------
# 11. TRAIN FINAL MODEL & EVALUATE ON TEST SET
# ---------------------------------------------------------
def train_and_test_final_model(
    best_params, 
    train_ds, 
    val_ds, 
    test_ds, 
    tokenizer, 
    model_name="microsoft/mdeberta-v3-base", 
    output_dir="./best_mdeberta_detector"
):
    print(f"\nTraining Final Production Model on FULL Training Set with Best Hyperparameters...")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2,
        id2label={0: "Human", 1: "LLM"},
        label2id={"Human": 0, "LLM": 1},
        use_safetensors=True
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=best_params["learning_rate"],
        per_device_train_batch_size=best_params["per_device_train_batch_size"],
        per_device_eval_batch_size=16,
        num_train_epochs=best_params["num_train_epochs"],
        weight_decay=best_params["weight_decay"],
        warmup_ratio=best_params["warmup_ratio"],
        label_smoothing_factor=best_params["label_smoothing_factor"],
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n[MODEL SAVED] Final best model and tokenizer successfully saved to '{output_dir}'.")

    plot_training_curves(trainer.state.log_history, save_path="final_training_curves.png")

    print("\n" + "="*60)
    print("EVALUATING FINAL MODEL ON HELD-OUT TEST SET")
    print("="*60)
    
    test_metrics = trainer.evaluate(eval_dataset=test_ds)
    
    print("\nFINAL TEST SET PERFORMANCE METRICS:")
    print(f"  - Test ROC-AUC   : {test_metrics['eval_roc_auc']:.4f}")
    print(f"  - Test Accuracy  : {test_metrics['eval_accuracy']:.4f}")
    print(f"  - Test F1 Score  : {test_metrics['eval_f1']:.4f}")
    print(f"  - Test Precision : {test_metrics['eval_precision']:.4f}")
    print(f"  - Test Recall    : {test_metrics['eval_recall']:.4f}")
    print("="*60 + "\n")
    
    return model, tokenizer

# ---------------------------------------------------------
# 12. LOGIT EXTRACTION
# ---------------------------------------------------------
def extract_logits_for_analysis(df, model, tokenizer, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.to(device)
    model.eval()
    
    all_logits = []
    all_probs = []
    
    texts = df['text'].tolist()
    batch_size = 32

    print(f"Extracting logits for {len(texts)} samples...")
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            
            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    logits_arr = np.concatenate(all_logits, axis=0)
    probs_arr = np.concatenate(all_probs, axis=0)

    result_df = df.copy()
    result_df['logit_human'] = logits_arr[:, 0]
    result_df['logit_llm'] = logits_arr[:, 1]
    result_df['logit_diff'] = logits_arr[:, 1] - logits_arr[:, 0]
    result_df['prob_llm'] = probs_arr[:, 1]

    return result_df

# ---------------------------------------------------------
# CLI ARGUMENT PARSER
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune mDeBERTa-v3 for Human vs. LLM Text Detection")
    
    parser.add_argument("--parquet_path", type=str, default="/home/gderijck/internship/data/gold/llm_added.parquet", help="Path to parquet dataset")
    parser.add_argument("--llm_col", nargs="+", required=True, help="LLM model name(s) or column name(s) e.g. gpt4 llama3 or gpt4_single")
    parser.add_argument("--sample", type=int, default=500, help="Number of sentences PER CLASS for Optuna tuning. Default: 500 (1000 total). Set -1 to use full training data.")
    parser.add_argument("--full", type=int, default=2000, help="Number of sentences PER CLASS for full training & testing (e.g. 2000 -> 2000 Human + 2000 LLM = 4000 total). Set -1 for all available.")
    parser.add_argument("--n_trials", type=int, default=10, help="Number of Optuna tuning trials")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # 1. Load Parquet Data with Document Diversity
    df = load_abstracts_dataset(
        parquet_path=args.parquet_path,
        llm_columns=args.llm_col,
        samples_per_class=args.full,
        min_words=10,
        seed=args.seed
    )

    # 2. Run Optuna Hyperparameter Optimization on Document-Diverse Sample
    best_params, train_ds, val_ds, test_ds, tokenizer, test_df = run_hyperparameter_tuning(
        df,
        model_name="microsoft/mdeberta-v3-base",
        n_trials=args.n_trials,
        optuna_sample_per_class=args.sample,
        seed=args.seed
    )

    # 3. Train Final Model on FULL Dataset & Evaluate on Held-Out Test Set
    final_model, final_tokenizer = train_and_test_final_model(
        best_params, 
        train_ds, 
        val_ds, 
        test_ds,
        tokenizer,
        model_name="microsoft/mdeberta-v3-base",
        output_dir="./best_mdeberta_detector"
    )

    # 4. GENERATE FULL PAPER EVALUATION REPORT & LATEX TABLES
    overall_metrics, per_model_df = evaluate_paper_results(
        df_test=test_df,
        model=final_model,
        tokenizer=final_tokenizer,
        save_dir="./paper_results"
    )

    # 4. Extract Logits on Full Dataset for Downstream Analysis
    analysis_df = extract_logits_for_analysis(df, final_model, final_tokenizer)
    
    # 5. Save Output CSV
    analysis_df.to_csv("mdeberta_test_logits_analysis.csv", index=False)
    print("Logits saved to mdeberta_test_logits_analysis.csv!")