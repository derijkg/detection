# Copyright (c) Guangsheng Bao.
# Modified for custom model pair calibration, threshold tuning, and inference.

import argparse
import gc
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import norm
import sklearn.metrics
from transformers import AutoTokenizer, AutoModelForCausalLM
import tqdm
import sys
from pathlib import Path
# Add project root (DETECTION) to Python search path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Now import your data manager
from src.data.data_loader import DetectionDataManager, DataFilter

# =========================================================================
# Reimplemented Model Loading Helpers
# =========================================================================

def load_tokenizer(model_name, cache_dir=None):
    """Loads a HuggingFace AutoTokenizer with right-side padding and slow tokenization."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_name, device, cache_dir=None):
    """Loads model in FP16 to fit within 12GB TITAN X VRAM."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    if device != "cpu":
        model = model.to(device)
    model.eval()
    return model


# =========================================================================
# Reimplemented Evaluation Metric Helpers
# =========================================================================

def get_roc_metrics(real_preds, sample_preds):
    """Computes False Positive Rate, True Positive Rate curves and ROC AUC score."""
    if len(real_preds) == 0 or len(sample_preds) == 0:
        return [], [], 0.0
    labels = [0] * len(real_preds) + [1] * len(sample_preds)
    predictions = list(real_preds) + list(sample_preds)
    fpr, tpr, _ = sklearn.metrics.roc_curve(labels, predictions, pos_label=1)
    roc_auc = float(sklearn.metrics.auc(fpr, tpr))
    return fpr.tolist(), tpr.tolist(), roc_auc


def get_precision_recall_metrics(real_preds, sample_preds):
    """Computes Precision, Recall curves and Precision-Recall AUC score."""
    if len(real_preds) == 0 or len(sample_preds) == 0:
        return [], [], 0.0
    labels = [0] * len(real_preds) + [1] * len(sample_preds)
    predictions = list(real_preds) + list(sample_preds)
    precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, predictions, pos_label=1)
    pr_auc = float(sklearn.metrics.auc(recall, precision))
    return precision.tolist(), recall.tolist(), pr_auc


# =========================================================================
# Core Fast-DetectGPT Discrepancy Calculations
# =========================================================================

def get_samples(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    nsamples = 10000
    lprobs = torch.log_softmax(logits, dim=-1)
    distrib = torch.distributions.categorical.Categorical(logits=lprobs)
    samples = distrib.sample([nsamples]).permute([1, 2, 0])
    return samples


def get_likelihood(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
    lprobs = torch.log_softmax(logits, dim=-1)
    log_likelihood = lprobs.gather(dim=-1, index=labels)
    return log_likelihood.mean(dim=1)


def get_sampling_discrepancy(logits_ref, logits_score, labels):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    samples = get_samples(logits_ref, labels)
    log_likelihood_x = get_likelihood(logits_score, labels)
    log_likelihood_x_tilde = get_likelihood(logits_score, samples)
    miu_tilde = log_likelihood_x_tilde.mean(dim=-1)
    sigma_tilde = log_likelihood_x_tilde.std(dim=-1)
    discrepancy = (log_likelihood_x.squeeze(-1) - miu_tilde) / sigma_tilde
    return discrepancy.item()


def get_sampling_discrepancy_analytic(logits_ref, logits_score, labels):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
    discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()
    discrepancy = discrepancy.mean()
    return discrepancy.item()


def compute_score_for_sample(
    text,
    scoring_model,
    scoring_tokenizer,
    sampling_model,
    sampling_tokenizer,
    criterion_fn,
    device,
    is_same_model=False,
    max_length=None,
):
    """Tokenizes text and calculates Fast-DetectGPT raw discrepancy score strictly matching original behavior."""
    if not text or len(str(text).strip()) == 0:
        return None

    tok_kwargs = {"return_tensors": "pt", "padding": True, "return_token_type_ids": False}
    if max_length is not None:
        tok_kwargs.update({"truncation": True, "max_length": max_length})

    tokenized_score = scoring_tokenizer(text, **tok_kwargs).to(scoring_model.device)
    labels = tokenized_score.input_ids[:, 1:]
    if labels.shape[1] == 0:
        return None

    with torch.no_grad():
        logits_score = scoring_model(**tokenized_score).logits[:, :-1]

        if is_same_model:
            logits_ref = logits_score
        else:
            tokenized_ref = sampling_tokenizer(text, **tok_kwargs).to(sampling_model.device)

            # Move tensors to CPU for equality check
            assert torch.all(
                tokenized_ref.input_ids[:, 1:].cpu() == labels.cpu()
            ), "Tokenizer mismatch between scoring and sampling model. Token sequences must match exactly."

            logits_ref = sampling_model(**tokenized_ref).logits[:, :-1]
            logits_ref = logits_ref.to(logits_score.device)

        score = criterion_fn(logits_ref, logits_score, labels)
    return score


def compute_calibrated_prob(score, mu0, sigma0, mu1, sigma1):
    """Calculates posterior AI probability P(AI | score) using Gaussian calibration parameters."""
    sigma0 = max(sigma0, 1e-6)
    sigma1 = max(sigma1, 1e-6)
    pdf0 = norm.pdf(score, loc=mu0, scale=sigma0)
    pdf1 = norm.pdf(score, loc=mu1, scale=sigma1)
    if pdf0 + pdf1 == 0:
        return 0.5
    return float(pdf1 / (pdf0 + pdf1))


def sanitize_value(v):
    """Ensures value is JSON serializable."""
    if isinstance(v, (np.generic, np.ndarray)):
        return v.item() if np.isscalar(v) else v.tolist()
    if pd.isna(v):
        return None
    return v


# =========================================================================
# Pipeline Execution Function
# =========================================================================

def run_pipeline(args):
    # Set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Resolve Scope, Sample Sizes, and Output Directory
    if args.scope in ["sentence", "single"]:
        target_scopes = ["sentence", "single"]
        scope_folder_name = "sentence"
        calib_sample_size = args.calib_sample_size if args.calib_sample_size is not None else args.calib_sample_size_sentence
        dev_sample_size = args.dev_sample_size if args.dev_sample_size is not None else args.dev_sample_size_sentence
    elif args.scope in ["full", "abstract"]:
        target_scopes = ["full", "abstract"]
        scope_folder_name = "abstract"
        calib_sample_size = args.calib_sample_size if args.calib_sample_size is not None else args.calib_sample_size_abstract
        dev_sample_size = args.dev_sample_size if args.dev_sample_size is not None else args.dev_sample_size_abstract
    else:
        raise ValueError(f"Invalid scope '{args.scope}'. Choose from ['sentence', 'single', 'full', 'abstract'].")

    test_sample_size = args.test_sample_size

    output_dir = Path(args.output_dir) / scope_folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"--> Target Scope: {args.scope} (mapped to '{scope_folder_name}')")
    print(f"--> Configured Sample Sizes: Calibration={calib_sample_size}, Dev={dev_sample_size}, Test={test_sample_size}")
    print(f"--> Output Directory: {output_dir}")

    # 2. Load Models and Tokenizers
    print(f"\n[1/5] Loading Models...")
    print(f"  - Scoring Model:  {args.scoring_model_name}")
    print(f"  - Sampling Model: {args.sampling_model_name}")

    is_same_model = args.sampling_model_name == args.scoring_model_name

    if torch.cuda.is_available() and torch.cuda.device_count() >= 2 and not is_same_model:
        scoring_device = "cuda:0"
        sampling_device = "cuda:1"
        print("  ✓ Assigning Scoring Model to cuda:0 and Sampling Model to cuda:1")
    else:
        scoring_device = args.device
        sampling_device = args.device

    scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.cache_dir)
    scoring_model = load_model(args.scoring_model_name, scoring_device, args.cache_dir)

    if is_same_model:
        sampling_tokenizer = scoring_tokenizer
        sampling_model = scoring_model
    else:
        sampling_tokenizer = load_tokenizer(args.sampling_model_name, args.cache_dir)
        sampling_model = load_model(args.sampling_model_name, sampling_device, args.cache_dir)

    criterion_fn = get_sampling_discrepancy_analytic if args.discrepancy_analytic else get_sampling_discrepancy

    # 3. Load Data Splits
    print(f"\n[2/5] Loading Splits via DetectionDataManager...")
    data_manager = DetectionDataManager(data_path=args.data_path)

    # Test split (Always loaded)
    test_df = data_manager.filter_dataframe(
        splits=["test"], scopes=target_scopes, sample_size=test_sample_size, seed=args.seed
    )
    print(f"  - Test Split (Final Testing):         {len(test_df)} samples")

    # Check if calibration parameters are supplied via file
    calib_file_path = Path(args.calibration_file) if args.calibration_file else (output_dir / "calibration_params.json")

    if args.calibration_file or (args.skip_calibration and calib_file_path.exists()):
        print(f"\n[3/5 & 4/5] Loading Existing Calibration Metadata from: {calib_file_path}")
        with open(calib_file_path, "r") as f:
            calib_params = json.load(f)

        mu0 = float(calib_params["distribution_params"]["mu0"])
        sigma0 = float(calib_params["distribution_params"]["sigma0"])
        mu1 = float(calib_params["distribution_params"]["mu1"])
        sigma1 = float(calib_params["distribution_params"]["sigma1"])

        threshold_raw = float(calib_params["threshold_calibration"]["threshold_raw"])
        threshold_prob = float(calib_params["threshold_calibration"]["threshold_prob"])

        print(f"  ✓ Distribution Parameters Loaded:")
        print(f"    Human (Label 0): mu0 = {mu0:.4f}, sigma0 = {sigma0:.4f}")
        print(f"    AI    (Label 1): mu1 = {mu1:.4f}, sigma1 = {sigma1:.4f}")
        print(f"  ✓ Operational Threshold Loaded:")
        print(f"    Raw Threshold (T_raw):        {threshold_raw:.4f}")
        print(f"    Calibrated Prob Threshold:    {threshold_prob * 100:.2f}%")

    else:
        # Full Calibration Pipeline (Train & Dev Splits)
        half_calib = max(1, calib_sample_size // 2) if calib_sample_size > 0 else -1
        df_human_train = data_manager.filter_dataframe(
            splits=["train"], scopes=target_scopes, labels=[0], sample_size=half_calib, seed=args.seed
        )
        df_ai_train = data_manager.filter_dataframe(
            splits=["train"], scopes=target_scopes, labels=[1], sample_size=half_calib, seed=args.seed
        )
        train_df = pd.concat([df_human_train, df_ai_train]).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        print(f"  - Train Split (Calib Distribution): {len(train_df)} samples ({len(df_human_train)} Human, {len(df_ai_train)} AI)")

        dev_df = data_manager.filter_dataframe(
            splits=["dev", "val"], scopes=target_scopes, sample_size=dev_sample_size, seed=args.seed
        )
        print(f"  - Dev Split (Threshold Calibration): {len(dev_df)} samples")

        # Step 1: Probability Distribution Calibration (Train Split)
        print(f"\n[3/5] Calibrating Probability Distribution (Train Split)...")
        human_train_scores = []
        ai_train_scores = []

        for _, row in tqdm.tqdm(train_df.iterrows(), total=len(train_df), desc="Calibrating (Train)"):
            score = compute_score_for_sample(
                text=row["text"],
                scoring_model=scoring_model,
                scoring_tokenizer=scoring_tokenizer,
                sampling_model=sampling_model,
                sampling_tokenizer=sampling_tokenizer,
                criterion_fn=criterion_fn,
                device=args.device,
                is_same_model=is_same_model,
                max_length=args.max_length,
            )
            if score is not None:
                if int(row["label"]) == 0:
                    human_train_scores.append(score)
                else:
                    ai_train_scores.append(score)

        mu0 = float(np.mean(human_train_scores))
        sigma0 = float(np.std(human_train_scores))
        mu1 = float(np.mean(ai_train_scores))
        sigma1 = float(np.std(ai_train_scores))

        print(f"  ✓ Distribution Parameters:")
        print(f"    Human (Label 0): mu0 = {mu0:.4f}, sigma0 = {sigma0:.4f}")
        print(f"    AI    (Label 1): mu1 = {mu1:.4f}, sigma1 = {sigma1:.4f}")

        # Step 2: Threshold Calibration (Dev Split @ target FPR <= 0.01)
        print(f"\n[4/5] Calibrating Operational Threshold on Dev Split (Target FPR <= {args.target_fpr})...")
        human_dev_scores = []
        ai_dev_scores = []

        for _, row in tqdm.tqdm(dev_df.iterrows(), total=len(dev_df), desc="Threshold Tuning (Dev)"):
            score = compute_score_for_sample(
                text=row["text"],
                scoring_model=scoring_model,
                scoring_tokenizer=scoring_tokenizer,
                sampling_model=sampling_model,
                sampling_tokenizer=sampling_tokenizer,
                criterion_fn=criterion_fn,
                device=args.device,
                is_same_model=is_same_model,
                max_length=args.max_length,
            )
            if score is not None:
                if int(row["label"]) == 0:
                    human_dev_scores.append(score)
                else:
                    ai_dev_scores.append(score)

        threshold_raw = float(np.quantile(human_dev_scores, 1.0 - args.target_fpr))
        threshold_prob = compute_calibrated_prob(threshold_raw, mu0, sigma0, mu1, sigma1)

        dev_fpr = float(np.mean(np.array(human_dev_scores) >= threshold_raw))
        dev_tpr = float(np.mean(np.array(ai_dev_scores) >= threshold_raw))

        calib_params = {
            "scoring_model": args.scoring_model_name,
            "sampling_model": args.sampling_model_name,
            "scope": scope_folder_name,
            "distribution_params": {
                "mu0": mu0,
                "sigma0": sigma0,
                "mu1": mu1,
                "sigma1": sigma1,
            },
            "threshold_calibration": {
                "target_fpr": args.target_fpr,
                "threshold_raw": threshold_raw,
                "threshold_prob": threshold_prob,
                "dev_fpr_achieved": dev_fpr,
                "dev_tpr_achieved": dev_tpr,
                "dev_human_samples": len(human_dev_scores),
                "dev_ai_samples": len(ai_dev_scores),
            },
        }

        with open(calib_file_path, "w") as f:
            json.dump(calib_params, f, indent=4)
        print(f"    Saved calibration metadata to: {calib_file_path}")

    # 4. Step 3: Test Inference Phase
    print(f"\n[5/5] Running Inference on Test Split...")
    test_results = []
    real_scores = []
    sample_scores = []

    for idx, row in tqdm.tqdm(test_df.iterrows(), total=len(test_df), desc="Testing"):
        # Periodically clear CUDA cache every 50 iterations to prevent fragmentation
        if idx % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()

        text = str(row["text"])
        
        # OOM Safeguard per sample
        try:
            raw_score = compute_score_for_sample(
                text=text,
                scoring_model=scoring_model,
                scoring_tokenizer=scoring_tokenizer,
                sampling_model=sampling_model,
                sampling_tokenizer=sampling_tokenizer,
                criterion_fn=criterion_fn,
                device=args.device,
                is_same_model=is_same_model,
                max_length=args.max_length,
            )
        except torch.OutOfMemoryError:
            print(f"\n[Warning] CUDA OOM on sample idx {idx} (length={len(text)} chars). Skipping sample...")
            torch.cuda.empty_cache()
            gc.collect()
            raw_score = None

        if raw_score is not None:
            calibrated_prob = compute_calibrated_prob(raw_score, mu0, sigma0, mu1, sigma1)
            label = int(row["label"])

            pred_label = 1 if raw_score >= threshold_raw else 0
            is_correct = bool(pred_label == label)

            if label == 1 and pred_label == 1:
                prediction_type = "TP"
            elif label == 0 and pred_label == 0:
                prediction_type = "TN"
            elif label == 0 and pred_label == 1:
                prediction_type = "FP"
            else:
                prediction_type = "FN"

            if label == 0:
                real_scores.append(raw_score)
            else:
                sample_scores.append(raw_score)

            row_dict = row.to_dict()
            metadata = {
                k: sanitize_value(v)
                for k, v in row_dict.items()
                if k not in ["text", "label", "_id", "id"]
            }

            sample_id = str(row.get("_id", row.get("id", f"sample_{idx}")))

            test_results.append({
                "id": sample_id,
                "text": text,
                "label": label,
                "predicted_label": pred_label,
                "prediction_type": prediction_type,
                "is_correct": is_correct,
                "raw_discrepancy_score": raw_score,
                "calibrated_ai_probability": calibrated_prob,
                "scope": scope_folder_name,
                "metadata": metadata,
            })

    # Metrics computation
    fpr, tpr, roc_auc = get_roc_metrics(real_scores, sample_scores)
    p, r, pr_auc = get_precision_recall_metrics(real_scores, sample_scores)

    test_human_scores = np.array(real_scores)
    test_ai_scores = np.array(sample_scores)

    test_fpr = float(np.mean(test_human_scores >= threshold_raw)) if len(test_human_scores) > 0 else 0.0
    test_tpr = float(np.mean(test_ai_scores >= threshold_raw)) if len(test_ai_scores) > 0 else 0.0

    tp = sum(1 for item in test_results if item["prediction_type"] == "TP")
    fp = sum(1 for item in test_results if item["prediction_type"] == "FP")
    tn = sum(1 for item in test_results if item["prediction_type"] == "TN")
    fn = sum(1 for item in test_results if item["prediction_type"] == "FN")

    accuracy = (tp + tn) / len(test_results) if test_results else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\n" + "=" * 60)
    print(f" TEST EVALUATION SUMMARY [{scope_folder_name.upper()}]")
    print("=" * 60)
    print(f" Total Test Samples:   {len(test_results)}")
    print(f" ROC AUC:              {roc_auc:.4f}")
    print(f" PR AUC:               {pr_auc:.4f}")
    print(f" F1 Score:             {f1_score:.4f}")
    print(f" Accuracy (@ T_raw):   {accuracy * 100:.2f}%")
    print(f" Test FPR (Target <= {args.target_fpr * 100:.1f}%): {test_fpr * 100:.2f}%")
    print(f" Test TPR / Recall:    {test_tpr * 100:.2f}%")
    print(f" Confusion Matrix:     TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print("=" * 60)

    # Save complete test results
    test_file = output_dir / "test_results.json"
    output_data = {
        "run_metadata": {
            "scoring_model": args.scoring_model_name,
            "sampling_model": args.sampling_model_name,
            "scope": scope_folder_name,
            "target_scopes": target_scopes,
            "seed": args.seed,
            "discrepancy_analytic": args.discrepancy_analytic,
            "max_length": args.max_length,
            "calib_sample_size_used": calib_sample_size,
            "dev_sample_size_used": dev_sample_size,
            "test_sample_size_evaluated": len(test_results),
        },
        "metrics": {
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1_score),
            "test_fpr": float(test_fpr),
            "test_tpr": float(test_tpr),
            "confusion_matrix": {
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            },
            "fpr_curve": [float(x) for x in fpr],
            "tpr_curve": [float(x) for x in tpr],
        },
        "calibration_used": calib_params,
        "predictions": test_results,
    }

    with open(test_file, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"Saved test evaluation results to: {test_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast-DetectGPT Calibration, Threshold Tuning, and Evaluation")

    # Model Pair Defaults
    parser.add_argument("--scoring_model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--sampling_model_name", type=str, default="Qwen/Qwen2.5-3B")

    # Scope & Data Config
    parser.add_argument("--scope", type=str, default="sentence", choices=["sentence", "single", "full", "abstract"])
    parser.add_argument("--data_path", type=str, default=None, help="Path to parquet file")
    parser.add_argument("--output_dir", type=str, default="output/fdgpt")

    # Reuse Existing Calibration File
    parser.add_argument("--calibration_file", type=str, default=None, help="Path to calibration_params.json to skip Train/Dev calibration")
    parser.add_argument("--skip_calibration", action="store_true", default=False, help="Automatically use calibration_params.json if it exists in output_dir")

    # Scope-Specific Default Sample Sizes
    parser.add_argument("--calib_sample_size_sentence", type=int, default=1000, help="Train samples for sentence scope calibration")
    parser.add_argument("--calib_sample_size_abstract", type=int, default=300, help="Train samples for abstract scope calibration")
    parser.add_argument("--dev_sample_size_sentence", type=int, default=-1, help="Dev samples for sentence scope thresholding (-1 for all)")
    parser.add_argument("--dev_sample_size_abstract", type=int, default=-1, help="Dev samples for abstract scope thresholding (-1 for all)")

    # Direct Overrides
    parser.add_argument("--calib_sample_size", type=int, default=None, help="Direct override for calibration sample size")
    parser.add_argument("--dev_sample_size", type=int, default=None, help="Direct override for dev sample size")
    parser.add_argument("--test_sample_size", type=int, default=-1, help="Test split sample size (-1 for all)")

    parser.add_argument("--target_fpr", type=float, default=0.01, help="Target False Positive Rate on Dev split (default: 0.01)")

    # Fast-DetectGPT Engine Flags
    parser.add_argument("--discrepancy_analytic", action="store_true", default=True)
    parser.add_argument("--max_length", type=int, default=None, help="Max sequence length (None for un-truncated original behavior)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache_dir", type=str, default="../cache")

    args = parser.parse_args()
    run_pipeline(args)