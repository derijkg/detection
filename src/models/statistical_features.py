# src/models/statistical_features.py

import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import joblib
import nltk
import numpy as np
import pandas as pd
import scipy.special
import scipy.stats as stats
from huggingface_hub import hf_hub_download
from tqdm import tqdm

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

nltk.download('punkt', quiet=True)


def compute_vectorized_gini(probs: np.ndarray, top_k: int = 500) -> Union[float, np.ndarray]:
    probs_arr = np.asarray(probs, dtype=np.float64)
    is_1d = probs_arr.ndim == 1
    probs_2d = np.atleast_2d(probs_arr)
    M, V = probs_2d.shape
    actual_k = min(top_k, V)
    if actual_k < 2:
        return 0.0 if is_1d else np.zeros(M)

    topk_probs = np.partition(probs_2d, -actual_k, axis=-1)[:, -actual_k:]
    sorted_topk = np.sort(topk_probs, axis=-1)
    k_idx = np.arange(1, actual_k + 1, dtype=np.float64)
    weights = (actual_k - k_idx + 0.5) / actual_k
    total_mass = np.sum(sorted_topk, axis=-1, keepdims=True)
    lorenz_area = np.sum(sorted_topk * weights, axis=-1, keepdims=True) / (total_mass + 1e-12)
    gini = (1.0 - 2.0 * lorenz_area).squeeze(-1)
    return gini[0] if is_1d else gini


def compute_zipf_exponent(v_logits: np.ndarray, top_k: int = 20) -> Union[float, np.ndarray]:
    v_logits_arr = np.asarray(v_logits, dtype=np.float64)
    is_1d = v_logits_arr.ndim == 1
    v_logits_2d = np.atleast_2d(v_logits_arr)
    M, V = v_logits_2d.shape
    actual_k = min(top_k, V)
    if actual_k < 2:
        return 0.0 if is_1d else np.zeros(M)

    topk_logits = np.partition(v_logits_2d, -actual_k, axis=-1)[:, -actual_k:]
    sorted_topk = np.sort(topk_logits, axis=-1)[:, ::-1]
    log_ranks = np.log(np.arange(1, actual_k + 1, dtype=np.float64))

    mean_x = np.mean(log_ranks)
    var_x = np.var(log_ranks)
    mean_y = np.mean(sorted_topk, axis=-1, keepdims=True)
    cov_xy = np.mean((log_ranks - mean_x) * (sorted_topk - mean_y), axis=-1)
    zipf_alpha = -cov_xy / (var_x + 1e-12)
    return zipf_alpha[0] if is_1d else zipf_alpha


def compute_zipf_mandelbrot_params(v_logits: np.ndarray, top_k: int = 20) -> Tuple[Any, Any]:
    v_logits_arr = np.asarray(v_logits, dtype=np.float64)
    is_1d = (v_logits_arr.ndim == 1)
    v_logits_2d = np.atleast_2d(v_logits_arr)
    M, V = v_logits_2d.shape
    actual_k = min(top_k, V)
    if actual_k < 2:
        return (0.0, 0.0) if is_1d else (np.zeros(M), np.zeros(M))

    topk_logits = np.partition(v_logits_2d, -actual_k, axis=-1)[:, -actual_k:]
    sorted_topk = np.sort(topk_logits, axis=-1)[:, ::-1]
    ranks = np.arange(1, actual_k + 1, dtype=np.float64)
    beta_grid = np.linspace(0.0, 10.0, 101, dtype=np.float64)

    log_r_beta = np.log(ranks[None, :] + beta_grid[:, None])
    mean_x = np.mean(log_r_beta, axis=-1, keepdims=True)
    x_cent = log_r_beta - mean_x
    var_x = np.mean(x_cent**2, axis=-1)

    mean_y = np.mean(sorted_topk, axis=-1, keepdims=True)
    y_cent = sorted_topk - mean_y
    var_y = np.mean(y_cent**2, axis=-1, keepdims=True)
    cov_xy = (y_cent @ x_cent.T) / actual_k

    alphas_grid = np.clip(-cov_xy / (var_x[None, :] + 1e-12), 1e-4, 20.0)
    mse_grid = var_y + 2.0 * alphas_grid * cov_xy + (alphas_grid**2) * var_x[None, :]
    best_beta_idx = np.argmin(mse_grid, axis=-1)
    best_betas = beta_grid[best_beta_idx]

    log_r_ref = np.log(ranks[None, :] + best_betas[:, None])
    x_ref_cent = log_r_ref - np.mean(log_r_ref, axis=-1, keepdims=True)
    var_x_ref = np.mean(x_ref_cent**2, axis=-1)
    cov_xy_ref = np.mean(y_cent * x_ref_cent, axis=-1)
    best_alphas = np.clip(-cov_xy_ref / (var_x_ref + 1e-12), 1e-4, 20.0)

    return (best_alphas[0], best_betas[0]) if is_1d else (best_alphas, best_betas)


def extract_array_trajectory_features(norm_pos: np.ndarray, array_vals: np.ndarray, feature_prefix: str, num_bins: int = 10) -> Dict[str, float]:
    features = {}
    if len(array_vals) == 0:
        return features

    target_bins = np.linspace(0.1, 1.0, num_bins)
    interpolated = np.interp(target_bins, norm_pos, array_vals)

    for i in range(num_bins):
        features[f"{feature_prefix}_step_{i+1:02d}"] = float(interpolated[i])

    adj_diffs = np.abs(np.diff(interpolated))
    for i in range(len(adj_diffs)):
        features[f"{feature_prefix}_diff_step_{i+1:02d}_{i+2:02d}"] = float(adj_diffs[i])

    features[f"{feature_prefix}_total_variation"] = float(np.sum(adj_diffs))
    features[f"{feature_prefix}_max_local_jump"] = float(np.max(adj_diffs)) if len(adj_diffs) > 0 else 0.0
    features[f"{feature_prefix}_mean_local_jump"] = float(np.mean(adj_diffs)) if len(adj_diffs) > 0 else 0.0
    features[f"{feature_prefix}_span_start_to_end"] = float(interpolated[-1] - interpolated[0])

    centered_interp = interpolated - np.mean(interpolated)
    fft_raw = np.fft.rfft(centered_interp)[1:]
    power_spectrum = np.abs(fft_raw) ** 2

    if len(power_spectrum) > 0:
        mid = max(1, len(power_spectrum) // 2)
        low_energy = float(np.sum(power_spectrum[:mid]))
        high_energy = float(np.sum(power_spectrum[mid:]))
        features[f"{feature_prefix}_fft_spectral_ratio"] = float(high_energy / (low_energy + 1e-8))
    else:
        features[f"{feature_prefix}_fft_spectral_ratio"] = 0.0

    return features


def extract_ranks_and_entropies_fast(v_logits: np.ndarray, v_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    T, V = v_logits.shape
    lse = scipy.special.logsumexp(v_logits, axis=-1, keepdims=True)
    log_probs = v_logits - lse
    probs = np.exp(log_probs)

    raw_log_probs = log_probs[np.arange(T), v_labels]
    surprisals = -raw_log_probs
    entropies = -np.sum(probs * log_probs, axis=-1)

    target_logits = v_logits[np.arange(T), v_labels]
    ranks = np.empty(T, dtype=np.float64)
    
    chunk_size = 64
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        ranks[start:end] = np.sum(v_logits[start:end] > target_logits[start:end, None], axis=-1) + 1

    log_ranks = np.log(ranks)
    return raw_log_probs, surprisals, entropies, log_ranks, probs


def get_default_feature_dict() -> Dict[str, float]:
    """Provides a complete schema with 0.0 values when text is too short or unparseable."""
    dummy_pos = np.linspace(0.1, 1.0, 10)
    dummy_vals = np.zeros(10)
    d = {
        "token_length": 0.0,
        "mean_log_prob": 0.0, "std_log_prob": 0.0,
        "mean_surprisal": 0.0, "std_surprisal": 0.0,
        "mean_entropy": 0.0, "std_entropy": 0.0,
        "mean_log_rank": 0.0, "std_log_rank": 0.0,
        "mean_gini_coef": 0.0, "std_gini_coef": 0.0,
        "mean_zipf_alpha": 0.0, "std_zipf_alpha": 0.0,
        "mean_mandelbrot_beta": 0.0,
        "mean_top1_top2_margin": 0.0,
        "fano_factor_burstiness": 0.0,
    }
    for pfx in ["zipf", "gini", "ent", "lp"]:
        d.update(extract_array_trajectory_features(dummy_pos, dummy_vals, pfx))
    return d


def extract_text_statistics(text: str, llm: Any, max_tokens: int = 1024) -> Dict[str, float]:
    text_clean = str(text).strip()
    if not text_clean:
        return get_default_feature_dict()

    tokens = llm.tokenize(text_clean.encode("utf-8"))
    bos_id = llm.token_bos()
    eos_id = llm.token_eos()
    start_id = bos_id if (bos_id is not None and bos_id != -1) else eos_id

    if start_id is not None and start_id != -1 and (len(tokens) == 0 or tokens[0] != start_id):
        tokens = [start_id] + tokens

    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    if len(tokens) < 3:
        return get_default_feature_dict()

    llm.reset()
    llm.eval(tokens)

    logits = np.array(llm.eval_logits, dtype=np.float32)
    shift_logits = logits[:-1, :]
    shift_labels = np.array(tokens[1:], dtype=np.int64)

    n_vocab = llm.n_vocab()
    valid_mask = (shift_labels < n_vocab)
    valid_positions = np.where(valid_mask)[0]
    total_valid_tokens = len(valid_positions)

    if total_valid_tokens < 2:
        return get_default_feature_dict()

    v_logits = shift_logits[valid_positions]
    v_labels = shift_labels[valid_positions]

    raw_log_probs, surprisal, entropies, log_rank, probs = extract_ranks_and_entropies_fast(v_logits, v_labels)

    top_k_partition = min(2, n_vocab)
    top2_logits = np.partition(v_logits, -top_k_partition, axis=-1)[:, -top_k_partition:]
    sorted_top2 = np.sort(top2_logits, axis=-1)
    lse = scipy.special.logsumexp(v_logits, axis=-1, keepdims=True)
    p_top1 = np.exp(sorted_top2[:, -1] - lse.squeeze(-1))
    p_top2 = np.exp(sorted_top2[:, -2] - lse.squeeze(-1)) if top_k_partition >= 2 else np.zeros_like(p_top1)
    margins = p_top1 - p_top2

    gini_coefs = compute_vectorized_gini(probs)
    zipf_alphas = compute_zipf_exponent(v_logits, top_k=20)
    mb_alphas, mb_betas = compute_zipf_mandelbrot_params(v_logits, top_k=20)

    norm_pos = np.linspace(1.0 / total_valid_tokens, 1.0, total_valid_tokens)

    stats_dict = {
        "token_length": float(total_valid_tokens),
        "mean_log_prob": float(np.mean(raw_log_probs)),
        "std_log_prob": float(np.std(raw_log_probs, ddof=1)) if total_valid_tokens > 1 else 0.0,
        "mean_surprisal": float(np.mean(surprisal)),
        "std_surprisal": float(np.std(surprisal, ddof=1)) if total_valid_tokens > 1 else 0.0,
        "mean_entropy": float(np.mean(entropies)),
        "std_entropy": float(np.std(entropies, ddof=1)) if total_valid_tokens > 1 else 0.0,
        "mean_log_rank": float(np.mean(log_rank)),
        "std_log_rank": float(np.std(log_rank, ddof=1)) if total_valid_tokens > 1 else 0.0,
        "mean_gini_coef": float(np.mean(gini_coefs)),
        "std_gini_coef": float(np.std(gini_coefs, ddof=1)) if total_valid_tokens > 1 else 0.0,
        "mean_zipf_alpha": float(np.mean(zipf_alphas)),
        "std_zipf_alpha": float(np.std(zipf_alphas, ddof=1)) if total_valid_tokens > 1 else 0.0,
        "mean_mandelbrot_beta": float(np.mean(mb_betas)),
        "mean_top1_top2_margin": float(np.mean(margins)),
        "fano_factor_burstiness": float(np.var(surprisal) / (np.mean(surprisal) + 1e-8)),
    }

    stats_dict.update(extract_array_trajectory_features(norm_pos, zipf_alphas, "zipf"))
    stats_dict.update(extract_array_trajectory_features(norm_pos, gini_coefs, "gini"))
    stats_dict.update(extract_array_trajectory_features(norm_pos, entropies, "ent"))
    stats_dict.update(extract_array_trajectory_features(norm_pos, raw_log_probs, "lp"))

    return stats_dict


def extract_or_load_statistical_dataset(
    df: pd.DataFrame, 
    scope: str, 
    split_name: str, 
    cache_dir: Path,
    repo_id: str = "QuantFactory/Qwen2.5-3B-GGUF",
    filename: str = "Qwen2.5-3B.Q8_0.gguf"
) -> pd.DataFrame:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"stat_features_{scope}_{split_name}_{len(df)}.parquet"

    if cache_file.exists():
        return pd.read_parquet(cache_file)

    if Llama is None:
        raise ImportError("llama-cpp-python is required for Statistical Trajectory extraction.")

    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=2048,
        n_batch=512,
        logits_all=True,
        verbose=False
    )

    records = []
    max_len = 128 if scope == "sentence" else 512
    meta_cols = ["_id", "text", "label", "llm_ratio", "model_name", "generation_type", "source", "year"]

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting Stats [{scope.upper()}]"):
        feats = extract_text_statistics(row.get("text", ""), llm, max_tokens=max_len)
        for mc in meta_cols:
            if mc in row:
                feats[mc] = row[mc]

        feats["label"] = int(row.get("label", 0))
        records.append(feats)

    del llm
    feat_df = pd.DataFrame(records)
    feat_df.to_parquet(cache_file, index=False)
    return feat_df