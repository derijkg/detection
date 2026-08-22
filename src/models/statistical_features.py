"""
src/models/statistical_features.py
Extracts thermodynamic and information-theoretic trajectory signatures:
- Critical Temperature Rank & Entropy Discrepancies
- Vectorized Gini Mass Coefficients
- Zipf Exponent & Zipf-Mandelbrot Beta Parameters
- Specific Heat Analogues ($dE/dT$) & Thermal Margin Elasticity.
"""

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
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
    """Computes exact Gini inequality coefficient over top-k token probability mass."""
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
    return float(gini[0]) if is_1d else gini


def compute_zipf_exponent(v_logits: np.ndarray, top_k: int = 20) -> Union[float, np.ndarray]:
    """Calculates empirical Zipf power-law scaling exponent alpha via OLS on log-ranks."""
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
    return float(zipf_alpha[0]) if is_1d else zipf_alpha


def compute_zipf_mandelbrot_params(v_logits: np.ndarray, top_k: int = 20) -> Tuple[Any, Any]:
    """Fits Zipf-Mandelbrot parameters (alpha, beta) over a dense grid search."""
    v_logits_arr = np.asarray(v_logits, dtype=np.float64)
    is_1d = v_logits_arr.ndim == 1
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
    var_x = np.mean(x_cent ** 2, axis=-1)

    mean_y = np.mean(sorted_topk, axis=-1, keepdims=True)
    y_cent = sorted_topk - mean_y
    var_y = np.mean(y_cent ** 2, axis=-1, keepdims=True)

    cov_xy = (y_cent @ x_cent.T) / actual_k
    alphas_grid = np.clip(-cov_xy / (var_x[None, :] + 1e-12), 1e-4, 20.0)
    mse_grid = var_y + 2.0 * alphas_grid * cov_xy + (alphas_grid ** 2) * var_x[None, :]

    best_beta_idx = np.argmin(mse_grid, axis=-1)
    best_betas = beta_grid[best_beta_idx]

    log_r_ref = np.log(ranks[None, :] + best_betas[:, None])
    x_ref_cent = log_r_ref - np.mean(log_r_ref, axis=-1, keepdims=True)
    var_x_ref = np.mean(x_ref_cent ** 2, axis=-1)
    cov_xy_ref = np.mean(y_cent * x_ref_cent, axis=-1)
    best_alphas = np.clip(-cov_xy_ref / (var_x_ref + 1e-12), 1e-4, 20.0)

    return (float(best_alphas[0]), float(best_betas[0])) if is_1d else (best_alphas, best_betas)


def extract_array_trajectory_features(
    norm_pos: np.ndarray,
    array_vals: np.ndarray,
    feature_prefix: str,
    num_bins: int = 10
) -> Dict[str, float]:
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


def extract_ranks_and_entropies_fast(
    v_logits: np.ndarray,
    v_labels: np.ndarray,
    temp: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scaled_logits = v_logits / max(float(temp), 1e-4)
    T, V = scaled_logits.shape
    lse = scipy.special.logsumexp(scaled_logits, axis=-1, keepdims=True)
    log_probs = scaled_logits - lse
    probs = np.exp(log_probs)

    raw_log_probs = log_probs[np.arange(T), v_labels]
    surprisals = -raw_log_probs
    safe_entropy_terms = np.where(probs > 0.0, -probs * log_probs, 0.0)
    entropies = np.sum(safe_entropy_terms, axis=-1)

    target_logits = scaled_logits[np.arange(T), v_labels]
    ranks = np.empty(T, dtype=np.float64)
    chunk_size = 64
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        ranks[start:end] = np.sum(scaled_logits[start:end] > target_logits[start:end, None], axis=-1) + 1
    log_ranks = np.log(np.maximum(ranks, 1.0))

    return (raw_log_probs, surprisals, entropies, log_ranks, probs)


def get_default_feature_dict() -> Dict[str, float]:
    dummy_pos = np.linspace(0.1, 1.0, 10)
    dummy_vals = np.zeros(10)
    d = {
        'token_length': 0.0,
        'mean_log_prob_crit': 0.0, 'std_log_prob_crit': 0.0,
        'mean_surprisal_crit': 0.0, 'std_surprisal_crit': 0.0,
        'mean_entropy_crit': 0.0, 'std_entropy_crit': 0.0,
        'mean_log_rank_crit': 0.0, 'std_log_rank_crit': 0.0,
        'mean_gini_coef_crit': 0.0, 'std_gini_coef_crit': 0.0,
        'mean_zipf_alpha_crit': 0.0, 'std_zipf_alpha_crit': 0.0,
        'mean_mandelbrot_beta_crit': 0.0, 'mean_top1_top2_margin_crit': 0.0,
        'fano_factor_burstiness': 0.0,
        'mean_entropy_frozen': 0.0, 'mean_entropy_gas': 0.0,
        'mean_specific_heat_response': 0.0, 'std_specific_heat_response': 0.0,
        'mean_zipf_critical_residual': 0.0, 'max_zipf_critical_residual': 0.0,
        'mean_thermal_margin_elasticity': 0.0
    }
    for pfx in ['zipf_crit', 'gini_crit', 'ent_crit', 'lp_crit', 'spec_heat']:
        d.update(extract_array_trajectory_features(dummy_pos, dummy_vals, pfx))
    return d


def extract_text_statistics(
    text: str,
    llm: Any,
    max_tokens: int = 1024,
    t_critical: float = 0.88,
    t_frozen: float = 0.5,
    t_gas: float = 1.3
) -> Dict[str, float]:
    text_clean = str(text).strip()
    if not text_clean:
        return get_default_feature_dict()

    tokens = llm.tokenize(text_clean.encode('utf-8'))
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
    valid_mask = shift_labels < n_vocab
    valid_positions = np.where(valid_mask)[0]
    total_valid_tokens = len(valid_positions)

    if total_valid_tokens < 2:
        return get_default_feature_dict()

    v_logits = shift_logits[valid_positions]
    v_labels = shift_labels[valid_positions]

    lp_c, surp_c, ent_c, rank_c, probs_c = extract_ranks_and_entropies_fast(v_logits, v_labels, temp=t_critical)

    top_k_partition = min(2, n_vocab)
    top2_logits_c = np.partition(v_logits / t_critical, -top_k_partition, axis=-1)[:, -top_k_partition:]
    sorted_top2_c = np.sort(top2_logits_c, axis=-1)
    lse_c = scipy.special.logsumexp(v_logits / t_critical, axis=-1, keepdims=True)
    p_top1_c = np.exp(sorted_top2_c[:, -1] - lse_c.squeeze(-1))
    p_top2_c = np.exp(sorted_top2_c[:, -2] - lse_c.squeeze(-1)) if top_k_partition >= 2 else np.zeros_like(p_top1_c)
    margins_c = p_top1_c - p_top2_c

    gini_coefs_c = compute_vectorized_gini(probs_c)
    zipf_alphas_unscaled = compute_zipf_exponent(v_logits, top_k=20)
    zipf_alphas_c = zipf_alphas_unscaled / t_critical
    mb_alphas_c, mb_betas_c = compute_zipf_mandelbrot_params(v_logits / t_critical, top_k=20)

    _, _, ent_frozen, _, _ = extract_ranks_and_entropies_fast(v_logits, v_labels, temp=t_frozen)
    _, _, ent_gas, _, _ = extract_ranks_and_entropies_fast(v_logits, v_labels, temp=t_gas)

    top2_logits_f = np.partition(v_logits / t_frozen, -top_k_partition, axis=-1)[:, -top_k_partition:]
    sorted_top2_f = np.sort(top2_logits_f, axis=-1)
    lse_f = scipy.special.logsumexp(v_logits / t_frozen, axis=-1, keepdims=True)
    margins_frozen = np.exp(sorted_top2_f[:, -1] - lse_f.squeeze(-1)) - (
        np.exp(sorted_top2_f[:, -2] - lse_f.squeeze(-1)) if top_k_partition >= 2 else 0.0
    )

    specific_heat_proxy = (ent_gas - ent_frozen) / (t_gas - t_frozen)
    zipf_critical_residuals = np.abs(zipf_alphas_c - 1.0)
    margin_elasticity = margins_frozen - margins_c
    norm_pos = np.linspace(1.0 / total_valid_tokens, 1.0, total_valid_tokens)

    stats_dict = {
        'token_length': float(total_valid_tokens),
        'mean_log_prob_crit': float(np.mean(lp_c)),
        'std_log_prob_crit': float(np.std(lp_c, ddof=1)) if total_valid_tokens > 1 else 0.0,
        'mean_surprisal_crit': float(np.mean(surp_c)),
        'std_surprisal_crit': float(np.std(surp_c, ddof=1)) if total_valid_tokens > 1 else 0.0,
        'mean_entropy_crit': float(np.mean(ent_c)),
        'std_entropy_crit': float(np.std(ent_c, ddof=1)) if total_valid_tokens > 1 else 0.0,
        'mean_log_rank_crit': float(np.mean(rank_c)),
        'std_log_rank_crit': float(np.std(rank_c, ddof=1)) if total_valid_tokens > 1 else 0.0,
        'mean_gini_coef_crit': float(np.mean(gini_coefs_c)),
        'std_gini_coef_crit': float(np.std(gini_coefs_c, ddof=1)) if total_valid_tokens > 1 else 0.0,
        'mean_zipf_alpha_crit': float(np.mean(zipf_alphas_c)),
        'std_zipf_alpha_crit': float(np.std(zipf_alphas_c, ddof=1)) if total_valid_tokens > 1 else 0.0,
        'mean_mandelbrot_beta_crit': float(np.mean(mb_betas_c)),
        'mean_top1_top2_margin_crit': float(np.mean(margins_c)),
        'fano_factor_burstiness': float(np.var(surp_c) / (np.mean(surp_c) + 1e-8)),
        'mean_entropy_frozen': float(np.mean(ent_frozen)),
        'mean_entropy_gas': float(np.mean(ent_gas)),
        'mean_specific_heat_response': float(np.mean(specific_heat_proxy)),
        'std_specific_heat_response': float(np.std(specific_heat_proxy, ddof=1)) if total_valid_tokens > 1 else 0.0,
        'mean_zipf_critical_residual': float(np.mean(zipf_critical_residuals)),
        'max_zipf_critical_residual': float(np.max(zipf_critical_residuals)),
        'mean_thermal_margin_elasticity': float(np.mean(margin_elasticity))
    }

    stats_dict.update(extract_array_trajectory_features(norm_pos, zipf_alphas_c, 'zipf_crit'))
    stats_dict.update(extract_array_trajectory_features(norm_pos, gini_coefs_c, 'gini_crit'))
    stats_dict.update(extract_array_trajectory_features(norm_pos, ent_c, 'ent_crit'))
    stats_dict.update(extract_array_trajectory_features(norm_pos, lp_c, 'lp_crit'))
    stats_dict.update(extract_array_trajectory_features(norm_pos, specific_heat_proxy, 'spec_heat'))

    return stats_dict


def extract_or_load_statistical_dataset(
    df: pd.DataFrame,
    scope: str,
    split_name: str,
    cache_dir: Path,
    repo_id: str = 'QuantFactory/Qwen2.5-3B-GGUF',
    filename: str = 'Qwen2.5-3B.Q8_0.gguf',
    t_critical: float = 0.88
) -> pd.DataFrame:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Collision-free fingerprint
    text_sample = ''.join(df['text'].dropna().astype(str).values[:50])
    content_hash = hashlib.md5(f"{len(df)}_{text_sample}".encode('utf-8')).hexdigest()[:8]
    cache_file = cache_dir / f"stat_features_{scope}_{split_name}_{len(df)}_{content_hash}_tc{int(t_critical*100)}.parquet"

    if cache_file.exists():
        return pd.read_parquet(cache_file)

    if Llama is None:
        raise ImportError("llama-cpp-python is required for Statistical Trajectory extraction.")

    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    llm = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=2048, n_batch=512, logits_all=True, verbose=False)

    records = []
    max_len = 128 if scope == 'sentence' else 512
    meta_cols = ['_id', 'text', 'label', 'llm_ratio', 'model_name', 'generation_type', 'source', 'year']

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting Stats [{scope.upper()}]"):
        feats = extract_text_statistics(row.get('text', ''), llm, max_tokens=max_len, t_critical=t_critical)
        for mc in meta_cols:
            if mc in row:
                feats[mc] = row[mc]
        feats['label'] = int(row.get('label', 0))
        records.append(feats)

    del llm
    feat_df = pd.DataFrame(records)
    feat_df.to_parquet(cache_file, index=False)
    return feat_df


FEATURE_CHANNELS = [
    'surprisal_crit', 'entropy_crit', 'log_rank_crit', 'gini_crit', 'zipf_alpha_crit',
    'top1_top2_margin_crit', 'entropy_frozen', 'entropy_gas', 'specific_heat_proxy', 'margin_elasticity'
]


def extract_text_2d_trajectory(
    text: str,
    llm: Any,
    max_tokens: int = 256,
    t_critical: float = 0.88,
    t_frozen: float = 0.5,
    t_gas: float = 1.3
) -> np.ndarray:
    text_clean = str(text).strip()
    num_channels = len(FEATURE_CHANNELS)
    if not text_clean:
        return np.zeros((1, num_channels), dtype=np.float32)

    tokens = llm.tokenize(text_clean.encode('utf-8'))
    bos_id = llm.token_bos()
    eos_id = llm.token_eos()
    start_id = bos_id if (bos_id is not None and bos_id != -1) else eos_id
    if start_id is not None and start_id != -1 and (len(tokens) == 0 or tokens[0] != start_id):
        tokens = [start_id] + tokens

    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    if len(tokens) < 3:
        return np.zeros((1, num_channels), dtype=np.float32)

    llm.reset()
    llm.eval(tokens)
    logits = np.array(llm.eval_logits, dtype=np.float32)

    shift_logits = logits[:-1, :]
    shift_labels = np.array(tokens[1:], dtype=np.int64)

    n_vocab = llm.n_vocab()
    valid_positions = np.where(shift_labels < n_vocab)[0]
    total_valid = len(valid_positions)
    if total_valid < 2:
        return np.zeros((1, num_channels), dtype=np.float32)

    v_logits = shift_logits[valid_positions]
    v_labels = shift_labels[valid_positions]

    lp_c, surp_c, ent_c, rank_c, probs_c = extract_ranks_and_entropies_fast(v_logits, v_labels, temp=t_critical)

    top_k_partition = min(2, n_vocab)
    top2_logits_c = np.partition(v_logits / t_critical, -top_k_partition, axis=-1)[:, -top_k_partition:]
    sorted_top2_c = np.sort(top2_logits_c, axis=-1)
    lse_c = scipy.special.logsumexp(v_logits / t_critical, axis=-1, keepdims=True)
    p_top1_c = np.exp(sorted_top2_c[:, -1] - lse_c.squeeze(-1))
    p_top2_c = np.exp(sorted_top2_c[:, -2] - lse_c.squeeze(-1)) if top_k_partition >= 2 else np.zeros_like(p_top1_c)
    margins_c = p_top1_c - p_top2_c

    gini_c = compute_vectorized_gini(probs_c)
    zipf_alphas_unscaled = compute_zipf_exponent(v_logits, top_k=20)
    zipf_c = zipf_alphas_unscaled / t_critical

    _, _, ent_f, _, _ = extract_ranks_and_entropies_fast(v_logits, v_labels, temp=t_frozen)
    _, _, ent_g, _, _ = extract_ranks_and_entropies_fast(v_logits, v_labels, temp=t_gas)

    top2_logits_f = np.partition(v_logits / t_frozen, -top_k_partition, axis=-1)[:, -top_k_partition:]
    sorted_top2_f = np.sort(top2_logits_f, axis=-1)
    lse_f = scipy.special.logsumexp(v_logits / t_frozen, axis=-1, keepdims=True)
    margins_f = np.exp(sorted_top2_f[:, -1] - lse_f.squeeze(-1)) - (
        np.exp(sorted_top2_f[:, -2] - lse_f.squeeze(-1)) if top_k_partition >= 2 else 0.0
    )

    cv_proxy = (ent_g - ent_f) / (t_gas - t_frozen)
    margin_elasticity = margins_f - margins_c

    trajectory_matrix = np.column_stack([
        surp_c, ent_c, rank_c, gini_c, zipf_c, margins_c, ent_f, ent_g, cv_proxy, margin_elasticity
    ]).astype(np.float32)

    return trajectory_matrix