# detection/src/training/tuner.py

import os
import gc
import json
import argparse
import optuna
import torch
from typing import Dict, Any, List

from src.utils.config import Config
from src.utils.seed import set_seed
from src.utils.logger import setup_logger
from src.models.old.factory import ModelFactory
from src.data.data_loader import DetectionDataManager

optuna.logging.set_verbosity(optuna.logging.WARNING)


def suggest_float_log(trial: optuna.Trial, name: str, bounds: List[float], default_log: bool = True) -> float:
    low, high = float(bounds[0]), float(bounds[1])
    use_log = default_log and (low > 0)
    return trial.suggest_float(name, low, high, log=use_log)


def optuna_objective(trial: optuna.Trial, train_samples: Any, val_samples: Any, 
                     model_name: str, search_space: Dict[str, Any], cfg: Config, logger: Any) -> float:
    training_params = vars(cfg.training).copy()

    # 1. DEBERTA / TRANSFORMER SAMPLING
    if model_name.lower() in ["deberta", "mdeberta", "mdeberta-v3"]:
        if "num_train_epochs" in search_space:
            training_params["num_train_epochs"] = trial.suggest_int(
                "num_train_epochs", search_space["num_train_epochs"][0], search_space["num_train_epochs"][1]
            )
        
        if "per_device_train_batch_size" in search_space:
            training_params["per_device_train_batch_size"] = trial.suggest_categorical(
                "per_device_train_batch_size", search_space["per_device_train_batch_size"]
            )

        if "learning_rate" in search_space:
            training_params["learning_rate"] = suggest_float_log(
                trial, "learning_rate", search_space["learning_rate"], default_log=True
            )
        
        if "weight_decay" in search_space:
            training_params["weight_decay"] = suggest_float_log(
                trial, "weight_decay", search_space["weight_decay"], default_log=True
            )

        if "warmup_ratio" in search_space:
            training_params["warmup_ratio"] = suggest_float_log(
                trial, "warmup_ratio", search_space["warmup_ratio"], default_log=False
            )
        
        if "label_smoothing_factor" in search_space:
            training_params["label_smoothing_factor"] = suggest_float_log(
                trial, "label_smoothing_factor", search_space["label_smoothing_factor"], default_log=False
            )

        training_params["output_dir"] = f"outputs/checkpoints/trial_{trial.number}"

    # 2. SVM SAMPLING
    elif model_name.lower() in ["svm", "linear_svm"]:
        if "C" in search_space:
            training_params["C"] = suggest_float_log(
                trial, "C", search_space["C"], default_log=True
            )

        kernel = trial.suggest_categorical("kernel", search_space.get("kernel", ["linear"]))
        training_params["kernel"] = kernel

        if kernel in ["rbf", "sigmoid"] and "gamma" in search_space:
            training_params["gamma"] = suggest_float_log(
                trial, "gamma", search_space["gamma"], default_log=True
            )

        if "sty_weight" in search_space:
            training_params["sty_weight"] = suggest_float_log(
                trial, "sty_weight", search_space["sty_weight"], default_log=True
            )

        weight_mode = trial.suggest_categorical("weight_mode", search_space.get("weight_mode", ["balanced"]))
        if weight_mode == "custom" and "human_class_weight" in search_space:
            human_w = suggest_float_log(
                trial, "human_class_weight", search_space["human_class_weight"], default_log=True
            )
            training_params["class_weight"] = {0: human_w, 1: 1.0}
        else:
            training_params["class_weight"] = "balanced"

        training_params["use_stylometrics"] = True

    # 3. MODEL TRAINING & SCORE EXTRACTION
    detector = ModelFactory.create(model_name, granularity=getattr(cfg.model, "granularity", "full"))
    eval_metrics = detector.train(train_samples, val_samples, training_params)

    score_metric = getattr(cfg.optuna, "score_metric", "pauc").lower()

    if "pauc" in score_metric or "tpr" in score_metric:
        val_score = eval_metrics.get("TPR @ 1% FPR", eval_metrics.get("ROC-AUC", eval_metrics.get("eval_roc_auc", 0.5)))
    elif "f1" in score_metric:
        val_score = eval_metrics.get("F1-Score", eval_metrics.get("eval_f1", 0.5))
    else:
        val_score = eval_metrics.get("ROC-AUC", eval_metrics.get("eval_roc_auc", 0.5))

    logger.info(f"Trial #{trial.number} Finished | {score_metric.upper()} Score: {val_score:.4f}")

    del detector
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return float(val_score)


def run_tuning(model_name: str, scope: str, seed: int = 42) -> str:
    """Runs Optuna hyperparameter search and saves best parameters to a scope-isolated JSON file."""
    config_path = Config.resolve_config_path(model_name, scope)
    cfg = Config.from_yaml(str(config_path))
    cfg.model.granularity = scope

    set_seed(seed)
    logger = setup_logger(name="tuner", log_file=f"tuner_{model_name}_{scope}.log")

    optuna_out_dir = f"outputs/metrics/{model_name}_{scope}_optuna"
    os.makedirs(optuna_out_dir, exist_ok=True)
    best_json_path = os.path.join(optuna_out_dir, "best_hyperparameters.json")

    tune_size = getattr(cfg.optuna, "tune_sample_size", 1000)
    logger.info(f"Starting Optuna search for '{model_name}' ({scope}) | Config: '{config_path}' | Tune Size: {tune_size}...")

    data_mgr = DetectionDataManager()

    if model_name.lower() in ["deberta", "mdeberta", "mdeberta-v3"]:
        train_df = data_mgr.filter_dataframe(splits=["train"], scopes=[scope], sample_size=tune_size, seed=seed)
        val_df = data_mgr.filter_dataframe(splits=["dev"], scopes=[scope], sample_size=max(200, int(tune_size*0.3)), seed=seed)
        
        from datasets import Dataset as HFDataset
        train_ds = HFDataset.from_pandas(train_df, preserve_index=False)
        val_ds = HFDataset.from_pandas(val_df, preserve_index=False)
    else:
        train_df = data_mgr.filter_dataframe(splits=["train"], scopes=[scope], sample_size=tune_size, seed=seed)
        val_df = data_mgr.filter_dataframe(splits=["dev"], scopes=[scope], sample_size=max(200, int(tune_size*0.3)), seed=seed)
        
        train_ds = (train_df['text'].tolist(), train_df['label'].values)
        val_ds = (val_df['text'].tolist(), val_df['label'].values)

    study = optuna.create_study(direction="maximize")

    # ENQUEUE INITIAL TRIAL
    if cfg.optuna.enqueue_params:
        logger.info(f"Enqueueing warmstart trial parameters: {cfg.optuna.enqueue_params}")
        study.enqueue_trial(cfg.optuna.enqueue_params)

    study.optimize(
        lambda trial: optuna_objective(trial, train_ds, val_ds, model_name, cfg.optuna.search_space, cfg, logger),
        n_trials=cfg.optuna.n_trials
    )

    best_data = {
        "model": model_name,
        "scope": scope,
        "best_trial_number": study.best_trial.number,
        "best_val_score": float(study.best_value),
        "score_metric": getattr(cfg.optuna, "score_metric", "pauc"),
        "max_fpr": getattr(cfg.optuna, "max_fpr", 0.01),
        "best_params": study.best_params
    }

    with open(best_json_path, "w", encoding="utf-8") as f:
        json.dump(best_data, f, indent=4)

    logger.info(f"Optuna Tuning Complete!")
    logger.info(f"Best Trial #{study.best_trial.number} | Best Score: {study.best_value:.4f}")
    logger.info(f"[SAVED] Best hyperparameters exported to: '{best_json_path}'")

    return best_json_path


def main():
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Search")
    parser.add_argument("--config", type=str, default=None, help="Optional path to config file")
    parser.add_argument("--model", type=str, default="deberta", help="Model name ('deberta', 'svm')")
    parser.add_argument("--scope", type=str, default="full", choices=["full", "sentence"], help="Scope ('full', 'sentence')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    model_name = args.model
    if args.config:
        cfg = Config.from_yaml(args.config)
        model_name = cfg.model.name

    run_tuning(model_name, args.scope, args.seed)


if __name__ == "__main__":
    main()