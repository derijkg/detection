# detection/src/training/trainer.py

import os
import argparse
import json
import shutil
from pathlib import Path

from src.utils.config import Config
from src.utils.seed import set_seed
from src.utils.logger import setup_logger
from src.models.old.factory import ModelFactory
from src.data.data_loader import DetectionDataManager


def main():
    parser = argparse.ArgumentParser(description="Train Detector Model")
    parser.add_argument("--config", type=str, default="configs/models/deberta.yaml", help="Path to YAML config")
    parser.add_argument("--scope", type=str, default="full", choices=["full", "single"], help="Scope ('full' or 'single')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)

    cfg = Config.from_yaml(args.config)
    model_name = cfg.model.name
    scope = args.scope

    logger = setup_logger(name="trainer", log_file=f"trainer_{model_name}_{scope}.log")

    # Scope-isolated paths
    checkpoint_dir = f"outputs/checkpoints/{model_name}_{scope}"
    tuned_params_file = f"outputs/metrics/{model_name}_{scope}_optuna/best_hyperparameters.json"

    logger.info(f"Initializing training for Model='{model_name}' | Scope='{scope}'")

    training_dict = vars(cfg.training).copy()
    training_dict["output_dir"] = checkpoint_dir

    # Automatically load tuned parameters if available
    if os.path.exists(tuned_params_file):
        logger.info(f"Found tuned parameters in '{tuned_params_file}'. Overriding defaults.")
        with open(tuned_params_file, "r", encoding="utf-8") as f:
            tuned_data = json.load(f)
            training_dict.update(tuned_data.get("best_params", {}))

    data_mgr = DetectionDataManager()

    if model_name.lower() in ["deberta", "mdeberta", "mdeberta-v3"]:
        hf_datasets = data_mgr.get_hf_dataset(scopes=[scope])
        train_ds = hf_datasets["train"]
        val_ds = hf_datasets["dev"]
    else:
        train_ds = data_mgr.get_sklearn_data(splits=["train"], scopes=[scope])
        val_ds = data_mgr.get_sklearn_data(splits=["dev"], scopes=[scope])

    detector = ModelFactory.create(
        model_name, 
        max_length=getattr(cfg.model, "max_length", 256),
        granularity=scope,
        calibrate=getattr(cfg.model, "calibrate", True)
    )
    
    eval_metrics = detector.train(train_ds, val_ds, training_dict)
    detector.save(checkpoint_dir)

    # Save copy of best parameters inside checkpoint folder
    if os.path.exists(tuned_params_file):
        shutil.copy(tuned_params_file, os.path.join(checkpoint_dir, "best_hyperparameters.json"))
        logger.info(f"[SAVED] Copied best_hyperparameters.json to checkpoint folder '{checkpoint_dir}'")

    logger.info("Training completed successfully!")
    logger.info(f"Validation Metrics: {eval_metrics}")


if __name__ == "__main__":
    main()