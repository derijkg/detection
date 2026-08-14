# detection/src/utils/config.py

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field, fields
from typing import Dict, Any, Optional


def filter_dataclass_kwargs(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Filters dictionary keys to match only valid dataclass field names."""
    valid_fields = {f.name for f in fields(cls)}
    return {k: v for k, v in kwargs.items() if k in valid_fields}


@dataclass
class ModelConfig:
    name: str
    pretrained_model_name: str = "microsoft/mdeberta-v3-base"
    max_length: int = 256
    granularity: str = "full"
    calibrate: bool = True
    use_stylometrics: bool = True


@dataclass
class TrainingConfig:
    output_dir: str = "outputs/checkpoints/model"
    train_sample_size: int = -1
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    learning_rate: float = 1.5e-5
    weight_decay: float = 0.05
    warmup_ratio: float = 0.1
    label_smoothing_factor: float = 0.05
    early_stopping_patience: int = 2
    use_stylometrics: bool = True
    kernel: str = "linear"
    C: float = 1.0


@dataclass
class OptunaConfig:
    n_trials: int = 30
    output_dir: str = "outputs/metrics/optuna"
    tune_sample_size: int = 1000
    score_metric: str = "pauc"
    max_fpr: float = 0.01
    enqueue_params: Dict[str, Any] = field(default_factory=dict)
    search_space: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalConfig:
    save_dir: str = "outputs/metrics/paper_results"
    batch_size: int = 32


class Config:
    def __init__(self, config_dict: Dict[str, Any]):
        self.model = ModelConfig(**filter_dataclass_kwargs(ModelConfig, config_dict.get("model", {})))
        self.training = TrainingConfig(**filter_dataclass_kwargs(TrainingConfig, config_dict.get("training", {})))
        self.optuna = OptunaConfig(**filter_dataclass_kwargs(OptunaConfig, config_dict.get("optuna", {})))
        self.eval = EvalConfig(**filter_dataclass_kwargs(EvalConfig, config_dict.get("eval", {})))

    @classmethod
    def resolve_config_path(cls, model_name: str, scope: str = "") -> Path:
        """
        Dynamically resolves path to a config file. Checks scope-specific configs 
        first (e.g., configs/models/svm_sentence.yaml), then falls back to model-level config.
        """
        project_root = Path(__file__).resolve().parent.parent.parent
        base_dir = project_root / "configs" / "models"
        
        candidates = []
        if scope:
            candidates.append(base_dir / f"{model_name}_{scope}.yaml")
            if scope in ['sentence', 'single']:
                candidates.append(base_dir / f"{model_name}_sentence.yaml")
                candidates.append(base_dir / f"{model_name}_single.yaml")
            elif scope == 'full':
                candidates.append(base_dir / f"{model_name}_full.yaml")

        candidates.append(base_dir / f"{model_name}.yaml")

        for cand in candidates:
            if cand.exists():
                return cand

        raise FileNotFoundError(
            f"Could not find config file for model '{model_name}' (scope='{scope}'). "
            f"Checked paths: {[str(c) for c in candidates]}"
        )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config file not found at: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(data)