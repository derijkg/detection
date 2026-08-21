# src/models/registry.py

from typing import Any, Dict, Type
from src.models.base import BaseDetector
from src.models.svm_pipeline import SVMDetector
from src.models.fast_detect_gpt import FastDetectGPTDetector
from src.models.statistical_detector import StatisticalTrajectoryDetector
from src.models.deberta import MDeBERTaDetector

MODEL_METADATA: Dict[str, Dict[str, Any]] = {
    "svm": {
        "class": SVMDetector,
        "canonical": "svm",
        "display_name": "Linear SVM (TF-IDF + Stylo)",
        "color": "#1f77b4",
    },
    "mdeberta": {
        "class": MDeBERTaDetector,
        "canonical": "deberta",
        "display_name": "mDeBERTa-v3 (CVaR-DRO)",
        "color": "#d62728",
    },
    "fdgpt": {
        "class": FastDetectGPTDetector,
        "canonical": "fdgpt",
        "display_name": "Fast-DetectGPT (Zero-Shot)",
        "color": "#2ca02c",
    },
    "stat_trajectory": {
        "class": StatisticalTrajectoryDetector,
        "canonical": "stat",
        "display_name": "LLM Trajectory (Ours)",
        "color": "#9467bd",
    },
}

# Alias resolution mapping
ALIAS_MAP: Dict[str, str] = {
    "svm": "svm",
    "mdeberta": "mdeberta",
    "deberta": "mdeberta",
    "fdgpt": "fdgpt",
    "fast_detect_gpt": "fdgpt",
    "fast_detectgpt": "fdgpt",
    "stat_trajectory": "stat_trajectory",
    "stat": "stat_trajectory",
}


def normalize_model_name(name: str) -> str:
    key = name.lower()
    if key not in ALIAS_MAP:
        raise ValueError(f"Unknown model '{name}'. Available: {list(ALIAS_MAP.keys())}")
    return ALIAS_MAP[key]


def get_detector_class(name: str) -> Type[BaseDetector]:
    canonical_key = normalize_model_name(name)
    return MODEL_METADATA[canonical_key]["class"]


def get_model_display_name(name: str) -> str:
    canonical_key = normalize_model_name(name)
    return MODEL_METADATA[canonical_key]["display_name"]


def get_model_color(name: str) -> str:
    canonical_key = normalize_model_name(name)
    return MODEL_METADATA[canonical_key]["color"]


def get_canonical_directory_name(name: str) -> str:
    canonical_key = normalize_model_name(name)
    return MODEL_METADATA[canonical_key]["canonical"]