from typing import Dict, Type
from src.models.base_model import BaseDetector
from src.models.deberta import DeBERTaDetector
from src.models.svm import SVMDetector


class ModelFactory:
    """Factory registry for instantiating detectors by name."""

    _registry: Dict[str, Type[BaseDetector]] = {
        "deberta": DeBERTaDetector,
        "mdeberta": DeBERTaDetector,
        "mdeberta-v3": DeBERTaDetector,
        "svm": SVMDetector,
        "linear_svm": SVMDetector,
    }

    @classmethod
    def register(cls, model_name: str, model_cls: Type[BaseDetector]) -> None:
        """Registers a new model class into the factory."""
        cls._registry[model_name.lower()] = model_cls

    @classmethod
    def create(cls, model_name: str, **kwargs) -> BaseDetector:
        """Instantiates a model by registered key name."""
        key = model_name.lower()
        if key not in cls._registry:
            raise ValueError(f"Model '{model_name}' not in registry. Available: {list(cls._registry.keys())}")
        return cls._registry[key](**kwargs)