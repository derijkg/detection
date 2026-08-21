# src/models/base.py

from abc import ABC, abstractmethod
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from src.utils.logger import setup_logger
from src.utils.seed import set_seed


class BaseDetector(ABC):
    """
    Abstract Base Class for all thesis detectors (SVM, DeBERTa, Fast-DetectGPT, Statistical Trajectory).
    """
    def __init__(
        self,
        model_name: str,
        scope: str = "full",
        seed: int = 42,
        log_dir: Optional[Union[str, Path]] = None,
        log_level: int = logging.INFO
    ):
        self.model_name = model_name
        self.scope = scope
        self.seed = seed
        self.calibrated_threshold: float = 0.5

        set_seed(self.seed)

        log_file = (Path(log_dir) / "run.log") if log_dir else None
        self.logger = setup_logger(
            name=f"{self.model_name}_{self.scope}",
            log_file=log_file,
            level=log_level
        )
        self.logger.info(f"Initialized {self.model_name.upper()} detector [Scope: {self.scope.upper()}, Seed: {self.seed}]")

    @abstractmethod
    def fit(self, train_data: Any, y_train: Optional[Any] = None, **kwargs) -> "BaseDetector":
        """Fits/calibrates the model on training data."""
        pass

    @abstractmethod
    def predict_proba(self, texts: Union[List[str], List[Dict[str, Any]], pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Polymorphic inference interface returning 1D numpy array of probabilities [0, 1] for class 1 (LLM).
        """
        pass

    def predict(
        self, 
        texts: Union[List[str], List[Dict[str, Any]], pd.DataFrame, np.ndarray], 
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """Converts probabilities to binary predictions using the calibrated decision threshold."""
        thresh = threshold if threshold is not None else self.calibrated_threshold
        probs = self.predict_proba(texts)
        return (probs >= thresh).astype(int)

    @abstractmethod
    def save(self, path: Union[str, Path]):
        """Saves model weights, pipelines, and calibration parameters to disk."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Union[str, Path], **kwargs) -> "BaseDetector":
        """Loads model weights/pipeline from disk."""
        pass