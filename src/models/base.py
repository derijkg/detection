# src/models/base.py
from abc import ABC, abstractmethod
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from src.utils.logger import setup_logger
from src.utils.seed import set_seed

if TYPE_CHECKING:
    from src.utils.config import ExperimentConfig


class BaseDetector(ABC):

    def __init__(
        self,
        model_name: str,
        scope: str = "full",
        seed: int = 42,
        log_dir: Optional[Union[str, Path]] = None,
        log_level: int = logging.INFO,
    ):
        self.model_name = model_name
        self.scope = scope
        self.seed = seed
        self.calibrated_threshold: float = 0.5
        set_seed(self.seed)

        log_file = Path(log_dir) / "run.log" if log_dir else None
        self.logger = setup_logger(
            name=f"{self.model_name}_{self.scope}",
            log_file=log_file,
            level=log_level
        )
        self.logger.info(f"Initialized {self.model_name.upper()} detector [Scope: {self.scope.upper()}, Seed: {self.seed}]")

    @classmethod
    def from_config(
        cls,
        config: "ExperimentConfig",
        log_dir: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> "BaseDetector":
        """Standard factory to instantiate a detector directly from an ExperimentConfig."""
        return cls(
            scope=config.scope,
            seed=config.seed,
            log_dir=log_dir,
            max_length=config.hyperparams.max_length,
            **kwargs
        )

    @abstractmethod
    def fit(
        self,
        train_data: Union[pd.DataFrame, List[Dict[str, Any]], Any],
        dev_data: Optional[pd.DataFrame] = None,
        config: Optional["ExperimentConfig"] = None,
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> "BaseDetector":
        """Standardized training/calibration entry point."""
        pass

    @abstractmethod
    def predict_proba(
        self,
        texts: Union[List[str], List[Dict[str, Any]], pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        pass

    def predict(
        self,
        texts: Union[List[str], List[Dict[str, Any]], pd.DataFrame, np.ndarray],
        threshold: Optional[float] = None
    ) -> np.ndarray:
        thresh = threshold if threshold is not None else self.calibrated_threshold
        probs = self.predict_proba(texts)
        return (probs >= thresh).astype(int)

    @abstractmethod
    def save(self, path: Union[str, Path]):
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Union[str, Path], **kwargs) -> "BaseDetector":
        pass