from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Any


class BaseDetector(ABC):
    """Abstract interface for all AI text detectors."""

    @abstractmethod
    def train(self, train_ds: Any, val_ds: Any, config: Any) -> Dict[str, Any]:
        """Trains detector on training set and evaluates on validation set."""
        pass

    @abstractmethod
    def predict_proba(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Returns array of shape (N, 2) where col 0 = P(Human) and col 1 = P(LLM)."""
        pass

    @abstractmethod
    def save(self, output_dir: str) -> None:
        """Saves model weights/artifacts to disk."""
        pass

    @abstractmethod
    def load(self, input_dir: str) -> None:
        """Loads model weights/artifacts from disk."""
        pass