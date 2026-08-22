# src/models/trajectory_nn.py
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.models.base import BaseDetector
from src.models.statistical_features import extract_text_2d_trajectory, FEATURE_CHANNELS
from src.training.trainer_deberta import RockafellarUryasevCVaRLoss

class TrajectoryDataset(Dataset):

    def __init__(self, matrices: List[np.ndarray], labels: Optional[List[int]] = None, max_len: int = 256):
        self.matrices = matrices
        self.labels = labels
        self.max_len = max_len
        self.num_channels = len(FEATURE_CHANNELS)

    def __len__(self):
        return len(self.matrices)

    def __getitem__(self, idx: int):
        mat = self.matrices[idx]
        seq_len = min(len(mat), self.max_len)
        padded = np.zeros((self.num_channels, self.max_len), dtype=np.float32)
        if seq_len > 0:
            padded[:, :seq_len] = mat[:seq_len].T
        mask = np.zeros((self.max_len,), dtype=np.float32)
        mask[:seq_len] = 1.0
        item = {
            'features': torch.tensor(padded, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.float32)
        }
        if self.labels is not None:
            item['label'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

class MultiScaleConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        branch_channels = out_channels // 3
        rem = out_channels - branch_channels * 2
        self.b1 = nn.Conv1d(in_channels, branch_channels, kernel_size=3, padding=1)
        self.b2 = nn.Conv1d(in_channels, branch_channels, kernel_size=5, padding=2)
        self.b3 = nn.Conv1d(in_channels, rem, kernel_size=7, dilation=2, padding=6)
        self.bn = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
        out = F.gelu(self.bn(out))
        return out + self.shortcut(x)

class Thermodynamic2DTrajectoryClassifier(nn.Module):

    def __init__(self, in_channels: int = 10, hidden_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.in_proj = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)
        self.layer1 = MultiScaleConvBlock(hidden_dim, hidden_dim)
        self.layer2 = MultiScaleConvBlock(hidden_dim, hidden_dim * 2)
        self.layer3 = MultiScaleConvBlock(hidden_dim * 2, hidden_dim * 4)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 4 * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, num_classes)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.in_proj(x))
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        mask_expanded = mask.unsqueeze(1)
        h_masked = h * mask_expanded
        mean_pool = h_masked.sum(dim=-1) / mask_expanded.sum(dim=-1).clamp(min=1.0)
        h_for_max = h.masked_fill(mask_expanded == 0, -1e9)
        max_pool = torch.max(h_for_max, dim=-1)[0]
        fused = torch.cat([mean_pool, max_pool], dim=-1)
        return self.head(fused)

class TrajectoryNNDetector(BaseDetector):

    def __init__(
        self,
        scope: str = 'full',
        hidden_dim: int = 64,
        max_len: int = 256,
        seed: int = 42,
        device: Optional[str] = None,
        log_dir: Optional[Union[str, Path]] = None,
        **kwargs
    ):
        super().__init__(model_name='trajectory_nn', scope=scope, seed=seed, log_dir=log_dir)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_len = max_len
        self.model = Thermodynamic2DTrajectoryClassifier(
            in_channels=len(FEATURE_CHANNELS),
            hidden_dim=hidden_dim,
            num_classes=2
        ).to(self.device)

    def fit(
        self,
        train_data: Any,
        y_train: Optional[Any] = None,
        dev_data: Optional[Any] = None,
        epochs: int = 10,
        lr: float = 1e-3,
        batch_size: int = 32,
        target_fpr: float = 0.01,
        **kwargs
    ) -> 'TrajectoryNNDetector':
        df_train = pd.DataFrame(train_data)
        if 'label' not in df_train.columns and y_train is not None:
            df_train['label'] = y_train

        labels = df_train['label'].astype(int).tolist()
        matrices = []
        for _, row in df_train.iterrows():
            if 'trajectory_matrix' in row and isinstance(row['trajectory_matrix'], np.ndarray):
                matrices.append(row['trajectory_matrix'])
            else:
                matrices.append(np.zeros((self.max_len, len(FEATURE_CHANNELS)), dtype=np.float32))

        dataset = TrajectoryDataset(matrices, labels=labels, max_len=self.max_len)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-2)
        loss_fn = RockafellarUryasevCVaRLoss(alpha=target_fpr, lambda_neg=2.0).to(self.device)

        self.model.train()
        for epoch in range(epochs):
            for batch in loader:
                x = batch['features'].to(self.device)
                mask = batch['mask'].to(self.device)
                y = batch['label'].to(self.device)
                optimizer.zero_grad()
                logits = self.model(x, mask)
                loss = loss_fn(logits, y)
                loss.backward()
                optimizer.step()

        self.model.eval()
        return self

    def predict_proba(self, texts: Union[List[str], List[Dict[str, Any]], pd.DataFrame, np.ndarray]) -> np.ndarray:
        self.model.eval()
        if isinstance(texts, pd.DataFrame):
            records = texts.to_dict(orient='records')
        elif isinstance(texts, list) and len(texts) > 0 and isinstance(texts[0], dict):
            records = texts
        else:
            records = [{'text': str(t)} for t in texts]

        if not records:
            return np.array([], dtype=np.float32)

        matrices = []
        for r in records:
            if 'trajectory_matrix' in r and isinstance(r['trajectory_matrix'], np.ndarray):
                matrices.append(r['trajectory_matrix'])
            else:
                matrices.append(np.zeros((self.max_len, len(FEATURE_CHANNELS)), dtype=np.float32))

        dataset = TrajectoryDataset(matrices, max_len=self.max_len)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        all_probs = []
        with torch.inference_mode():
            for batch in loader:
                x = batch['features'].to(self.device)
                mask = batch['mask'].to(self.device)
                logits = self.model(x, mask)
                probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                all_probs.append(probs)

        return np.concatenate(all_probs, axis=0) if all_probs else np.array([], dtype=np.float32)

    def save(self, path: Union[str, Path]):
        save_p = Path(path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            'scope': self.scope,
            'max_len': self.max_len,
            'calibrated_threshold': self.calibrated_threshold
        }
        torch.save({
            'state_dict': self.model.state_dict(),
            'meta': meta
        }, save_p if str(save_p).endswith('.pt') else save_p / 'model.pt')

    @classmethod
    def load(cls, path: Union[str, Path], scope: str = 'full', device: Optional[str] = None, **kwargs) -> 'TrajectoryNNDetector':
        load_p = Path(path)
        if load_p.is_dir():
            load_p = load_p / 'model.pt'
        checkpoint = torch.load(load_p, map_location=device or 'cpu')
        meta = checkpoint.get('meta', {})
        detector = cls(
            scope=meta.get('scope', scope),
            max_len=meta.get('max_len', 256),
            device=device,
            **kwargs
        )
        detector.model.load_state_dict(checkpoint['state_dict'])
        detector.calibrated_threshold = meta.get('calibrated_threshold', 0.5)
        detector.model.eval()
        return detector