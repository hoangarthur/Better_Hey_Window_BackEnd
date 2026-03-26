"""
core/ml_model.py
Temporal Convolutional Network (TCN) cho gesture recognition.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.ml_features import FEATURE_DIM


class ResidualBlock(nn.Module):
    """TCN block với residual connection và dilated convolution."""

    def __init__(self, in_ch: int, out_ch: int, dilation: int):
        super().__init__()
        pad = dilation  # same-padding cho kernel_size=3
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=pad, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(out_ch)
        self.norm2 = nn.BatchNorm1d(out_ch)
        self.drop  = nn.Dropout(0.2)
        self.proj  = (nn.Conv1d(in_ch, out_ch, 1)
                      if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.drop(x)
        x = self.norm2(self.conv2(x))
        return F.relu(x + residual)


class GestureTCN(nn.Module):
    """
    Temporal Convolutional Network cho gesture recognition.

    Input:  (batch, T, input_dim)  — sequence of feature vectors
    Output: (batch, n_classes)     — unnormalized logits

    Receptive field với 4 blocks dilation [1,2,4,8]:
      kernel=3, dilation=8 → RF ≈ 35 frames — đủ capture gesture ~1s ở 30fps
    """

    def __init__(self, input_dim: int = FEATURE_DIM,
                 num_classes: int = 20, channels: int = 128):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, channels)
        self.blocks = nn.Sequential(
            ResidualBlock(channels, channels, dilation=1),
            ResidualBlock(channels, channels, dilation=2),
            ResidualBlock(channels, channels, dilation=4),
            ResidualBlock(channels, channels, dilation=8),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(channels // 2, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        x = self.input_proj(x)         # (B, T, C)
        x = x.transpose(1, 2)          # (B, C, T)
        x = self.blocks(x)             # (B, C, T)
        x = self.pool(x).squeeze(-1)   # (B, C)
        return self.classifier(x)      # (B, n_classes)

    def predict_proba(self, x: torch.Tensor,
                      temperature: float = 1.0) -> torch.Tensor:
        """Return softmax probabilities, optionally temperature-scaled."""
        logits = self.forward(x)
        if temperature != 1.0:
            logits = logits / temperature
        return F.softmax(logits, dim=-1)


def count_params(model: nn.Module) -> str:
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"{n:,} params ({n * 4 / 1024:.1f} KB float32)"
