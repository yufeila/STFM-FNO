"""Temporal controls used in the paper's three-seed encoding comparison.

The Learned-frequency-FiLM control implements the bounded Time2Vec-style
encoding used in the revision experiments; it adds exactly 48 parameters.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..time import build_generic_fourier_feature_vector, build_time_feature_vector
from .fno2d_film import FNO2d_FiLM


TIME2VEC_FMAX = 1460.0
TIME_VARIANTS = ("generic_fourier_film", "time2vec_film", "ours_fourier_film")


class Time2VecEncoding(nn.Module):
    """One linear term and 23 learned sinusoids of normalized global time."""

    def __init__(self, nt_total: int = 2920):
        super().__init__()
        self.nt_total = nt_total
        initial = torch.logspace(np.log10(1.0), np.log10(1200.0), 23, dtype=torch.float32)
        self.theta = nn.Parameter(torch.logit(initial / TIME2VEC_FMAX))
        self.phase = nn.Parameter(torch.zeros(23, dtype=torch.float32))
        self.linear_a0 = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.linear_b0 = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def frequencies(self) -> torch.Tensor:
        return TIME2VEC_FMAX * torch.sigmoid(self.theta)

    def forward(self, global_idx: torch.Tensor) -> torch.Tensor:
        u = global_idx.to(dtype=self.theta.dtype).unsqueeze(-1) / float(self.nt_total - 1)
        linear = self.linear_a0 * u + self.linear_b0
        periodic = torch.sin(2.0 * np.pi * self.frequencies().unsqueeze(0) * u + self.phase.unsqueeze(0))
        return torch.cat((linear, periodic), dim=-1)

    def summary(self) -> dict:
        return {
            "formula": "[a0*u+b0, sin(2*pi*f_j*u+p_j)], u=global_idx/2919",
            "frequency_cap": TIME2VEC_FMAX,
            "frequencies": self.frequencies().detach().cpu().tolist(),
            "phases": self.phase.detach().cpu().tolist(),
            "linear_a0": float(self.linear_a0.detach().cpu()),
            "linear_b0": float(self.linear_b0.detach().cpu()),
        }


class TimeFiLMModel(nn.Module):
    """Keep the original encoder-before-FNO initialization and state keys."""

    def __init__(self, variant: str, modes1: int = 32, modes2: int = 128,
                 width: int = 64, padding: int = 20, dropout: float = 0.1):
        super().__init__()
        if variant not in TIME_VARIANTS:
            raise ValueError(f"Unknown temporal control: {variant}")
        self.time_encoder = Time2VecEncoding() if variant == "time2vec_film" else None
        self.fno = FNO2d_FiLM(modes1, modes2, width, time_dim=24,
                             padding=padding, dropout=dropout)
        self.variant = variant

    def time_features(self, global_idx: torch.Tensor) -> torch.Tensor:
        if self.variant == "time2vec_film":
            return self.time_encoder(global_idx)
        if self.variant == "generic_fourier_film":
            return build_generic_fourier_feature_vector(global_idx.float())
        return build_time_feature_vector(global_idx.float())

    def forward(self, x: torch.Tensor, global_idx: torch.Tensor) -> torch.Tensor:
        return self.fno(x, self.time_features(global_idx))
