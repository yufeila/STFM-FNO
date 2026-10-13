"""Build temporal inputs and labels for shuffled STFM-FNO samples.

Use build_time_feature_vector(global_idx) to create the FiLM time input t for
FNO2d_FiLM; use the date helpers for analysis and plotting.
Inputs are 0-based global sample indices as torch tensors or NumPy arrays.
Outputs are Fourier time vectors and month/season/day labels.
"""

from __future__ import annotations

import numpy as np
import torch

from .config import (
    NTOTAL,
    SAMPLES_PER_DAY,
    TIME_DAY_HARMONICS,
    TIME_FEATURE_DIM,
    TIME_YEAR_HARMONICS,
)


def fourier_features_1d(phase: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    two_pi = 2.0 * np.pi
    angles = two_pi * phase[:, None] * freqs[None, :]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


def build_time_feature_vector(
    global_idx: torch.Tensor,
    nt_total: int = NTOTAL,
    samples_per_day: int = SAMPLES_PER_DAY,
    day_harmonics: int = TIME_DAY_HARMONICS,
    year_harmonics: int = TIME_YEAR_HARMONICS,
) -> torch.Tensor:
    day_phase = (global_idx.remainder(float(samples_per_day))) / float(samples_per_day)
    year_phase = global_idx / float(nt_total - 1)

    freqs_day = torch.arange(
        1, day_harmonics + 1, dtype=torch.float32, device=global_idx.device
    )
    freqs_year = torch.arange(
        1, year_harmonics + 1, dtype=torch.float32, device=global_idx.device
    )
    feat_day = fourier_features_1d(day_phase, freqs_day)
    feat_year = fourier_features_1d(year_phase, freqs_year)
    features = torch.cat([feat_day, feat_year], dim=-1)
    if features.shape[-1] != TIME_FEATURE_DIM:
        raise RuntimeError(
            f"Expected time feature dim {TIME_FEATURE_DIM}, got {features.shape[-1]}"
        )
    return features


def build_generic_fourier_feature_vector(
    global_idx: torch.Tensor,
    nt_total: int = NTOTAL,
    min_frequency: float = 12.0,
    max_frequency: float = 1200.0,
    n_frequencies: int = TIME_FEATURE_DIM // 2,
) -> torch.Tensor:
    """Return a fixed, non-physical log-spaced Fourier control of matched dimension."""
    if n_frequencies * 2 != TIME_FEATURE_DIM:
        raise ValueError("Generic Fourier feature dimension must match TIME_FEATURE_DIM")
    if not 0.0 < min_frequency < max_frequency < (nt_total - 1) / 2:
        raise ValueError("Generic Fourier frequencies must be below the Nyquist limit")
    phase = global_idx / float(nt_total - 1)
    freqs = torch.logspace(
        np.log10(min_frequency),
        np.log10(max_frequency),
        steps=n_frequencies,
        dtype=torch.float32,
        device=global_idx.device,
    )
    return fourier_features_1d(phase, freqs)


def expand_time_features_to_grid(
    feature_vec: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    channels = feature_vec.shape[-1]
    return feature_vec.view(-1, 1, 1, channels).repeat(1, height, width, 1)


def build_time_scalar(global_idx: torch.Tensor, nt_total: int = NTOTAL) -> torch.Tensor:
    return (global_idx / float(nt_total - 1)).view(-1, 1)


def expand_time_scalar_to_grid(
    global_idx: torch.Tensor, height: int, width: int, nt_total: int = NTOTAL
) -> torch.Tensor:
    return build_time_scalar(global_idx, nt_total).view(-1, 1, 1, 1).repeat(
        1, height, width, 1
    )


def month_from_global_idx(
    global_idx: np.ndarray, samples_per_day: int = SAMPLES_PER_DAY
) -> np.ndarray:
    month_days = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=int)
    cumulative = np.cumsum(month_days)
    day_idx = (global_idx // samples_per_day).astype(int)
    return np.searchsorted(cumulative, day_idx + 1) + 1


def season_from_month(month: np.ndarray) -> np.ndarray:
    season = np.empty_like(month, dtype=object)
    season[np.isin(month, [12, 1, 2])] = "DJF"
    season[np.isin(month, [3, 4, 5])] = "MAM"
    season[np.isin(month, [6, 7, 8])] = "JJA"
    season[np.isin(month, [9, 10, 11])] = "SON"
    return season


def day_from_global_idx(global_idx: np.ndarray, samples_per_day: int = SAMPLES_PER_DAY) -> np.ndarray:
    return (global_idx // samples_per_day).astype(int)
