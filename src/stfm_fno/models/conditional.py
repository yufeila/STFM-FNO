"""Parameter-matched conditional FNO baselines for the frozen monthly split.

FNMFNO maps the 24-D time vector to four auxiliary spatial functions through a
fixed real Fourier basis, then passes SSP plus those functions to the unchanged
four-layer FNO trunk. HyperFNO applies a time-dependent CP-low-rank update to
the complex Fourier kernels; it never uses FiLM scale/shift.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fno2d import FNO2d


TIME_DIM = 24
HEIGHT = 199
WIDTH = 800
TRUNK_WIDTH = 64
MODES1 = 32
MODES2 = 128
N_LAYERS = 4


def _real_fourier_basis(height: int = HEIGHT, width: int = WIDTH) -> torch.Tensor:
    """Return 128 fixed real 2-D Fourier basis functions on the TL grid."""
    x = torch.linspace(0.0, 1.0, height)
    y = torch.linspace(0.0, 1.0, width)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    phases = []
    for kx in range(8):
        for ky in range(8):
            phases.append(2.0 * math.pi * (kx * xx + ky * yy))
    phase = torch.stack(phases, dim=0)
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=0)


class FourierVectorToField(nn.Module):
    """A V2F Fourier decoder: time vector -> four auxiliary function fields."""

    def __init__(self, time_dim: int = TIME_DIM, hidden_dim: int = 72, channels: int = 4,
                 height: int = HEIGHT, width: int = WIDTH):
        super().__init__()
        self.channels = channels
        self.n_basis = 128
        self.net = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels * self.n_basis),
        )
        self.register_buffer("basis", _real_fourier_basis(height, width), persistent=True)

    def forward(self, time_features: torch.Tensor) -> torch.Tensor:
        coeff = self.net(time_features).view(-1, self.channels, self.n_basis)
        return torch.einsum("bck,khw->bhwc", coeff, self.basis)


class FNMFNO(nn.Module):
    """FNO with a Fourier Neural Mapping V2F time-condition decoder."""

    def __init__(self, hidden_dim: int = 72, aux_channels: int = 4,
                 modes1: int = MODES1, modes2: int = MODES2, width: int = TRUNK_WIDTH,
                 height: int = HEIGHT, grid_width: int = WIDTH,
                 padding: int = 20, dropout: float = 0.1):
        super().__init__()
        self.decoder = FourierVectorToField(hidden_dim=hidden_dim, channels=aux_channels,
                                            height=height, width=grid_width)
        self.trunk = FNO2d(modes1, modes2, width, in_dim=1 + aux_channels,
                           padding=padding, dropout=dropout)

    def forward(self, x: torch.Tensor, time_features: torch.Tensor) -> torch.Tensor:
        aux = self.decoder(time_features)
        if aux.shape[:3] != x.shape[:3]:
            raise ValueError(f"FNM field shape {aux.shape} is incompatible with SSP {x.shape}")
        return self.trunk(torch.cat([x, aux], dim=-1))


class ConditionalCPSpectralConv2d(nn.Module):
    """FNO spectral convolution plus a factorized, time-gated kernel update.

    For sign s in {+, -}, the update is
      Delta R_s(t)[i,o,k1,k2] = sum_r g_s,r(t) A_s[r,i] B_s[r,o]
                                      C_s[r,k1] D_s[r,k2].
    All A/B/C/D factors are complex, while g is real and sample-dependent.
    The update is contracted without materializing B x I x O x K1 x K2 kernels.
    """

    def __init__(self, width: int = TRUNK_WIDTH, rank: int = 16,
                 modes1: int = MODES1, modes2: int = MODES2):
        super().__init__()
        self.width = width
        self.modes1, self.modes2 = modes1, modes2
        self.rank = rank
        self.scale = 1.0 / (width * width)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(width, width, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(width, width, modes1, modes2, dtype=torch.cfloat)
        )
        factor_scale = 0.02
        for suffix in ("1", "2"):
            self.register_parameter(
                f"a{suffix}", nn.Parameter(factor_scale * torch.randn(rank, width, dtype=torch.cfloat))
            )
            self.register_parameter(
                f"b{suffix}", nn.Parameter(factor_scale * torch.randn(rank, width, dtype=torch.cfloat))
            )
            self.register_parameter(
                f"c{suffix}", nn.Parameter(factor_scale * torch.randn(rank, modes1, dtype=torch.cfloat))
            )
            self.register_parameter(
                f"d{suffix}", nn.Parameter(factor_scale * torch.randn(rank, modes2, dtype=torch.cfloat))
            )

    @staticmethod
    def _base(x_ft: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x_ft, weights)

    def _delta(self, x_ft: torch.Tensor, gate: torch.Tensor, suffix: str) -> torch.Tensor:
        a = getattr(self, f"a{suffix}")
        b = getattr(self, f"b{suffix}")
        c = getattr(self, f"c{suffix}")
        d = getattr(self, f"d{suffix}")
        latent = torch.einsum("bixy,ri->brxy", x_ft, a)
        latent = latent * c.unsqueeze(0).unsqueeze(-1)
        latent = latent * d.unsqueeze(0).unsqueeze(-2)
        latent = latent * gate.to(latent.dtype).unsqueeze(-1).unsqueeze(-1)
        return torch.einsum("brxy,ro->boxy", latent, b)

    def forward(self, x: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        # gates: (batch, 2, rank), one condition coefficient per Fourier sign/rank.
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            x.shape[0], self.width, x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )
        top = x_ft[:, :, :self.modes1, :self.modes2]
        bottom = x_ft[:, :, -self.modes1:, :self.modes2]
        out_ft[:, :, :self.modes1, :self.modes2] = self._base(top, self.weights1) + self._delta(top, gates[:, 0], "1")
        out_ft[:, :, -self.modes1:, :self.modes2] = self._base(bottom, self.weights2) + self._delta(bottom, gates[:, 1], "2")
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class LowRankHyperFNO(nn.Module):
    """Fourier-kernel-conditioned FNO with a global time hypernetwork."""

    def __init__(self, rank: int = 16, hyper_hidden_dim: int = 18,
                 modes1: int = MODES1, modes2: int = MODES2, width: int = TRUNK_WIDTH,
                 padding: int = 20, dropout: float = 0.1):
        super().__init__()
        self.rank = rank
        self.width = width
        self.padding = padding
        self.fc0 = nn.Linear(3, width)
        self.conv0 = ConditionalCPSpectralConv2d(width, rank, modes1, modes2)
        self.conv1 = ConditionalCPSpectralConv2d(width, rank, modes1, modes2)
        self.conv2 = ConditionalCPSpectralConv2d(width, rank, modes1, modes2)
        self.conv3 = ConditionalCPSpectralConv2d(width, rank, modes1, modes2)
        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)
        self.hyper = nn.Sequential(
            nn.Linear(TIME_DIM, hyper_hidden_dim),
            nn.GELU(),
            nn.Linear(hyper_hidden_dim, N_LAYERS * 2 * rank),
        )
        self.fc1 = nn.Linear(width, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 1)

    @staticmethod
    def _grid(shape: torch.Size, device: torch.device) -> torch.Tensor:
        batch, height, width = shape[0], shape[1], shape[2]
        x = torch.linspace(0.0, 1.0, height, device=device).view(1, height, 1, 1)
        y = torch.linspace(0.0, 1.0, width, device=device).view(1, 1, width, 1)
        return torch.cat([x.repeat(batch, 1, width, 1), y.repeat(batch, height, 1, 1)], dim=-1)

    def forward(self, x: torch.Tensor, time_features: torch.Tensor) -> torch.Tensor:
        gates = self.hyper(time_features).view(-1, N_LAYERS, 2, self.rank)
        x = torch.cat([x, self._grid(x.shape, x.device)], dim=-1)
        x = self.fc0(x).permute(0, 3, 1, 2)
        x = F.pad(x, [0, self.padding, 0, self.padding])
        for layer, pointwise, gate, activate in (
            (self.conv0, self.w0, gates[:, 0], True),
            (self.conv1, self.w1, gates[:, 1], True),
            (self.conv2, self.w2, gates[:, 2], True),
            (self.conv3, self.w3, gates[:, 3], False),
        ):
            x = layer(x, gate) + pointwise(x)
            if activate:
                x = F.gelu(x)
        if self.padding:
            x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        return self.fc2(self.dropout(x))


def parameter_counts() -> dict[str, int]:
    # Do not instantiate six 134M-parameter models merely to audit candidates.
    # This is the exact PyTorch ``numel`` count of the existing SSP-only FNO2d.
    base_fno = 134_243_073

    def fnm_count(hidden: int) -> int:
        # The trunk lift sees four additional auxiliary fields (+4*64).  The
        # decoder is Linear(24,h) then Linear(h,4*128).
        return base_fno + 4 * TRUNK_WIDTH + (TIME_DIM + 1) * hidden + (hidden + 1) * (4 * 128)

    def hyper_count(rank: int, hidden: int) -> int:
        # Four FNO blocks, two Fourier signs each.  Each CP factor has
        # rank*(64 + 64 + 32 + 128) complex PyTorch parameters.  The shared
        # real hyper MLP emits 4*2*rank gates.
        cp = N_LAYERS * 2 * rank * (TRUNK_WIDTH + TRUNK_WIDTH + MODES1 + MODES2)
        gates = (TIME_DIM + 1) * hidden + (hidden + 1) * (N_LAYERS * 2 * rank)
        return base_fno + cp + gates

    return {
        "fnm_h64": fnm_count(64),
        "fnm_h72": fnm_count(72),
        "fnm_h80": fnm_count(80),
        "hyper_r16_h8": hyper_count(16, 8),
        "hyper_r16_h16": hyper_count(16, 16),
        "hyper_r16_h18": hyper_count(16, 18),
    }
