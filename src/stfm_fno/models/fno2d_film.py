"""Define the FiLM-conditioned 2D Fourier Neural Operator for STFM-FNO.

Instantiate FNO2d_FiLM(modes1, modes2, width, time_dim) for the time-aware variant.
Inputs are normalized SSP tensors shaped (batch, height, width, 1) and temporal
feature vectors shaped (batch, time_dim).
Output is a TL prediction tensor shaped (batch, height, width, 1).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

class H1Loss(object):
    def __init__(self, d=2, beta=0.1):
        super(H1Loss, self).__init__()
        self.d = d
        self.beta = beta
        self.l2 = LpLoss(d=2, size_average=False)

    def __call__(self, x, y):
        l2 = self.l2(x, y)
        dx_x = x[:, 1:, :] - x[:, :-1, :]
        dy_x = y[:, 1:, :] - y[:, :-1, :]
        l2_dx = self.l2(dx_x, dy_x)

        dx_y = x[:, :, 1:] - x[:, :, :-1]
        dy_y = y[:, :, 1:] - y[:, :, :-1]
        l2_dy = self.l2(dx_y, dy_y)

        return l2 + self.beta * (l2_dx + l2_dy)

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d_FiLM(nn.Module):
    def __init__(self, modes1, modes2, width, time_dim=24, film_hidden_dim=None, padding=20, dropout=0.1):
        super(FNO2d_FiLM, self).__init__()

        """
        FNO combined with Feature-wise Linear Modulation (FiLM).
        Input:
            x: (batchsize, s, s, 1) - SSP field
            t: (batchsize, time_dim) - Global temporal feature vector
        """
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = padding
        self.film_hidden_dim = width if film_hidden_dim is None else film_hidden_dim

        # in_dim = 1 (SSP) + 2 (grid_x, grid_y) = 3
        self.fc0 = nn.Linear(3, self.width)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        # FiLM Generators
        # Map time_dim -> 2 * width (gamma and beta) for each of the 4 FNO blocks
        self.film0 = nn.Sequential(nn.Linear(time_dim, self.film_hidden_dim), nn.ReLU(), nn.Linear(self.film_hidden_dim, 2 * width))
        self.film1 = nn.Sequential(nn.Linear(time_dim, self.film_hidden_dim), nn.ReLU(), nn.Linear(self.film_hidden_dim, 2 * width))
        self.film2 = nn.Sequential(nn.Linear(time_dim, self.film_hidden_dim), nn.ReLU(), nn.Linear(self.film_hidden_dim, 2 * width))
        self.film3 = nn.Sequential(nn.Linear(time_dim, self.film_hidden_dim), nn.ReLU(), nn.Linear(self.film_hidden_dim, 2 * width))

        self.fc1 = nn.Linear(self.width, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 1)

    def apply_film(self, x, t_feat, film_layer):
        # film_layer(t_feat) -> (batchsize, 2 * width)
        film_params = film_layer(t_feat)
        gamma = film_params[:, :self.width].unsqueeze(-1).unsqueeze(-1) # (B, W, 1, 1)
        beta  = film_params[:, self.width:].unsqueeze(-1).unsqueeze(-1) # (B, W, 1, 1)
        return x * (1 + gamma) + beta # Centered around 1 and 0

    def forward(self, x, t):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1) # (B, H, W, 3)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0,self.padding, 0,self.padding])

        # Block 0
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = self.apply_film(x, t, self.film0)
        x = F.gelu(x)

        # Block 1
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = self.apply_film(x, t, self.film1)
        x = F.gelu(x)

        # Block 2
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = self.apply_film(x, t, self.film2)
        x = F.gelu(x)

        # Block 3
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        x = self.apply_film(x, t, self.film3)

        x = x[..., :-self.padding, :-self.padding] if self.padding else x
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
