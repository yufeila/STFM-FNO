"""Per-grid-point Gaussian normalization fitted exclusively on training samples."""
from pathlib import Path
import torch


class UnitGaussianNormalizer:
    def __init__(self, x, eps=1e-5):
        if x.shape[0] < 2:
            raise ValueError("At least two training samples are needed for sample standard deviation")
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)  # correction=1, matching the experiment implementation
        self.eps = eps

    def encode(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x):
        return x * (self.std + self.eps) + self.mean

    def to(self, device):
        self.mean, self.std = self.mean.to(device), self.std.to(device)
        return self

    def cuda(self):
        return self.to("cuda")

    def cpu(self):
        return self.to("cpu")


def save_normalizers(path, x_norm, y_norm, meta):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"x_mean": x_norm.mean.cpu(), "x_std": x_norm.std.cpu(),
                "y_mean": y_norm.mean.cpu(), "y_std": y_norm.std.cpu(),
                "eps": x_norm.eps, "meta": meta}, path)


def load_normalizers(path, device=None):
    state = torch.load(path, map_location="cpu", weights_only=True)
    norms = []
    for prefix in ("x", "y"):
        norm = UnitGaussianNormalizer.__new__(UnitGaussianNormalizer)
        norm.mean, norm.std = state[prefix + "_mean"], state[prefix + "_std"]
        norm.eps = state["eps"]
        norms.append(norm.to(device) if device is not None else norm)
    return tuple(norms)
