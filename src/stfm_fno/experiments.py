"""Portable reproduction CLI for Tables I and II of the STFM-FNO paper.

Training uses normalized training targets and the historical H1Loss(beta=.01).
The final epoch is saved without consulting test-set metrics. Evaluation keeps
the paper's sample-wise metric definitions and the manifest's global indices.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from . import __version__
from .data import load_global_data
from .normalizer import UnitGaussianNormalizer, load_normalizers, save_normalizers
from .split import load_monthly_manifest, manifest_sha256
from .time import build_time_feature_vector, build_time_scalar
from .models.fno2d import FNO2d, H1Loss
from .models.fno2d_film import FNO2d_FiLM
from .models.conditional import FNMFNO, LowRankHyperFNO
from .models.time_encoding import TIME_VARIANTS, TimeFiLMModel


MAIN_VARIANTS = ("baseline", "scalar", "pe", "film", "fnm", "hyper")
VARIANTS = MAIN_VARIANTS + TIME_VARIANTS
LABELS = {
    "baseline": "Baseline FNO", "scalar": "Concat (Scalar)",
    "pe": "Concat (PE)", "film": "STFM-FNO", "fnm": "FNM-FNO",
    "hyper": "Low-rank Hyper-FNO", "generic_fourier_film": "Generic-Fourier-FiLM",
    "time2vec_film": "Learned-frequency-FiLM", "ours_fourier_film": "Daily-and-yearly-FiLM",
}
PAPER_PARAMETER_COUNTS = {
    "baseline": 134243073, "scalar": 134243137, "pe": 134244609,
    "film": 134282753, "fnm": 134282505, "hyper": 134282819,
    "generic_fourier_film": 134282753, "time2vec_film": 134282801,
    "ours_fourier_film": 134282753,
}
MODEL_DEFAULTS = {"modes1": 32, "modes2": 128, "width": 64, "padding": 20, "dropout": 0.1}
METRIC_KEYS = ("rmse", "rel_l2", "ssim", "pcc")


def resolve_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; pass --device cpu explicitly.")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Supported devices are cpu and cuda[:index].")
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    return device


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()


def paper_settings(config: dict, variant: str) -> bool:
    return (
        config["model"] == MODEL_DEFAULTS and config["epochs"] == 100
        and config["batch_size"] == 20 and config["optimizer"] == "Adam"
        and config["learning_rate"] == 1e-3 and config["weight_decay"] == 1e-4
        and config["scheduler"] == {"type": "StepLR", "step_size": 50, "gamma": 0.5}
        and config["loss"] == "H1Loss(d=2,beta=0.01)"
        and config["parameters"] == PAPER_PARAMETER_COUNTS[variant]
        and config["shape"] == [199, 800]
    )


def environment(device: torch.device) -> dict:
    import h5py
    import scipy
    import skimage

    return {
        "python": sys.version, "platform": platform.platform(),
        "torch": str(torch.__version__), "numpy": np.__version__,
        "scipy": scipy.__version__, "h5py": h5py.__version__,
        "scikit_image": skimage.__version__, "cuda": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def make_model(variant: str, settings: dict, shape: tuple[int, int]) -> torch.nn.Module:
    if variant in {"baseline", "scalar", "pe"}:
        return FNO2d(**settings, in_dim={"baseline": 1, "scalar": 2, "pe": 25}[variant])
    if variant == "film":
        return FNO2d_FiLM(**settings, time_dim=24)
    if variant == "fnm":
        return FNMFNO(**settings, hidden_dim=72, aux_channels=4,
                      height=shape[0], grid_width=shape[1])
    if variant == "hyper":
        return LowRankHyperFNO(**settings, rank=16, hyper_hidden_dim=18)
    if variant in TIME_VARIANTS:
        return TimeFiLMModel(variant, **settings)
    raise ValueError(variant)


def forward(model: torch.nn.Module, variant: str, normalized_ssp: torch.Tensor,
            global_idx: torch.Tensor) -> torch.Tensor:
    """Construct concatenated time fields per batch, keeping the original values."""
    if variant == "baseline":
        return model(normalized_ssp).squeeze(-1)
    if variant in {"scalar", "pe"}:
        # Table I source runners computed fixed features on CPU before transfer.
        cpu_idx = global_idx.detach().cpu().float()
        features = (build_time_scalar(cpu_idx) if variant == "scalar"
                    else build_time_feature_vector(cpu_idx)).to(normalized_ssp.device)
        h, w = normalized_ssp.shape[1:3]
        fields = features.view(-1, 1, 1, features.shape[-1]).expand(-1, h, w, -1)
        return model(torch.cat((normalized_ssp, fields), dim=-1)).squeeze(-1)
    if variant in {"film", "fnm", "hyper"}:
        features = build_time_feature_vector(global_idx.detach().cpu().float()).to(normalized_ssp.device)
        return model(normalized_ssp, features).squeeze(-1)
    return model(normalized_ssp, global_idx).squeeze(-1)


def metric_arrays(pred: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    from skimage.metrics import structural_similarity

    if pred.shape != target.shape or pred.ndim != 3:
        raise ValueError(f"Expected matching N x H x W fields, got {pred.shape}, {target.shape}")
    if not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError("Non-finite predictions or targets cannot be evaluated.")
    error = pred - target
    flat_error, flat_target = error.reshape(len(target), -1), target.reshape(len(target), -1)
    arrays = {
        "rmse": np.sqrt(np.mean(error ** 2, axis=(1, 2))),
        "rel_l2": np.linalg.norm(flat_error, axis=1) / np.maximum(np.linalg.norm(flat_target, axis=1), 1e-12),
        "ssim": np.asarray([structural_similarity(t, p, data_range=max(float(t.max() - t.min()), 1e-6))
                            for p, t in zip(pred, target)]),
        "pcc": np.asarray([np.corrcoef(p.ravel(), t.ravel())[0, 1] for p, t in zip(pred, target)]),
    }
    if not all(np.isfinite(values).all() for values in arrays.values()):
        raise ValueError("A metric is undefined or non-finite; check for constant fields.")
    return arrays


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_directory(args: argparse.Namespace) -> Path:
    return args.output_dir / args.variant / f"seed_{args.seed}"


def training_settings(args: argparse.Namespace) -> dict:
    settings = {name: getattr(args, name) for name in MODEL_DEFAULTS}
    if min(settings["modes1"], settings["modes2"], settings["width"]) < 1 or settings["padding"] < 0:
        raise ValueError("Modes and width must be positive; padding must be nonnegative.")
    if not 0 <= settings["dropout"] < 1:
        raise ValueError("Dropout must be in [0,1).")
    return settings


def load_data_contract(args: argparse.Namespace) -> tuple[dict, torch.Tensor, torch.Tensor, dict]:
    if args.raw_ssp is None or args.raw_tl is None:
        raise ValueError("Training and evaluation require --raw-ssp and --raw-tl.")
    manifest = load_monthly_manifest(args.manifest)
    identity = {
        "ssp_sha256": sha256_file(args.raw_ssp), "tl_sha256": sha256_file(args.raw_tl),
        "split_sha256": manifest_sha256(manifest),
    }
    x, y = load_global_data(args.raw_ssp, args.raw_tl)
    return manifest, x, y, identity


def train(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.batch_size < 1 or args.step_size < 1:
        raise ValueError("Epochs, batch size, and scheduler step size must be positive.")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.gamma <= 0:
        raise ValueError("Learning rate and gamma must be positive; weight decay must be nonnegative.")
    settings = training_settings(args)
    device = resolve_device(args.device)
    out = run_directory(args)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Run directory is nonempty: {out}. Choose another --output-dir.")
    manifest, x, y, identity = load_data_contract(args)
    index = np.asarray(manifest["train_idx"], dtype=np.int64)
    x_train, y_train = x[index].unsqueeze(-1), y[index]
    xn, yn = UnitGaussianNormalizer(x_train), UnitGaussianNormalizer(y_train)
    out.mkdir(parents=True, exist_ok=True)
    save_normalizers(out / "normalizer.pt", xn, yn,
                     {"fit": "training samples only", "ntrain": len(index), **identity})
    x_train, y_train = xn.encode(x_train), yn.encode(y_train)
    # Seed and model construction order match the corresponding source runners.
    set_seed(args.seed)
    dataset = torch.utils.data.TensorDataset(x_train, torch.from_numpy(index), y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                        generator=torch.Generator().manual_seed(args.seed))
    model = make_model(args.variant, settings, tuple(x.shape[1:])).to(device)
    count = sum(parameter.numel() for parameter in model.parameters())
    if settings == MODEL_DEFAULTS and count != PAPER_PARAMETER_COUNTS[args.variant]:
        raise RuntimeError(f"Unexpected parameter count: {count}")
    config = {
        "variant": args.variant, "label": LABELS[args.variant], "seed": args.seed,
        "epochs": args.epochs, "model": settings, "parameters": count,
        "shape": list(x.shape[1:]), "batch_size": args.batch_size,
        "optimizer": "Adam", "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "scheduler": {"type": "StepLR", "step_size": args.step_size, "gamma": args.gamma},
        "loss": "H1Loss(d=2,beta=0.01)", "checkpoint_policy": "final epoch; no test selection",
        "raw_ssp": str(args.raw_ssp.resolve()), "raw_tl": str(args.raw_tl.resolve()),
        "manifest": str(args.manifest.resolve()) if args.manifest else "built-in canonical monthly split",
        "split_counts": {key: len(manifest[f"{key}_idx"]) for key in ("train", "test", "purge")},
        "identity": identity, "environment": environment(device),
        "code_version": __version__, "normalizer_file_sha256": sha256_file(out / "normalizer.pt"),
    }
    _write_json(out / "config.json", config)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    loss_fn = H1Loss(d=2, beta=0.01)
    started, history = perf_counter(), []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        epoch_start = perf_counter()
        for xb, ib, yb in loader:
            xb, ib, yb = xb.to(device), ib.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(forward(model, args.variant, xb, ib), yb)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}.")
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu())
        scheduler.step()
        row = {"epoch": epoch, "seconds": perf_counter() - epoch_start,
               "train_loss_sum_per_sample": total / len(index)}
        history.append(row)
        _write_json(out / "history.json", {"epochs": history})
        print(json.dumps(row), flush=True)
    checkpoint = {"state_dict": model.state_dict(), "config": config, "epoch": args.epochs,
                  "config_sha256": config_hash(config)}
    if args.variant == "time2vec_film":
        checkpoint["time_encoding"] = model.time_encoder.summary()
    torch.save(checkpoint, out / "checkpoint.pt")
    _write_json(out / "training_summary.json", {
        "final_epoch": args.epochs, "elapsed_seconds": perf_counter() - started,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "parameters": count, "checkpoint": "checkpoint.pt", "final": history[-1],
    })
    print(f"Saved final checkpoint: {out / 'checkpoint.pt'}", flush=True)


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    out = run_directory(args)
    checkpoint = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if checkpoint["config_sha256"] != config_hash(config):
        raise ValueError("Checkpoint configuration hash is inconsistent.")
    if config["variant"] != args.variant or config["seed"] != args.seed:
        raise ValueError("Checkpoint variant or seed does not match the requested run.")
    if sha256_file(out / "normalizer.pt") != config["normalizer_file_sha256"]:
        raise ValueError("Normalizer file differs from the one recorded by the training checkpoint.")
    manifest, x, y, identity = load_data_contract(args)
    if identity != config["identity"]:
        raise ValueError("Raw data or split differs from the training checkpoint's recorded hashes.")
    xn, yn = load_normalizers(out / "normalizer.pt")
    normalizer_record = torch.load(out / "normalizer.pt", map_location="cpu", weights_only=True)
    if any(normalizer_record["meta"].get(k) != v for k, v in identity.items()):
        raise ValueError("Normalizer is not bound to this checkpoint's data and split.")
    model = make_model(args.variant, config["model"], tuple(config["shape"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    yn.mean, yn.std = yn.mean.to(device), yn.std.to(device)
    test_idx = np.asarray(manifest["test_idx"], dtype=np.int64)
    predictions = []
    for start in range(0, len(test_idx), args.eval_batch_size):
        ids = test_idx[start:start + args.eval_batch_size]
        xb = xn.encode(x[ids].unsqueeze(-1)).to(device)
        ib = torch.from_numpy(ids).to(device)
        predictions.append(yn.decode(forward(model, args.variant, xb, ib)).cpu())
    arrays = metric_arrays(torch.cat(predictions).numpy(), y[test_idx].numpy())
    report = {
        "variant": args.variant, "label": LABELS[args.variant], "seed": args.seed,
        "epoch": checkpoint["epoch"], "parameters": config["parameters"],
        "identity": identity, "ntest": len(test_idx),
        "config_sha256": checkpoint["config_sha256"],
        "normalizer_file_sha256": config["normalizer_file_sha256"],
        "code_version": __version__, "paper_settings": paper_settings(config, args.variant),
        "training_config": {key: config[key] for key in (
            "model", "epochs", "batch_size", "optimizer", "learning_rate", "weight_decay",
            "scheduler", "loss", "parameters", "shape")},
        "metrics": {key: float(values.mean()) for key, values in arrays.items()},
        "definitions": {"aggregation": "unweighted mean of per-sample metrics",
                        "ssim": "skimage defaults; target max-min data_range, minimum 1e-6",
                        "pcc": "Pearson correlation over flattened physical TL fields"},
        "environment": environment(device),
    }
    if args.variant == "time2vec_film":
        report["time_encoding"] = model.time_encoder.summary()
    np.savez_compressed(out / "per_sample_metrics.npz", test_idx=test_idx, **arrays)
    _write_json(out / "metrics.json", report)
    print(json.dumps(report, indent=2), flush=True)


def aggregate(args: argparse.Namespace) -> None:
    groups = {"table_i": (MAIN_VARIANTS, (1,)), "table_ii": (TIME_VARIANTS, (1, 2, 3))}
    selected = groups if args.table == "all" else {args.table: groups[args.table]}
    report = {"tables": {}, "identity": None}
    for table, (variants, seeds) in selected.items():
        report["tables"][table] = {}
        for variant in variants:
            records = [json.loads((args.output_dir / variant / f"seed_{seed}" / "metrics.json").read_text())
                       for seed in seeds]
            for seed, record in zip(seeds, records):
                if record["variant"] != variant or record["seed"] != seed or record["epoch"] != 100:
                    raise ValueError("Paper aggregation requires the named seeds and epoch-100 checkpoints.")
                if record["ntest"] != 584:
                    raise ValueError("Paper aggregation requires all 584 canonical test samples.")
                if not record["paper_settings"] or not paper_settings(record["training_config"], variant):
                    raise ValueError("Paper aggregation requires the published architecture and training settings.")
                if record["identity"]["split_sha256"] != manifest_sha256(load_monthly_manifest()):
                    raise ValueError("Paper aggregation requires the canonical monthly split hash.")
                if report["identity"] is None:
                    report["identity"] = record["identity"]
                if record["identity"] != report["identity"]:
                    raise ValueError("Cannot aggregate runs with different raw-data or split hashes.")
            report["tables"][table][variant] = {
                "label": LABELS[variant], "seeds": list(seeds),
                "mean": {key: float(np.mean([r["metrics"][key] for r in records])) for key in METRIC_KEYS},
                "std": {key: (float(np.std([r["metrics"][key] for r in records], ddof=1))
                              if len(records) > 1 else None) for key in METRIC_KEYS},
                "per_seed": [{"seed": r["seed"], **r["metrics"]} for r in records],
            }
    _write_json(args.output_dir / f"aggregate_{args.table}.json", report)
    print(json.dumps(report, indent=2), flush=True)


def smoke(args: argparse.Namespace) -> None:
    """Exercise every released model with small synthetic fields and state roundtrips."""
    device = resolve_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(2, torch.get_num_threads()))
    settings = {"modes1": 2, "modes2": 3, "width": 4, "padding": 2, "dropout": 0.1}
    results = {}
    for variant in VARIANTS if args.variant == "all" else (args.variant,):
        set_seed(args.seed)
        x, target = torch.randn(2, 12, 16, 1), 70 + torch.randn(2, 12, 16)
        xn, yn = UnitGaussianNormalizer(x), UnitGaussianNormalizer(target)
        xb, yb = xn.encode(x).to(device), yn.encode(target).to(device)
        indices = torch.tensor([10, 1830], device=device)
        model = make_model(variant, settings, (12, 16)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss = H1Loss(d=2, beta=0.01)(forward(model, variant, xb, indices), yb)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite smoke loss for {variant}.")
        loss.backward()
        if variant == "time2vec_film":
            for name, parameter in model.time_encoder.named_parameters():
                if parameter.grad is None or not torch.isfinite(parameter.grad).all() or not torch.count_nonzero(parameter.grad):
                    raise RuntimeError(f"Invalid learned-frequency gradient: {name}")
        optimizer.step()
        model.eval()
        with torch.no_grad():
            pred = forward(model, variant, xb, indices)
            buffer = io.BytesIO()
            torch.save({"state_dict": model.state_dict(), "epoch": 1}, buffer)
            buffer.seek(0)
            restored = make_model(variant, settings, (12, 16)).to(device)
            restored.load_state_dict(torch.load(buffer, map_location=device, weights_only=True)["state_dict"])
            restored.eval()
            torch.testing.assert_close(forward(restored, variant, xb, indices), pred, rtol=0, atol=0)
            yn.mean, yn.std = yn.mean.to(device), yn.std.to(device)
            metrics = metric_arrays(yn.decode(pred).cpu().numpy(), target.numpy())
        results[variant] = {"shape": list(pred.shape), "loss": float(loss.detach().cpu()),
                            "checkpoint_roundtrip": True, "parameters": sum(p.numel() for p in model.parameters()),
                            "metrics": {key: float(value.mean()) for key, value in metrics.items()}}
    print(json.dumps({"synthetic_smoke_only": True, "device": str(device), "models": results}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "evaluate"):
        p = sub.add_parser(command)
        p.add_argument("--variant", required=True, choices=VARIANTS)
        p.add_argument("--seed", type=int, default=1)
        p.add_argument("--device", default="cuda")
        p.add_argument("--output-dir", type=Path, default=Path("artifacts"))
        p.add_argument("--raw-ssp", type=Path)
        p.add_argument("--raw-tl", type=Path)
        p.add_argument("--manifest", type=Path)
        if command == "evaluate":
            p.add_argument("--eval-batch-size", type=int, default=8)
        else:
            p.add_argument("--epochs", type=int, default=100)
            p.add_argument("--batch-size", type=int, default=20)
            p.add_argument("--learning-rate", type=float, default=1e-3)
            p.add_argument("--weight-decay", type=float, default=1e-4)
            p.add_argument("--step-size", type=int, default=50)
            p.add_argument("--gamma", type=float, default=0.5)
            for name, value in MODEL_DEFAULTS.items():
                p.add_argument(f"--{name}", type=type(value), default=value)
    p = sub.add_parser("aggregate")
    p.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    p.add_argument("--table", choices=("table_i", "table_ii", "all"), default="all")
    p = sub.add_parser("smoke")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--variant", choices=("all",) + VARIANTS, default="all")
    args = parser.parse_args()
    if args.command == "evaluate" and args.eval_batch_size < 1:
        parser.error("--eval-batch-size must be positive")
    {"train": train, "evaluate": evaluate, "aggregate": aggregate, "smoke": smoke}[args.command](args)


if __name__ == "__main__":
    main()
