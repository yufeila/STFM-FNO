"""Deterministic monthly blocked split with a one-day purge on either side."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .config import NTOTAL, SAMPLES_PER_DAY, SPLIT_NAME

MONTH_LENGTHS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def build_monthly_manifest():
    test_days, test_ranges = set(), []
    start = 0
    for month, length in enumerate(MONTH_LENGTHS):
        block_length = 7 if month == 1 else 6
        first = start + (length - block_length) // 2
        test_ranges.append([first, first + block_length])
        test_days.update(range(first, first + block_length))
        start += length
    purge_days = {day + delta for day in test_days for delta in (-1, 1)
                  if 0 <= day + delta < 365} - test_days
    train_days = set(range(365)) - test_days - purge_days
    def samples(days):
        return np.array([day * SAMPLES_PER_DAY + slot for day in sorted(days)
                         for slot in range(SAMPLES_PER_DAY)], dtype=np.int64)
    return {"split": SPLIT_NAME, "purge_days": 1, "nt_total": NTOTAL,
            "ntrain": 2144, "ntest": 584, "npurge": 192,
            "train_idx": samples(train_days), "test_idx": samples(test_days),
            "purge_idx": samples(purge_days), "test_ranges": test_ranges}


def load_monthly_manifest(path=None):
    canonical = build_monthly_manifest()
    if path is None:
        return canonical
    path = Path(path)
    if path.suffix == ".json":
        manifest = json.loads(path.read_text())
    else:
        with np.load(path, allow_pickle=False) as data:
            manifest = {key: data[key] for key in data.files}
    for key in ("ntrain", "ntest", "npurge"):
        if int(manifest[key]) != canonical[key]:
            raise ValueError(f"{path}: incorrect {key}")
    for key in ("train_idx", "test_idx", "purge_idx"):
        if not np.array_equal(np.asarray(manifest[key]), canonical[key]):
            raise ValueError(f"{path}: {key} differs from the frozen paper split")
    return canonical


def manifest_sha256(manifest):
    digest = hashlib.sha256()
    for key in ("train_idx", "test_idx", "purge_idx"):
        digest.update(np.asarray(manifest[key], dtype="<i8").tobytes())
    return digest.hexdigest()


def save_monthly_manifest(path):
    manifest = build_monthly_manifest()
    serial = {key: value.tolist() if isinstance(value, np.ndarray) else value
              for key, value in manifest.items()}
    serial["index_sha256"] = manifest_sha256(manifest)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(serial, indent=2) + "\n")
