#!/usr/bin/env python3
"""Inspect the public data contract without allocating interpolated full-year fields."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from scipy.io import whosmat, loadmat


def check(path, key, expected, full):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as f:
            ds = f[key]
            shape = tuple(reversed(ds.shape))
            if shape != expected:
                raise ValueError(f"{path}:{key}: logical shape {shape}, expected {expected}")
            if not np.issubdtype(ds.dtype, np.number):
                raise ValueError(f"{path}:{key}: expected a numeric dataset")
            if full:
                for start in range(0, expected[0], 32):
                    if not np.isfinite(ds[..., start:start + 32]).all():
                        raise ValueError(f"{path}:{key}: non-finite values near sample {start}")
    else:
        entries = {name: shape for name, shape, _ in whosmat(path)}
        shape = entries.get(key)
        if shape != expected:
            raise ValueError(f"{path}:{key}: logical shape {shape}, expected {expected}")
        if full and not np.isfinite(loadmat(path, variable_names=[key])[key]).all():
            raise ValueError(f"{path}:{key}: non-finite values")
    record = {"file": str(path), "key": key, "logical_shape": list(expected), "finite_checked": full}
    if full:
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        record["file_sha256"] = h.hexdigest()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-ssp", type=Path, required=True)
    parser.add_argument("--raw-tl", type=Path, required=True)
    parser.add_argument("--full", action="store_true", help="Stream finite checks and SHA-256 hashes")
    args = parser.parse_args()
    records = [check(args.raw_ssp, "square_SSP", (2920, 21, 10), args.full),
               check(args.raw_tl, "square_TL", (2920, 199, 800), args.full)]
    print(json.dumps({"files": records, "chronology_and_pairing_verified": False}, indent=2))


if __name__ == "__main__":
    main()
