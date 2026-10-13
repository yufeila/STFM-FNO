"""Load chronological MATLAB SSP/TL pairs and interpolate SSP to the TL grid."""
from pathlib import Path
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from .config import NTOTAL, SSP_HEIGHT, SSP_WIDTH, TL_HEIGHT, TL_WIDTH


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_raw_mat_dataset(path, dataset_name):
    import h5py
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required data file is missing: {path}")
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as handle:
            if dataset_name not in handle:
                raise KeyError(f"{path}: missing {dataset_name}")
            array = handle[dataset_name][()]
        array = array.transpose(tuple(range(array.ndim - 1, -1, -1)))
    else:
        from scipy.io import loadmat
        array = loadmat(path, variable_names=[dataset_name])[dataset_name]
    array = np.asarray(array, dtype=np.float32)
    expected = (NTOTAL, SSP_HEIGHT, SSP_WIDTH) if dataset_name == "square_SSP" else (NTOTAL, TL_HEIGHT, TL_WIDTH)
    if array.shape != expected:
        raise ValueError(f"{path}:{dataset_name}: got {array.shape}, expected {expected}; check MATLAB axis order")
    for start in range(0, NTOTAL, 32):
        if not np.isfinite(array[start:start + 32]).all():
            raise ValueError(f"{path}:{dataset_name}: non-finite values near sample {start}")
    return np.ascontiguousarray(array)


def interpolate_ssp_to_tl_grid(ssp):
    if ssp.shape != (NTOTAL, SSP_HEIGHT, SSP_WIDTH):
        raise ValueError(f"Unexpected SSP shape: {ssp.shape}")
    result = np.empty((NTOTAL, TL_HEIGHT, TL_WIDTH), dtype=np.float32)
    for start in range(0, NTOTAL, 32):
        tensor = torch.from_numpy(ssp[start:start + 32]).unsqueeze(1)
        result[start:start + len(tensor)] = F.interpolate(
            tensor, size=(TL_HEIGHT, TL_WIDTH), mode="bilinear", align_corners=True
        )[:, 0].numpy()
    return result


def load_global_data(raw_ssp, raw_tl):
    x = interpolate_ssp_to_tl_grid(load_raw_mat_dataset(raw_ssp, "square_SSP"))
    y = load_raw_mat_dataset(raw_tl, "square_TL")
    return torch.from_numpy(x), torch.from_numpy(y)
