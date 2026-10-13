# Dataset

The experiments use 2,920 chronologically ordered sound-speed fields (SSF) and paired RAM-computed transmission-loss (TL) fields: 365 days × eight samples per day. The main acoustic configuration uses a 200-Hz source at 50-m depth over 0–200 m depth and 0–40 km range.

Place the paired MATLAB files in a local data directory:

```text
raw_data/
├── SSP.mat
└── TL.mat
```

| File | MATLAB key | Logical shape `(sample, depth, range)` | Units |
|---|---|---|---|
| `SSP.mat` | `square_SSP` | `(2920, 21, 10)` | m/s |
| `TL.mat` | `square_TL` | `(2920, 199, 800)` | dB |

MATLAB v7.3 stores reversed dimensions in HDF5: SSP `(10, 21, 2920)` and TL `(800, 199, 2920)`. The loader restores their logical dimensions; conventional MATLAB files are read with SciPy. SSP is bilinearly interpolated to the TL grid using PyTorch `align_corners=True`, and both arrays are converted to float32.

Each global index `0...2919` must identify the matching SSP/TL pair at the original sampling time. Keep the chronological order and finite values; do not renumber samples after splitting. The paper's monthly blocked indices and training normalization are described in [the training and evaluation guide](REPRODUCIBILITY.md).

Check the file keys and shapes with:

```bash
python scripts/validate_data.py --raw-ssp raw_data/SSP.mat --raw-tl raw_data/TL.mat
```

Add `--full` to check finite values and calculate file hashes.

The paper's environmental dataset source is S. Li, L. Cheng, J. Li, Z. Wang, and J. Li, “Learning data distribution of three-dimensional ocean sound speed fields via diffusion models,” *The Journal of the Acoustical Society of America*, 155(5), 3410–3425, 2024. The study-specific paired SSP/TL files are not bundled; no public download for these pairs is provided. Dataset access and redistribution remain subject to the owners' terms. The software license applies to code.
