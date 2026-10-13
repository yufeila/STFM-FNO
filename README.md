# STFM-FNO

Official implementation of [Spatio-Temporal Field-Modulated Fourier Neural Operator for Transmission Loss Estimation](https://ieeexplore.ieee.org/document/11723170).

STFM-FNO estimates underwater acoustic transmission loss from sound-speed fields using Fourier-encoded time to modulate latent spatial features.

[Project page](https://yufeila.github.io/STFM-FNO/) · [Training guide](docs/REPRODUCIBILITY.md) · [Data specification](docs/DATA.md)

## Method overview

STFM-FNO separates spatial feature extraction from temporal conditioning. The spatial branch processes the sound-speed field and range-depth coordinates through an FNO backbone, while the temporal branch encodes acquisition time with daily and yearly Fourier features. An MLP in each block maps these time features to channel-wise scale and shift parameters, which modulate the latent spatial features. The network then projects the conditioned features to a transmission-loss field.

![STFM-FNO architecture](docs/figures/figure_1_architecture.png)

**Figure 1.** STFM-FNO architecture with Fourier time encoding and feature-wise affine modulation. [Original PDF](docs/figures/figure_1_architecture.pdf).

## Installation

Use Python 3.10 and PyTorch 2.5.1. A CUDA GPU is recommended for training.

```bash
git clone https://github.com/yufeila/STFM-FNO.git
cd STFM-FNO
conda env create -f environment.yml
conda activate stfm_fno
pip install -e .
```

For CUDA 12.1:

```bash
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

## Data

The study uses the 3DSSF dataset with 2,920 sound-speed snapshots, sampled eight times per day over one year, and corresponding RAM-generated transmission-loss fields. Place the paired MATLAB files in `raw_data/`:

```text
raw_data/
├── SSP.mat   # square_SSP: (2920, 21, 10), sound speed in m/s
└── TL.mat    # square_TL:  (2920, 199, 800), transmission loss in dB
```

The dimensions above follow the loader's sample–depth–range order. See [data specification](docs/DATA.md) for the file format and dataset reference.

## Results

### TL predictions and errors

![Reference TL, Baseline FNO and STFM-FNO predictions, and absolute errors](docs/figures/figure_2_tl_predictions.png)

**Figure 2.** TL estimates and absolute errors for a monthly test sample (`t = 1830`, August 17). The black line marks the bottom at 180 m. [Original PDF](docs/figures/figure_2_tl_predictions.pdf).

**Table I.** Performance on the 584 test fields, using seed 1.

| Model | RMSE (dB) ↓ | Rel. L2 ↓ | SSIM ↑ | PCC ↑ |
|---|---:|---:|---:|---:|
| Baseline FNO | 4.15 | 0.052 | 0.787 | 0.952 |
| Concat (Scalar) | 4.22 | 0.053 | 0.781 | 0.951 |
| Concat (PE) | 4.77 | 0.059 | 0.734 | 0.938 |
| FNM-FNO | 4.42 | 0.055 | 0.755 | 0.947 |
| Low-rank Hyper-FNO | 4.14 | 0.052 | 0.787 | 0.952 |
| **STFM-FNO** | **3.95** | **0.049** | **0.800** | **0.956** |

### Temporal encoding

**Table II.** Mean RMSE across seeds 1, 2, and 3.

| Temporal encoding | RMSE (dB) ↓ |
|---|---:|
| Generic-Fourier-FiLM | 3.96 |
| Learned-frequency-FiLM | 3.91 |
| Ours (daily + yearly) | 3.90 |

The tables show the paper's reported values. Full-precision metrics are available in [reference_results](reference_results/).

### Inspecting temporal modulation

![First-block feature interception with a fixed sound-speed field and winter or summer conditioning](docs/figures/figure_3_feature_modulation.png)

**Figure 3.** The same sound-speed field is evaluated with winter (`t = 360`) and summer (`t = 1800`) conditioning. The top row shows the shared input; the following rows show unmodulated features and latent modulation fields in the first FNO block. [Original PDF](docs/figures/figure_3_feature_modulation.pdf).

## Repository layout

```text
src/stfm_fno/          Models, data utilities, and training/evaluation CLI
scripts/              Experiment and data-check helpers
configs/              Paper settings and monthly split indices
reference_results/    Paper experiment metrics
docs/                 Paper, figures, and detailed usage documentation
```

## Acknowledgments

This work was supported by the Zhejiang Provincial Natural Science Foundation of China (LR26F010004), the National Natural Science Foundation of China (62622127, 62371418), the Fundamental Research Funds for the Central Universities (226-2025-00168), and the Novo Nordisk Foundation (NNF24OC0089302).

## Citation and license

If you use this code, please cite our paper:

```bibtex
@article{yang2026stfmfno,
  title   = {Spatio-Temporal Field-Modulated Fourier Neural Operator for Transmission Loss Estimation},
  author  = {Yang, Yufei and Sun, Yifan and Cheng, Lei and Gerstoft, Peter},
  journal = {IEEE Signal Processing Letters},
  year    = {2026},
  doi     = {10.1109/LSP.2026.3740795},
  url     = {https://ieeexplore.ieee.org/document/11723170},
  note    = {Early Access}
}
```

The code is released under the [MIT License](LICENSE). See [third-party notices](THIRD_PARTY_NOTICES.md) for source attribution.
