# Training and evaluation

This guide describes the experimental settings used in the published STFM-FNO paper. Install the package following the [README](../README.md#installation) and prepare the paired files described in [DATA.md](DATA.md). Paper settings and chronological indices are saved in [configs/paper.json](../configs/paper.json) and [configs/monthly_split.json](../configs/monthly_split.json).

## Experimental settings

| Setting | Value |
|---|---|
| Main comparison (Table I) | Six models; seed 1; final epoch 100 |
| Temporal encodings (Table II) | Three encodings; seeds 1, 2, 3; final epoch 100 |
| Split | 2144 train / 584 test / 192 purge |
| Input normalization | Per-grid-point mean/sample std over train only; epsilon 1e-5 |
| Target normalization | Same rule, fitted on train TL only |
| FNO | Four blocks; width 64; depth/range modes 32/128; right/bottom zero padding 20 |
| Projection | 64 → 128 → 1; GELU then dropout 0.1 before output |
| FiLM generator | Four independent MLPs, 24 → 64 → 128 with hidden ReLU |
| Optimizer | Adam, lr 0.001, weight decay 0.0001 |
| Scheduler | StepLR, step_size 50, gamma 0.5, stepped after each epoch |
| Loss | Sum of per-sample relative L2, plus 0.01 times the relative L2 of each first grid difference |
| Batch/epochs | 20 / 100; shuffled train batches with seeded generator |

The monthly split selects a central test block of seven days in February and six days in every other month, with one day purged on each side. Global sample indices retain their original chronological positions. The manifest is checked against the canonical indices, whose SHA-256 is `1370f6e47141e577aca3b29de36a2af7b5408cddd4e5c753d57f81f89a55fb68` over the concatenated little-endian int64 `train_idx`, `test_idx`, and `purge_idx` arrays.

The loss is applied in standardized TL space. First differences are unscaled grid differences. Relative norms in the training loss have no denominator clamp; non-finite losses raise an error. Evaluation uses the final epoch-100 checkpoint.

## Commands

Run the commands below from the repository root. For a single STFM-FNO experiment:

```bash
python -m stfm_fno.experiments train \
  --variant film --seed 1 --epochs 100 --device cuda:0 \
  --raw-ssp raw_data/SSP.mat --raw-tl raw_data/TL.mat \
  --manifest configs/monthly_split.json --output-dir outputs/table1
python -m stfm_fno.experiments evaluate \
  --variant film --seed 1 --device cuda:0 \
  --raw-ssp raw_data/SSP.mat --raw-tl raw_data/TL.mat \
  --manifest configs/monthly_split.json --output-dir outputs/table1
```

To run all models in Table I and all temporal encodings in Table II, use separate output directories:

```bash
bash scripts/reproduce_table1.sh raw_data/SSP.mat raw_data/TL.mat outputs/table1_all cuda:0
bash scripts/reproduce_table2.sh raw_data/SSP.mat raw_data/TL.mat outputs/table2_all cuda:0
```

Each script trains, evaluates, and aggregates its experiment matrix sequentially. For manual aggregation:

```bash
python -m stfm_fno.experiments aggregate --output-dir outputs/table1_all --table table_i
python -m stfm_fno.experiments aggregate --output-dir outputs/table2_all --table table_ii
```

## Temporal encodings and model details

The proposed 24-channel vector concatenates daily sine/cosine harmonics 1–4 and yearly sine/cosine harmonics 1–8. Its implemented phases are `(t % 8)/8` and `t/2919` for zero-based global index `t`. Generic Fourier encoding uses 12 log-spaced frequencies from 12 to 1200 with the yearly normalized phase.

Learned-frequency-FiLM (`time2vec_film`) uses `[a0*u+b0, sin(2*pi*f_j*u+p_j)]`, where `u=t/2919`, with 23 learned periodic channels. Frequencies are parameterized as `1460*sigmoid(theta)` and initialized log-spaced from 1 to 1200. It adds 48 parameters to STFM-FNO.

The paper's yearly-encoding equation uses 2920, while the experiments and this implementation use 2919. The paper's block equation presents a GELU after modulation; the implementation applies block GELU after the first three blocks and sends the fourth directly to the output projection, which has its own GELU.

FNM-FNO uses four auxiliary fields generated from 128 fixed real Fourier basis functions and a hidden width of 72. Low-rank Hyper-FNO uses complex CP rank 16 and a global time hypernetwork with hidden width 18. These are the parameter-matched comparison models used in the paper.

| Model | CLI variant | Parameter count |
|---|---|---:|
| Baseline FNO | `baseline` | 134243073 |
| Concat-Scalar | `scalar` | 134243137 |
| Concat-PE | `pe` | 134244609 |
| FNM-FNO | `fnm` | 134282505 |
| Low-rank Hyper-FNO | `hyper` | 134282819 |
| STFM-FNO | `film` | 134282753 |

Counts use PyTorch `numel`, with each complex element counted once.

## Outputs

Each run writes `OUTPUT/VARIANT/seed_SEED/` containing `config.json`, `normalizer.pt`, `history.json`, `checkpoint.pt`, and `training_summary.json`. Evaluation adds `metrics.json` and `per_sample_metrics.npz` with the frozen test indices. The checkpoint stores model state, configuration, and epoch; evaluation verifies the data, split, and normalizer identities. Use a new output directory for each fresh training run.

Aggregation writes `aggregate_table_i.json` or `aggregate_table_ii.json` under the corresponding output directory. The [reference_results](../reference_results/) directory contains the recorded measurements from the paper experiments.

## Metrics

Predictions are decoded to dB before computing each metric per field:

- RMSE: square root of mean squared error over the 199 × 800 grid.
- Relative L2: Euclidean error norm divided by the target norm, with denominator at least 1e-12.
- SSIM: scikit-image defaults with `data_range=max(target.max()-target.min(), 1e-6)` for each target field.
- PCC: Pearson correlation over flattened field values; undefined correlations raise an error.

Table I averages the 584 per-field metric values. Table II first averages fields within each seed, then averages the three seed-level results. Its sample standard deviations use `ddof=1`.

The paper displays FNM-FNO PCC as `0.947`; its full-precision value in the reference JSON is `0.946498973425821`.

## Dataset availability

The paired dataset and trained checkpoints are not bundled. The 100-Hz result reported in the paper requires its corresponding RAM labels; the commands above use the main 200-Hz dataset.
