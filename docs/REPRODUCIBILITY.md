# Reproduction protocol and implementation details

## What can be reproduced

The release implements the final Table I comparison and Table II temporal-encoding comparison with the original chronological data supplied by the user. Reference JSON files are historical experiment summaries. No trained weights are included and no full table retraining was performed while preparing the release.

The canonical split is exactly equal to the historical frozen NPZ manifest. Its index SHA-256 is `1370f6e47141e577aca3b29de36a2af7b5408cddd4e5c753d57f81f89a55fb68`, computed over the concatenated little-endian int64 `train_idx`, `test_idx`, and `purge_idx` arrays in that order. A supplied manifest must match every canonical index, not only the counts.

## Experimental settings

| Setting | Value |
|---|---|
| Table I | Six variants; seed 1; final epoch 100 |
| Table II | Three encodings; seeds 1, 2, 3; final epoch 100 |
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

The loss is applied in standardized TL space. First differences are unscaled grid differences, rather than derivatives divided by physical grid spacing. Relative norms in the historical loss have no denominator clamp; non-finite losses are detected by the release. The reported relative L2 evaluation denominator is clamped to at least 1e-12.

Table I model parameter counts (PyTorch `numel`, counting each complex element once): baseline 134243073, scalar 134243137, PE concat 134244609, FNM-FNO 134282505, low-rank Hyper-FNO 134282819, STFM-FNO 134282753. These counts should not be interpreted as real scalar counts or memory consumption.

FNM-FNO uses four auxiliary fields generated from 128 fixed real Fourier basis functions and a hidden width of 72. Low-rank Hyper-FNO uses complex CP rank 16 and a global time hypernetwork with hidden width 18. These are the project's parameter-matched comparison implementations, not imported third-party model packages.

## Temporal representations

The proposed vector concatenates daily sine/cosine harmonics 1–4 and yearly sine/cosine harmonics 1–8. The implemented phases are `(t % 8)/8` and `t/2919` for zero-based global index `t`. Generic Fourier encoding uses 12 log-spaced frequencies from 12 to 1200 with the latter normalized phase.

Learned-frequency-FiLM (`time2vec_film`) uses `[a0*u+b0, sin(2*pi*f_j*u+p_j)]`, where `u=t/2919`, with 23 learned periodic channels. Frequencies are parameterized as `1460*sigmoid(theta)` and initialized log-spaced from 1 to 1200. It adds 48 parameters to the proposed model. The supplied project code calls this a Time2Vec control; the final paper uses the broader name Learned-frequency-FiLM.

## Paper/code differences retained for traceability

The paper's yearly-encoding equation uses a denominator of 2920, while the recorded experiments use 2919. The release retains 2919. The paper's block equation presents a GELU after modulation; the experiment code applies block GELU after the first three blocks, while the fourth is followed directly by the output projection (which has its own GELU). The release retains this architecture.

The historical Table I four-model runner logged test relative L2 after each epoch, but trained for a fixed 100 epochs and reported the final model. Conditional baselines and Table II evaluated the fixed test set after training. The consolidated release trains without consulting test metrics and evaluates on request; it does not retroactively claim that the original runs never inspected the test set.

The release also uses `.squeeze(-1)` to retain a batch dimension of size one, constructs time-concatenation channels per batch, loads raw chronological data directly, and records configuration/provenance with checkpoints. These fixes preserve the intended inputs and calculations, but the release does not guarantee bit-for-bit equivalence to all historical training scripts. Use the retained full-precision reference JSON to distinguish recorded measurements from reruns.

## Generated artifact layout

Each run writes `OUTPUT/VARIANT/seed_SEED/` with `config.json`, `normalizer.pt`, `history.json`, `checkpoint.pt`, and `training_summary.json`. Evaluation adds `metrics.json` and `per_sample_metrics.npz` with the frozen test indices. The final checkpoint stores model state plus its configuration and epoch; evaluation loads the network settings from that checkpoint and verifies data/split/normalizer identities. A nonempty training directory is rejected; select a new output directory for a fresh run.

Aggregation reads the completed epoch-100 evaluation files. For example, `python -m stfm_fno.experiments aggregate --output-dir outputs/table1 --table table_i` writes `outputs/table1/aggregate_table_i.json`; use `--table table_ii` for the three-seed matrix. Table scripts use separate roots so recorded results and newly computed results stay independently inspectable.

## Metrics and uncertainty

Predictions are decoded to dB before metrics are computed. Per field:

- RMSE: square root of mean squared error over the 199 × 800 grid.
- Relative L2: Euclidean error norm divided by the target norm, with denominator at least 1e-12.
- SSIM: scikit-image defaults with `data_range=max(target.max()-target.min(), 1e-6)` for each target field.
- PCC: Pearson correlation over flattened field values; undefined correlations are treated as errors.

Each Table I summary is the arithmetic mean of the 584 per-field values. Table II first averages over those fields for each seed, then averages the three seed-level results; reported standard deviations use `ddof=1`. This is not a confidence interval, a significance test, or a pooled pixel RMSE.

## Scope limits and running long jobs

The paper's frequency-change experiment requires its original 100-Hz RAM labels and complete generation settings. Existing cross-configuration exploratory scripts cannot by themselves establish that exact frequency-only comparison. This release does not bundle a solver, reproduce those labels, or recreate the final publication figures. The accepted manuscript is supplied for reference; its figures and numerical claims remain historical research outputs.

Run full table jobs on a suitable GPU server, preferably inside `tmux`, for example `tmux new -s stfm-table1` and then the README command. Monitor the terminal/log and output directory. Stop a run if data validation fails, loss becomes non-finite, or it exhausts available memory. The sequential table scripts avoid scheduling several full models on a single GPU. CPU smoke tests are deliberately small.
