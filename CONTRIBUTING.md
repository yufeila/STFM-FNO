# Contributing

Please describe the scientific or implementation issue and provide a minimal reproducible example, software versions, the command used, and the relevant split/configuration. Use the repository issue tracker; do not attach private data or credentials.

Changes to model initialization, time encoding, normalization, the loss, split indices, or metric aggregation affect comparisons with the paper. Document such changes and distinguish new experiments from the recorded reference results. Keep the final-epoch checkpoint policy and train-only normalization explicit.

Do not commit raw data, model checkpoints, generated predictions, machine-specific paths, or caches. Preserve applicable license and attribution notices. Run `python -m stfm_fno.experiments smoke --device cpu` before submitting an implementation change.
