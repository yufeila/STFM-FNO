# Release validation — October 13, 2026

The release was checked in the existing Linux Python 3.10.20 / PyTorch 2.5.1 environment, using CPU and two CPU threads for numerical checks. No GPU training was started and no dependencies were installed into the existing environment.

Passed checks:

- All nine released variants: small-grid forward pass, finite H1 loss, backward pass, optimizer step, metric calculation, and exact checkpoint reload agreement. All four Learned-frequency encoder parameter groups had finite nonzero gradients.
- All nine default paper architectures: exact recorded PyTorch parameter counts, using the meta device to avoid allocating full networks.
- Baseline and FiLM backbones: exact state initialization and exact forward-output agreement with the original experiment implementations on identical small inputs, including a batch of size one.
- Frozen split: every train/test/purge index equals the historical NPZ manifest. Counts are 2144/584/192 and the documented index SHA-256 matches.
- Actual research files: keys and logical shapes checked for SSP and TL; SSP loader values and axis order exactly match the original loader. The full TL finite check, chronology, and physical-generation metadata were not independently audited in this preparation.
- Known numerical example: a constant 2-dB offset gives RMSE 2 and PCC 1.
- Complete training/evaluation functions: one epoch on generated small-grid fields with all 2920 canonical sample indices, followed by evaluation of all 584 test indices. These are synthetic integration results, not paper metrics.
- Protection checks: a nonempty run cannot be overwritten; changed data or normalizer files are rejected during evaluation.
- Packaging: wheel built successfully without downloading dependencies, installed into an isolated target directory, and all nine smoke models passed through the installed package. The main and third-party license notices were present in the wheel.
- Syntax and metadata: Python AST parsing, shell syntax, TOML, JSON, YAML, and citation parsing passed.

The recorded paper tables were not retrained, so this validation does not establish numerical reproduction of their reported accuracy. Validation identity and original source snapshot hashes are recorded in `RELEASE_PROVENANCE.json`.
