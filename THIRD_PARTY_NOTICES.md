# Third-party notices

## Fourier Neural Operator reference implementation

The `SpectralConv2d` and `FNO2d` building blocks in
`src/stfm_fno/models/fno2d.py` and `src/stfm_fno/models/fno2d_film.py`, the
`LpLoss` utility, and `UnitGaussianNormalizer` in
`src/stfm_fno/normalizer.py` are adapted from the Fourier Neural Operator
reference implementation associated with Li et al. (ICLR 2021). This project
adds temporal conditioning, experiment protocols, and acoustic data handling.

The historical upstream URL,
<https://github.com/zongyi-li/fourier_neural_operator>, redirects to the
[official NeuralOperator repository](https://github.com/neuraloperator/neuraloperator).
The exact historical revision used when these building blocks were first
adapted has not been recorded. The notice therefore does not assert that the
local code was copied from a particular commit of the present library.

The upstream MIT copyright notice is **Copyright (c) 2023 NeuralOperator
developers**. Its complete license is retained in
[`LICENSES/neuraloperator-MIT.txt`](LICENSES/neuraloperator-MIT.txt).
The license text was verified on 2026-10-13 against
[the upstream license commit](https://github.com/neuraloperator/neuraloperator/blob/951f55280b463e3f12ad4085b63bc673eb36d48c/LICENSE)
and the current `main` branch. Its SHA-256 is
`5f335b0603acfb1d180bf3117492e2076562075cc9b6353a3b94d5d5bc03ef33`.

Reference: Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu,
Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. *Fourier Neural
Operator for Parametric Partial Differential Equations*. ICLR, 2021.
[Paper](https://arxiv.org/abs/2010.08895).

## External data and RAM

The study uses acoustic data derived from the 3DSSF work of Siyuan Li, Lei
Cheng, Jun Li, Zichen Wang, and Jianlong Li, *Learning data distribution of
three-dimensional ocean sound speed fields via diffusion models*, JASA
155(5), 3410–3425, 2024. The authors provide dataset access through the
[DiffusionSSF repository](https://github.com/OceanSTARLab/DiffusionSSF).
Its repository code is licensed under Apache-2.0; that fact alone does not
establish a license for every externally hosted dataset or for this study's
derived `SSP.mat` and `TL.mat` files. Those files are not distributed in this
source release. Obtain the study-specific data and any redistribution
permission from the relevant data owners.

RAM is the Range-dependent Acoustic Model developed by Michael D. Collins.
Its split-step Padé method is described in *A split-step Padé solution for
the parabolic equation method*, JASA 93(4), 1736–1742, 1993.
The [Ocean Acoustics Library](https://oalib-acoustics.org/website_resources/PE/archive/)
provides RAM resources. RAM executables and external MATLAB/Fortran solver
sources are not bundled with this release, and the project MIT license does
not apply to them. Precomputed TL targets are sufficient to run the released
learning and evaluation workflow.

## Installed dependencies

Python dependencies are installed separately. Their own licenses and
copyright notices apply; this project's MIT license does not replace them.
