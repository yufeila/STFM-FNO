# Data specification

The release consumes paired SSP/TL arrays; it does not download or generate the original experiment dataset.

| Property | Required value |
|---|---|
| Sample count | 2920, chronologically ordered, eight samples per day for a non-leap year |
| Sample index | Zero-based global index; do not renumber after splitting |
| SSP file/key | `SSP.mat`, `square_SSP` |
| SSP logical shape | `(sample, depth, range) = (2920, 21, 10)` |
| SSP units | m/s |
| TL file/key | `TL.mat`, `square_TL` |
| TL logical shape | `(sample, depth, range) = (2920, 199, 800)` |
| TL units | dB |
| Main acoustic condition | 200 Hz, source depth 50 m; remaining conditions follow the cited 3DSSF setup |
| Field domain | Paper describes 0–200 m depth and 0–40 km range |

MATLAB v7.3 stores reversed dimensions in the HDF5 representation: SSP `(10, 21, 2920)` and TL `(800, 199, 2920)`. The loader reverses these dimensions; conventional MATLAB files are read with SciPy in their logical dimensions. All values must be finite. No heuristic sample reordering or axis inference is performed.

SSP is bilinearly interpolated to `(199, 800)` with PyTorch `align_corners=True`; values are float32. Interpolation is batched to reduce transient memory. The full interpolated input and target arrays each occupy about 1.86 GB in decimal units, before train/test copies, normalization, optimizer states, and concatenation features. Temporal concatenation is constructed per minibatch in the release.

The supplied arrays must already contain the matching SSP and RAM output at each global index. Key/shape/finite checks cannot verify that pairing, sampling time, frequency, source depth, seabed conditions, or solver settings are correct. Keep those generation records alongside the dataset.

The reference dataset is cited as S. Li, L. Cheng, J. Li, Z. Wang, and J. Li, “Learning data distribution of three-dimensional ocean sound speed fields via diffusion models,” *The Journal of the Acoustical Society of America*, 155(5), 3410–3425, 2024. The local records do not confirm a public download or distribution license for the study-specific paired SSP/TL files. The original 3DSSF project provides separate environmental data, as noted in the third-party notices; these are not a verified drop-in replacement for the required pairs. Obtain access and redistribution rights from the relevant dataset owners. MIT licensing of this software does not license external data or the RAM solver.

For the paper's 100-Hz frequency-change experiment, the corresponding labels and generation contract are required. The provided release does not establish the complete original 100-Hz data-generation procedure and does not advertise its main-data CLI as an exact reproduction of that experiment.
