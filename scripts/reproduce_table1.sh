#!/usr/bin/env bash
# Run the fixed paper matrix sequentially. All paths can be supplied explicitly.
set -euo pipefail
if [[ $# -ne 4 ]]; then
  echo "Usage: $0 SSP.mat TL.mat OUTPUT_DIR DEVICE" >&2
  exit 2
fi
stfm_ssp="$1"
stfm_tl="$2"
stfm_output="$3"
stfm_device="$4"
stfm_python="${STFM_PYTHON:-python}"
mkdir -p "$stfm_output/logs"
for stfm_seed in 1; do
  for stfm_variant in baseline scalar pe fnm hyper film; do
    "$stfm_python" -m stfm_fno.experiments train \
      --raw-ssp "$stfm_ssp" --raw-tl "$stfm_tl" \
      --variant "$stfm_variant" --seed "$stfm_seed" --epochs 100 \
      --output-dir "$stfm_output" --device "$stfm_device" \
      | tee "$stfm_output/logs/train_${stfm_variant}_seed${stfm_seed}.log"
    "$stfm_python" -m stfm_fno.experiments evaluate \
      --raw-ssp "$stfm_ssp" --raw-tl "$stfm_tl" \
      --variant "$stfm_variant" --seed "$stfm_seed" \
      --output-dir "$stfm_output" --device "$stfm_device" \
      | tee "$stfm_output/logs/evaluate_${stfm_variant}_seed${stfm_seed}.log"
  done
done
"$stfm_python" -m stfm_fno.experiments aggregate --output-dir "$stfm_output" --table table_i
