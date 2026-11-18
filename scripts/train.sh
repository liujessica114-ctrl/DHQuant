#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
scale=${1:-4}
bit=${2:-4}
if (( $# >= 2 )); then shift 2; elif (( $# == 1 )); then shift; fi
case "$scale" in 2|3|4) ;; *) echo "Scale must be 2, 3, or 4." >&2; exit 2 ;; esac
case "$bit" in 2|3|4) ;; *) echo "Bit-width must be 2, 3, or 4." >&2; exit 2 ;; esac
iterations=4000
seed=2026
if [[ "$bit" == 2 ]]; then iterations=10000; fi
if [[ "$scale" == 4 && "$bit" != 2 ]]; then seed=3407; fi
python basicsr/train.py -opt "options/train/train_DHQuant_x${scale}.yml" \
  --force_yml "bit=$bit" "name=train_DHQuant_x${scale}_bit${bit}" \
  "train:total_iter=$iterations" "manual_seed=$seed" "$@"
