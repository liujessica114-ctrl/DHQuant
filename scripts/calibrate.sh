#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
scale=${1:-4}
if (( $# )); then shift; fi
case "$scale" in 2|3|4) ;; *) echo "Scale must be 2, 3, or 4." >&2; exit 2 ;; esac
python basicsr/getcalidata.py -opt "options/calibration/calibration_x${scale}.yml"   --force_yml "scale=$scale" "$@"
