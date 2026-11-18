#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if (( $# < 3 )); then
  echo "Usage: bash scripts/test.sh SCALE BIT CHECKPOINT [key=value ...]" >&2
  exit 2
fi
scale=$1
bit=$2
checkpoint=$3
shift 3
case "$scale" in 2|3|4) ;; *) echo "Scale must be 2, 3, or 4." >&2; exit 2 ;; esac
case "$bit" in 2|3|4) ;; *) echo "Bit-width must be 2, 3, or 4." >&2; exit 2 ;; esac
python basicsr/test.py -opt "options/test/test_DHQuant_x${scale}.yml"   --force_yml "bit=$bit" "name=test_DHQuant_x${scale}_bit${bit}" "path:pretrain_network_Q=$checkpoint" "$@"
