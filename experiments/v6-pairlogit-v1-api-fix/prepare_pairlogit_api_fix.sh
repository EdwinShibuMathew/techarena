#!/usr/bin/env bash
set -euo pipefail

test "${CONDA_DEFAULT_ENV:-}" = techarena
export PYTHONDONTWRITEBYTECODE=1

experiment=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$experiment/../.." && pwd)
source_dir="${1:-$repo/artifacts/experiments/v6-pairlogit-v1-inputs/source}"
work="$repo/artifacts/experiments/v6-pairlogit-v1"
prep_log="$repo/artifacts/experiments/v6-pairlogit-v1-prepare-api-fix.log"

cd "$repo"
test -d "$source_dir"
test ! -e "$work"
test ! -e "$prep_log"
git check-ignore -q artifacts/experiments/v6-pairlogit-v1/probe.npz
(cd "$experiment" && sha256sum -c SHA256SUMS_PAIRLOGIT.txt)
python -m unittest discover -s "$experiment" -p test_pairlogit.py -v
timeout --signal=TERM --kill-after=15s 45m bash -c '
  set -euo pipefail
  python -u "$1/pairlogit_trial.py" verify --repo "$2" --source-dir "$3"
  python -u "$1/pairlogit_trial.py" prepare --repo "$2"
' _ "$experiment" "$repo" "$source_dir" 2>&1 | tee "$prep_log"
