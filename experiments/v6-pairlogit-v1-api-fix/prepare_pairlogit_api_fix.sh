#!/usr/bin/env bash
# API-only repair. Preserve the original failed code folder and preparation log.
set -euo pipefail
test "${CONDA_DEFAULT_ENV:-}" = techarena
export PYTHONDONTWRITEBYTECODE=1

repo=/home/edwin/projects/techarena-team
bundle='/mnt/c/Users/Edwin/Documents/ChatGPT/TechArena — ChaiScript/pairlogit-api-fix'
previous="$repo/experiments/v6-pairlogit-v1"
experiment="$repo/experiments/v6-pairlogit-v1-api-fix"
work="$repo/artifacts/experiments/v6-pairlogit-v1"
prep_log="$repo/artifacts/experiments/v6-pairlogit-v1-prepare-api-fix.log"

cd "$repo"
test -f "$repo/artifacts/experiments/v6-pairlogit-v1-prepare.log"
test ! -e "$work"
test ! -e "$experiment"
test ! -e "$prep_log"
test ! -e "$repo/docs/results/v6-pairlogit-v1/pairlogit.json"
git check-ignore -q artifacts/experiments/v6-pairlogit-v1/probe.npz

# Confirm this is the same failed source shown in the supplied log.
(
  cd "$previous"
  printf '%s\n' \
    'e2a289ba09279da4c452b5d3167e68c19df96fc709b400b5323f97267790b11a  pairlogit_trial.py' \
    'f8b4163b577e5bec58197ba0c0345b97802e95bc5a1dd1cf032c574e7524d7ec  test_pairlogit.py' \
    | sha256sum -c -
)

mkdir "$experiment"
for name in pairlogit_trial.py pairlogit_core.py test_pairlogit.py \
            event_scale_diagnostic.py attention_allocation_test.py \
            conditional_choice_trial.py RUN_PAIRLOGIT.md \
            prepare_pairlogit_api_fix.sh SHA256SUMS_PAIRLOGIT.txt; do
  cp -- "$bundle/$name" "$experiment/"
done
cd "$experiment"
sha256sum -c SHA256SUMS_PAIRLOGIT.txt

(
  set -euo pipefail
  python -c 'import sys,inspect,numpy,torch,sklearn,catboost; print(sys.version); print("NumPy",numpy.__version__,"torch",torch.__version__,"sklearn",sklearn.__version__,"CatBoost",catboost.__version__); print("Installed ranker predict:",inspect.signature(catboost.CatBoostRanker.predict)); print(inspect.getsource(catboost.CatBoostRanker.predict))'
  python -m unittest discover -s "$experiment" -p test_pairlogit.py -v
  timeout --signal=TERM --kill-after=15s 45m bash -c '
    set -euo pipefail
    python -u "$1/pairlogit_trial.py" verify --repo "$2"
    python -u "$1/pairlogit_trial.py" prepare --repo "$2"
  ' _ "$experiment" "$repo"
) 2>&1 | tee "$prep_log"
