#!/usr/bin/env bash
# Generate dfl_sup_v11 (v10 with the four extra feature families of possession/extra_features.py,
# every model retrained on them: docs/MIGLIORAMENTI_SPELLF1.md, section 34.3) twice, check that both
# runs give the same file, validate it, score it on the public 5 minutes and redraw the comparison
# with v10. Run it yourself, locally: it reads the restricted data in data/. Prints aggregates only.
#
#     bash scripts/run_local_v11.sh
#
# Local guard (section 21.1, point 6): v11 replaces v10 only if (1) valid and reproducible,
# (2) whole-match D share in 40-50% and 2-4.5 possessions per minute, (3) public-window dead-ball
# recall >= 0.66. Otherwise submit dfl_sup_v10.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
cd "$ROOT"
ARGS=(--networks light_symsw --evidence-weight 0.25 --extra-features)

uv pip install --python "$PY" -r requirements-pipeline.txt   # the pinned CPU PyTorch
bash scripts/fetch_dfl.sh                                     # no-op when already downloaded
"$PY" scripts/make_submission.py --name dfl_sup_v11 "${ARGS[@]}"
"$PY" scripts/make_submission.py --name dfl_sup_v11_rerun "${ARGS[@]}"
sha256sum evaluation/results/dfl_sup_v11.csv evaluation/results/dfl_sup_v11_rerun.csv
cmp -s evaluation/results/dfl_sup_v11.csv evaluation/results/dfl_sup_v11_rerun.csv \
  && echo "REPRODUCIBLE: identical files" || echo "NOT REPRODUCIBLE: files differ"
rm evaluation/results/dfl_sup_v11_rerun.csv

"$PY" evaluation/validate_submission.py evaluation/results/dfl_sup_v11.csv
"$PY" evaluation/run_evaluation.py | sed -n '/^rank/,/^$/p;/^dfl_sup_v11$/,/abstained/p'
"$PY" scripts/plot_possessions.py dfl_sup_v11 dfl_sup_v10
