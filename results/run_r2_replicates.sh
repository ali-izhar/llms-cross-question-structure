#!/usr/bin/env bash
# R2: 24 models x 4 rotations x 500, greedy, two replicates with separate engine starts (A then B per model).
cd "$(dirname "$0")/.."
MODELS=$(.venv/bin/python -c "import sys; sys.path.insert(0,'src'); from plan import ENSEMBLE_MODELS; print(' '.join(ENSEMBLE_MODELS))")
for m in $MODELS; do
  s=${m//\//_}
  bash src/run_v3.sh results/r2 0 "$m:mmlu_greedyrot_$s"
  bash src/run_v3.sh results/r2_rep 0 "$m:mmlu_greedyrot_$s"
done
echo R2DONE
