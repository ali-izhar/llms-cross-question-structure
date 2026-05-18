#!/bin/bash
# Full data-collection pipeline for the day.
#
# Stage 1: wait for current ensemble run (PID ${WAIT_PID}) to finish
# Stage 2: relaunch ensemble — picks up gated models with fresh token
# Stage 3: within-model batch — 3 additional models at K=100
#
# Each stage writes its own console log. Checkpoints in each script mean any
# interrupted stage can be relaunched without losing work.

set -e
cd "$(dirname "$0")/.."

PROJ_ROOT="$(pwd)"
VENV="$PROJ_ROOT/.venv/bin/python"
LOG_DIR="$PROJ_ROOT/results"
WAIT_PID="${WAIT_PID:-359694}"

echo "[pipeline] starting at $(date -Iseconds)"
echo "[pipeline] waiting for PID $WAIT_PID (current ensemble run) to finish..."

while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 60
done
echo "[pipeline] PID $WAIT_PID finished at $(date -Iseconds)"

# Give GPU a moment to release memory
sleep 30

echo "[pipeline] Stage 2: relaunching ensemble to pick up gated models"
"$VENV" src/run_ensemble_expanded.py > "$LOG_DIR/ensemble_stage2_console.log" 2>&1
echo "[pipeline] Stage 2 done at $(date -Iseconds)"

sleep 30

echo "[pipeline] Stage 3: within-model batch (K=100 on 3 models)"
"$VENV" src/run_within_batch.py > "$LOG_DIR/within_batch_console.log" 2>&1
echo "[pipeline] Stage 3 done at $(date -Iseconds)"

echo "[pipeline] ALL STAGES COMPLETE at $(date -Iseconds)"
