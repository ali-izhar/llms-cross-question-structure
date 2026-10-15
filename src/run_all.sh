#!/usr/bin/env bash
# Sampled runs: every planned model, one gb10-vllm container at a time; resumable.
#   nohup bash src/run_all.sh > results/logs/run_all.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
RUN="src/docker_run.sh gb10-vllm"
LOGS=results/logs
mkdir -p "$LOGS"
STATUS=$LOGS/model_status.tsv
MODELS=$(.venv/bin/python -c "import sys; sys.path.insert(0,'src'); from plan import MODEL_ORDER; print('\n'.join(MODEL_ORDER))")
IMG=$(docker image inspect gb10-vllm --format '{{.Id}}')
echo "start $(date -Is) image=$IMG"
for m in $MODELS; do
  s=${m//\//_}
  # one model at a time: wait until no collection container runs and used memory is near baseline
  while [ -n "$(docker ps -q --filter label=collect=1)" ]; do sleep 5; done
  until [ "$(awk '/MemTotal/{t=$2}/MemAvailable/{a=$2}END{print int((t-a)/1048576)}' /proc/meminfo)" -lt 35 ]; do
    echo "$(date -Is) waiting for memory release"; sleep 15; done
  for attempt in 1 2; do
    echo "$(date -Is) model=$m attempt=$attempt"
    t0=$(date +%s)
    $RUN python src/collect.py --model "$m" > "$LOGS/vllm_${s}.console.log" 2>&1
    rc=$?
    echo -e "$(date -Is)\t$m\tvllm\tattempt=$attempt\trc=$rc\tsec=$(( $(date +%s) - t0 ))" >> "$STATUS"
    [ $rc -eq 0 ] && break
    # make sure nothing is left holding the unified pool before a retry
    docker ps -q --filter label=collect=1 | xargs -r docker kill >/dev/null 2>&1
  done
done
echo "done $(date -Is)"
