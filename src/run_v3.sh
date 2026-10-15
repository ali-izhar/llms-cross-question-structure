#!/usr/bin/env bash
# Greedy MMLU option-rotation runs (R2), one gb10-vllm container at a time.
#   bash src/run_v3.sh OUT_DIR N_QUESTIONS SPEC...      SPEC = model:runs  (runs: v3 run ids, comma-separated, or "all")
# N_QUESTIONS = 0 means all questions. A memory watchdog (src/memwatch.sh) logs to OUT_DIR/logs/mem.log.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=$1; NQ=$2; shift 2
for spec in "$@"; do  # refuse a spec without ':' before anything runs
  case "$spec" in ?*:?*) ;; *) echo "bad spec '$spec' (need model:runs)"; exit 2;; esac
done
mkdir -p "$OUT/logs"
MEMWATCH_LOG="$OUT/logs/mem.log" bash src/memwatch.sh & MW=$!
trap 'kill $MW 2>/dev/null' EXIT
STATUS=$OUT/logs/model_status.tsv
for spec in "$@"; do
  m=${spec%%:*}; runs=${spec#*:}; s=${m//\//_}
  while [ -n "$(docker ps -q --filter label=collect=1)" ]; do sleep 5; done
  until [ "$(awk '/MemTotal/{t=$2}/MemAvailable/{a=$2}END{print int((t-a)/1048576)}' /proc/meminfo)" -lt 25 ]; do
    echo "$(date -Is) waiting for memory release"; sleep 15; done
  args=(--model "$m" --plan v3 --out-dir "$OUT")
  [ "$runs" != "all" ] && args+=(--runs "$runs")
  [ "$NQ" != "0" ] && args+=(--n-questions "$NQ")
  echo "$(date -Is) model=$m runs=$runs"
  # flag fallback: keep a determinism flag only if the model runs with it (recorded per attempt)
  for extra in "" "--batch-invariant off" "--batch-invariant off --blocks-override off"; do
    t0=$(date +%s)
    # shellcheck disable=SC2086
    src/docker_run.sh gb10-vllm python src/collect.py "${args[@]}" $extra > "$OUT/logs/vllm_${s}.console.log" 2>&1
    rc=$?
    echo -e "$(date -Is)\t$m\t$runs\textra=[${extra}]\trc=$rc\tsec=$(( $(date +%s) - t0 ))" | tee -a "$STATUS"
    docker ps -q --filter label=collect=1 | xargs -r docker kill >/dev/null 2>&1
    [ $rc -eq 0 ] && break
    cp "$OUT/logs/vllm_${s}.console.log" "$OUT/logs/vllm_${s}.console.failed_$(date +%s).log"
    # a partial file from the failed attempt must not be resumed with other flags
    rm -f "$OUT"/*"${s}".partial.npz
    until [ "$(awk '/MemTotal/{t=$2}/MemAvailable/{a=$2}END{print int((t-a)/1048576)}' /proc/meminfo)" -lt 25 ]; do sleep 15; done
  done
done
echo "done $(date -Is)"
