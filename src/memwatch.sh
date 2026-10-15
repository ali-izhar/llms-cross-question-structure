#!/usr/bin/env bash
# Every 10 s: log used memory (MemTotal - MemAvailable); stop collection containers if used > 100 GB.
LOG="${MEMWATCH_LOG:-$(dirname "$0")/../results/logs/mem.log}"
LIMIT=$((100*1000*1000*1000))
while true; do
  read t a < <(awk '/MemTotal/{t=$2}/MemAvailable/{a=$2}END{print t*1024, a*1024}' /proc/meminfo)
  used=$((t - a))
  c=$(docker ps -q --filter label=collect=1 | tr '\n' ' ')
  echo "$(date -Is) used_GB=$(awk -v u=$used 'BEGIN{printf "%.1f", u/1e9}') containers=[$c]" >> "$LOG"
  if [ "$used" -gt "$LIMIT" ] && [ -n "$c" ]; then
    echo "$(date -Is) LIMIT EXCEEDED -> docker stop $c" >> "$LOG"
    docker stop -t 15 $c >/dev/null 2>&1
  fi
  sleep 10
done
