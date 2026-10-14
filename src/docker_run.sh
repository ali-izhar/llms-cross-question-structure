#!/usr/bin/env bash
# gb10-run.sh equivalent (same flags and mounts) plus a hard memory cap.
#   src/docker_run.sh <image> <cmd...>     (run from the project root)
set -euo pipefail
IMAGE="$1"; shift
exec docker run --rm -i --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --memory=100g --memory-swap=100g --label v2collect=1 \
  -e HF_HOME=/root/.cache/huggingface -e HF_TOKEN="${HF_TOKEN:-}" -e PYTHONUNBUFFERED=1 \
  -e OMP_NUM_THREADS=8 -e V2_DOCKER_MEMORY=100g -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 \
  -v "$(pwd):/workspace" -v "$HOME/.cache/huggingface:/root/.cache/huggingface" -w /workspace \
  "$IMAGE" "$@"
