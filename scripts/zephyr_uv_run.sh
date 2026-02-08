#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

"$SCRIPT_DIR/zephyr_uv_sync.sh" >/dev/null
source .venv/bin/activate

# Prefer Zephyr CUDA runtime libs before any wheel-provided CUDA libraries.
export LD_LIBRARY_PATH="/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec uv run --active --no-sync "$@"
