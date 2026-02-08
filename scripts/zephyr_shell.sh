#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="monarch-gpu-mode"
ENTRYPOINT="default"
IMAGE="${SYGALDRY_IMAGE:-sygaldry/zephyr:spack}"
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
LAUNCHER="/mnt/data_infra/workspace/sygaldry/container/launch_container.sh"

usage() {
  cat <<'USAGE'
Launch an interactive shell in Zephyr container infra.

Usage:
  scripts/zephyr_shell.sh [--project-id ID] [--entrypoint NAME] [--image IMAGE] [--repo-root PATH] [-- CMD...]

Examples:
  scripts/zephyr_shell.sh
  scripts/zephyr_shell.sh --project-id monarch-gpu-mode
  scripts/zephyr_shell.sh --image sygaldry/zephyr:base
  scripts/zephyr_shell.sh -- bash -lc "cd /workspace/monarch-gpu-mode && ./scripts/zephyr_uv_run.sh pytest -q"
USAGE
}

PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-id)
      PROJECT_ID="$2"; shift 2 ;;
    --entrypoint)
      ENTRYPOINT="$2"; shift 2 ;;
    --image)
      IMAGE="$2"; shift 2 ;;
    --repo-root)
      REPO_ROOT="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    --)
      shift
      PASSTHROUGH=("$@")
      break ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [[ ! -x "$LAUNCHER" ]]; then
  echo "error: launcher not found: $LAUNCHER" >&2
  exit 1
fi

if [[ ! -d "$REPO_ROOT" ]]; then
  echo "error: repo root not found: $REPO_ROOT" >&2
  exit 1
fi

if [[ ${#PASSTHROUGH[@]} -eq 0 ]]; then
  PASSTHROUGH=(
    bash
    -lc
    'export LD_LIBRARY_PATH="/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; exec bash'
  )
fi

exec env \
  SYGALDRY_PROJECT_ID="$PROJECT_ID" \
  SYGALDRY_REPO="$REPO_ROOT" \
  SYGALDRY_BUILD_IMAGE=never \
  SYGALDRY_ENTRYPOINT="$ENTRYPOINT" \
  SYGALDRY_IMAGE="$IMAGE" \
  "$LAUNCHER" "${PASSTHROUGH[@]}"
