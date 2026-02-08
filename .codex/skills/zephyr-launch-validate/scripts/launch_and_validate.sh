#!/usr/bin/env bash
set -euo pipefail

SESSION="zephyr-marimo-nb01"
PROJECT_ID="monarch-gpu-mode"
PORT="2718"
NOTEBOOK="notebooks/01_history_and_vision.py"
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
LAUNCHER="/mnt/data_infra/workspace/sygaldry/container/launch_container.sh"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session)
      SESSION="$2"; shift 2 ;;
    --project-id)
      PROJECT_ID="$2"; shift 2 ;;
    --port)
      PORT="$2"; shift 2 ;;
    --notebook)
      NOTEBOOK="$2"; shift 2 ;;
    --repo-root)
      REPO_ROOT="$2"; shift 2 ;;
    -h|--help)
      cat <<USAGE
Usage:
  launch_and_validate.sh [--session NAME] [--project-id ID] [--port PORT] [--notebook PATH] [--repo-root PATH]
USAGE
      exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2 ;;
  esac
done

if [[ ! -x "$LAUNCHER" ]]; then
  echo "error: launcher not found: $LAUNCHER" >&2
  exit 1
fi

if [[ ! -f "$REPO_ROOT/$NOTEBOOK" ]]; then
  echo "error: notebook not found: $REPO_ROOT/$NOTEBOOK" >&2
  exit 1
fi

if [[ ! -x "$REPO_ROOT/scripts/zephyr_uv_run.sh" ]]; then
  echo "error: missing executable: $REPO_ROOT/scripts/zephyr_uv_run.sh" >&2
  exit 1
fi

# 1) Validate Zephyr torch provenance before launching long-lived notebook server.
VALIDATE_CMD='cd /workspace/'"$(basename "$REPO_ROOT")"' && ./scripts/zephyr_uv_run.sh python - <<"PY"
import importlib.util as u
spec = u.find_spec("torch")
origin = spec.origin if spec else ""
print("torch_spec_origin", origin)
if not origin.startswith("/opt/spack_store/view/"):
    raise SystemExit("torch must come from /opt/spack_store/view")
PY'

env \
  SYGALDRY_PROJECT_ID="$PROJECT_ID" \
  SYGALDRY_REPO="$REPO_ROOT" \
  SYGALDRY_BUILD_IMAGE=never \
  "$LAUNCHER" bash -lc "$VALIDATE_CMD" >/tmp/zephyr-validate-$SESSION.log 2>&1 || {
    echo "validation_failed=/tmp/zephyr-validate-$SESSION.log" >&2
    tail -n 80 /tmp/zephyr-validate-$SESSION.log >&2 || true
    exit 1
  }

# 2) Launch notebook in tmux.
tmux kill-session -t "$SESSION" 2>/dev/null || true
LAUNCH_CMD='env SYGALDRY_PROJECT_ID='"$PROJECT_ID"' SYGALDRY_REPO='"$REPO_ROOT"' SYGALDRY_BUILD_IMAGE=never '"$LAUNCHER"' bash -lc "cd /workspace/'"$(basename "$REPO_ROOT")"' && ./scripts/zephyr_uv_run.sh marimo edit '"$NOTEBOOK"' --host 0.0.0.0 --port '"$PORT"'"'

tmux new-session -d -s "$SESSION" "$LAUNCH_CMD"

echo "started_session=$SESSION"
echo "project_id=$PROJECT_ID"
echo "port=$PORT"
echo "notebook=$NOTEBOOK"
echo "validate_log=/tmp/zephyr-validate-$SESSION.log"
echo "monitor: tmux capture-pane -pt $SESSION -S -120 | tail -n 120"
