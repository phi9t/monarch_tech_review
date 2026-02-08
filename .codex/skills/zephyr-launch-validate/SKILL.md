---
name: zephyr-launch-validate
description: Launch this repository in Zephyr container infra and validate the runtime wiring. Use when running marimo notebooks or repo commands in Zephyr, ensuring uv venv reuses Zephyr-provided packages (especially torch from /opt/spack_store/view) instead of reinstalling upstream wheels.
---

# Zephyr Launch + Validate

Use this skill to run this repo inside Zephyr container infra and verify environment correctness.

## Run Launch + Validation

From repo root, run:

```bash
.codex/skills/zephyr-launch-validate/scripts/launch_and_validate.sh \
  --notebook notebooks/01_history_and_vision.py \
  --session zephyr-marimo-nb01 \
  --project-id monarch-gpu-mode \
  --port 2718
```

## What It Enforces

- Launch via `/mnt/data_infra/workspace/sygaldry/container/launch_container.sh`.
- Mount this repo with `SYGALDRY_REPO`.
- Start notebook using `./scripts/zephyr_uv_run.sh ...`.
- Validate torch provenance in Zephyr runtime:
  - `torch.__version__` is printed.
  - `torch.__file__` must be under `/opt/spack_store/view/...`.
- Keep process alive in a named `tmux` session for monitoring.

## Monitoring

Use:

```bash
tmux capture-pane -pt zephyr-marimo-nb01 -S -120 | tail -n 120
```

## Stop

Use:

```bash
tmux kill-session -t zephyr-marimo-nb01
```
