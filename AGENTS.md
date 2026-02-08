# Repository Guidelines

## Project Structure & Module Organization
- `notebooks/`: Primary marimo notebooks (`01_...py` through `08_...py`) for the presentation flow; keep filenames numeric and topic-based (example: `09_new_topic.py`).
- `src/`: Reusable Python modules.
- `src/monarch_utils/`: service registry and routing utilities.
- `src/zorplex_rl/`: Zorplex task specs and evaluation logic.
- `src/rl_primitives.py`: shared RL dataclasses used across notebooks.
- `scripts/`: operational scripts (`export_html.sh`, benchmark/eval runners, sanity tests).
- `docs/`: generated HTML exports for GitHub Pages; treat as build artifacts from notebooks.

## Zephyr Container Infra Workflow
- Execute repository commands through Zephyr container infra, not directly on the host.
- Open an interactive container shell with `./scripts/zephyr_shell.sh` (defaults to snapshot image `sygaldry/zephyr:spack`).
- Override image if needed with `./scripts/zephyr_shell.sh --image sygaldry/zephyr:base`.
- Initialize/update the project venv with `./scripts/zephyr_uv_sync.sh`.
- Run all repo commands through `./scripts/zephyr_uv_run.sh ...` (example: `./scripts/zephyr_uv_run.sh pytest`).
- Use `.codex/skills/zephyr-launch-validate/scripts/launch_and_validate.sh` to launch notebooks with validation in one step.
- Do not install upstream PyTorch/CUDA wheel packages in `.venv`; rely on Zephyr-provided packages from `/opt/spack_store/view`.

## Build, Test, and Development Commands
- `./scripts/zephyr_shell.sh`: open interactive Zephyr container shell with this repo mounted.
- `./scripts/zephyr_uv_sync.sh`: sync `.venv` while reusing Zephyr-provided packages.
- `./scripts/zephyr_uv_run.sh marimo edit notebooks/01_history_and_vision.py`: open notebook in interactive edit mode.
- `./scripts/zephyr_uv_run.sh marimo run notebooks/01_history_and_vision.py`: run notebook as read-only app.
- `./scripts/export_html.sh`: export all notebooks to `docs/` and refresh `docs/index.html`.
- `./scripts/zephyr_uv_run.sh pytest`: run Python tests and script-based test files.
- `./scripts/zephyr_uv_run.sh ruff check .` and `./scripts/zephyr_uv_run.sh ruff format .`: lint and format code.

## Coding Style & Naming Conventions
- Target Python 3.10+; use 4-space indentation and type hints for reusable code in `src/` and `scripts/`.
- Keep notebook filenames prefixed with ordered numbers (`01_`, `02_`, ...).
- Prefer small, composable functions in `src/`; keep notebook cells focused on one concept.
- Run `ruff check` and `ruff format` before opening a PR.

## Testing Guidelines
- Framework: `pytest`.
- Test files should be named `test_*.py` (existing examples in `scripts/`).
- For GPU-dependent paths (e.g., Zorplex benchmark), include lightweight sanity checks that fail fast with clear messages when GPU/runtime requirements are missing.
- Before submitting, run `./scripts/zephyr_uv_run.sh pytest` and any touched script directly (example: `./scripts/zephyr_uv_run.sh scripts/run_zorplex_eval.py --task simple`).

## Commit & Pull Request Guidelines
- Commit messages in this repo are short, imperative, and lower-case (examples: `add nav`, `rerun export`, `notebooks updates`).
- Keep commits focused (code vs. exported `docs/` changes should be easy to review).
- PRs should include: what changed, why it changed, impacted notebooks/modules, and screenshots or HTML diffs when presentation output changes.
- Link related issues/tasks and list exact verification commands you ran.
