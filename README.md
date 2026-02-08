# Monarch GPU Mode

Interactive marimo notebooks for the GPU Mode presentation on Monarch.

## Overview

This repo contains a series of notebooks that progressively introduce Monarch concepts through hands-on examples:

| Notebook | Topic |
|----------|-------|
| `01_history_and_vision.py` | Tensor engine origins, Hyperactor, the actor model |
| `02_interactive_devx.py` | SPMDJob, HostMesh, remote torchrun experience |
| `03_fault_tolerance.py` | Error handling, supervision, TorchFT, semi-sync training |
| `04_distributed_tensors.py` | Mesh activation, tensor compute, collectives, device mesh dimensions |
| `05_rl_intro.py` | RL at scale: sync vs async, on-policy vs off-policy, Zorplex benchmark |
| `06_services.py` | Services: round-robin routing, health tracking, failure recovery |
| `07_rdma_weight_sync.py` | RDMA weight sync, CPU staging, circular buffers |
| `07b_weight_sync_deep_dive.py` | ibverbs internals, RDMA buffer patterns |
| `08_rl_e2e.py` | Full end-to-end async RL loop with weight sync |

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (for full examples)
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Installation

### Install uv (if you don't have it)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv

# Or with Homebrew
brew install uv
```

### Clone and install dependencies

```bash
git clone <this-repo>
cd monarch-gpu-mode

# Install dependencies with uv
uv sync
```

### Zephyr container mode

On Zephyr infra, use the wrappers below to enforce `.venv` on Spack Python with system site-packages and avoid installing upstream Torch wheels:

- `./scripts/zephyr_shell.sh` defaults to snapshot image `sygaldry/zephyr:spack`.
- Override image when needed, e.g. `./scripts/zephyr_shell.sh --image sygaldry/zephyr:base`.

```bash
./scripts/zephyr_uv_sync.sh
./scripts/zephyr_uv_run.sh marimo edit notebooks/01_history_and_vision.py
```

## Running the Notebooks

### Option 1: Interactive mode (recommended)

This opens the notebook in your browser with live editing:

```bash
# Run a specific notebook
uv run marimo edit notebooks/01_history_and_vision.py

# Or run any notebook
uv run marimo edit notebooks/08_rl_e2e.py
```

On Zephyr infra, prefer:

```bash
./scripts/zephyr_uv_run.sh marimo edit notebooks/01_history_and_vision.py
```

### Option 2: App mode (read-only)

This runs the notebook as a read-only app:

```bash
uv run marimo run notebooks/01_history_and_vision.py
```

### Option 3: Export to HTML

```bash
uv run marimo export html notebooks/01_history_and_vision.py -o output.html
```

On Zephyr infra:

```bash
./scripts/zephyr_uv_run.sh marimo export html notebooks/01_history_and_vision.py -o output.html
```

## Development

### Adding new notebooks

```bash
uv run marimo new notebooks/05_new_topic.py
```

### Running tests

```bash
uv run pytest
```

### Linting

```bash
uv run ruff check .
uv run ruff format .
```

## Resources

- [Monarch Documentation](https://meta-pytorch.org/monarch/)
- [Monarch Examples](https://meta-pytorch.org/monarch/generated/examples/index.html)
- [PyTorch Monarch Tutorial](https://docs.pytorch.org/tutorials/intermediate/monarch_distributed_tutorial.html)
