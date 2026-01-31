# Monarch GPU Mode

Interactive marimo notebooks for the GPU Mode presentation on Monarch.

## Overview

This repo contains a series of notebooks that progressively introduce Monarch concepts through hands-on examples:

| Notebook | Topic |
|----------|-------|
| `01_history_and_vision.py` | Tensor engine origins, Hyperactor, the actor model |
| `02_interactive_devx.py` | SPMDJob, HostMesh, remote torchrun experience |
| `03_fault_tolerance.py` | Error handling, supervision, TorchFT, semi-sync training |
| `04_async_rl.py` | Services, weight sync, RDMA, full async RL loop |

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

## Running the Notebooks

### Option 1: Interactive mode (recommended)

This opens the notebook in your browser with live editing:

```bash
# Run a specific notebook
uv run marimo edit notebooks/01_history_and_vision.py

# Or run any notebook
uv run marimo edit notebooks/04_async_rl.py
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
