#!/usr/bin/env python3
"""Demo 07: Distributed Tensors — first-class (Monarch) vs DDP wrapper (Ray).

Shows Monarch's DeviceMesh.activate() and distributed tensor ops
vs Ray Train's TorchTrainer wrapping DDP.
Run: ./scripts/zephyr_uv_run.sh python scripts/demos/07_distributed_tensors.py [--dry-run]
"""

from __future__ import annotations

import argparse
import time

from demo_tui import ComparisonPanel, DashboardBase, KeyValuePanel, LogPanel, PanelConfig, ProgressPanel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Simulate without GPUs")
    parser.add_argument("--steps", type=int, default=5, help="Training steps to simulate")
    return parser.parse_args()


def _simulate_monarch_dtensor(
    dashboard: DashboardBase, comparison: ComparisonPanel, kv: KeyValuePanel,
    progress: ProgressPanel, steps: int,
) -> None:
    lines: list[str] = [
        "# Monarch: first-class distributed tensors",
        ">>> mesh = this_host().spawn_procs({'gpu': 4})",
        ">>> with mesh.activate():",
        ">>>     t = torch.rand(3, 4, device='cuda')",
        "    # t exists on ALL 4 GPUs simultaneously",
        "",
    ]
    comparison.set_left(lines, title="Monarch (distributed tensors)")
    kv.set("Monarch mesh", "{gpu: 4}")
    kv.set("Monarch tensor", "Distributed across mesh")
    dashboard.log("[Monarch] Created mesh, activated distributed tensors")
    dashboard.update_live()
    time.sleep(0.5)

    for step in range(1, steps + 1):
        loss = 2.5 / (step + 0.5)
        lines.append(f"  step {step}: loss={loss:.4f}")
        lines.append(f"    >>> p.grad.reduce_('dp', reduction='avg')")
        comparison.set_left(lines)
        kv.set("Monarch step", str(step))
        kv.set("Monarch loss", f"{loss:.4f}")
        progress.update(step, steps, extra_text=f"Monarch step {step}")
        dashboard.log(f"[Monarch] step={step} loss={loss:.4f} (reduce over 'dp' dim)")
        dashboard.update_live()
        time.sleep(0.4)

    lines.append("")
    lines.append(">>> local = inspect(t, gpu=0)  # fetch to controller")
    lines.append(">>> print(local.shape)  # torch.Size([3, 4])")
    comparison.set_left(lines)
    dashboard.update_live()


def _simulate_ray_ddp(
    dashboard: DashboardBase, comparison: ComparisonPanel, kv: KeyValuePanel,
    progress: ProgressPanel, steps: int,
) -> None:
    lines: list[str] = [
        "# Ray Train: wraps PyTorch DDP",
        ">>> trainer = TorchTrainer(",
        ">>>     train_func,",
        ">>>     scaling_config=ScalingConfig(num_workers=4, use_gpu=True))",
        ">>> result = trainer.fit()",
        "",
        "# Inside train_func (runs on each worker):",
        ">>>   model = ray.train.torch.prepare_model(model)  # wraps DDP",
        "",
    ]
    comparison.set_right(lines, title="Ray (DDP wrapper)")
    kv.set("Ray workers", "4 (TorchTrainer)")
    kv.set("Ray model", "Wrapped in DDP")
    dashboard.log("[Ray] Created TorchTrainer with 4 workers")
    dashboard.update_live()
    time.sleep(0.5)

    for step in range(1, steps + 1):
        loss = 2.5 / (step + 0.5)
        lines.append(f"  step {step}: loss={loss:.4f}")
        lines.append(f"    # DDP handles gradient sync automatically")
        comparison.set_right(lines)
        kv.set("Ray step", str(step))
        kv.set("Ray loss", f"{loss:.4f}")
        progress.update(step, steps, extra_text=f"Ray step {step}")
        dashboard.log(f"[Ray] step={step} loss={loss:.4f} (DDP gradient sync)")
        dashboard.update_live()
        time.sleep(0.4)

    lines.append("")
    lines.append(">>> ray.train.report({'loss': loss})")
    comparison.set_right(lines)
    dashboard.update_live()


def main() -> None:
    args = parse_args()

    dashboard = DashboardBase(title="Demo 07: Distributed Tensors", refresh_hz=4.0)
    comparison = ComparisonPanel(PanelConfig(title="Training Approach", border_style="blue"))
    kv = KeyValuePanel(PanelConfig(title="Tensor Metadata", border_style="cyan"))
    progress = ProgressPanel(PanelConfig(title="Training Progress", border_style="green"))
    log_panel = LogPanel(PanelConfig(title="Training Log", border_style="magenta"), max_lines=50)
    dashboard.add_panel("kv", kv)
    dashboard.add_panel("progress", progress)
    dashboard.add_panel("comparison", comparison)
    dashboard.add_panel("log", log_panel)

    def callback(db: DashboardBase) -> None:
        db.log("Starting Distributed Tensors demo")
        db.update_live()

        _simulate_monarch_dtensor(db, comparison, kv, progress, args.steps)
        _simulate_ray_ddp(db, comparison, kv, progress, args.steps)

        db.log("Demo complete")
        db.update_live()
        time.sleep(1.0)

    dashboard.run(callback)


if __name__ == "__main__":
    main()
