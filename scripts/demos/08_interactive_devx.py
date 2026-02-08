#!/usr/bin/env python3
"""Demo 08: Interactive DevX — Single Controller (Monarch) vs Job Submission (Ray).

Shows Monarch's notebook-as-controller paradigm vs Ray's job submission model.
Run: ./scripts/zephyr_uv_run.sh python scripts/demos/08_interactive_devx.py [--dry-run]
"""

from __future__ import annotations

import argparse
import time

from demo_tui import ComparisonPanel, DashboardBase, LogPanel, PanelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Simulate without clusters")
    return parser.parse_args()


def _simulate_monarch_devx(dashboard: DashboardBase, comparison: ComparisonPanel) -> None:
    """Simulate Monarch's single-controller interactive development."""
    session = [
        ("# marimo notebook cell 1", ""),
        ("host = this_host()", "Acquired local HostMesh"),
        ("procs = host.spawn_procs({'gpus': 4})", "ProcMesh with 4 GPUs"),
        ("researchers = procs.spawn('r', Researcher)", "ActorMesh spawned"),
        ("", ""),
        ("# Cell 2: train interactively", ""),
        ("result = researchers.train_step.call(lr=1e-4).get()", "All 4 GPUs trained, results back"),
        ("print(result)", "ValueMesh{0: {loss: 2.1}, 1: {loss: 2.0}, ...}"),
        ("", ""),
        ("# Cell 3: inspect state", ""),
        ("state = researchers.get_state.call().get()", "Distributed state fetched"),
        ("", ""),
        ("# Cell 4: change hyperparameters — no restart!", ""),
        ("result2 = researchers.train_step.call(lr=5e-5).get()", "New LR applied instantly"),
        ("", ""),
        ("# The notebook IS the controller.", ""),
    ]
    lines: list[str] = []
    for code, desc in session:
        if code:
            lines.append(f">>> {code}")
        if desc:
            lines.append(f"    {desc}")
        if not code and not desc:
            lines.append("")
        comparison.set_left(lines, title="Monarch (single controller)")
        if desc:
            dashboard.log(f"[Monarch] {desc}")
        dashboard.update_live()
        time.sleep(0.4)


def _simulate_ray_devx(dashboard: DashboardBase, comparison: ComparisonPanel) -> None:
    """Simulate Ray's job submission model."""
    session = [
        ("# notebook / script", ""),
        ("ray.init()", "Connected to Ray cluster"),
        ("trainers = [Trainer.remote() for _ in range(4)]", "4 independent actors"),
        ("", ""),
        ("# Training: fan-out + gather", ""),
        ("refs = [t.train_step.remote(lr=1e-4) for t in trainers]", "4 remote calls"),
        ("results = ray.get(refs)", "Gathered: [{'loss': 2.1}, ...]"),
        ("", ""),
        ("# Inspect: another remote call", ""),
        ("states = ray.get([t.get_state.remote() for t in trainers])", "Fetched states"),
        ("", ""),
        ("# Change hyperparameters", ""),
        ("refs2 = [t.train_step.remote(lr=5e-5) for t in trainers]", "New calls with new LR"),
        ("results2 = ray.get(refs2)", "Results gathered"),
        ("", ""),
        ("# Production: Ray Jobs CLI", ""),
        ("# $ ray job submit -- python train.py", "Submit job to cluster"),
        ("# $ ray job logs <job_id>", "Monitor externally"),
    ]
    lines: list[str] = []
    for code, desc in session:
        if code:
            lines.append(f">>> {code}")
        if desc:
            lines.append(f"    {desc}")
        if not code and not desc:
            lines.append("")
        comparison.set_right(lines, title="Ray (job submission)")
        if desc:
            dashboard.log(f"[Ray] {desc}")
        dashboard.update_live()
        time.sleep(0.4)


def main() -> None:
    args = parse_args()

    dashboard = DashboardBase(title="Demo 08: Interactive DevX", refresh_hz=4.0)
    comparison = ComparisonPanel(PanelConfig(title="Development Workflow", border_style="blue"))
    log_panel = LogPanel(PanelConfig(title="Interactive Session", border_style="magenta"), max_lines=50)
    dashboard.add_panel("comparison", comparison)
    dashboard.add_panel("log", log_panel)

    def callback(db: DashboardBase) -> None:
        db.log("Starting Interactive DevX demo")
        db.update_live()

        _simulate_monarch_devx(db, comparison)
        _simulate_ray_devx(db, comparison)

        db.log("Demo complete")
        db.update_live()
        time.sleep(1.0)

    dashboard.run(callback)


if __name__ == "__main__":
    main()
