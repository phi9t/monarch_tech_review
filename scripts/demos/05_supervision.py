#!/usr/bin/env python3
"""Demo 05: Supervision and Fault Tolerance — Erlang-style (Monarch) vs max_restarts (Ray).

Shows supervision trees and failure propagation vs flat restart policies.
Run: ./scripts/zephyr_uv_run.sh python scripts/demos/05_supervision.py [--dry-run]
"""

from __future__ import annotations

import argparse
import time

from demo_tui import ComparisonPanel, DashboardBase, KeyValuePanel, LogPanel, PanelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Simulate without clusters")
    return parser.parse_args()


def _simulate_monarch_supervision(
    dashboard: DashboardBase, comparison: ComparisonPanel, kv: KeyValuePanel
) -> None:
    events = [
        ("Spawn supervisor", "supervisor = procs.spawn('sup', Supervisor)", "healthy"),
        ("Spawn worker", "worker = this_proc().spawn('worker', FlakyWorker)", "healthy"),
        ("Worker crashes!", "FlakyWorker.do_work raises RuntimeError", "FAULT"),
        ("__supervise__ called", "Supervisor.__supervise__(MeshFailure) -> return True", "recovering"),
        ("Fault handled", "Supervisor handled failure, stopped propagation", "recovered"),
        ("Respawn worker", "new_worker = this_proc().spawn('worker2', FlakyWorker)", "healthy"),
    ]
    lines: list[str] = []
    for event_name, detail, state in events:
        lines.append(f"[{state.upper()}] {event_name}")
        lines.append(f"  {detail}")
        lines.append("")
        comparison.set_left(lines, title="Monarch (supervision tree)")
        kv.set("Monarch actor state", state)
        kv.set("Monarch model", "Supervision tree (Erlang-style)")
        kv.set("Monarch failure handler", "__supervise__(MeshFailure)")
        dashboard.log(f"[Monarch] {event_name}: {state}")
        dashboard.update_live()
        time.sleep(0.6)


def _simulate_ray_supervision(
    dashboard: DashboardBase, comparison: ComparisonPanel, kv: KeyValuePanel
) -> None:
    events = [
        ("Spawn actor", "@ray.remote(max_restarts=3) class Worker", "healthy"),
        ("Actor crashes!", "Worker.do_work raises RuntimeError", "FAULT"),
        ("GCS restarts actor", "GcsActorManager reschedules (restart 1/3)", "restarting"),
        ("Actor restarted", "Worker.__init__() called again", "healthy"),
        ("In-flight task retried", "max_task_retries=5: auto-retry the failed call", "healthy"),
        ("RayActorError", "After max_restarts: RayActorError to caller", "dead"),
    ]
    lines: list[str] = []
    for event_name, detail, state in events:
        lines.append(f"[{state.upper()}] {event_name}")
        lines.append(f"  {detail}")
        lines.append("")
        comparison.set_right(lines, title="Ray (flat restart)")
        kv.set("Ray actor state", state)
        kv.set("Ray model", "Flat restart (max_restarts)")
        kv.set("Ray failure handler", "RayActorError exception")
        dashboard.log(f"[Ray] {event_name}: {state}")
        dashboard.update_live()
        time.sleep(0.6)


def main() -> None:
    args = parse_args()

    dashboard = DashboardBase(title="Demo 05: Supervision & Fault Tolerance", refresh_hz=4.0)
    comparison = ComparisonPanel(PanelConfig(title="Fault Tolerance", border_style="blue"))
    kv = KeyValuePanel(PanelConfig(title="Actor State", border_style="cyan"))
    log_panel = LogPanel(PanelConfig(title="Fault/Recovery Events", border_style="magenta"), max_lines=50)
    dashboard.add_panel("kv", kv)
    dashboard.add_panel("comparison", comparison)
    dashboard.add_panel("log", log_panel)

    def callback(db: DashboardBase) -> None:
        db.log("Starting Supervision demo")
        db.update_live()

        _simulate_monarch_supervision(db, comparison, kv)
        _simulate_ray_supervision(db, comparison, kv)

        db.log("Demo complete")
        db.update_live()
        time.sleep(1.0)

    dashboard.run(callback)


if __name__ == "__main__":
    main()
