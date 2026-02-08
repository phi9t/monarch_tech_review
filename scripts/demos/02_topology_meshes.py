#!/usr/bin/env python3
"""Demo 02: Topology — Meshes (Monarch) vs Flat Handles (Ray).

Shows mesh creation, slicing, and reshape vs placement groups.
Run: ./scripts/zephyr_uv_run.sh python scripts/demos/02_topology_meshes.py [--dry-run]
"""

from __future__ import annotations

import argparse
import time

from demo_tui import ComparisonPanel, DashboardBase, KeyValuePanel, LogPanel, PanelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Simulate without clusters")
    return parser.parse_args()


def _simulate_monarch_topology(dashboard: DashboardBase, comparison: ComparisonPanel, kv: KeyValuePanel) -> None:
    steps = [
        ("mesh = this_host().spawn_procs({'gpu': 4})", "Created ProcMesh with 4 GPUs"),
        ("mesh.to_table()", "gpu | host | rank\n 0  |  0   |  0\n 1  |  0   |  1\n 2  |  0   |  2\n 3  |  0   |  3"),
        ("mesh_2d = mesh.rename(gpu='tp')", "Renamed gpu->tp dimension"),
        ("mesh_3d = mesh.split(gpu=('dp','tp'), dp=2)", "Split into 2D mesh: dp=2, tp=2"),
        ("actors.slice(dp=0)", "Sliced: actors at dp=0 (2 actors)"),
        ("actors.slice(tp=1)", "Sliced: actors at tp=1 (2 actors)"),
    ]
    lines: list[str] = []
    for code, desc in steps:
        lines.append(f">>> {code}")
        lines.append(f"    {desc}")
        comparison.set_left(lines, title="Monarch (Meshes)")
        dashboard.log(f"[Monarch] {desc}")
        dashboard.update_live()
        time.sleep(0.5)

    kv.set("Monarch topology", "ProcMesh{gpu: 4} -> split -> {dp: 2, tp: 2}")
    kv.set("Monarch addressing", "Dimension-based slicing")
    kv.set("Monarch broadcast", "O(log N) tree-based")
    dashboard.update_live()


def _simulate_ray_topology(dashboard: DashboardBase, comparison: ComparisonPanel, kv: KeyValuePanel) -> None:
    steps = [
        ("pg = placement_group([{'GPU':1}]*4, 'SPREAD')", "Created PlacementGroup (4 bundles)"),
        ("ray.get(pg.ready())", "PlacementGroup ready"),
        ("actors = [W.options(...).remote() for i in range(4)]", "Spawned 4 actors individually"),
        ("# No slice() or reshape — index by list position", "Actors: actors[0], actors[1], ..."),
        ("refs = [a.work.remote() for a in actors]", "Fan-out: O(N) calls"),
        ("results = ray.get(refs)", "Gather: plain list"),
    ]
    lines: list[str] = []
    for code, desc in steps:
        lines.append(f">>> {code}")
        lines.append(f"    {desc}")
        comparison.set_right(lines, title="Ray (Flat Handles)")
        dashboard.log(f"[Ray] {desc}")
        dashboard.update_live()
        time.sleep(0.5)

    kv.set("Ray topology", "PlacementGroup (scheduling constraint)")
    kv.set("Ray addressing", "List index: actors[i]")
    kv.set("Ray broadcast", "O(N) list comprehension")
    dashboard.update_live()


def main() -> None:
    args = parse_args()

    dashboard = DashboardBase(title="Demo 02: Topology", refresh_hz=4.0)
    comparison = ComparisonPanel(PanelConfig(title="Topology Comparison", border_style="blue"))
    kv = KeyValuePanel(PanelConfig(title="Topology Shape", border_style="cyan"))
    log_panel = LogPanel(PanelConfig(title="Timeline", border_style="magenta"), max_lines=50)
    dashboard.add_panel("topology", kv)
    dashboard.add_panel("comparison", comparison)
    dashboard.add_panel("log", log_panel)

    def callback(db: DashboardBase) -> None:
        db.log("Starting Topology demo")
        db.update_live()

        if args.dry_run:
            _simulate_monarch_topology(db, comparison, kv)
            _simulate_ray_topology(db, comparison, kv)
        else:
            _simulate_monarch_topology(db, comparison, kv)
            _simulate_ray_topology(db, comparison, kv)

        db.log("Demo complete")
        db.update_live()
        time.sleep(1.0)

    dashboard.run(callback)


if __name__ == "__main__":
    main()
