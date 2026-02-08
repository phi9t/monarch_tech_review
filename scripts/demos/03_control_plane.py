#!/usr/bin/env python3
"""Demo 03: Control Plane Messaging — call_one, call, stream, broadcast.

Shows Monarch's 4 messaging patterns vs Ray's ObjectRef model.
Run: ./scripts/zephyr_uv_run.sh python scripts/demos/03_control_plane.py [--dry-run]
"""

from __future__ import annotations

import argparse
import time

from demo_tui import ComparisonPanel, DashboardBase, LogPanel, PanelConfig, ProgressPanel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Simulate without clusters")
    return parser.parse_args()


def _simulate_monarch_messaging(
    dashboard: DashboardBase, comparison: ComparisonPanel, progress: ProgressPanel
) -> None:
    patterns = [
        (
            "call_one()",
            "actor.echo.call_one('hello').get()",
            "Single actor, request-response -> 'echo: hello'",
        ),
        (
            "call() (broadcast)",
            "actors.echo.call('hello').get()",
            "Broadcast to mesh -> ValueMesh{0:'echo: hello', 1:'echo: hello'}",
        ),
        (
            "stream()",
            "async for item in actor.stream_data.stream(count=5):",
            "Async stream: received items 0..4 incrementally",
        ),
        (
            "broadcast() (fire-and-forget)",
            "actors.echo.broadcast('hello')",
            "Fire-and-forget to all actors (no response)",
        ),
    ]
    lines: list[str] = []
    for i, (name, code, desc) in enumerate(patterns):
        lines.append(f"Pattern: {name}")
        lines.append(f"  >>> {code}")
        lines.append(f"  -> {desc}")
        lines.append("")
        comparison.set_left(lines, title="Monarch (4 patterns)")
        progress.update(i + 1, len(patterns), extra_text=f"Pattern: {name}")
        dashboard.log(f"[Monarch] {name}: {desc}")
        dashboard.update_live()
        time.sleep(0.7)


def _simulate_ray_messaging(
    dashboard: DashboardBase, comparison: ComparisonPanel, progress: ProgressPanel
) -> None:
    patterns = [
        (
            ".remote() + ray.get()",
            "ref = actor.echo.remote('hello'); ray.get(ref)",
            "Submit task -> ObjectRef -> resolve -> 'echo: hello'",
        ),
        (
            "Fan-out + ray.get()",
            "refs = [a.echo.remote('hello') for a in actors]; ray.get(refs)",
            "Manual fan-out: O(N) calls -> list of results",
        ),
        (
            "ray.wait() (polling)",
            "done, pending = ray.wait(refs, num_returns=1)",
            "Poll for completion (no native streaming)",
        ),
        (
            "Discard ObjectRef",
            "_ = actor.echo.remote('hello')",
            "Closest to fire-and-forget (ObjectRef still created)",
        ),
    ]
    lines: list[str] = []
    for i, (name, code, desc) in enumerate(patterns):
        lines.append(f"Pattern: {name}")
        lines.append(f"  >>> {code}")
        lines.append(f"  -> {desc}")
        lines.append("")
        comparison.set_right(lines, title="Ray (ObjectRef model)")
        progress.update(i + 1, len(patterns), extra_text=f"Pattern: {name}")
        dashboard.log(f"[Ray] {name}: {desc}")
        dashboard.update_live()
        time.sleep(0.7)


def main() -> None:
    args = parse_args()

    dashboard = DashboardBase(title="Demo 03: Control Plane Messaging", refresh_hz=4.0)
    comparison = ComparisonPanel(PanelConfig(title="Messaging Patterns", border_style="blue"))
    progress = ProgressPanel(PanelConfig(title="Progress", border_style="green"))
    log_panel = LogPanel(PanelConfig(title="Message Flow", border_style="magenta"), max_lines=50)
    dashboard.add_panel("progress", progress)
    dashboard.add_panel("comparison", comparison)
    dashboard.add_panel("log", log_panel)

    def callback(db: DashboardBase) -> None:
        db.log("Starting Control Plane demo")
        db.update_live()

        _simulate_monarch_messaging(db, comparison, progress)
        _simulate_ray_messaging(db, comparison, progress)

        db.log("Demo complete")
        db.update_live()
        time.sleep(1.0)

    dashboard.run(callback)


if __name__ == "__main__":
    main()
