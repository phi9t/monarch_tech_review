#!/usr/bin/env python3
"""Demo 04: Message Ordering — per-sender FIFO (Monarch) vs optional ordering (Ray).

Shows how Monarch guarantees message ordering via Rust-level sequence numbers,
while Ray provides ordering only as a side effect of max_concurrency=1.
Run: ./scripts/zephyr_uv_run.sh python scripts/demos/04_message_ordering.py [--dry-run]
"""

from __future__ import annotations

import argparse
import time

from demo_tui import ComparisonPanel, DashboardBase, KeyValuePanel, LogPanel, PanelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Simulate without clusters")
    return parser.parse_args()


def _simulate_monarch_ordering(
    dashboard: DashboardBase, comparison: ComparisonPanel, kv: KeyValuePanel
) -> None:
    lines: list[str] = [
        "# Monarch: per-sender FIFO guaranteed",
        "# (Rust ordering.rs: BufferState + OrderedSender)",
        "",
    ]
    sent = []
    received = []

    for i in range(8):
        sent.append(i)
        received.append(i)
        lines.append(f"  send(seq={i}) -> received in order: seq={i}")
        kv.set("Sent sequence", str(sent))
        kv.set("Received sequence", str(received))
        kv.set("In order?", "YES (guaranteed)")
        comparison.set_left(lines, title="Monarch (ordered)")
        dashboard.log(f"[Monarch] sent={i}, received={i} (in order)")
        dashboard.update_live()
        time.sleep(0.3)

    lines.append("")
    lines.append("assert received == list(range(8))  # always True")
    comparison.set_left(lines)
    dashboard.update_live()


def _simulate_ray_ordering(
    dashboard: DashboardBase, comparison: ComparisonPanel, kv: KeyValuePanel
) -> None:
    lines: list[str] = [
        "# Ray: no ordering guarantee by default",
        "# max_concurrency=1 gives sequential execution",
        "# _ray_allow_out_of_order_execution = True for throughput",
        "",
    ]

    # Simulate sequential (max_concurrency=1)
    lines.append("# With max_concurrency=1:")
    for i in range(4):
        lines.append(f"  task {i} submitted -> executed sequentially")
        comparison.set_right(lines, title="Ray (configurable)")
        dashboard.log(f"[Ray] max_concurrency=1: task {i} sequential")
        dashboard.update_live()
        time.sleep(0.3)

    lines.append("")
    lines.append("# With allow_out_of_order_execution=True:")
    # Simulate out-of-order
    ooo_order = [2, 0, 3, 1]
    for i, actual in enumerate(ooo_order):
        lines.append(f"  task {i} submitted -> executed as task {actual}")
        comparison.set_right(lines)
        dashboard.log(f"[Ray] out-of-order: submitted {i}, executed {actual}")
        dashboard.update_live()
        time.sleep(0.3)

    kv.set("Ray default", "No ordering guarantee")
    kv.set("Ray max_concurrency=1", "Sequential (side effect)")
    kv.set("Ray out-of-order flag", "_ray_allow_out_of_order_execution")
    dashboard.update_live()


def main() -> None:
    args = parse_args()

    dashboard = DashboardBase(title="Demo 04: Message Ordering", refresh_hz=4.0)
    comparison = ComparisonPanel(PanelConfig(title="Ordering Behavior", border_style="blue"))
    kv = KeyValuePanel(PanelConfig(title="Sequence Numbers", border_style="cyan"))
    log_panel = LogPanel(PanelConfig(title="Send/Receive Log", border_style="magenta"), max_lines=50)
    dashboard.add_panel("kv", kv)
    dashboard.add_panel("comparison", comparison)
    dashboard.add_panel("log", log_panel)

    def callback(db: DashboardBase) -> None:
        db.log("Starting Message Ordering demo")
        db.update_live()

        _simulate_monarch_ordering(db, comparison, kv)
        _simulate_ray_ordering(db, comparison, kv)

        db.log("Demo complete")
        db.update_live()
        time.sleep(1.0)

    dashboard.run(callback)


if __name__ == "__main__":
    main()
