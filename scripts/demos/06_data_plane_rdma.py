#!/usr/bin/env python3
"""Demo 06: Data Plane — RDMA (Monarch) vs Object Store (Ray).

Shows one-sided RDMA reads vs Plasma object store for weight transfer.
Run: ./scripts/zephyr_uv_run.sh python scripts/demos/06_data_plane_rdma.py [--dry-run]
"""

from __future__ import annotations

import argparse
import time

from demo_tui import ComparisonPanel, DashboardBase, KeyValuePanel, LogPanel, PanelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Simulate without RDMA hardware")
    return parser.parse_args()


def _simulate_monarch_rdma(
    dashboard: DashboardBase, comparison: ComparisonPanel, kv: KeyValuePanel
) -> None:
    lines: list[str] = [
        "Architecture: Control Plane + Data Plane separated",
        "",
        "Control plane: actor endpoint messages (~small)",
        "Data plane: RDMABuffer (~150-byte handle)",
        "",
    ]
    comparison.set_left(lines, title="Monarch (RDMA)")
    dashboard.update_live()
    time.sleep(0.5)

    steps = [
        ("Register weights", "buf = RDMABuffer(weights.view(uint8))", "150 bytes registered"),
        ("Get handle", "handle = trainer.get_weight_handle.call_one().get()", "~150B via control plane"),
        ("RDMA read", "handle.read_into(local_buf).get()", "One-sided: bypass remote CPU"),
        ("Load weights", "model.load_state_dict(local_buf)", "Zero-copy transfer complete"),
    ]
    for name, code, desc in steps:
        lines.append(f"  >>> {code}")
        lines.append(f"      {desc}")
        comparison.set_left(lines)
        kv.set("Monarch handle size", "~150 bytes")
        kv.set("Monarch transfer", "One-sided RDMA (CPU-bypass)")
        kv.set("Monarch interruption", "None to trainer")
        dashboard.log(f"[Monarch] {name}: {desc}")
        dashboard.update_live()
        time.sleep(0.5)

    lines.append("")
    lines.append("  +-----------+     RDMA read     +-----------+")
    lines.append("  | Inference | =================> |  Trainer  |")
    lines.append("  | (pulls)   |   (one-sided)      | (GPU mem) |")
    lines.append("  +-----------+                     +-----------+")
    comparison.set_left(lines)
    dashboard.update_live()


def _simulate_ray_object_store(
    dashboard: DashboardBase, comparison: ComparisonPanel, kv: KeyValuePanel
) -> None:
    lines: list[str] = [
        "Architecture: Plasma Object Store (shared memory)",
        "",
        "All data goes through object store serialization",
        "",
    ]
    comparison.set_right(lines, title="Ray (Object Store)")
    dashboard.update_live()
    time.sleep(0.5)

    steps = [
        ("Serialize weights", "weights = trainer.get_weights.remote()", "Full state_dict serialized"),
        ("Store in Plasma", "ray.put() -> ObjectRef", "Copied to shared memory"),
        ("Fetch weights", "w = ray.get(weights_ref)", "Deserialized from Plasma"),
        ("Load weights", "model.load_state_dict(w)", "Full copy complete"),
    ]
    for name, code, desc in steps:
        lines.append(f"  >>> {code}")
        lines.append(f"      {desc}")
        comparison.set_right(lines)
        kv.set("Ray transfer size", "Full model state_dict")
        kv.set("Ray transfer", "Plasma serialize/deserialize")
        kv.set("Ray interruption", "Trainer must serialize")
        dashboard.log(f"[Ray] {name}: {desc}")
        dashboard.update_live()
        time.sleep(0.5)

    lines.append("")
    lines.append("  +-----------+     ray.get()      +-----------+")
    lines.append("  | Inference | <---- Plasma -----> |  Trainer  |")
    lines.append("  | (fetches) |   (serialize)       | (CPU+GPU) |")
    lines.append("  +-----------+                     +-----------+")
    comparison.set_right(lines)
    dashboard.update_live()


def main() -> None:
    args = parse_args()

    dashboard = DashboardBase(title="Demo 06: Data Plane", refresh_hz=4.0)
    comparison = ComparisonPanel(PanelConfig(title="Data Transfer Architecture", border_style="blue"))
    kv = KeyValuePanel(PanelConfig(title="Transfer Stats", border_style="cyan"))
    log_panel = LogPanel(PanelConfig(title="Transfer Log", border_style="magenta"), max_lines=50)
    dashboard.add_panel("kv", kv)
    dashboard.add_panel("comparison", comparison)
    dashboard.add_panel("log", log_panel)

    def callback(db: DashboardBase) -> None:
        db.log("Starting Data Plane demo")
        db.update_live()

        _simulate_monarch_rdma(db, comparison, kv)
        _simulate_ray_object_store(db, comparison, kv)

        db.log("Demo complete")
        db.update_live()
        time.sleep(1.0)

    dashboard.run(callback)


if __name__ == "__main__":
    main()
