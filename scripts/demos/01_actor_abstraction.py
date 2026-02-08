#!/usr/bin/env python3
"""Demo 01: Actor Abstraction — Monarch vs Ray.

Shows actor creation and endpoint calls side by side.
Run: ./scripts/zephyr_uv_run.sh python scripts/demos/01_actor_abstraction.py [--dry-run]
"""

from __future__ import annotations

import argparse
import time

from demo_tui import ComparisonPanel, DashboardBase, LogPanel, PanelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Simulate without Monarch/Ray")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dry-run simulations
# ---------------------------------------------------------------------------

def _simulate_monarch_actor(dashboard: DashboardBase, comparison: ComparisonPanel) -> None:
    """Simulate Monarch actor lifecycle."""
    steps = [
        ("host = this_host()", "Acquiring local HostMesh"),
        ("procs = host.spawn_procs(per_host={'gpus': 2})", "Spawning ProcMesh (2 procs)"),
        ("actors = procs.spawn('pinger', PingPong)", "Spawning ActorMesh 'pinger'"),
        ("result = actors.ping.call_one('hello').get()", "call_one -> 'pong from Ping (count=1)'"),
        ("results = actors.ping.call('hello').get()", "call (broadcast) -> ValueMesh{0: 'pong...', 1: 'pong...'}"),
    ]
    lines: list[str] = []
    for code, desc in steps:
        lines.append(f">>> {code}")
        lines.append(f"    {desc}")
        comparison.set_left(lines, title="Monarch")
        dashboard.log(f"[Monarch] {desc}")
        dashboard.update_live()
        time.sleep(0.5)


def _simulate_ray_actor(dashboard: DashboardBase, comparison: ComparisonPanel) -> None:
    """Simulate Ray actor lifecycle."""
    steps = [
        ("ray.init()", "Connecting to Ray cluster"),
        ("counter = Counter.remote()", "Spawning remote actor (scheduler picks node)"),
        ("ref = counter.increment.remote()", "Submitting remote task -> ObjectRef"),
        ("result = ray.get(ref)", "Resolving ObjectRef -> 1"),
        ("refs = [a.increment.remote() for a in actors]", "Fan-out to multiple actors"),
        ("results = ray.get(refs)", "Gather results -> [1, 1]"),
    ]
    lines: list[str] = []
    for code, desc in steps:
        lines.append(f">>> {code}")
        lines.append(f"    {desc}")
        comparison.set_right(lines, title="Ray")
        dashboard.log(f"[Ray] {desc}")
        dashboard.update_live()
        time.sleep(0.5)


# ---------------------------------------------------------------------------
# Live-cluster execution (requires Monarch + Ray)
# ---------------------------------------------------------------------------

def _run_monarch_actor(dashboard: DashboardBase, comparison: ComparisonPanel) -> None:
    """Run actual Monarch actor demo."""
    from monarch.actor import Actor, current_rank, endpoint, this_host

    class PingPong(Actor):
        def __init__(self):
            self.name = "Ping" if current_rank().rank == 0 else "Pong"
            self.count = 0

        @endpoint
        def ping(self, message: str) -> str:
            self.count += 1
            return f"pong from {self.name} (count={self.count})"

    lines: list[str] = []

    host = this_host()
    lines.append("host = this_host()  # acquired HostMesh")
    comparison.set_left(lines, title="Monarch")
    dashboard.log("[Monarch] Acquired HostMesh")
    dashboard.update_live()

    procs = host.spawn_procs(per_host={"gpus": 2})
    lines.append("procs = host.spawn_procs(per_host={'gpus': 2})")
    comparison.set_left(lines)
    dashboard.log("[Monarch] Spawned ProcMesh")
    dashboard.update_live()

    actors = procs.spawn("pinger", PingPong)
    lines.append("actors = procs.spawn('pinger', PingPong)")
    comparison.set_left(lines)
    dashboard.log("[Monarch] Spawned ActorMesh")
    dashboard.update_live()

    result = actors.ping.call_one("hello").get()
    lines.append(f"call_one -> {result!r}")
    comparison.set_left(lines)
    dashboard.log(f"[Monarch] call_one result: {result}")
    dashboard.update_live()

    results = actors.ping.call("hello").get()
    lines.append(f"call (broadcast) -> {results}")
    comparison.set_left(lines)
    dashboard.log(f"[Monarch] call (broadcast) result: {results}")
    dashboard.update_live()

    procs.stop().get()
    lines.append("procs.stop().get()  # cleanup")
    comparison.set_left(lines)
    dashboard.log("[Monarch] ProcMesh stopped")
    dashboard.update_live()


def _run_ray_actor(dashboard: DashboardBase, comparison: ComparisonPanel) -> None:
    """Run actual Ray actor demo."""
    import ray

    @ray.remote
    class Counter:
        def __init__(self):
            self.count = 0

        def increment(self):
            self.count += 1
            return self.count

    lines: list[str] = []

    ray.init(ignore_reinit_error=True)
    lines.append("ray.init()  # connected")
    comparison.set_right(lines, title="Ray")
    dashboard.log("[Ray] Connected to cluster")
    dashboard.update_live()

    counter = Counter.remote()
    lines.append("counter = Counter.remote()")
    comparison.set_right(lines)
    dashboard.log("[Ray] Spawned Counter actor")
    dashboard.update_live()

    ref = counter.increment.remote()
    result = ray.get(ref)
    lines.append(f"ray.get(counter.increment.remote()) -> {result}")
    comparison.set_right(lines)
    dashboard.log(f"[Ray] increment result: {result}")
    dashboard.update_live()

    actors = [Counter.remote() for _ in range(2)]
    refs = [a.increment.remote() for a in actors]
    results = ray.get(refs)
    lines.append(f"fan-out to 2 actors -> {results}")
    comparison.set_right(lines)
    dashboard.log(f"[Ray] fan-out results: {results}")
    dashboard.update_live()

    ray.shutdown()
    lines.append("ray.shutdown()")
    comparison.set_right(lines)
    dashboard.log("[Ray] Shutdown")
    dashboard.update_live()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    dashboard = DashboardBase(title="Demo 01: Actor Abstraction", refresh_hz=4.0)
    comparison = ComparisonPanel(PanelConfig(title="Actor Abstraction", border_style="blue"))
    comparison.set_left(["(waiting...)"], title="Monarch")
    comparison.set_right(["(waiting...)"], title="Ray")
    log_panel = LogPanel(PanelConfig(title="Timeline", border_style="magenta"), max_lines=50)
    dashboard.add_panel("comparison", comparison)
    dashboard.add_panel("log", log_panel)

    def callback(db: DashboardBase) -> None:
        db.log("Starting Actor Abstraction demo")
        db.update_live()

        if args.dry_run:
            _simulate_monarch_actor(db, comparison)
            _simulate_ray_actor(db, comparison)
        else:
            _run_monarch_actor(db, comparison)
            _run_ray_actor(db, comparison)

        db.log("Demo complete")
        db.update_live()
        time.sleep(1.0)

    dashboard.run(callback)


if __name__ == "__main__":
    main()
