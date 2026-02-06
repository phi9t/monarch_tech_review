import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Services: Managing Worker Pools

    In the last notebook, we saw how generation variance makes synchronous RL
    wasteful — the trainer sits idle waiting for the slowest generator. The fix
    is running many generators in parallel, decoupled from training.

    But managing a pool of workers brings its own problems. Let's see what
    happens when you try to do it by hand.
    """)
    return


@app.cell
def _():
    import random
    from monarch.actor import Actor, endpoint, current_rank, this_host, get_or_spawn_controller
    return (
        Actor,
        current_rank,
        endpoint,
        get_or_spawn_controller,
        random,
        this_host,
    )


@app.cell
def _(Actor, current_rank, endpoint, random):
    class Worker(Actor):
        """A worker that processes requests. May fail randomly."""

        def __init__(self, fail_rate: float = 0.0):
            self.rank = current_rank().rank
            self.fail_rate = fail_rate
            self.calls = 0

        @endpoint
        def ping(self) -> bool:
            return True

        @endpoint
        def process(self, data: str) -> str:
            """Process a request. Might fail based on fail_rate."""
            self.calls += 1
            if random.random() < self.fail_rate:
                raise RuntimeError(f"Worker failed!")
            return f"Processed '{data}'"
    return (Worker,)


@app.cell
def _(Worker, this_host):
    # === The Manual Approach ===
    host = this_host()
    procs = host.spawn_procs(per_host={"procs": 4})

    # Spawn 4 workers on individual proc slices
    manual_workers = []
    for i in range(4):
        w = procs.slice(procs=i).spawn(f"w_{i}", Worker, fail_rate=0.3)
        manual_workers.append(w)

    # Manual round-robin with manual failure tracking
    healthy = list(range(4))
    current_idx = 0

    print("=== Manual Worker Management ===")
    for req in range(6):
        if not healthy:
            print(f"  Request {req}: No workers left!")
            break

        worker_idx = healthy[current_idx % len(healthy)]
        try:
            result = manual_workers[worker_idx].process.call_one(f"req_{req}").get()
            print(f"  Request {req}: Worker {worker_idx} -> OK")
            current_idx += 1
        except Exception:
            print(f"  Request {req}: Worker {worker_idx} FAILED — removing from pool")
            healthy.remove(worker_idx)
            # But is it really dead? Or just a transient error?
            # How do we check later? When do we add it back?

    print(f"\n{len(healthy)}/4 workers still in pool")
    print("Questions:")
    print("  Who checks if failed workers recovered?")
    print("  What if multiple clients need this same pool?")
    print("  What if we have 50 workers across 10 nodes?")
    return (host,)


@app.cell
def _(mo):
    mo.md(r"""
    ## The Problems with Manual Management

    That works for 4 workers in a notebook. But at scale:

    - **Shared state**: Multiple clients need the same pool, but each tracks
      its own `healthy` list — when one client discovers a failure, others
      don't know
    - **Recovery**: No mechanism to check if failed workers come back
    - **Discovery**: Clients must know exactly which workers exist at spawn time
    - **Multi-GPU replicas**: A 70B model needs tensor parallelism across 8
      GPUs — each "replica" is a group of workers, not a single process

    The fix: centralize pool management in a **Service** actor. Since it's an
    actor, its state (which replicas are healthy, the round-robin index) is
    automatically shared across all callers — no duplicate tracking.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The Service Actor

    The Service centralizes everything the manual approach did ad-hoc:

    1. Takes a `worker_class` and a ProcMesh, slices it into replica-sized
       chunks, and spawns an ActorMesh on each
    2. Routes requests round-robin across healthy replicas
    3. Tracks health — callers mark replicas unhealthy, the Service can
       probe for recovery

    We treat **1 replica = 1 ActorMesh** — each replica can span multiple
    GPUs for tensor parallelism. If your model fits on one GPU, a replica is
    just size 1.

    The methods fall into three groups:

    - **Routing**: `get_replica()` / `get_replica_with_idx()` — round-robin
      across healthy replicas
    - **Health**: `mark_unhealthy()` / `check_health()` — callers report
      failures, Service probes for recovery
    - **Lifecycle**: `__init__` slices the ProcMesh and spawns workers
    """)
    return


@app.cell
def _(Actor, endpoint):
    class Service(Actor):
        """Manages a pool of worker replicas with routing and health tracking.

        The Service owns worker lifecycle: it slices a ProcMesh into
        replica-sized chunks and spawns an ActorMesh on each.
        """

        def __init__(
            self,
            service_name: str,
            worker_class: type,
            procs,
            procs_per_replica: int = 1,
            **worker_kwargs,
        ):
            self.service_name = service_name
            total = len(procs)
            self.num_replicas = total // procs_per_replica

            # Slice ProcMesh and spawn a replica on each chunk
            self.replicas = []
            for i in range(self.num_replicas):
                start = i * procs_per_replica
                end = start + procs_per_replica
                replica_slice = procs.slice(procs=slice(start, end))
                replica = replica_slice.spawn(
                    f"replica_{i}", worker_class, **worker_kwargs,
                )
                self.replicas.append(replica)

            self.healthy = set(range(self.num_replicas))
            self.unhealthy: set[int] = set()
            self.next_idx = 0

            print(f"[SERVICE:{service_name}] {self.num_replicas} replicas "
                  f"x {procs_per_replica} procs each")

        # --- Routing ---

        @endpoint
        def get_replica(self):
            """Get a healthy replica (round-robin)."""
            if not self.healthy:
                raise RuntimeError("No healthy replicas available!")
            healthy_list = sorted(self.healthy)
            idx = self.next_idx % len(healthy_list)
            self.next_idx += 1
            return self.replicas[healthy_list[idx]]

        @endpoint
        def get_replica_with_idx(self):
            """Get (replica, index). Use the index for mark_unhealthy()."""
            if not self.healthy:
                raise RuntimeError("No healthy replicas available!")
            healthy_list = sorted(self.healthy)
            idx = self.next_idx % len(healthy_list)
            self.next_idx += 1
            replica_idx = healthy_list[idx]
            return self.replicas[replica_idx], replica_idx

        # --- Health tracking ---

        @endpoint
        def mark_unhealthy(self, replica_idx: int) -> None:
            """Remove a replica from rotation."""
            if replica_idx in self.healthy:
                self.healthy.discard(replica_idx)
                self.unhealthy.add(replica_idx)
                print(f"[SERVICE:{self.service_name}] Replica {replica_idx} unhealthy. "
                      f"Healthy: {len(self.healthy)}/{self.num_replicas}")

        @endpoint
        def mark_healthy(self, replica_idx: int) -> None:
            """Reinstate a replica."""
            if replica_idx < self.num_replicas:
                self.unhealthy.discard(replica_idx)
                self.healthy.add(replica_idx)

        @endpoint
        def check_health(self) -> dict:
            """Probe unhealthy replicas. Reinstate those that respond to ping()."""
            recovered = []
            still_unhealthy = []
            for replica_idx in list(self.unhealthy):
                try:
                    # Use .call() not .call_one() — works for multi-GPU replicas too
                    self.replicas[replica_idx].ping.call().get()
                    self.unhealthy.discard(replica_idx)
                    self.healthy.add(replica_idx)
                    recovered.append(replica_idx)
                    print(f"[SERVICE:{self.service_name}] Replica {replica_idx} recovered!")
                except Exception:
                    still_unhealthy.append(replica_idx)
            return {
                "recovered": recovered,
                "still_unhealthy": still_unhealthy,
                "healthy_count": len(self.healthy),
            }

        @endpoint
        def get_health_status(self) -> dict:
            return {
                "total": self.num_replicas,
                "healthy": len(self.healthy),
                "unhealthy": len(self.unhealthy),
                "healthy_indices": sorted(self.healthy),
                "unhealthy_indices": sorted(self.unhealthy),
            }

        # --- Lifecycle ---

        @endpoint
        def get_all_replicas(self) -> list:
            """Get all replicas (for operations that touch every replica)."""
            return list(self.replicas)

        @endpoint
        def ping(self) -> bool:
            return True
    return (Service,)


@app.cell
def _(mo):
    mo.md(r"""
    Key design decisions:

    - **Service spawns workers** — it takes a `worker_class` and a ProcMesh,
      slices it into replica-sized chunks, and spawns an ActorMesh on each.
      Because the Service owns the lifecycle, it could also respawn failed
      replicas.
    - **Caller marks unhealthy** — the Service doesn't auto-detect failures.
      The caller catches the exception and tells the Service. This is the same
      try/except pattern from notebook 03.
    - **Caller registers** — we call `register_service()` after spawning
      because actors can't block on Futures during `__init__`.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Service Discovery

    Multiple parts of the system need the same services — the trainer needs
    generators, the orchestrator needs the trainer. Rather than threading
    references everywhere, we use a **singleton registry**. Monarch's
    `get_or_spawn_controller` guarantees exactly one exists across all
    processes.
    """)
    return


@app.cell
def _(Actor, endpoint, get_or_spawn_controller):
    class ServiceRegistry(Actor):
        """Singleton registry for service discovery."""

        def __init__(self):
            self.services: dict[str, Actor] = {}
            print("[REGISTRY] ServiceRegistry spawned")

        @endpoint
        def register(self, name: str, service) -> None:
            self.services[name] = service
            print(f"[REGISTRY] Registered '{name}'")

        @endpoint
        def get(self, name: str):
            if name not in self.services:
                raise KeyError(f"Service '{name}' not found")
            return self.services[name]

        @endpoint
        def list_services(self) -> list[str]:
            return list(self.services.keys())


    def _get_registry():
        return get_or_spawn_controller("service_registry", ServiceRegistry).get()

    def get_service(name: str):
        """Get a service by name."""
        return _get_registry().get.call_one(name).get()

    def register_service(name: str, service) -> None:
        """Register a service for discovery."""
        _get_registry().register.call_one(name, service).get()

    def list_services() -> list[str]:
        """List all registered services."""
        return _get_registry().list_services.call_one().get()
    return get_service, list_services, register_service


@app.cell
def _(mo):
    mo.md(r"""
    ## Seeing It Work

    First, the happy path — 4 replicas, no failures. Watch the round-robin
    distribute requests evenly.
    """)
    return


@app.cell
def _(Service, Worker, host, register_service):
    # === Demo 1: Round-Robin Routing ===
    svc_proc = host.spawn_procs(per_host={"procs": 1})
    worker_procs_demo = host.spawn_procs(per_host={"procs": 4})

    demo_svc = svc_proc.spawn(
        "demo_svc", Service,
        service_name="demo",
        worker_class=Worker,
        procs=worker_procs_demo,
        procs_per_replica=1,
        fail_rate=0.0,
    )
    register_service("demo", demo_svc)

    print("=== Round-Robin Demo ===")
    for req_num in range(4):
        worker, idx = demo_svc.get_replica_with_idx.call_one().get()
        response = worker.process.call_one(f"req_{req_num}").get()
        print(f"  Request {req_num}: replica {idx} -> {response}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Handling Failures

    Now with a 30% failure rate. The pattern: try, catch, `mark_unhealthy`,
    retry with a different replica. Then `check_health` to probe for recovery.
    """)
    return


@app.cell
def _(Service, Worker, host, register_service):
    # === Demo 2: Failure Handling + Recovery ===
    flaky_svc_proc = host.spawn_procs(per_host={"procs": 1})
    flaky_worker_procs = host.spawn_procs(per_host={"procs": 4})

    flaky_svc = flaky_svc_proc.spawn(
        "flaky_svc", Service,
        service_name="flaky",
        worker_class=Worker,
        procs=flaky_worker_procs,
        procs_per_replica=1,
        fail_rate=0.3,
    )
    register_service("flaky", flaky_svc)

    def process_with_retry(svc, data, max_retries=3):
        """The canonical pattern: try/except + mark_unhealthy + retry."""
        for attempt in range(max_retries):
            replica, idx = None, None
            try:
                replica, idx = svc.get_replica_with_idx.call_one().get()
                return replica.process.call_one(data).get()
            except Exception:
                if idx is not None:
                    print(f"    Attempt {attempt + 1}: replica {idx} failed")
                    svc.mark_unhealthy.call_one(idx).get()
                else:
                    print(f"    Attempt {attempt + 1}: no healthy replicas")
        raise RuntimeError("All retries exhausted")

    print("=== Failure Handling Demo ===")
    for j in range(5):
        print(f"  Request {j}:")
        try:
            _result = process_with_retry(flaky_svc, f"data_{j}")
            print(f"    Success: {_result}")
        except RuntimeError as e:
            print(f"    {e}")

    # Check health after failures
    status = flaky_svc.get_health_status.call_one().get()
    print(f"\nAfter failures: {status['healthy']}/{status['total']} healthy")
    print(f"Unhealthy: {status['unhealthy_indices']}")

    # Recover — workers are still alive, they just had transient errors
    print("\nProbing unhealthy replicas...")
    recovery = flaky_svc.check_health.call_one().get()
    print(f"Recovered: {recovery['recovered']}")

    status_after = flaky_svc.get_health_status.call_one().get()
    print(f"Now: {status_after['healthy']}/{status_after['total']} healthy")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Discovery in Action

    A completely separate part of the system can find services by name —
    no need to pass actor references around.
    """)
    return


@app.cell
def _(get_service, list_services):
    # === Demo 3: Service Discovery ===
    print("=== Service Discovery Demo ===")
    print(f"Registered services: {list_services()}")

    # Find the demo service by name — no reference needed
    discovered = get_service("demo")
    found_worker = discovered.get_replica.call_one().get()
    discovery_result = found_worker.process.call_one("found via discovery!").get()
    print(f"Discovered 'demo' service, got: {discovery_result}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## What We Built

    | Component | Responsibility |
    |-----------|---------------|
    | **Worker** | Processes requests, has `ping()` for health checks |
    | **Service** | Spawns replicas, routes round-robin, tracks health |
    | **ServiceRegistry** | Singleton for discovery (`get_service("name")`) |

    The pattern:
    1. Allocate procs and spawn a Service (which spawns its own workers)
    2. Register the service so clients can discover it by name
    3. Clients get replicas, make calls, handle failures inline
    4. On failure: `mark_unhealthy` + retry with a different replica
    5. Periodically: `check_health` to recover transient failures

    This Service class lives in `src/monarch_utils/services.py` — in notebook
    07, we'll import it and use it to manage real generator workers.

    ## What's Missing?

    The generators need **updated weights** after the trainer takes a gradient
    step. Right now they'd use stale weights forever.

    How do we efficiently move hundreds of megabytes of model weights from
    trainer to generators — without blocking training?

    **Next: RDMA Weight Sync** — using Monarch's RDMA primitives for direct
    GPU-to-GPU weight transfer.
    """)
    return


if __name__ == "__main__":
    app.run()
