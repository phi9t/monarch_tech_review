import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    # Shared imports for the notebook
    import sys
    import torch
    from monarch.actor import Actor, endpoint, current_rank, this_host
    return Actor, current_rank, endpoint, sys, this_host, torch


@app.cell
def _(mo):
    mo.md(r"""
    # Interactive DevX: Monarch as Remote Torchrun

    What you'll learn:

    1. The pain of traditional distributed development (sbatch → wait → debug → repeat)
    2. SPMDJob as "remote torchrun" - allocate once, iterate fast
    3. The three-layer stack: Boot → Coordination → Application
    4. Running distributed programs interactively with `this_host()`
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The Pain of Traditional Development

    ```
    Traditional Distributed Development Loop:

    Developer                    SLURM                     Cluster
        │                          │                          │
        ├── sbatch job.sh ────────►│                          │
        │                          ├── queue... ──────────────►
        │                          │   (minutes to hours)     │
        │                          │◄─────────────────────────┤
        │◄── job started ──────────┤                          │
        │                          │                          │
        │   ...wait for completion...                         │
        │                          │                          │
        ├── cat slurm-*.out ──────►│                          │
        │◄── scattered logs ───────┤                          │
        │                          │                          │
        │   "Found bug on line 42"                            │
        │                          │                          │
        └── sbatch again... ──────►│   (repeat forever)       │
    ```

    **Key problems:**

    - Queue wait time dominates iteration time
    - Logs scattered across nodes
    - Each fix requires full resubmission
    - No interactive debugging
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The Monarch Solution

    ```
    Monarch Development Loop:

    Developer                    Monarch                   Cluster
        │                          │                          │
        ├── allocate hosts ───────►│                          │
        │   (one time, slow)       ├── gang schedule ────────►│
        │                          │◄─────────────────────────┤
        │◄── HostMesh ready ───────┤                          │
        │                          │                          │
        │   === Fast iteration loop ===                       │
        │                          │                          │
        ├── spawn_procs() ────────►│   (instant)              │
        ├── spawn actors ─────────►│   (instant)              │
        ├── call endpoints ───────►│   (instant)              │
        │◄── aggregated logs ──────┤                          │
        │                          │                          │
        │   "Found bug, fixing..."                            │
        │                          │                          │
        ├── spawn_procs() again ──►│   (instant, same hosts!) │
        └── ...                    │                          │
    ```

    **Key insight:** Allocation is slow, but spawning on existing hosts is fast.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## What's Really Happening Under the Hood

    When you call `SlurmJob(meshes={"trainers": 4}).apply()`, here's what actually happens:

    ```
    1. Job Submission: Monarch submits an sbatch script to SLURM
       ┌────────────────────────────────────────────────────────────┐
       │  #!/bin/bash                                               │
       │  #SBATCH --nodes=4                                         │
       │  srun python -c 'from monarch.actor import                 │
       │       run_worker_loop_forever;                             │
       │       run_worker_loop_forever(                             │
       │           address="tcp://$(hostname):22222",               │
       │           ca="trust_all_connections")'                     │
       └────────────────────────────────────────────────────────────┘

    2. Worker Startup: Each node runs run_worker_loop_forever
       ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
       │  Node 0  │  │  Node 1  │  │  Node 2  │  │  Node 3  │
       │  :22222  │  │  :22222  │  │  :22222  │  │  :22222  │
       │ waiting  │  │ waiting  │  │ waiting  │  │ waiting  │
       └──────────┘  └──────────┘  └──────────┘  └──────────┘

    3. Client Connection: job.state() calls attach_to_workers(...)
       ┌─────────────────────────────────────────────────────────┐
       │  Your notebook / script                                 │
       │                                                         │
       │  host_mesh = attach_to_workers(                         │
       │      workers=["tcp://node0:22222", "tcp://node1:22222", │
       │               "tcp://node2:22222", "tcp://node3:22222"],│
       │      ca="trust_all_connections"                         │
       │  )                                                      │
       └─────────────────────────────────────────────────────────┘
    ```

    **The key insight: Workers are just servers waiting for work.**

    The scheduler (SLURM, Kubernetes, etc.) handles launching them.
    The client (your notebook/script) connects and orchestrates.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Live Demo: Workers as Servers

    Let's see this in action. We'll use `create_local_host_mesh` which spawns
    worker processes locally - exactly what SlurmJob does on a real cluster.
    """)
    return


@app.cell
def _(Actor, current_rank, endpoint, sys):
    from monarch._src.actor.host_mesh import create_local_host_mesh
    from monarch._src.actor.endpoint import Extent

    class BootstrapDemo(Actor):
        """Actor that reports where it's running."""

        def __init__(self):
            self.rank = current_rank().rank
            print(f"[BootstrapDemo] Actor initialized on rank {self.rank}")

        @endpoint
        def whoami(self) -> dict:
            return {"rank": self.rank, "python": sys.executable}

    # This simulates what SlurmJob does:
    # 1. Spawn worker processes (like run_worker_loop_forever on each node)
    # 2. Connect to them (like attach_to_workers)
    # 3. Return a HostMesh
    print("Creating host mesh with 3 workers...")
    demo_host_mesh = create_local_host_mesh(
        extent=Extent(["hosts"], [3]),
        env={"PYTHONPATH": ":".join(sys.path)},
    )
    print("Host mesh created - workers are now listening!\n")

    # Spawn processes and actors
    demo_procs = demo_host_mesh.spawn_procs(per_host={"gpus": 1})
    demo_actors = demo_procs.spawn("bootstrap_demo", BootstrapDemo)

    # Call all actors
    print("\nCalling whoami() on all actors:")
    demo_results = demo_actors.whoami.call().get()
    for point, info in demo_results.items():
        print(f"  {point}: rank={info['rank']}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The Three-Layer Stack

    ```
    ┌─────────────────────────────────────────────────────┐
    │               APPLICATION LAYER                     │
    │   ActorMesh[T] - Your training code as typed actors │
    │   Example: trainer.train.call(batch).get()          │
    ├─────────────────────────────────────────────────────┤
    │              COORDINATION LAYER                     │
    │   ProcMesh - Manages process lifecycle, ranks       │
    │   HostMesh - Manages host lifecycle, resources      │
    │   Example: host_mesh.spawn_procs(name="run_1")      │
    ├─────────────────────────────────────────────────────┤
    │               BOOT LAYER                            │
    │   Bootstrap - Config serialization via env vars     │
    │   Channel - Async message transport (hyperactor)    │
    │   Safety: pdeathsig ensures no orphaned processes   │
    └─────────────────────────────────────────────────────┘
    ```

    **Code pattern:**

    ```python
    from monarch.job import SlurmJob

    # Coordination layer - allocate hosts (slow, once)
    job = SlurmJob(meshes={"workers": 4}, gpus_per_node=8)
    job.apply()
    host_mesh = job.state().workers

    # Spawn processes (fast)
    proc_mesh = host_mesh.spawn_procs(per_host={"gpus": 8})

    # Application layer
    trainers = proc_mesh.spawn("trainer", TrainerActor)      # Instant
    trainers.train.call(dataset).get()                       # Your code

    # Iterate without re-allocating - spawn new procs on same hosts
    proc_mesh_2 = host_mesh.spawn_procs(per_host={"gpus": 8})  # Fast!
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The Simple API: `this_host()`

    The bootstrap demo above used internal APIs to show the mechanics.
    For actual development, Monarch provides a simpler interface: `this_host()`.

    This gives you a local host that can spawn multiple processes - same pattern,
    cleaner API. This is how we'll run all our examples in this presentation.
    """)
    return


@app.cell
def _(Actor, current_rank, endpoint, this_host, torch):
    class Worker(Actor):
        """A simple worker that knows its rank."""

        def __init__(self):
            self.rank = current_rank().rank
            print(f"Worker initialized on rank {self.rank}")

        @endpoint
        def compute(self, data: torch.Tensor) -> dict:
            """Do some computation and return info about this worker."""
            result = data.sum().item() * (self.rank + 1)
            return {
                "rank": self.rank,
                "input_shape": tuple(data.shape),
                "result": result,
            }

    # Spawn 4 worker processes locally
    host = this_host()
    procs = host.spawn_procs(per_host={"gpus": 4})
    workers = procs.spawn("workers", Worker)

    print("\n=== Calling all workers ===")
    # Call all workers with the same data
    data = torch.randn(10, 10)
    results = workers.compute.call(data).get()

    for _point, _result in results.items():
        print(f"Worker {_result['rank']}: computed {_result['result']:.2f}")
    return (workers,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Working with Individual Actors

    When you spawn actors on a mesh, you get an ActorMesh.
    You can slice it to talk to individual actors or subsets.
    """)
    return


@app.cell
def _(torch, workers):
    # Using workers from previous cell

    # Call all workers
    all_results = workers.compute.call(torch.ones(5, 5)).get()
    print(f"All workers returned: {list(all_results.values())}")

    # Call just worker 0
    worker_0 = workers.slice(gpus=0)
    result_0 = worker_0.compute.call_one(torch.ones(5, 5)).get()
    print(f"Worker 0 alone: {result_0}")

    # Call workers 1 and 2
    subset = workers.slice(gpus=slice(1, 3))
    subset_results = subset.compute.call(torch.ones(5, 5)).get()
    print(f"Workers 1-2 returned: {list(subset_results.values())}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## All Logs in One Place

    When actors print, their output is aggregated back to your controller.
    No more SSH-ing into nodes to check logs!
    """)
    return


@app.cell
def _(Actor, current_rank, endpoint, this_host):
    import time

    class VerboseWorker(Actor):
        def __init__(self):
            self.rank = current_rank().rank
            print(f"[Rank {self.rank}] Initializing...")

        @endpoint
        def do_work(self, task_id: int) -> str:
            print(f"[Rank {self.rank}] Starting task {task_id}")
            time.sleep(0.1 * (self.rank + 1))  # Simulate varying work
            print(f"[Rank {self.rank}] Completed task {task_id}")
            return f"rank_{self.rank}_task_{task_id}"

    # Spawn workers
    verbose_procs = this_host().spawn_procs(per_host={"gpus": 3})
    verbose_workers = verbose_procs.spawn("verbose", VerboseWorker)

    print("\n=== Running tasks ===")
    # All workers work on the same task - watch the interleaved logs!
    verbose_results = verbose_workers.do_work.call(42).get()

    print("\n=== Results ===")
    for _point, _result in verbose_results.items():
        print(f"  {_result}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## When Things Go Wrong

    Errors from remote actors are brought back to your controller with full tracebacks.
    """)
    return


@app.cell
def _(Actor, current_rank, endpoint, this_host):
    class FlakyWorker(Actor):
        def __init__(self):
            self.rank = current_rank().rank

        @endpoint
        def risky_operation(self, fail_on_rank: int) -> str:
            if self.rank == fail_on_rank:
                raise ValueError(f"Intentional failure on rank {self.rank}!")
            return f"success on rank {self.rank}"

    flaky_procs = this_host().spawn_procs(per_host={"gpus": 3})
    flaky_workers = flaky_procs.spawn("flaky", FlakyWorker)

    # This will fail on rank 1
    try:
        flaky_results = flaky_workers.risky_operation.call(1).get()
    except Exception as e:
        print(f"Caught error: {type(e).__name__}")
        print(f"Message: {e}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Putting It Together

    The Monarch development workflow:

    1. **Start session**: `this_host().spawn_procs()` (or allocate real hosts)
    2. **Write actor code**: Define your `Actor` classes with `@endpoint` methods
    3. **Spawn and test**: `procs.spawn("name", MyActor)` then call endpoints
    4. **See errors immediately**: Full tracebacks in your terminal
    5. **Fix and respawn**: No queue wait, no reallocation

    This is what we mean by "interactive distributed development."
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    **Key takeaways:**

    1. **Decouple allocation from iteration**: Allocate hosts once, spawn fast
    2. **Three-layer stack**: Boot → Coordination → Application
    3. **`this_host()`**: Local development with the same patterns as real clusters
    4. **Mesh operations**: `call()`, `slice()`, `call_one()` for flexible actor interaction
    5. **Aggregated everything**: Logs and errors come back to you

    **Next:** Fault Tolerance - what happens when things go wrong at scale
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    **TODO:** Live demo of `monarch run`
    """)
    return


if __name__ == "__main__":
    app.run()
