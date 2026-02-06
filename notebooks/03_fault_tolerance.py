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
    import random
    from monarch.actor import Actor, endpoint, current_rank, MeshFailure, this_host
    return Actor, MeshFailure, current_rank, endpoint, random, this_host


@app.cell
def _(mo):
    mo.md(r"""
    # Fault Tolerance in Monarch

    In the last notebook, you caught a single error with try/except on a
    FlakyWorker. That works when you know which actor failed and have a backup
    ready. But when failures hit every 3 hours across 16,000 GPUs, you need
    patterns that scale.

    This notebook covers three layers of fault handling — from simple to
    industrial:

    1. **Try/except with retry** — the 90% solution you'll reach for first
    2. **Supervision trees** — centralized error surfacing for infrastructure builders
    3. **TorchFT** — quorum-based training that continues through failures
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The Simple Pattern: Try/Except

    This is what you should reach for first. It's familiar, explicit, and works great.
    """)
    return


@app.cell
def _(Actor, current_rank, endpoint, random, this_host):
    class UnreliableWorker(Actor):
        """A worker that sometimes fails."""

        def __init__(self, failure_rate: float = 0.3):
            self.rank = current_rank().rank
            self.failure_rate = failure_rate
            self.call_count = 0

        @endpoint
        def process(self, data: int) -> dict:
            self.call_count += 1
            if random.random() < self.failure_rate:
                raise RuntimeError(f"Random failure on rank {self.rank}, call {self.call_count}")
            return {"rank": self.rank, "result": data * 2, "calls": self.call_count}

    # Spawn workers
    procs = this_host().spawn_procs(per_host={"gpus": 3})
    workers = procs.spawn("unreliable", UnreliableWorker, 0.3)

    # The canonical pattern: try/except with fallback
    def process_with_retry(workers_mesh, data, max_retries=3):
        """Try each worker until one succeeds."""
        errors = []
        for attempt in range(max_retries):
            worker = workers_mesh.slice(gpus=attempt % 3)  # Round-robin through workers
            try:
                result = worker.process.call_one(data).get()
                print(f"  Attempt {attempt + 1}: Success! {result}")
                return result
            except Exception as e:
                errors.append(str(e))
                print(f"  Attempt {attempt + 1}: Failed - {type(e).__name__}")
        raise RuntimeError(f"All {max_retries} attempts failed")

    print("=== Processing with retry ===")
    for i in range(5):
        print(f"\nTask {i}:")
        try:
            _result = process_with_retry(workers, i * 10)
        except RuntimeError as e:
            print(f"  All retries exhausted: {e}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Why This Simple Pattern Works

    Actors are designed to be **stateless from the caller's perspective**:

    - If one fails, try another
    - The caller doesn't need to know implementation details
    - Each actor can maintain its own state internally

    This is the **90% solution**. Use it unless you have a specific reason not to.

    ```python
    # The pattern in one line
    result = try_call(actor1) or try_call(actor2) or try_call(actor3)
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Supervision Trees: Centralized Error Surfacing

    Try/except works when the **caller** catches failures. But what about failures
    you don't directly trigger — a child actor crashing in the background, a
    cascading failure deep in a hierarchy?

    Monarch's answer is **supervision**. When an actor spawns children, it becomes
    their supervisor. If a child fails, the supervisor's `__supervise__` method is
    called with a `MeshFailure` describing what went wrong.

    Think of it like a management chain: when an employee hits a problem they can't
    solve, it escalates to their manager. The manager can handle it (return `True`)
    or escalate further (return `False`).
    """)
    return


@app.cell
def _(Actor, MeshFailure, endpoint, this_host):
    import asyncio
    import monarch.actor

    class ActorCrash(BaseException):
        """Inherits from BaseException (not Exception) to trigger actor death.
        Regular Exceptions are application errors — returned to the caller,
        actor stays alive. BaseException causes the actor to crash, which
        triggers the supervision tree."""
        pass

    class FragileWorker(Actor):
        """A worker that crashes when asked."""

        @endpoint
        async def do_work(self) -> str:
            raise ActorCrash("GPU memory error on this worker!")

    class Supervisor(Actor):
        """Spawns and supervises children. Catches their failures automatically."""

        def __init__(self, child_procs):
            # Supervisor owns these children because it calls spawn()
            self.workers = child_procs.spawn("fragile", FragileWorker)
            self.failure_log = []
            self._supervised = asyncio.Event()

        def __supervise__(self, failure: MeshFailure) -> bool:
            """Called automatically when an owned child actor fails."""
            self.failure_log.append(str(failure))
            print(f"[Supervisor] Caught failure: {failure}")
            self._supervised.set()
            return True  # Handled — don't propagate further

        @endpoint
        async def crash_a_child(self):
            """Tell the child to crash, then wait for supervision to fire."""
            self.workers.do_work.broadcast()  # fire-and-forget — triggers supervision
            await asyncio.wait_for(self._supervised.wait(), timeout=30)

        @endpoint
        async def get_failure_log(self) -> list:
            """Return failures caught by __supervise__ (as strings)."""
            return list(self.failure_log)

    # Override the default fault hook (which calls sys.exit) so unhandled
    # supervision events don't kill our notebook
    original_hook = monarch.actor.unhandled_fault_hook
    monarch.actor.unhandled_fault_hook = lambda f: print(f"[Client] Unhandled fault: {f}")

    # Create two separate proc meshes (supervisor and children MUST be on different meshes)
    supervisor_procs = this_host().spawn_procs(per_host={"gpus": 1})
    child_procs = this_host().spawn_procs(per_host={"gpus": 1})

    # Spawn supervisor, passing child procs for it to own
    supervisor = supervisor_procs.spawn("supervisor", Supervisor, child_procs)

    # Tell supervisor to trigger the child — broadcast() means the error has
    # no caller to return to, so the actor crashes and supervision fires
    supervisor.crash_a_child.call_one().get()

    # Query the supervisor's failure log
    failures = supervisor.get_failure_log.call_one().get()
    print(f"\n=== Supervisor's failure log ({len(failures)} entries) ===")
    for j, f in enumerate(failures):
        print(f"  [{j}] {f}")

    if failures:
        print("\n__supervise__ caught the child's failure automatically!")

    # Restore original fault hook
    monarch.actor.unhandled_fault_hook = original_hook
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### What Just Happened

    The supervisor **owns** the `FragileWorker` because it called `spawn()`. When
    the child raised `ActorCrash` (a `BaseException`), the actor died and Monarch
    delivered a `MeshFailure` to the supervisor's `__supervise__` method.

    **Why `BaseException`?** In Monarch, regular `Exception` subclasses are
    application errors — they're returned to the caller and the actor stays alive.
    `BaseException` subclasses signal a fatal crash: the actor dies, and the
    supervision tree is notified.

    Key points:

    - `__supervise__` receives a `MeshFailure` with `.report()` for readable output
    - Return `True` -> "I handled it" (stops propagation up the tree)
    - Return `False` -> escalate to the supervisor's supervisor
    - If no one handles it, the failure reaches the client as an unhandled fault

    This gives you **one structured place** to see what went wrong — instead of
    grepping through logs scattered across nodes.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### From One Supervisor to a Tree

    You just saw a single supervisor catch one child's failure. Now imagine this
    at scale — supervision forms a **tree** where failures propagate upward:

    ```
    Your Script (root)
    └── Orchestrator
        ├── TrainerSupervisor
        │   ├── Trainer Rank 0  ✓
        │   ├── Trainer Rank 1  ✓
        │   └── Trainer Rank 2  ✗ OOM
        └── GeneratorSupervisor
            ├── Generator 0     ✓
            └── Generator 1     ✓
    ```

    When Trainer Rank 2 crashes, `TrainerSupervisor.__supervise__` fires first.
    If it returns `True`, the failure is handled locally — the Orchestrator never
    even knows. If it returns `False`, the TrainerSupervisor itself fails, and the
    event propagates up to the Orchestrator.

    This is just like exception handling in a call stack:

    ```python
    try:                          # Orchestrator.__supervise__
        try:                      # TrainerSupervisor.__supervise__
            raise OOMError()      # Trainer Rank 2 crashes
        except OOMError:          #   -> handle or re-raise
            handle_or_reraise()
    except Exception:             #   -> Orchestrator handles or crashes
        handle_or_crash()
    ```

    The payoff: instead of grepping through 16,000 log files, you get **one
    structured chain** showing exactly what failed and how the failure propagated.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## TorchFT: Fault-Tolerant Training at Scale

    Recall the 419 interruptions from Notebook 1 — at that scale, restarting
    the entire job each time is unacceptable. You need training to **continue
    with healthy replicas** while failed ones recover.

    The result: **60% faster recovery** than full SLURM restarts, with process
    failures recovering in **~90 seconds**. Here's how.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### How TorchFT Works

    TorchFT provides quorum-based training — instead of stopping the whole job
    when a replica fails, the remaining replicas continue:

    ```
    Traditional Training:
    ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
    │Replica 0│ │Replica 1│ │Replica 2│ │Replica 3│
    └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
         │           │           │           │
         └───────────┴─────┬─────┴───────────┘
                           ▼
                  All-reduce every step
                  Single failure → restart everything

    TorchFT Quorum-Based Training:
    ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
    │Replica 0│ │Replica 1│ │Replica 2│ │Replica 3│
    │ healthy │ │  FAILED │ │ healthy │ │ healthy │
    └────┬────┘ └─────────┘ └────┬────┘ └────┬────┘
         │                       │           │
         └───────────┬───────────┴───────────┘
                     ▼
            Quorum shrinks to 3 replicas
            Training continues without stopping
            Failed replica recovers and rejoins
    ```

    On failure, TorchFT catches collective errors through its own
    ManagedProcessGroup, isolates the failure to a single replica group, and
    continues training without restarting the entire job.

    A Lighthouse service (Rust gRPC) coordinates quorum membership via
    heartbeats (~100ms), while a Manager handles checkpoint recovery when
    replicas join or leave.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Monarch + TorchFT: Separation of Concerns

    TorchFT handles **step-level fault tolerance** — catching errors mid-training
    and shrinking the quorum. Monarch handles **orchestration-level recovery** —
    respawning crashed replicas, managing job allocation, and escalating from
    fast process restarts to full reallocation when needed.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Case Study: Qwen3-32B Training with Fault Injection

    We ran this on a **30 node (240 H100s) Coreweave cluster**, training
    Qwen3-32B using torchtitan and TorchFT.

    **Test conditions:**

    - Failures injected at ~3-minute intervals, 100 total events
    - Multiple failure modes: segfaults, process kills, NCCL abort, host
      eviction, GIL deadlock

    **Results:**

    | Metric | Result |
    |--------|--------|
    | Recovery speed vs full SLURM restart | **60% faster** |
    | Avg recovery time (process failures) | **90 seconds** |
    | Avg recovery time (machine failures) | **2.5 minutes** |

    **Why the improvement?**
    Monarch allows for configurable recovery strategies based on failure type —
    we avoid unnecessary job rescheduling by attempting fast process-level
    restarts first.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    **Key takeaways:**

    1. **Try/except first**: The canonical pattern covers 90% of cases
    2. **Supervision trees**: Centralized error surfacing — one place to see
       what went wrong
    3. **Quorum-based training**: Continue with healthy replicas while failed
       ones recover
    4. **Monarch + TorchFT**: Orchestration-level recovery + step-level fault
       tolerance

    These three patterns compose — try/except is your first line of defense,
    supervision gives you visibility, and TorchFT handles training-specific
    recovery at scale.

    We've built patterns to handle failures. Next: putting it all together —
    services, weight sync, and async RL loops.
    """)
    return


if __name__ == "__main__":
    app.run()
