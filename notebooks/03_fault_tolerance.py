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
    from monarch.actor import Actor, endpoint, current_rank, this_host
    return Actor, current_rank, endpoint, random, this_host


@app.cell
def _(mo):
    mo.md(r"""
    # Fault Tolerance in Monarch

    At scale, failures are constant. This notebook shows how to handle them gracefully.

    What you'll learn:

    1. The canonical try/except pattern (covers 90% of cases)
    2. Supervision trees for centralized error surfacing
    3. TorchFT integration for fault-tolerant training at scale
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## When You're Running on Thousands of GPUs

    Something is always failing:

    - GPU memory errors (ECC, OOM)
    - Network hiccups (transient, routing issues)
    - Node crashes (hardware, kernel panics)
    - Software bugs (race conditions, edge cases)

    **The real cost:** A single undetected failure can burn millions of dollars. Silent failures are especially dangerous - your training looks fine but is actually diverging or stalled.

    **The observability challenge:**
    You need monitoring and observability at every layer to detect when systems are healthy or unhealthy. Of course, this is a prime target for automation.

    **Why SPMD makes this hard:**
    With traditional SPMD, automating failure detection and recovery is difficult:

    - Making sense of thousands of ranks, each with their own view
    - Plumbing through logs scattered across nodes
    - Correlating failures across distributed processes
    - No centralized place to catch and handle errors

    **Monarch's approach:** Supervision trees provide centralized error surfacing. When an actor fails, the error bubbles up through a structured hierarchy - giving you one place to see what went wrong and why.
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
    host = this_host()
    procs = host.spawn_procs(per_host={"gpus": 3})
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

    Actors are **stateless from the caller's perspective**:

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

    For infrastructure builders who want **centralized failure handling**.

    The `__supervise__` hook intercepts child actor failures before they propagate.
    Think of it as middleware for failure handling.

    ```
                        ┌─────────────┐
                        │ Supervisor  │
                        │             │
                        │ __supervise__() ◄── failure event
                        └──────┬──────┘           │
                               │ spawns           │
               ┌───────────────┼───────────────┐  │
               ▼               ▼               ▼  │
          [Worker 0]      [Worker 1]      [Worker 2]
             ✓              ✗ FAIL ────────────┘
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### How Supervision Works

    When a child fails, the supervisor's `__supervise__` is called with:

    ```python
    SupervisionEvent:
      - actor_id: Which actor failed
      - actor_status: Running | Stopped | Failed(error)
      - occurred_at: Timestamp
      - message_headers: Context from triggering message
    ```

    The supervisor can:

    - **Return True**: "I handled it" (stops propagation)
    - **Return False**: "Propagate up" (parent's `__supervise__` called)

    ```python
    class WorkerSupervisor(Actor):
        def __init__(self):
            self.failure_count = 0

        async def __supervise__(self, event: SupervisionEvent) -> bool:
            self.failure_count += 1
            print(f"Caught failure #{self.failure_count}: {event.actor_id}")

            if self.failure_count < 5:
                print("  -> Handling locally (restarting worker)")
                return True  # Handled - don't propagate
            else:
                print("  -> Too many failures, propagating up")
                return False  # Propagate to our supervisor
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Failure Chain Visibility

    When failures propagate, Monarch maintains the full chain of causality:

    ```
    ERROR [orchestrator] Actor 'orchestrator' failed:
      └─ UnhandledSupervisionEvent from 'trainer_controller':
           └─ UnhandledSupervisionEvent from 'trainer_rank_3':
                └─ ActorPanic: "CUDA out of memory allocating 2.1GB"
                   at train_step() line 142
                   message_id: "batch_12847"
    ```

    This gives you:

    - **Root cause**: The actual error (OOM on rank 3)
    - **Propagation path**: How the failure bubbled up
    - **Context**: Which message triggered the failure
    - **Actor identity**: Exact actor name and hierarchy

    Traditional distributed systems show scattered logs. Monarch shows **one structured chain**.

    Supervision trees help with error detection - they're foundational to how Monarch integrates with fault-tolerant training systems like TorchFT.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## TorchFT: Fault-Tolerant Training at Scale

    **Why does fault tolerance matter at scale?**

    At large scales, failures become expected, not exceptional:

    > In our Llama3 training runs we experienced **419 interruptions across a 54 day training window** for a 16k GPU training job. This averages to about **one failure every 3 hours**.
    >
    > If we project this out to 10s of thousands of GPUs, this represents a **failure once every hour or more frequently**.

    (from [Introducing PyTorch Monarch](https://pytorch.org/blog/introducing-pytorch-monarch/))

    Restarting the entire job for each of these failures will reduce the effective training time dramatically.

    **TorchFT's approach:** Instead of stopping the whole job, continue training with the healthy replicas while failed ones recover.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Approaches to Fault-Tolerant Training

    The solution is to make training tolerant of replica failures without stopping the entire job.

    **TorchFT** (from PyTorch) provides a way to withstand GPU failures and allow training to continue.

    **Strategy: Hybrid Sharded Data Parallelism**
    Combines fault-tolerant DDP with FSDP v2 and Pipeline Parallelism.

    On failure:

    - Use torchcomms to catch collective-related errors and bubble them up through Monarch
    - Continue training on the next batch without downtime
    - Isolate failures to a single "replica group"
    - Continue training with a subset of the original job (quorum shrinks)

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
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## How Monarch Integrates with TorchFT

    Monarch and TorchFT have a clear separation of concerns:

    **TorchFT: Step-Level Fault Tolerance**

    - Lighthouse (Rust gRPC service) coordinates quorum membership via heartbeats (~100ms)
    - Manager handles checkpoint recovery when replicas join/leave
    - Training continues with reduced world size - no global stop

    **Monarch: Orchestration-Level Recovery**

    - Job allocation (SLURM) and process mesh spawning
    - When a replica completely crashes, Monarch respins it with configurable retry logic
    - First attempts fast process-level restarts, then escalates to new allocation if needed

    **How they connect:**

    1. Monarch spawns trainer processes and sets `TORCHFT_LIGHTHOUSE` env var
    2. TorchFT trainers connect to Lighthouse using that address
    3. If individual trainer crashes → Monarch's orchestration respins the replica
    4. If GPU fails during a training step → TorchFT handles it via quorum shrink

    ```
    ┌─────────────────────────────────────────────────────────────┐
    │              Monarch Orchestration Layer                    │
    │                                                             │
    │  Job Allocation          Replica Lifecycle                  │
    │  ┌──────────────┐       ┌─────────────────────┐             │
    │  │ SLURM /      │       │ Crash detected?     │             │
    │  │ Allocation   │       │ → Respin replica    │             │
    │  └──────────────┘       │ → Retry logic       │             │
    │                         └─────────────────────┘             │
    └─────────────────────────────────────────────────────────────┘
                                  │
                  ┌───────────────┼───────────────┐
                  ▼               ▼               ▼
            ┌──────────┐    ┌──────────┐    ┌──────────┐
            │ Replica  │    │ Replica  │    │ Replica  │
            │ Group 0  │    │ Group 1  │    │ Group 2  │
            └────┬─────┘    └────┬─────┘    └────┬─────┘
                 │               │               │
                 └───────────────┼───────────────┘
                                 ▼
    ┌─────────────────────────────────────────────────────────────┐
    │              TorchFT Fault Tolerance Layer                  │
    │                                                             │
    │  Lighthouse (gRPC)       Per-Step Recovery                  │
    │  ┌──────────────┐       ┌─────────────────────┐             │
    │  │ Heartbeats   │       │ Quorum shrink/grow  │             │
    │  │ (~100ms)     │       │ Checkpoint transfer │             │
    │  │ Quorum       │       │ No global stop      │             │
    │  └──────────────┘       └─────────────────────┘             │
    └─────────────────────────────────────────────────────────────┘
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Case Study: Qwen3-32B Training with Fault Injection

    We ran this code on a **30 node (240 H100s) Coreweave cluster**, using the SLURM scheduler to train Qwen3-32B using torchtitan and TorchFT.

    **Test conditions:**

    - 100 injected failures every 3 minutes
    - Multiple failure modes: segfaults, process kills, NCCL abort, host eviction, GIL deadlock

    **Results:**

    | Metric | Result |
    |--------|--------|
    | Recovery speed vs full SLURM restart | **60% faster** |
    | Avg recovery time (process failures) | **90 seconds** |
    | Avg recovery time (machine failures) | **2.5 minutes** |

    **Why the improvement?**
    Monarch allows for configurable recovery strategies based on failure type — we avoid unnecessary job rescheduling by attempting fast process-level restarts first.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    **Key takeaways:**

    1. **Try/except first**: The canonical pattern covers 90% of cases
    2. **Supervision trees**: Centralized error surfacing - one place to see what went wrong
    3. **Quorum-based training**: Continue with healthy replicas while failed ones recover
    4. **Monarch + TorchFT**: Orchestration-level recovery + step-level fault tolerance

    **When to use what:**

    | Situation | Pattern |
    |-----------|---------|
    | Simple actor calls | Try/except |
    | Building infrastructure | Supervision trees |
    | Massive scale training | TorchFT + Monarch |

    **Coming up:** In the Services notebook, we'll see the try/except pattern in action - building a generator pool that routes around failures and recovers unhealthy replicas.

    **Next:** Async RL - putting it all together with services, RDMA, and async loops
    """)
    return


if __name__ == "__main__":
    app.run()
