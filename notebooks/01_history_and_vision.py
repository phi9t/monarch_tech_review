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
    # Monarch: History & Vision

    What you'll learn:

    1. Why Monarch exists (the tensor engine origin story)
    2. The actor model and why it matters for distributed ML
    3. How Monarch scales to millions of actors without a global routing table
    4. Your first Monarch program: ping-pong actors
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The Tensor Engine Origin Story

    Monarch began as a **tensor engine** for distributed PyTorch.

    Key insight: **Single controller** vs SPMD (every rank runs same script)

    ```
    ┌──────────────────────────────────────────────────────────────────┐
    │                    CONTROLLER (Python)                           │
    │                                                                  │
    │   # Create a mesh: 4 hosts × 8 GPUs = 32 GPUs total              │
    │   mesh = DeviceMesh(hosts=4, gpus_per_host=8, dims=("dp", "tp")) │
    │                                                                  │
    │   with mesh.activate():                                          │
    │       loss = model(X)                                            │
    │       loss.backward()                                            │
    │       p.grad.reduce_("dp", reduction="avg")  # all-reduce        │
    │       optimizer.step()                                           │
    └───────────────────────────┬──────────────────────────────────────┘
                                │ tensor commands
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
       ┌─────────┐         ┌─────────┐         ┌─────────┐
       │ Host 0  │         │ Host 1  │         │ Host 2  │  ...
       │ 8 GPUs  │         │ 8 GPUs  │         │ 8 GPUs  │
       │ dp=0    │         │ dp=1    │         │ dp=2    │
       └─────────┘         └─────────┘         └─────────┘
    ```

    **What makes this special:**

    Monarch was designed as a **single-controller, DTensor-native** execution framework. One Python script on your laptop orchestrates thousands of GPUs - no SPMD rank-checking, no scattered logs.

    **Example pipeline parallelism implementation:**

    ```python
    # Controller script - runs on one machine, orchestrates many
    with pp_meshes[0].activate():
        Y = Y.to_mesh(pp_meshes[-1])       # move data between pipeline stages
        logits = model(X)
        loss = loss_fn(logits, Y)

    with mesh.activate():
        for p in model.parameters():
            p.grad.reduce_("dp", reduction="avg")  # all-reduce across data parallel
        optimizer.step()
    ```

    The Tensor Engine still exists today but is no longer the only mode of execution - Monarch also exposes the underlying actor primitives directly.

    **Key point:** Complex control flow (branching, loops, state machines) is natural in Monarch.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Evolution to Actors

    As we looked to support **RL workloads**, we discovered something:

    The **actor model** that underpins the tensor engine design was more generalizable. RL systems need:

    - Multiple heterogeneous actors (trainers, generators, reward models)
    - Async communication patterns (not just all-reduce)
    - Dynamic composition (add/remove actors at runtime)

    So Monarch evolved to bring those actor capabilities out to Python, ultimately becoming a **general-purpose, high-performance actor framework**.

    **What this means in practice:**

    ```python
    # The tensor engine is still there for training
    with mesh.activate():
        loss = model(X)
        loss.backward()
        p.grad.reduce_("dp", reduction="avg")

    # But now you can also build arbitrary actor systems
    # Spawn actors across processes
    host = this_host()
    trainer_procs = host.spawn_procs({"procs": 1})
    generator_procs = host.spawn_procs({"procs": 4})
    reward_procs = host.spawn_procs({"procs": 1})

    trainer = trainer_procs.spawn("trainer", TrainerActor)
    generators = generator_procs.spawn("generators", GeneratorActor)
    reward_model = reward_procs.spawn("reward", RewardActor)

    # Compose them however you want
    for batch in training_loop:
        # Call all generators (get list of results)
        samples = generators.generate.call(prompts).get()
        rewards = reward_model.score.call_one(samples).get()
        trainer.train_step.call_one(samples, rewards).get()
    ```

    Monarch was designed with ambitions to run truly large-scale pre-training workloads. This requires support for millions of actors with serious attention to scaling and performance.

    Let's demystify Monarch by looking at the architecture that underpins everything.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Hyperactor - The Rust Runtime

    Hyperactor is the Rust actor runtime built on [Tokio](https://tokio.rs/), designed to scale to millions of actors.

    **Why Rust + Tokio:**

    | Feature | Benefit |
    |---------|---------|
    | No GC pauses | Training can't afford stop-the-world |
    | Lightweight async | Millions of actors per machine (not OS threads) |
    | Work-stealing | Tokio balances load across cores |
    | Memory safety | Rust prevents data races at compile time |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The Monarch Ontology

    The strict hierarchy that makes everything work:

    ```
    WORLD (gang-scheduled group of processes)
    ├── PROC 0 (single actor runtime instance)
    │   ├── Actor "trainer"
    │   │   ├── Port 0 (train endpoint)
    │   │   ├── Port 1 (get_weights endpoint)
    │   │   └── Port 2 (sync endpoint)
    │   └── Actor "metrics"
    │       └── Port 0 (report endpoint)
    ├── PROC 1
    │   └── Actor "generator_0"
    │       ├── Port 0 (generate endpoint)
    │       └── Port 1 (update_weights endpoint)
    └── PROC 2
        └── Actor "generator_1"
    ```

    **Definitions:**

    - **World**: Fixed group of processes scheduled together (gang scheduling)
    - **Proc**: Single actor runtime instance, owns MailboxMuxer for local routing
    - **Actor**: Independent async unit with its own mailbox, communicates only through messages
    - **Port**: Typed message endpoint (`@endpoint` decorator creates a port)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Why It Scales - Hierarchical Addressing

    Every entity has a hierarchical ID that **encodes the routing path**:

    ```
    ActorId = (ProcId, actor_name, pid)
    ProcId  = Ranked(WorldId, rank) | Direct(ChannelAddr, name)
    ```

    **The full stack (HostMesh → ProcMesh → ActorMesh):**

    ```
    ┌─────────────────────────────────────────────────────────────────────┐
    │  HostMesh: hosts=4                                                  │
    │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐    │
    │  │   Host 0    │ │   Host 1    │ │   Host 2    │ │   Host 3    │    │
    │  │  Agent @    │ │  Agent @    │ │  Agent @    │ │  Agent @    │    │
    │  │  addr0:port │ │  addr1:port │ │  addr2:port │ │  addr3:port │    │
    │  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘    │
    │         │               │               │               │           │
    │  ┌──────┴───────────────┴───────────────┴───────────────┴──────┐    │
    │  │  ProcMesh: WorldId("my_job"), procs_per_host=2              │    │
    │  │                                                             │    │
    │  │  Proc rank=0   Proc rank=1   Proc rank=2   ...  Proc rank=7 │    │
    │  │  (host 0)      (host 0)      (host 1)           (host 3)    │    │
    │  └─────────────────────────────────────────────────────────────┘    │
    │                                                                     │
    │  ActorMesh: actors spawned with full hierarchical addresses         │
    │                                                                     │
    │    my_job[0].worker[0]  my_job[2].worker[0]  my_job[4].worker[0]    │
    │    my_job[0].worker[1]  my_job[2].worker[1]  my_job[4].worker[1]    │
    │    my_job[1].trainer[0] my_job[3].trainer[0] ...                    │
    └─────────────────────────────────────────────────────────────────────┘
    ```

    **The key insight: ActorId encodes location**

    ```
    Example ActorId: my_job[5].worker[3]

      ActorId(
        ProcId::Ranked(WorldId("my_job"), rank=5),   ← Which proc (implies which host)
        name="worker",                               ← Actor type
        pid=3                                        ← Instance index
      )

    The rank encodes everything needed to route:
      rank 5 → host 2 (if 2 procs per host) → dial addr2:port
    ```

    **Routing is the same at every layer:**

    ```
    ┌─────────────────────────────────────────────────────────────────────┐
    │  Message from my_job[0].trainer[0] → my_job[5].worker[3]            │
    │                                                                     │
    │  Step 1: Extract ProcId from destination                            │
    │          my_job[5] ← target proc                                    │
    │                                                                     │
    │  Step 2: Am I on proc 5?                                            │
    │          NO → look up proc 5 in MailboxRouter (O(log procs))        │
    │               Router has: my_job[5] → RemoteClient(addr2:port)      │
    │                                                                     │
    │  Step 3: Send to Host 2's agent                                     │
    │          Agent routes to local proc 5                               │
    │                                                                     │
    │  Step 4: Proc 5's MailboxMuxer delivers locally                     │
    │          O(1) DashMap lookup → worker[3]'s mailbox                  │
    └─────────────────────────────────────────────────────────────────────┘

    No central registry! The WorldId + rank tells you exactly where to go.
    Multi-host routing uses the same principle as cross-proc routing.
    ```

    **Why this scales:**

    | Operation | Global Table | Monarch Hierarchical |
    |-----------|--------------|---------------------|
    | Spawn | O(consensus) | O(1) local |
    | Route (local) | O(cache miss) | O(1) hash |
    | Route (cross-host) | O(cache miss) | O(log procs) + O(1) local |
    | Failure | Global invalidation | Local supervision |
    | Memory | O(actors) per node | O(procs) per node |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Per-Sender Message Ordering

    Messages are delivered in per-sender FIFO order - critical for determinism without global locks.

    ```
    Multiple senders → One receiver

    Sender A ──[M1]──[M3]──[M5]────┐
                                   │
    Sender B ──[M2]──[M4]───────┐  │    Actor X
                                │  │    ┌─────────────────────┐
    Sender C ──[M6]─────────┐   │  │    │ MailboxMuxer:       │
                            │   │  │    │                     │
                            │   │  │    │ A: [M1, M3, M5]     │
                            └───┴──┴───►│ B: [M2, M4]         │
                                        │ C: [M6]             │
                                        │                     │
                                        │ Fair dequeue:       │
                                        │ M1, M2, M6, M3, M4..│
                                        └─────────────────────┘

    Per-sender FIFO: A's messages always in order (M1 < M3 < M5)
    But A and B can interleave: M1, M2, M3, M4, M5...
    ```

    **How it works:**

    1. Each sender has a separate queue per receiver
    2. DashMap: `SenderId → MessageQueue` (no lock contention between senders)
    3. Actor's event loop fairly drains all queues
    4. Messages from same sender always delivered in order

    **Why it matters:**

    - Deterministic replay: same message sequence = same actor state
    - No global sequencer bottleneck
    - Scales to millions of concurrent senders
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Your First Monarch Program

    Let's see the simplest possible Monarch program - two actors playing ping pong.

    Reference: [ping_pong example](https://meta-pytorch.org/monarch/generated/examples/ping_pong.html)
    """)
    return


@app.cell
def _():
    from monarch.actor import Actor, endpoint, current_rank, this_host

    class PingPong(Actor):
        def __init__(self, name: str):
            self.name = name
            self.count = 0

        @endpoint
        def ping(self, message: str) -> str:
            self.count += 1
            print(f"{self.name} received: {message} (count: {self.count})")
            return f"pong from {self.name}"

    # Spawn two actors on separate processes
    host = this_host()
    procs = host.spawn_procs({"procs": 2})
    actors = procs.spawn("players", PingPong, "Player")

    # Get individual actors via slicing
    alice = actors.slice(procs=0)
    bob = actors.slice(procs=1)

    # Play ping pong
    for i in range(3):
        response = alice.ping.call_one(f"ping {i} to Alice").get()
        print(f"Got: {response}")
        response = bob.ping.call_one(f"ping {i} to Bob").get()
        print(f"Got: {response}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    **Key takeaways:**

    1. **Single controller**: One Python script orchestrates distributed compute
    2. **Tensor engine + actors**: Real tensor APIs (`to_mesh`, `reduce_`, `slice_mesh`) plus general actor patterns
    3. **Hierarchical addressing**: O(1) local routing, no global registry
    4. **Per-sender ordering**: Deterministic without global locks
    5. **Rust + Tokio**: Performance without GC pauses

    **Next:** Interactive DevX - using Monarch for fast distributed development
    """)
    return


if __name__ == "__main__":
    app.run()
