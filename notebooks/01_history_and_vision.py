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
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Training at the Frontier Hurts

    Before we look at any code, let's talk about **why Monarch exists**.

    During Llama 3 pre-training, Meta ran 16,384 GPUs for 54 days and hit
    **419 unexpected interruptions** — roughly one failure every 3 hours.
    The breakdown tells you a lot about what goes wrong at scale:

    | Cause | % of interruptions | Count |
    |-------|-------------------|-------|
    | Faulty GPUs | 30.1% | 148 |
    | GPU HBM3 errors | 17.2% | 72 |
    | Software bugs | 12.9% | 54 |
    | Network / cables | 8.4% | 35 |
    ...

    This is "just" for 16K GPUs. If you further scale workloads to tens of thousands of GPUs, you should expect failures
    every hour or more frequently. The distributed system must handle this gracefully — detect, checkpoint, recover, keep going
    automatically, without requiring someone to SSH in to restart things manually.

    **Monarch was built for this reality.** It's a PyTorch-native distributed
    systems framework designed from the ground up for fault tolerance, flexible
    communication patterns, and scale. Let's see how it got here.

    *(Failure data from the [Llama 3 paper](https://arxiv.org/abs/2407.21783);
    see also [Introducing PyTorch Monarch](https://pytorch.org/blog/introducing-pytorch-monarch/))*
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **What you'll learn:**

    1. Why Monarch exists (the pain that drove its creation)
    2. The tensor engine origin story
    3. The actor model and why it matters for distributed ML
    4. Your first Monarch program: ping-pong actors
    5. The Monarch ontology: World, Proc, Actor, Port
    6. How Monarch scales
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The Tensor Engine Origin Story

    The first step toward solving this was rethinking how we orchestrate
    distributed computation.

    Monarch began as a **tensor engine** for distributed PyTorch.
    It was built as a "single controller" that executed DTensor operations.

    In other words, in Monarch we have a **single controller** that orchestrates many GPUs,
    instead of SPMD (Single Program, Multiple Data) where every rank runs the same script.

    ```
    ┌──────────────────────────────────────────────────────────────────┐
    │                    CONTROLLER (Python)                           │
    │                                                                  │
    │   # Create a mesh: 4 hosts x 8 GPUs = 32 GPUs total              │
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

    This allows one Python script to orchestrate thousands of GPUs - bypassing SPMD-imposed complexities like per-rank checks, scattered logging, etc. Complex control flow becomes more natural to express in code.

    The tensor engine still exists today, but Monarch has evolved beyond it.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Evolution to Actors

    While building the tensor engine, the Monarch team realized that the system
    underneath — the Rust runtime managing processes, message routing, and
    scheduling — was far more general than just tensor orchestration. The
    **actor model** underpinning everything was powerful on its own.

    So the APIs shifted to bring those primitives directly to Python. Instead of
    only exposing tensor operations, Monarch now lets you define arbitrary
    **actors** that communicate via **messages**.

    **What is an actor?** In the formal sense, an actor is a concurrent unit of
    computation that:

    1. Has **private state** — no shared memory with other actors
    2. Communicates exclusively by **sending and receiving messages**
    3. Can **create new actors**, send messages, and decide how to handle the next
       message it receives

    A useful analogy: think of actors as workers in separate offices. They can't
    walk over and read each other's notebooks — they can only communicate by
    passing notes through mail slots.

    This model composes naturally with PyTorch's existing ecosystem. You can wrap any SPMD code with Monarch's actors,
    and command a group or "gang" of actors as a single addressable unit, and wire them together however your workload requires.

    Take RL for example (don't worry about the details of this snippet —
    we'll cover these APIs hands-on throughout the series):

    ```python
    # Spawn different actor types across processes
    host = this_host()
    trainer_procs = host.spawn_procs(per_host={"gpus": 1})
    generator_procs = host.spawn_procs(per_host={"gpus": 4})

    trainer = trainer_procs.spawn("trainer", TrainerActor)
    generators = generator_procs.spawn("generators", GeneratorActor)

    # Wire them together
    for batch in training_loop:
        # Call all generators (returns a ValueMesh)
        sample_mesh = generators.generate.call(prompts).get()
        # Extract values from the ValueMesh before passing along
        samples = list(sample_mesh.values())
        trainer.train_step.call_one(samples).get()
    ```

    Let's make this concrete with a real program you can run.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Your First Monarch Program

    The simplest possible Monarch program — two actors playing ping pong.

    Reference: [ping_pong example](https://meta-pytorch.org/monarch/generated/examples/ping_pong.html)
    """)
    return


@app.cell
def _():
    from monarch.actor import Actor, endpoint, current_rank, this_host

    class PingPong(Actor):
        def __init__(self):
            rank = current_rank().rank
            self.name = "Ping" if rank == 0 else "Pong"
            self.count = 0

        @endpoint
        def ping(self, message: str) -> str:
            self.count += 1
            print(f"{self.name} received: {message} (count: {self.count})")
            return f"pong from {self.name}"

    # Spawn two actors on separate processes
    host = this_host()
    procs = host.spawn_procs(per_host={"gpus": 2})
    actors = procs.spawn("players", PingPong)

    # Get individual actors via slicing
    ping_actor = actors.slice(gpus=0)
    pong_actor = actors.slice(gpus=1)

    # Play ping pong — call_one targets a single actor
    for i in range(3):
        response = ping_actor.ping.call_one(f"round {i}").get()
        print(f"Got: {response}")
        response = pong_actor.ping.call_one(f"round {i}").get()
        print(f"Got: {response}")

    # .call() broadcasts to ALL actors and returns a ValueMesh — a dict-like
    # container mapping each actor's position to its return value.
    results = actors.ping.call("hello everyone").get()
    for point, response in results.items():
        print(f"Actor at {point}: {response}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The Monarch Ontology

    Now that you've run code, let's name the pieces. Monarch has a strict
    hierarchy:

    ```
    WORLD (gang-scheduled group of processes)
    ├── PROC 0 (single actor runtime instance)
    │   └── Actor "players" (pid=0, a.k.a. "Ping")
    │       └── Port 0 (ping endpoint)
    └── PROC 1
        └── Actor "players" (pid=1, a.k.a. "Pong")
            └── Port 0 (ping endpoint)
    ```

    In Monarch, a **mesh** is a named, multi-dimensional collection of identical
    resources that you can address and operate on as a group. A HostMesh contains
    hosts, a ProcMesh contains processes, an ActorMesh contains actor instances —
    each layer spawns the next.

    **Definitions:**

    - **World**: A fixed group of processes launched together via gang scheduling
      (all processes start together as a group — if any fails to start, none do)
    - **Proc**: A single actor runtime instance. One proc runs on one GPU (or CPU).
    - **Actor**: An independent async unit with its own mailbox. Communicates only
      through messages — never shares memory.
    - **Port**: A typed message endpoint. Each `@endpoint` decorator creates a port.

    In the ping-pong example above:
    - The **World** is the group of 2 procs we spawned
    - Each **Proc** hosts one PingPong **Actor**
    - The `ping` method is a **Port**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Why It Scales

    Monarch is designed to run millions of actors. Most actor systems use a
    global routing table — Monarch doesn't. Instead, every entity has a
    hierarchical ID that **encodes the routing path directly**:

    ```
    ActorId = (ProcId, actor_name, pid)
    ProcId  = Ranked(WorldId, rank)
    ```

    A HostMesh contains hosts, each host runs procs (a ProcMesh), and each proc
    hosts actors (an ActorMesh). The ActorId tells you exactly where to route
    without any lookup:

    ```
    Example: my_job[5].worker[3]

      rank 5 -> host 2 (if 2 procs per host) -> dial addr2:port
      Then local delivery to worker[3]'s mailbox: O(1)
    ```

    **Why this beats a global table:**

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
    ## Summary

    **Key takeaways:**

    1. **Born from pain**: Monarch was built for the reality of frontier training —
       hundreds of failures across thousands of GPUs
    2. **Single controller**: One Python script orchestrates distributed compute
    3. **Actors, not threads**: Independent workers communicating via messages,
       never sharing memory
    4. **Hierarchical addressing**: O(1) local routing, no global registry, scales
       to millions of actors
    5. **Rust + Tokio**: Performance without GC pauses

    You've seen how Monarch's actors work. But right now, developing distributed
    systems means SSH, submit, wait, check logs, repeat. What if you could iterate
    as fast as local development? That's what we'll build next.
    """)
    return


if __name__ == "__main__":
    app.run()
