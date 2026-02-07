import marimo

__generated_with = "0.19.9"
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
    6. Scalable messaging via meshes
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
    ## Scalable Messaging

    Meshes of actors enable **scalable** messaging through tree-based routing.

    Instead of sending messages individually to every actor, Monarch routes through a
    **tree of intermediate hosts**, giving **O(log N) broadcast**. The same tree
    aggregates responses on the way back up, giving **O(log N) reduce** as well.

    This means primitives like barriers and all-reduces are as simple as
    waiting for an aggregate response — and they scale to thousands of actors
    without bottlenecking the client.

    *(See the animated diagrams in the
    [Monarch presentation notebook](https://github.com/meta-pytorch/monarch/blob/main/examples/presentation/presentation.ipynb))*
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Channels and Low-level Messaging

    It is sometimes useful to establish direct channels between two points, or forward
    the handling of some messages from one actor to another. To enable this, all messaging
    in Monarch is built out of `Port` objects.

    An actor can create a new `Channel`, which provides a `Port` for sending and a
    `PortReceiver` for receiving messages. The `Port` object can then be sent to any endpoint.

    ```python
    from monarch.actor import Channel, Port

    port, recv = Channel.open()

    port.send(3)
    print(recv.recv().get())
    ```

    Ports can be passed as arguments to actors and sent a response remotely. We can also
    directly ask an endpoint to send its response to a port using the `send` messaging primitive.

    ```python
    from monarch.actor import send

    with trainer_procs.activate():
        send(check_memory, args=(), kwargs={}, port=port)
    ```

    The port will receive a response from each actor sent the message:

    ```python
    for _ in range(4):
        print(recv.recv().get())
    ```

    The other adverbs like `call`, `stream`, and `broadcast` are just implemented in terms
    of ports and `send`.

    ### Message Ordering

    Messages from an actor are delivered to the destination actor in the order in which they
    are sent. If actor A sends message M0 to actor B, then later sends M1 to B, actor B will
    receive M0 before M1. Messages sent to a mesh of actors behave as if sent individually
    to each destination.

    Each actor handles its messages **sequentially** — it must finish handling a message before
    the next one is delivered. Different actors in the same process handle messages **concurrently**.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    Monarch uniquely provides:

    1. Scalable messaging using multidimensional meshes of actors
    2. Fault tolerance through supervision trees and `__supervise__`
    3. Point-to-point low-level RDMA
    4. Built-in distributed tensors

    This foundation enables building sophisticated multi-machine training programs
    with clear semantics for distribution, fault tolerance, and communication patterns.

    The remaining sections fill in more details about how to accomplish common
    patterns with the above features.

    ---

    **Next:** [NB02 — Interactive DevX](./02_interactive_devx.html)
    """)
    return


if __name__ == "__main__":
    app.run()
