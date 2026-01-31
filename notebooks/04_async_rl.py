import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Async RL: Building a Training Loop from Scratch

        Async RL is hard. This notebook walks through building an RL loop from the ground up, progressively disclosing Monarch concepts as we need them.

        ## Why RL?

        RL is a core part of the LLM pipeline:
        - **RLHF** originally for alignment
        - **Reasoning** capabilities (o1-style thinking)
        - **Agentic training** for tool use and multi-step tasks

        It's how we align models to the behavior we care about.

        ## When Should You RL?

        Today's LLMs are already capable. It's not always clear you *should* RL - it's notoriously finicky.

        The gold standard is **verifiable rewards**. What separates RL from SFT is the ability for the model to learn from exploration. You need:
        1. A task where the model can try different approaches
        2. A way to verify if the approach worked
        3. Room for improvement (the model isn't already perfect)

        ## What We'll Build

        We'll start simple and add complexity:
        1. A synthetic task where Qwen 0.7B can learn
        2. A basic sync RL loop
        3. Services and orchestration
        4. Weight synchronization with RDMA
        5. The full async loop with utilization visualization

        ---

        *By the end, you'll have a working async RL system built on Monarch.*
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Finding a Good Task

        The typical "hello world" for RL is GSM8k math problems. But Qwen 2.5 is already pretty good at math.

        We need a task where:
        - The model can do it *some* of the time
        - There's clear room for improvement
        - We can verify success programmatically

        TODO: Demonstrate a synthetic task - possibly involving tool use or pattern discovery

        Ideas:
        - Hidden pattern game (model calls `check(guess)` to discover rules)
        - Multi-step arithmetic with calculator tool
        - Information lookup that requires exploration
        """
    )
    return


@app.cell
def _():
    # TODO: Define the synthetic task
    # TODO: Show baseline Qwen 0.7B performance
    pass
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The Basic RL Loop

        Before async, let's build sync:

        ```python
        while training:
            # Generate trajectories (SLOW - this dominates)
            trajectories = model.generate(prompts)

            # Score them (usually fast)
            rewards = evaluate(trajectories)

            # Compute loss and update
            loss = rl_loss(trajectories, rewards)
            loss.backward()
            optimizer.step()
        ```

        The problem: generation is ~10x slower than training. The trainer sits idle.
        """
    )
    return


@app.cell
def _():
    # TODO: Basic sync RL loop implementation
    pass
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Why Async?

        ```
        SYNC RL:
        Generator:  ├── gen ──┤         ├── gen ──┤
        Trainer:              ├─ train ─┤         (idle)

        ASYNC RL:
        Generator:  ├── gen ──┼── gen ──┼── gen ──┤
        Buffer:     ════════════════════════════════
        Trainer:    ├─ train ─┼─ train ─┼─ train ─┤
        ```

        Decouple generation from training. Everything overlaps. GPUs stay busy.

        Reference: [GRPO Actor example](https://meta-pytorch.org/monarch/generated/examples/grpo_actor.html)
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Orchestration: Services & Actor Meshes

        To build async RL, we need:
        - Multiple generator replicas (keep generating in parallel)
        - A way to route requests to healthy replicas
        - Service discovery so components can find each other

        TODO: Build a simple service abstraction
        """
    )
    return


@app.cell
def _():
    # TODO: Service implementation
    # - Round-robin routing
    # - Health tracking
    # - Discoverable via registry
    pass
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Weight Sync: RDMA Fundamentals

        The trainer updates weights. Generators need fresh weights. How do we sync fast?

        **The key insight**: Separate control plane from data plane.

        - **Control plane** (actor messages): "Here's a handle to my weights" (~100 bytes)
        - **Data plane** (RDMA): Bulk transfer of actual weights (~10 GB)

        Think of it as a "magic pointer" - send a tiny handle, receiver pulls the big data.

        TODO: RDMA transfer example with timing comparison
        """
    )
    return


@app.cell
def _():
    # TODO: RDMA weight sync implementation
    # Pattern 1: CPU staging
    # Pattern 2: Direct GPU (if time permits)
    pass
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## DTensor & Resharding

        What if the trainer and generator have different parallelism?

        - Trainer: FSDP across 8 GPUs (Shard(0))
        - Generator: Tensor parallel across 2 GPUs (Shard(1))

        DTensor provides a unified language for describing sharding. The transfer layer figures out which bytes go where.

        TODO: Brief resharding example (stretch goal)
        """
    )
    return


@app.cell
def _():
    # TODO: DTensor resharding example (stretch goal)
    pass
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The Full Async Loop

        Putting it all together:

        ```
                ┌─────────────┐
                │  GENERATOR  │──────────────┐
                │   (model)   │              │
                └──────┬──────┘              │
                       │ episodes            │ weights (RDMA)
                       ▼                     │
                ┌─────────────┐              │
                │   BUFFER    │◀─ sample ────┤
                │(policy ver) │              │
                └──────┬──────┘              │
                       │ batches             │
                       ▼                     │
                ┌─────────────┐              │
                │   TRAINER   │──────────────┘
                └─────────────┘

        Loop 1: generate → score → add to buffer (continuous)
        Loop 2: sample → train → push weights (continuous)
        ```

        TODO: Full async RL implementation
        """
    )
    return


@app.cell
def _():
    # TODO: Full async RL loop
    pass
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Interactive Visualization: Where Does Time Go?

        Use the sliders below to see how different configurations affect GPU utilization.
        """
    )
    return


@app.cell
def _(mo):
    # Interactive sliders for the visualization
    gen_time = mo.ui.slider(100, 2000, value=800, label="Generation time (ms)")
    train_time = mo.ui.slider(50, 500, value=200, label="Training time (ms)")
    sync_time = mo.ui.slider(10, 200, value=50, label="Weight sync time (ms)")
    num_generators = mo.ui.slider(1, 8, value=4, label="Number of generators")

    mo.vstack([gen_time, train_time, sync_time, num_generators])
    return gen_time, num_generators, sync_time, train_time


@app.cell
def _(gen_time, mo, num_generators, sync_time, train_time):
    # TODO: Compute utilization and show pie charts
    # Sync RL vs Async RL comparison

    # Placeholder calculation
    total_sync = gen_time.value + train_time.value + sync_time.value
    sync_util = train_time.value / total_sync * 100

    # Async can overlap generation across multiple generators
    async_gen_throughput = gen_time.value / num_generators.value
    total_async = max(async_gen_throughput, train_time.value) + sync_time.value
    async_util = train_time.value / total_async * 100

    mo.md(f"""
    ### Results

    **Sync RL**: {sync_util:.1f}% trainer utilization, {1000/total_sync:.2f} steps/sec

    **Async RL**: {async_util:.1f}% trainer utilization, {1000/total_async:.2f} steps/sec

    **Speedup**: {total_sync/total_async:.1f}x

    TODO: Replace with proper pie chart visualization
    """)
    return async_gen_throughput, async_util, sync_util, total_async, total_sync


if __name__ == "__main__":
    app.run()
