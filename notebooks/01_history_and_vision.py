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

    This notebook covers the origins of Monarch and the design philosophy that shapes it today.

    ## What You'll Learn

    Monarch started as a **tensor engine** - a system for running op-level tensor compute across distributed machines. Understanding this origin story is key to understanding why Monarch actors are designed the way they are.

    We'll cover:

    1. **The Original Vision**: Tensor engine for distributed PyTorch, single-controller architecture
    2. **Evolution to Actors**: How Monarch grew from "PyTorch single controller" to a general-purpose actor framework where you can build your own tensor engine
    3. **Hyperactor**: The Rust actor runtime built on Tokio that underpins everything
    4. **The Monarch Ontology**: World → Proc → Actor → Port - what this hierarchy means and why it scales

    ## Why This Matters

    The actor model is a powerful abstraction. By the end of this notebook, you'll understand:

    - Why Monarch chose Rust + Tokio (performance, no GC pauses)
    - How prefix-based routing avoids a global routing table (scaling to millions of actors)
    - How deterministic replay is achieved and why it matters for debugging
    - The lack of global locks that enables massive parallelism

    ---

    *The next cells will walk through these concepts with interactive examples.*
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The Tensor Engine Origins

    TODO: Diagram of what tensor engine looks like - distributed tensors sharded across processes, operated as a single logical tensor

    TODO: Worker processes, control plane orchestration
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Hyperactor & The Actor Model

    TODO: What is the actor model? How do actors interact with each other?

    TODO: The Monarch ontology - World → Proc → Actor → Port

    TODO: Why Rust + Tokio? Performance characteristics.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Scaling to Millions

    TODO: Prefix-based routing / no global routing table

    TODO: Per-sender ordering without global locks

    TODO: Deterministic replay
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 101 Example: Ping Pong

    Let's see the simplest possible Monarch program - two actors playing ping pong.

    Reference: [ping_pong example](https://meta-pytorch.org/monarch/generated/examples/ping_pong.html)
    """)
    return


@app.cell
def _():
    # TODO: Ping pong example code
    pass
    return


if __name__ == "__main__":
    app.run()
