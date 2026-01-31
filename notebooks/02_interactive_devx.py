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
        # Interactive DevX: Monarch as Remote Torchrun

        This notebook demonstrates the "aha moment" of Monarch: a controller architecture that revolutionizes how we interact with distributed machines.

        ## What You'll Learn

        The typical distributed development workflow is painful:
        - Submit a job to SLURM, wait in queue
        - Check logs, find a bug, fix it
        - Submit again, wait again...

        Monarch's SPMDJob changes this. Think of it as **"remote torchrun"**:
        - Spin up a remote cluster once
        - Run torchrun-like commands against it continuously
        - Same familiar interface, but pointing at remote resources
        - No job queue, no waiting - direct iteration

        ## Topics Covered

        1. **SPMDJob**: How Monarch boots up and connects to distributed resources
        2. **HostMesh**: What a "host mesh" is and how it represents your cluster
        3. **The Development Loop**: Fast iteration on distributed code
        4. **Aggregated Logging**: All your nodes' output in one place

        ---

        *By the end of this notebook, you'll be running distributed programs interactively.*
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The Problem: Traditional Multi-Node Development

        ```
        Developer
           │
           ├── sbatch job.sh ──► SLURM Queue ──► [wait...]
           │
           │    (minutes to hours later)
           │
           ├── squeue, sacct, cat logs
           │
           └── make changes, sbatch again ──► [wait...]
        ```

        This is slow and frustrating. Let's fix it.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## SPMDJob: Remote Torchrun

        TODO: Show how SPMDJob parses familiar torchrun args

        TODO: Demonstrate the fast iteration loop

        Reference: [Hello SLURM tutorial](https://docs.pytorch.org/tutorials/intermediate/monarch_distributed_tutorial.html)
        """
    )
    return


@app.cell
def _():
    # TODO: SPMDJob example
    pass
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## HostMesh & Cluster Simulation

        For this presentation, we'll use a **HostMeshSimulator** that spins up containers locally to simulate a multi-host environment.

        This lets you experience the distributed development workflow without needing a real cluster.

        TODO: Implement HostMeshSimulator using podman/docker
        """
    )
    return


@app.cell
def _():
    # TODO: HostMeshSimulator implementation
    pass
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Debugging Distributed Programs

        Reference: [Monarch debugger example](https://meta-pytorch.org/monarch/generated/examples/debugging.html)

        TODO: Show debugging workflow
        """
    )
    return


@app.cell
def _():
    # TODO: Debugging example
    pass
    return


if __name__ == "__main__":
    app.run()
