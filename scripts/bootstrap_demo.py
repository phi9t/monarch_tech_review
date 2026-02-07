#!/usr/bin/env python3
"""
Demonstration of Monarch cluster bootstrapping concepts.

This script explains what happens "under the hood" when you use SlurmJob:
1. Worker processes run `run_worker_loop_forever` on each node
2. The client connects via `attach_to_workers`
3. You get a HostMesh that can spawn processes and actors

Due to environment complexities with virtual environments, this demo uses
`create_local_host_mesh` which handles the subprocess spawning correctly.
The concepts are the same - the difference is just in how workers are launched.
"""

import sys
from typing import Dict

# Monarch imports
from monarch._src.actor.host_mesh import create_local_host_mesh
from monarch._src.actor.endpoint import Extent
from monarch.actor import Actor, endpoint, current_rank


# =============================================================================
# What happens in a real cluster
# =============================================================================

CLUSTER_EXPLANATION = """
## What happens when you use SlurmJob

When you call `SlurmJob(meshes={"trainers": 4}).apply()`, here's what happens:

1. **Job Submission**: Monarch submits an sbatch script to SLURM:
   ```bash
   #!/bin/bash
   #SBATCH --nodes=4
   srun python -c 'from monarch.actor import run_worker_loop_forever; \\
                   run_worker_loop_forever(address="tcp://$(hostname):22222", \\
                                          ca="trust_all_connections")'
   ```

2. **Worker Startup**: Each node runs `run_worker_loop_forever`:
   - Opens a listening port (e.g., 22222)
   - Waits for client connections
   - Can host processes and actors

3. **Client Connection**: When you call `job.state()`:
   - Monarch queries SLURM for allocated hostnames
   - Calls `attach_to_workers(workers=["tcp://node1:22222", "tcp://node2:22222", ...])`
   - Returns a HostMesh connected to all workers

4. **Actor Execution**:
   - `host_mesh.spawn_procs()` spawns Python processes on each worker
   - `proc_mesh.spawn("name", Actor)` instantiates actors in those processes
   - Endpoint calls are routed to the correct actors

The key insight: **Workers are just servers waiting for work**.
The scheduler (SLURM, Kubernetes, etc.) handles launching them.
The client (your notebook/script) connects and orchestrates.
"""


# =============================================================================
# Simplified Demo using create_local_host_mesh
# =============================================================================

class HelloActor(Actor):
    """Simple actor that demonstrates distributed execution."""

    def __init__(self, config: Dict):
        self.rank = current_rank().rank
        self.config = config
        print(f"[HelloActor rank={self.rank}] Initialized with config: {config}")

    @endpoint
    def greet(self, name: str) -> str:
        return f"Hello {name} from rank {self.rank}!"

    @endpoint
    def get_info(self) -> Dict:
        return {
            "rank": self.rank,
            "config": self.config,
            "python": sys.executable,
        }


def main():
    print("=" * 70)
    print("Monarch Cluster Bootstrap Demo")
    print("=" * 70)
    print(CLUSTER_EXPLANATION)
    print("=" * 70)
    print("\n## Live Demo with Local Processes\n")

    # This is equivalent to what SlurmJob does, but locally.
    # create_local_host_mesh spawns workers as local processes.
    print("Step 1: Create local host mesh (simulates cluster allocation)")
    print("        This is like calling job.state() after SlurmJob.apply()")

    # Create a host mesh with 3 "hosts" (actually local processes)
    host_mesh = create_local_host_mesh(
        extent=Extent(["hosts"], [3]),
        env={"PYTHONPATH": ":".join(sys.path)},  # Pass current environment
    )
    print(f"        Created host mesh with extent: hosts=3")

    # Spawn processes on each host
    print("\nStep 2: Spawn processes on the host mesh")
    print("        This creates worker processes, each can run actors")
    proc_mesh = host_mesh.spawn_procs(per_host={"gpus": 1})
    print(f"        Spawned procs")

    # Spawn actors
    print("\nStep 3: Spawn actors on the process mesh")
    print("        Each actor runs on a different process")
    actors = proc_mesh.spawn("hello", HelloActor, {"role": "demo"})
    print(f"        Spawned HelloActor on all processes")

    # Call endpoints
    print("\nStep 4: Call actor endpoints")
    print("        Messages are routed to the correct processes")

    results = actors.greet.call("World").get()
    print(f"\n        Results from greet.call('World'):")
    for point, result in results.items():
        print(f"          {point}: {result}")

    # Call individual actor
    print("\nStep 5: Slice and call individual actors")
    # Note: mesh shape is {hosts: 3, gpus: 1}, so we slice by hosts dimension
    actor_0 = actors.slice(hosts=0)
    info = actor_0.get_info.call_one().get()
    print(f"        Info from actor 0: rank={info['rank']}")

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
