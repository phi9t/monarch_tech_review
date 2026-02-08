# Monarch vs Ray: Actors, Messaging, and Supervision

A technical comparison based on the Monarch notebook series (this repo) and the Ray codebase at `/home/phi9t/workspace/ray`.

---

## 1. The Actor Abstraction

**Monarch** — actors are Python classes that inherit from `Actor`. Methods are exposed via the `@endpoint` decorator, creating typed, named RPC entry points:

```python
# Monarch (from notebooks/01_history_and_vision.py:183-196)
class PingPong(Actor):
    def __init__(self):
        self.name = "Ping" if current_rank().rank == 0 else "Pong"
        self.count = 0

    @endpoint
    def ping(self, message: str) -> str:
        self.count += 1
        return f"pong from {self.name}"

# Spawn: mesh-aware, returns an ActorMesh
procs = host.spawn_procs(per_host={"gpus": 2})
actors = procs.spawn("players", PingPong)
```

**Ray** — actors are regular Python classes decorated with `@ray.remote`. Every public method is implicitly callable remotely:

```python
# Ray (from python/ray/actor.py:1201-1368)
@ray.remote
class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1
        return self.count

# Spawn: returns an ActorHandle (opaque proxy)
counter = Counter.remote()
```

**Key differences:**

| Aspect | Monarch | Ray |
|--------|---------|-----|
| **Base class** | Explicit `Actor` inheritance | `@ray.remote` decorator on any class |
| **Method exposure** | Opt-in via `@endpoint` | Opt-out (all methods are remote by default) |
| **Spawn model** | Two-step: `spawn_procs()` then `spawn()` on a ProcMesh | Single-step: `Class.remote()`, scheduler picks node |
| **Return type** | `ActorMesh` (collective-addressable group) | `ActorHandle` (single actor proxy) |
| **Invocation** | `actor.method.call_one().get()` / `actors.method.call().get()` | `actor.method.remote()` returns `ObjectRef`, then `ray.get()` |
| **Message ordering** | **Guaranteed**: messages from A to B arrive in order; each actor processes messages **sequentially** | **Not guaranteed** by default; `allow_out_of_order_execution` flag exists (`actor_handle.h:51`) |
| **Concurrency within actor** | Sequential by default (one message at a time). Different actors in the same process run concurrently | Configurable via `max_concurrency`. Defaults differ for sync vs async actors |
| **Actor inheritance** | Supported (normal Python inheritance from `Actor`) | **Explicitly blocked** — `ActorClassInheritanceException` raised if you try to subclass an `ActorClass` (`actor.py:1226`) |

---

## 2. Topology: Meshes vs Flat Handles

This is the deepest architectural divergence.

**Monarch** builds around **multidimensional meshes**. A `ProcMesh` is a structured grid of processes with named dimensions (e.g., `{"dp": 4, "tp": 8}`). An `ActorMesh` spawned on it inherits that structure. You address actors by slicing:

```python
# From NB04 (04_distributed_tensors.py:49-51, 228-233)
mesh = this_host().spawn_procs({"gpu": 8})
mesh_2d = remote_mesh.rename(host="dp", gpu="tp")
mesh_3d = remote_mesh.split(host=("dp", "pp"), gpu=("tp",), pp=2)

# Addressing: slice by dimension
ping_actor = actors.slice(gpus=0)          # single actor
dp_group = actors.slice(dp=2)             # all actors at dp=2
```

The mesh enables **O(log N) broadcast** via tree-based routing and **O(log N) reduce** for aggregating responses — the runtime routes through intermediate hosts automatically.

**Ray** has no native mesh concept. Actors are individually addressed by `ActorHandle` (backed by `ActorID` — `actor_handle.h:64`). Grouping is done ad-hoc:

- **Placement Groups** (`python/ray/util/placement_group.py:42`) — reserve bundles of resources with strategies like `PACK`, `SPREAD`, `STRICT_PACK`, `STRICT_SPREAD`. But they're a scheduling constraint, not an addressable collective.
- **Actor Pools** — Ray Serve builds actor pools on top, but there's no built-in `ActorMesh` equivalent.

To broadcast to N actors in Ray, you call `.remote()` N times and gather N `ObjectRef`s:

```python
# Ray: manual fan-out
refs = [actor.method.remote(data) for actor in actors]
results = ray.get(refs)

# Monarch: single broadcast, structured response
results = actors.method.call(data).get()  # returns ValueMesh
```

The `ValueMesh` maps each actor's position (coordinates in the mesh) to its return value — structure is preserved.

---

## 3. Control Plane and Messaging

**Ray's control plane** is a three-tier architecture:

1. **GCS (Global Control Store)** — centralized metadata server (`src/ray/gcs/gcs_server.h`). Manages `GcsActorManager`, `GcsNodeManager`, `GcsJobManager`, `GcsWorkerManager`. Backed by Redis or in-memory store. Tracks actor locations, node liveness, resource availability.

2. **Raylet (Node Manager)** — per-node daemon (`src/ray/raylet/node_manager.h`). Manages local worker pool, object store (Plasma), local scheduling. Leases workers from the cluster scheduler.

3. **CoreWorker** — per-process runtime (`src/ray/core_worker/core_worker.h`). Handles task submission, object resolution, reference counting. Each actor lives inside a CoreWorker.

Message flow for a method call:
```
caller CoreWorker -> gRPC -> target node's Raylet -> target CoreWorker -> execute method
                                                                       -> return ObjectRef
caller: ray.get(ref) -> resolve via ownership protocol -> fetch object
```

The **ownership model** is key: the process that creates an `ObjectRef` is the **owner** of that object (`actor_handle.h:40-41` — `owner_id`, `owner_address`). Object metadata and lineage live with the owner, not in GCS. This distributes the metadata load but means if the owner dies, the object's lineage is lost.

**Monarch's control plane** is fundamentally different — it's **actors all the way down**:

- No separate GCS/Raylet daemons. The Rust runtime manages processes and message routing.
- **Single controller** paradigm: one Python process (the notebook/script) orchestrates all actors via endpoint calls. There's no distributed scheduler deciding where actors go — you explicitly create `ProcMesh` topologies and spawn actors on them.
- Messaging uses **Ports** (`Channel.open()` gives a `Port` + `PortReceiver`) — the primitive underneath `call`, `broadcast`, `send`, `stream`. All higher-level patterns compose from this.
- Even infrastructure uses actors: `RdmaController` is a `get_or_spawn_controller` singleton actor (`07b_weight_sync_deep_dive.py:197-222`), `ServiceRegistry` is a singleton actor for discovery.

| Aspect | Monarch | Ray |
|--------|---------|-----|
| **Architecture** | Single controller + Rust actor runtime | GCS + Raylet + CoreWorker (3-tier) |
| **Scheduling** | Explicit: you build `ProcMesh` topology, spawn actors on it | Implicit: `@ray.remote(num_gpus=1)`, scheduler picks node |
| **Message routing** | Tree-based through mesh hierarchy | Direct gRPC between CoreWorkers |
| **Object model** | Return values are inline (`.get()` blocks for result) | `ObjectRef` with distributed reference counting + ownership |
| **Metadata store** | None (controller holds all references) | GCS (Redis-backed or in-memory) |
| **Data transport** | RDMA for bulk transfers (`RDMABuffer`), actor messages for control | Object store (Plasma/shared memory) for data, gRPC for control. NCCL/Gloo via `ray.util.collective` |

---

## 4. Fault Tolerance and Supervision

**Monarch** has an explicit **Erlang-style supervision tree**:

```python
# From NB03 (03_fault_tolerance.py:107-114)
class SupervisorActor(Actor):
    def __supervise__(self, failure: MeshFailure):
        print(f"Failure: {failure}")
        return True   # handled -- stop propagation
        # return False -> escalate to my supervisor
```

The semantics are precise:
- Every actor has a **parent** (the actor that called `spawn()`). The parent is the supervisor.
- When a child crashes, `__supervise__` is called on the parent with a `MeshFailure`.
- Return `True` -> handled locally. Return `False` -> propagate up the tree.
- If nothing handles it, `unhandled_fault_hook` fires at the client (default: crash with exit code 1).
- This gives you **one structured chain** showing exactly what failed and how it propagated — instead of grepping 16,000 log files.

The supervision tree composes with **TorchFT** for training-specific fault tolerance: quorum-based training continues with healthy replicas while Monarch respawns crashed ones.

**Ray** has **no supervision tree**. Fault tolerance is flat and configuration-driven:

```python
# Ray: restart policy at actor creation (from actor.py:1394-1409)
@ray.remote(max_restarts=3, max_task_retries=5)
class MyActor:
    pass

# Per-method override
@ray.method(max_task_retries=10)
def fragile_method(self):
    pass
```

When an actor dies:
1. Ray checks `max_restarts`. If restarts remain, the GCS `GcsActorManager` reschedules the actor on a (possibly different) node.
2. In-flight tasks fail with `RayActorError`. Callers catch this or it propagates as a Python exception.
3. `max_task_retries` controls automatic retry of individual method calls.
4. **Detached actors** (`lifetime="detached"`) survive driver exit and are tracked by GCS.
5. No parent notification, no structured propagation, no `__supervise__`-style callback.

| Aspect | Monarch | Ray |
|--------|---------|-----|
| **Model** | Supervision tree (Erlang-style) | Flat restart policy per actor |
| **Failure notification** | `__supervise__(MeshFailure)` on parent actor | `RayActorError` exception on caller |
| **Propagation** | Structured: up the tree, each level can handle or escalate | Flat: exception to caller, no tree |
| **Restart** | Parent decides (can respawn, log, ignore) | Automatic by GCS up to `max_restarts` |
| **Per-method control** | No (supervision is at actor level) | Yes (`@ray.method(max_task_retries=N)`) |
| **Quorum training** | Built-in via TorchFT integration | Not built-in (use external libraries) |
| **Unhandled failures** | `unhandled_fault_hook` at client | Actor marked as dead, `RayActorError` on future calls |

---

## 5. Data Plane: RDMA vs Object Store

This matters enormously for the async RL use case the Monarch notebooks target.

**Monarch** separates control plane from data plane explicitly:
- **Control plane**: actor endpoint messages (small, serialized Python objects)
- **Data plane**: `RDMABuffer` — a ~150-byte handle that represents remote GPU/CPU memory. One-sided RDMA reads/writes bypass the remote CPU entirely. The "magic pointer" pattern from NB07.

```python
# Monarch: zero-copy weight transfer
buffer = RDMABuffer(weights.view(torch.uint8))     # register once
handle = trainer.get_weight_handle.call_one().get() # ~150 bytes
handle.read_into(local_weights).get()               # bulk RDMA, no trainer involvement
```

**Ray** uses the **Plasma object store** (shared memory) for intra-node data sharing, and gRPC + object transfer for cross-node. For GPU collectives, there's `ray.util.collective` (`python/ray/util/collective/collective.py`) wrapping NCCL and Gloo — but these are **collective** (two-sided, blocking) rather than one-sided RDMA. Ray also has an experimental `tensor_transport` mechanism (`actor_handle.h:52`, `actor.py:1921`) for NCCL/Gloo-backed method returns, but it's opt-in and method-level.

| Aspect | Monarch | Ray |
|--------|---------|-----|
| **Intra-node** | Shared process mesh, direct memory access | Plasma object store (shared memory) |
| **Cross-node bulk** | One-sided RDMA (`RDMABuffer.read_into`) | gRPC object transfer, or NCCL collectives |
| **GPU-GPU** | RDMA (bypasses CPU on remote side) | NCCL via `ray.util.collective` (two-sided) |
| **Weight sync pattern** | Handle message (control) + RDMA pull (data) | `ray.get(ref)` fetches through object store |
| **Blocking** | One-sided: receiver pulls when ready, sender uninvolved | Two-sided: collective requires all participants |

---

## 6. Summary: Design Philosophy

**Monarch** is a **structured, topology-aware actor system** built for the specific reality of large-scale GPU training. You explicitly build mesh topologies, spawn actors with awareness of their position, and get automatic tree-routing and supervision. The Erlang heritage is clear: message ordering guarantees, supervision trees, single-controller orchestration. The trade-off is that you manage topology yourself.

**Ray** is a **general-purpose distributed computing framework** with a flat actor model layered on top of a sophisticated distributed object store. Scheduling is automatic, the programming model is simpler (just `@ray.remote` and `.remote()`), and it serves a broader range of workloads (ML training, serving, data processing, simulations). The trade-off is that you lose structured messaging, mesh-aware collectives, supervision trees, and native RDMA — you build these patterns yourself or use higher-level libraries (Ray Train, Ray Serve).

The core tension: **Monarch gives you structured primitives that compose** (meshes, tree-routing, supervision, RDMA handles) at the cost of explicitness. **Ray gives you simplicity and generality** at the cost of leaving structure to the application layer.

---

## Source Files Referenced

### Monarch (this repo)
- `notebooks/01_history_and_vision.py` — Actor model, PingPong, meshes
- `notebooks/03_fault_tolerance.py` — Supervision trees, `__supervise__`, TorchFT
- `notebooks/04_distributed_tensors.py` — Mesh reshape, collectives, pipelining
- `notebooks/06_services.py` — Service actor, ServiceRegistry
- `notebooks/07_rdma_weight_sync.py` — RDMABuffer, magic pointer pattern
- `notebooks/07b_weight_sync_deep_dive.py` — ibverbs, RdmaController as actor

### Ray (`/home/phi9t/workspace/ray`)
- `python/ray/actor.py` — `ActorClass`, `ActorHandle`, `max_restarts`, `max_task_retries`
- `src/ray/core_worker/actor_handle.h` — C++ `ActorHandle` with owner tracking
- `src/ray/core_worker/core_worker.h` — Per-process runtime
- `src/ray/gcs/gcs_server.h` — Global Control Store
- `src/ray/raylet/node_manager.h` — Per-node Raylet
- `python/ray/util/placement_group.py` — Placement groups
- `python/ray/util/collective/collective.py` — NCCL/Gloo collectives
