# Monarch vs Ray: A Comprehensive Technical Comparison

A deep technical comparison of [PyTorch Monarch](https://github.com/pytorch/monarch) and [Ray](https://github.com/ray-project/ray), covering actors, control plane messaging, supervision, data plane (RDMA), distributed tensors, and interactive development. Each section includes side-by-side code examples and links to the actual source implementations.

**Monarch commit**: [`febefda`](https://github.com/pytorch/monarch/tree/febefda69e1f9e9a696596b3de368ac557320b21)
**Ray commit**: [`25da710`](https://github.com/ray-project/ray/tree/25da7108abbec7606e8e0ce057e3a5d22421850f)

**Companion notebooks**: The Monarch side of each concept maps to a notebook in this repo's `notebooks/` directory. See [01–08](../notebooks/) for interactive walkthroughs.

---

## Table of Contents

1. [Actor Abstraction](#1-actor-abstraction)
2. [Topology — Meshes vs Flat Handles](#2-topology--meshes-vs-flat-handles)
3. [Control Plane Messaging](#3-control-plane-messaging)
4. [Message Ordering Guarantees](#4-message-ordering-guarantees)
5. [Supervision and Fault Tolerance](#5-supervision-and-fault-tolerance)
6. [Data Plane — RDMA vs Object Store](#6-data-plane--rdma-vs-object-store)
7. [Distributed Tensors and Training](#7-distributed-tensors-and-training)
8. [Interactive DevX (Single Controller)](#8-interactive-devx-single-controller)
9. [Design Philosophy Summary](#9-design-philosophy-summary)

---

## 1. Actor Abstraction

> **Notebook ref**: [`notebooks/01_history_and_vision.py`](../notebooks/01_history_and_vision.py) — PingPong actor, mesh spawning, endpoint calls.

### Monarch

Actors are Python classes that inherit from `Actor`. Methods are exposed via the `@endpoint` decorator, creating typed, named RPC entry points. Spawning is topology-aware: you first create a `ProcMesh` (a structured grid of processes), then spawn actors on it, getting back an `ActorMesh`.

```python
# Monarch actor definition
from monarch.actor import Actor, endpoint, current_rank, this_host

class PingPong(Actor):
    def __init__(self):
        self.name = "Ping" if current_rank().rank == 0 else "Pong"
        self.count = 0

    @endpoint
    def ping(self, message: str) -> str:
        self.count += 1
        return f"pong from {self.name} (count={self.count})"

# Spawn: two-step, mesh-aware
host = this_host()
procs = host.spawn_procs(per_host={"gpus": 2})
actors = procs.spawn("players", PingPong)

# Call: returns structured result
result = actors.ping.call_one("hello").get()
```

**Source links:**
- [`Actor` base class](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/python/monarch/_src/actor/actor_mesh.py#L1569) — `class Actor(MeshTrait)`
- [`ActorMesh`](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/python/monarch/_src/actor/actor_mesh.py#L1606) — `class ActorMesh(Generic[T])`
- [`@endpoint` decorator](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/python/monarch/_src/actor/endpoint.py#L509)

### Ray

Actors are regular Python classes decorated with `@ray.remote`. Every public method is implicitly callable remotely. Spawning is a single step: `Class.remote()` returns an opaque `ActorHandle`, and the scheduler picks a node.

```python
# Ray actor definition
import ray

@ray.remote
class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1
        return self.count

# Spawn: single-step, scheduler picks node
counter = Counter.remote()

# Call: returns ObjectRef, resolve with ray.get()
ref = counter.increment.remote()
result = ray.get(ref)
```

**Source links:**
- [`ActorClass`](https://github.com/ray-project/ray/blob/25da7108abbec7606e8e0ce057e3a5d22421850f/python/ray/actor.py#L1201) — `class ActorClass(Generic[T])`
- [`ActorClass.remote()`](https://github.com/ray-project/ray/blob/25da7108abbec7606e8e0ce057e3a5d22421850f/python/ray/actor.py#L1356) — actor creation entry point
- [`ActorHandle`](https://github.com/ray-project/ray/blob/25da7108abbec7606e8e0ce057e3a5d22421850f/python/ray/actor.py#L1886) — `class ActorHandle(Generic[T])`
- [`ActorClass._remote()`](https://github.com/ray-project/ray/blob/25da7108abbec7606e8e0ce057e3a5d22421850f/python/ray/actor.py#L1511) — full creation path with all options

### Comparison

| Aspect | Monarch | Ray |
|--------|---------|-----|
| **Base class** | Explicit `Actor` inheritance | `@ray.remote` decorator on any class |
| **Method exposure** | Opt-in via `@endpoint` | Opt-out (all public methods are remote by default) |
| **Spawn model** | Two-step: `spawn_procs()` then `spawn()` on a ProcMesh | Single-step: `Class.remote()`, scheduler picks node |
| **Return type** | `ActorMesh` (collective-addressable group) | `ActorHandle` (single actor proxy) |
| **Invocation** | `actor.method.call_one().get()` | `ray.get(actor.method.remote())` |
| **Concurrency** | Sequential by default (one message at a time) | Configurable via `max_concurrency` |
| **Actor inheritance** | Supported (normal Python inheritance from `Actor`) | Blocked — `ActorClassInheritanceException` raised |

---

## 2. Topology — Meshes vs Flat Handles

> **Notebook ref**: [`notebooks/04_distributed_tensors.py`](../notebooks/04_distributed_tensors.py) — mesh creation, reshape, slicing, distributed tensor ops.

This is the deepest architectural divergence between Monarch and Ray.

### Monarch

Monarch builds around **multidimensional meshes**. A `ProcMesh` is a structured grid of processes with named dimensions (e.g., `{"dp": 4, "tp": 8}`). An `ActorMesh` spawned on it inherits that structure. You address actors by slicing dimensions:

```python
# Create a mesh — structured grid with named dimensions
host = this_host()
mesh = host.spawn_procs({"gpu": 8})
print(mesh.to_table())
# gpu | host | rank
# 0   | 0    | 0
# 1   | 0    | 1
# ...

# Reshape: split a dimension into multiple
mesh_2d = mesh.rename(gpu="tp")  # just rename
mesh_3d = mesh.split(host=("dp", "pp"), gpu=("tp",), pp=2)

# Addressing: slice by dimension
single_actor = actors.slice(gpus=0)       # one actor
dp_group = actors.slice(dp=2)             # all actors at dp=2

# Broadcast: O(log N) tree-based routing
results = actors.method.call(data).get()  # returns ValueMesh
```

The mesh enables **O(log N) broadcast** via tree-based routing and **O(log N) reduce** for aggregating responses — the runtime routes through intermediate hosts automatically.

**Source links:**
- [`ProcMesh`](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/python/monarch/_src/actor/proc_mesh.py#L234)
- [`HostMesh.spawn_procs()`](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/python/monarch/_src/actor/host_mesh.py#L145)
- [`this_host()`](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/python/monarch/_src/actor/host_mesh.py#L48)
- [Rust `ProcMesh`](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/hyperactor_mesh/src/proc_mesh.rs)
- [Rust `ActorMesh`](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/hyperactor_mesh/src/actor_mesh.rs)
- [Tree-based broadcast (`multicast.rs`)](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/hyperactor_mesh/src/comm/multicast.rs)

### Ray

Ray has no native mesh concept. Actors are individually addressed by `ActorHandle` (backed by `ActorID`). Grouping is done ad-hoc:

```python
# Ray: PlacementGroup for co-location (scheduling constraint, not a collective)
from ray.util.placement_group import placement_group

pg = placement_group(
    [{"CPU": 1, "GPU": 1}] * 8,
    strategy="SPREAD"  # or PACK, STRICT_PACK, STRICT_SPREAD
)
ray.get(pg.ready())

# Actors placed within the group
actors = [
    Worker.options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg, placement_group_bundle_index=i
        )
    ).remote()
    for i in range(8)
]

# Broadcast: manual fan-out, O(N) calls
refs = [actor.method.remote(data) for actor in actors]
results = ray.get(refs)  # list, no structure
```

Placement groups are a **scheduling constraint**, not an addressable collective. There's no `slice()`, `reshape()`, or tree-routing.

**Source links:**
- [`PlacementGroup`](https://github.com/ray-project/ray/blob/25da7108abbec7606e8e0ce057e3a5d22421850f/python/ray/util/placement_group.py#L42)
- [`placement_group()` function](https://github.com/ray-project/ray/blob/25da7108abbec7606e8e0ce057e3a5d22421850f/python/ray/util/placement_group.py#L146)
- [C++ Placement Group Resource Manager](https://github.com/ray-project/ray/blob/25da7108abbec7606e8e0ce057e3a5d22421850f/src/ray/raylet/placement_group_resource_manager.h#L49)

### Comparison

| Aspect | Monarch | Ray |
|--------|---------|-----|
| **Topology model** | Multidimensional named mesh (`ProcMesh`) | Flat list of `ActorHandle`s |
| **Grouping** | First-class: meshes with dimensions, slice, reshape | Ad-hoc: `PlacementGroup` (scheduling only) |
| **Addressing** | By dimension coordinates: `actors.slice(dp=2)` | By index in a Python list |
| **Broadcast** | O(log N) tree-based via `call()` / `broadcast()` | O(N) manual fan-out with list comprehension |
| **Aggregation** | `ValueMesh` preserves mesh structure in results | Plain Python list of results |
| **Reshape** | Runtime: `mesh.split()`, `mesh.rename()` | N/A — rebuild placement group |
| **Multi-host** | `HostMesh` -> `ProcMesh` spans hosts natively | PlacementGroup with `SPREAD` strategy |

---

## 3. Control Plane Messaging

> **Notebook ref**: [`notebooks/01_history_and_vision.py`](../notebooks/01_history_and_vision.py) — Channels, Ports, message ordering, endpoint patterns.

### Monarch

Monarch provides four messaging patterns, all built on the primitive of **Ports** (typed channels):

```python
from monarch.actor import Actor, endpoint, Channel

class EchoActor(Actor):
    @endpoint
    def echo(self, msg: str) -> str:
        return f"echo: {msg}"

    @endpoint
    def stream_data(self, port: Port[str], count: int) -> None:
        for i in range(count):
            port.send(f"item {i}")

# Pattern 1: call_one() — single actor, request-response
result = actor.echo.call_one("hello").get()

# Pattern 2: call() — broadcast to all actors in mesh, collect responses
results = actors.echo.call("hello").get()  # returns ValueMesh

# Pattern 3: stream() — receive async stream of values
async for item in actors.stream_data.stream(count=10):
    print(item)

# Pattern 4: broadcast() — fire-and-forget to all actors
actors.echo.broadcast("hello")  # no response expected

# Low-level: Ports and Channels
port, receiver = Channel.open(str)
actor.stream_data.call_one(port, count=5)
async for msg in receiver:
    print(msg)
```

The underlying messaging infrastructure is built on a **Rust mailbox system** — every actor has a `Mailbox` with typed `Port`s for sending and `PortReceiver`s for receiving.

**Source links:**
- [`call_one()`](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/python/monarch/_src/actor/endpoint.py#L241)
- [`call()` (broadcast)](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/python/monarch/_src/actor/endpoint.py#L268)
- [`stream()`](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/python/monarch/_src/actor/endpoint.py#L305)
- [`broadcast()` (fire-and-forget)](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/python/monarch/_src/actor/endpoint.py#L343)
- [Port, PortReceiver, Mailbox type stubs](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/python/monarch/_rust_bindings/monarch_hyperactor/mailbox.pyi)
- [Rust `mailbox.rs`](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/hyperactor/src/mailbox.rs) (3977 lines — the core messaging engine)

### Ray

Ray has a simpler messaging model: every method call returns an `ObjectRef`, which you resolve with `ray.get()` or check with `ray.wait()`. There's no native streaming or fire-and-forget.

```python
import ray

@ray.remote
class EchoActor:
    def echo(self, msg: str) -> str:
        return f"echo: {msg}"

    def generate_items(self, count: int) -> list[str]:
        return [f"item {i}" for i in range(count)]

# Pattern 1: Single call — .remote() returns ObjectRef
ref = actor.echo.remote("hello")
result = ray.get(ref)

# Pattern 2: Fan-out to multiple actors
refs = [a.echo.remote("hello") for a in actors]
results = ray.get(refs)

# Pattern 3: No native streaming — return a list or use ray.wait() for polling
refs = [actor.generate_items.remote(10)]
done, pending = ray.wait(refs, num_returns=1, timeout=5.0)

# Pattern 4: No fire-and-forget — every call produces an ObjectRef
# (Closest: call .remote() and discard the ref)
_ = actor.echo.remote("hello")  # still creates an ObjectRef
```

Under the hood, Ray uses a **three-tier architecture**: GCS (Global Control Store) for metadata, Raylet (per-node daemon) for local scheduling, and CoreWorker (per-process runtime) for task execution and object management.

**Source links:**
- [`ActorMethod._remote()`](https://github.com/ray-project/ray/blob/25da7108abbec7606e8e0ce057e3a5d22421850f/python/ray/actor.py#L802) — remote method invocation
- [`ObjectRef`](https://github.com/ray-project/ray/blob/25da7108abbec7606e8e0ce057e3a5d22421850f/python/ray/includes/object_ref.pxi#L37) (Cython)
- [`ray.get()`](https://github.com/ray-project/ray/blob/25da7108abbec7606e8e0ce057e3a5d22421850f/python/ray/_private/worker.py#L2858)
- [`ray.wait()`](https://github.com/ray-project/ray/blob/25da7108abbec7606e8e0ce057e3a5d22421850f/python/ray/_private/worker.py#L3074)
- [C++ `CoreWorker`](https://github.com/ray-project/ray/blob/25da7108abbec7606e8e0ce057e3a5d22421850f/src/ray/core_worker/core_worker.h#L167)
- [GCS Server](https://github.com/ray-project/ray/blob/25da7108abbec7606e8e0ce057e3a5d22421850f/src/ray/gcs/gcs_server.h#L98)
- [Raylet NodeManager](https://github.com/ray-project/ray/blob/25da7108abbec7606e8e0ce057e3a5d22421850f/src/ray/raylet/node_manager.h#L70)
- [C++ `ActorHandle` (ownership)](https://github.com/ray-project/ray/blob/25da7108abbec7606e8e0ce057e3a5d22421850f/src/ray/core_worker/actor_handle.h#L34)

### Comparison

| Aspect | Monarch | Ray |
|--------|---------|-----|
| **Messaging patterns** | 4: `call_one`, `call`, `stream`, `broadcast` | 1: `.remote()` → `ObjectRef` |
| **Fire-and-forget** | Native `broadcast()` | Discard `ObjectRef` (still created) |
| **Streaming** | Native `stream()` with async iteration | No native support (return lists, poll with `ray.wait`) |
| **Result type** | Inline `.get()` or `ValueMesh` | `ObjectRef` → `ray.get()` |
| **Low-level primitive** | `Port` / `PortReceiver` (typed channels) | `ObjectRef` with ownership protocol |
| **Architecture** | Single controller + Rust actor runtime | GCS + Raylet + CoreWorker (3-tier) |
| **Metadata store** | None (controller holds references) | GCS (Redis or in-memory) |
| **Message routing** | Tree-based through mesh hierarchy | Direct gRPC between CoreWorkers |

---

## 4. Message Ordering Guarantees

> **Notebook ref**: [`notebooks/01_history_and_vision.py`](../notebooks/01_history_and_vision.py) — message ordering section demonstrating FIFO guarantees.

### Monarch

Monarch provides **per-sender FIFO ordering**: messages from actor A to actor B are guaranteed to arrive in the order they were sent. This is enforced at the Rust level via per-client sequence numbers:

```python
# Monarch: messages arrive in send order — guaranteed
class OrderedReceiver(Actor):
    def __init__(self):
        self.received = []

    @endpoint
    def receive(self, seq: int) -> None:
        self.received.append(seq)

    @endpoint
    def get_order(self) -> list[int]:
        return self.received

# Send messages 0..9 — they arrive in order
for i in range(10):
    receiver.receive.broadcast(i)

order = receiver.get_order.call_one().get()
assert order == list(range(10))  # guaranteed
```

The ordering is implemented in the Rust runtime via `BufferState` and `OrderedSender`, which maintain per-client sequence numbers and buffer out-of-order messages until gaps are filled.

**Source links:**
- [`ordering.rs`](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/hyperactor/src/ordering.rs#L25-L50) — `BufferState`, `OrderedSender` with per-client sequence numbers

### Ray

Ray does **not guarantee message ordering by default**. Actors have an `allow_out_of_order_execution` flag that explicitly opts in to out-of-order processing for performance:

```python
# Ray: ordering depends on configuration
@ray.remote(max_concurrency=1)
class OrderedActor:
    """With max_concurrency=1, tasks execute sequentially in FIFO order."""
    def __init__(self):
        self.received = []

    def receive(self, seq: int):
        self.received.append(seq)

    def get_order(self) -> list[int]:
        return self.received

# With max_concurrency=1, sequential execution gives apparent ordering
# But this is NOT a messaging guarantee — it's a concurrency constraint

# Explicit opt-in to out-of-order for performance:
@ray.remote
class FastActor:
    _ray_allow_out_of_order_execution = True
    # Tasks may execute in any order for higher throughput
```

**Source links:**
- [C++ `allow_out_of_order_execution` flag](https://github.com/ray-project/ray/blob/25da7108abbec7606e8e0ce057e3a5d22421850f/src/ray/core_worker/actor_handle.h#L51)
- [Python `_ray_allow_out_of_order_execution`](https://github.com/ray-project/ray/blob/25da7108abbec7606e8e0ce057e3a5d22421850f/python/ray/actor.py#L1937)

### Comparison

| Aspect | Monarch | Ray |
|--------|---------|-----|
| **Default ordering** | Per-sender FIFO guaranteed | Not guaranteed (FIFO with `max_concurrency=1` as side effect) |
| **Implementation** | Rust-level sequence numbers per sender | Concurrency control at task level |
| **Out-of-order opt-in** | Not available (ordering is fundamental) | `_ray_allow_out_of_order_execution = True` |
| **Multi-sender** | Each sender has independent ordering | No ordering guarantees across senders |
| **Performance trade-off** | Ordered delivery may buffer messages | Opt-in unordering for throughput |

---

## 5. Supervision and Fault Tolerance

> **Notebook ref**: [`notebooks/03_fault_tolerance.py`](../notebooks/03_fault_tolerance.py) — supervision trees, `__supervise__`, TorchFT integration.

### Monarch

Monarch has an explicit **Erlang-style supervision tree**. Every actor has a parent (the actor that called `spawn()`). When a child crashes, `__supervise__` is called on the parent with a `MeshFailure` describing what failed.

```python
# Monarch: Erlang-style supervision tree
from monarch.actor import Actor, endpoint, MeshFailure

class Supervisor(Actor):
    def __supervise__(self, failure: MeshFailure):
        """Called when a child actor crashes."""
        print(f"Child failed: {failure}")
        # Return True -> handled, stop propagation
        # Return False -> escalate to MY supervisor
        return True

    @endpoint
    def spawn_worker(self) -> None:
        worker = this_proc().spawn("worker", FlakyWorker)
        worker.do_work.broadcast()

class FlakyWorker(Actor):
    @endpoint
    def do_work(self):
        raise RuntimeError("GPU memory error!")

# Failure flow:
# 1. FlakyWorker.do_work raises RuntimeError
# 2. Supervisor.__supervise__(MeshFailure) is called
# 3. Supervisor returns True -> failure handled
# 4. If Supervisor returned False -> propagate to Supervisor's parent
# 5. If nobody handles it -> unhandled_fault_hook fires (default: exit 1)
```

The supervision tree composes with **TorchFT** for training-specific fault tolerance: quorum-based training continues with healthy replicas while Monarch respawns crashed ones.

**Source links:**
- [`unhandled_fault_hook()`](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/python/monarch/_src/actor/supervision.py#L17)
- [`MeshFailure` type stubs](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/python/monarch/_rust_bindings/monarch_hyperactor/supervision.pyi)

### Ray

Ray has **no supervision tree**. Fault tolerance is flat and configuration-driven, managed by the GCS `GcsActorManager`:

```python
# Ray: flat restart policy per actor
@ray.remote(max_restarts=3, max_task_retries=5)
class ResilientWorker:
    def __init__(self):
        self.state = self._load_checkpoint()

    def do_work(self):
        # If this crashes, Ray restarts the actor (up to max_restarts)
        # In-flight tasks retry (up to max_task_retries)
        pass

    @ray.method(max_task_retries=10)
    def fragile_method(self):
        """Per-method retry override."""
        pass

# When an actor dies:
try:
    ray.get(worker.do_work.remote())
except ray.exceptions.RayActorError as e:
    print(f"Actor died: {e}")
    # No parent notification, no structured propagation
    # Actor may have been restarted by GCS already

# Detached actors survive driver exit
worker = ResilientWorker.options(
    lifetime="detached", name="persistent_worker"
).remote()
```

**Source links:**
- [`max_restarts`, `max_task_retries` options](https://github.com/ray-project/ray/blob/25da7108abbec7606e8e0ce057e3a5d22421850f/python/ray/actor.py#L1394-L1412)
- [`RayActorError`](https://github.com/ray-project/ray/blob/25da7108abbec7606e8e0ce057e3a5d22421850f/python/ray/exceptions.py#L332)
- [`ActorDiedError`](https://github.com/ray-project/ray/blob/25da7108abbec7606e8e0ce057e3a5d22421850f/python/ray/exceptions.py#L371)
- [GCS `GcsActorManager`](https://github.com/ray-project/ray/blob/25da7108abbec7606e8e0ce057e3a5d22421850f/src/ray/gcs/gcs_actor_manager.h#L93)
- [GCS actor restart logic](https://github.com/ray-project/ray/blob/25da7108abbec7606e8e0ce057e3a5d22421850f/src/ray/gcs/gcs_actor_manager.cc) (2032 lines)

### Comparison

| Aspect | Monarch | Ray |
|--------|---------|-----|
| **Model** | Supervision tree (Erlang-style) | Flat restart policy per actor |
| **Failure notification** | `__supervise__(MeshFailure)` on parent | `RayActorError` exception on caller |
| **Propagation** | Structured: up the tree, each level handles or escalates | Flat: exception to caller, no tree |
| **Restart** | Parent decides (can respawn, log, ignore) | Automatic by GCS up to `max_restarts` |
| **Per-method control** | No (supervision is at actor level) | Yes (`@ray.method(max_task_retries=N)`) |
| **Quorum training** | Built-in via TorchFT integration | Not built-in (external libraries) |
| **Unhandled failures** | `unhandled_fault_hook` at client | Actor marked dead, `RayActorError` on future calls |
| **Detached actors** | N/A (single controller model) | `lifetime="detached"` survives driver exit |

---

## 6. Data Plane — RDMA vs Object Store

> **Notebook refs**: [`notebooks/07_rdma_weight_sync.py`](../notebooks/07_rdma_weight_sync.py), [`notebooks/07b_weight_sync_deep_dive.py`](../notebooks/07b_weight_sync_deep_dive.py) — RDMABuffer, magic pointer pattern, ibverbs deep dive.

This matters enormously for async RL and any workload that moves large tensors between actors.

### Monarch

Monarch separates control plane from data plane explicitly:
- **Control plane**: actor endpoint messages (small, serialized Python objects)
- **Data plane**: `RDMABuffer` — a ~150-byte handle that represents remote GPU/CPU memory

The key insight is the **"magic pointer" pattern**: you send a tiny handle (control plane) and the receiver does a one-sided RDMA read to pull the bulk data (data plane). The remote CPU is not involved in the transfer.

```python
# Monarch: zero-copy weight transfer via RDMA
from monarch.rdma import RDMABuffer

# Trainer side: register weights as RDMA buffer
class Trainer(Actor):
    @endpoint
    def get_weight_handle(self) -> RDMABuffer:
        # ~150-byte handle — just metadata, no data copy
        return RDMABuffer(self.model.state_dict_tensors().view(torch.uint8))

# Inference side: pull weights when ready
class Inference(Actor):
    @endpoint
    async def sync_weights(self, trainer_handle: RDMABuffer) -> None:
        # One-sided RDMA read: pulls directly from trainer's GPU memory
        # Trainer CPU is NOT involved — no interruption to training
        await trainer_handle.read_into(self.local_weight_buffer)
        self.model.load_state_dict(self.local_weight_buffer)

# Orchestration
handle = trainer.get_weight_handle.call_one().get()  # ~150 bytes via control plane
inference.sync_weights.broadcast(handle)              # RDMA pull via data plane
```

Under the hood, `RDMABuffer` uses InfiniBand verbs (ibverbs) for true one-sided RDMA — the `RdmaController` actor manages queue pairs and memory registration.

**Source links:**
- [`RDMABuffer`](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/python/monarch/_src/rdma/rdma.py#L210)
- [`RDMABuffer.read_into()`](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/python/monarch/_src/rdma/rdma.py#L263)
- [`RdmaController` actor](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/python/monarch/_src/rdma/rdma.py#L122)
- [Rust RDMA manager actor](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/monarch_rdma/src/rdma_manager_actor.rs)
- [Rust `RdmaBuffer`, `QueuePair`](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/monarch_rdma/src/rdma_components.rs)
- [InfiniBand verbs primitives](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/monarch_rdma/src/ibverbs_primitives.rs)

### Ray

Ray uses the **Plasma object store** (shared memory) for intra-node data sharing, and gRPC + object transfer for cross-node. For GPU collectives, `ray.util.collective` wraps NCCL and Gloo — but these are **collective** (two-sided, blocking) rather than one-sided RDMA.

```python
# Ray: weight transfer via object store
import ray

@ray.remote(num_gpus=1)
class Trainer:
    def get_weights(self) -> dict:
        # Serializes entire state dict through Plasma object store
        return self.model.state_dict()

@ray.remote(num_gpus=1)
class Inference:
    def load_weights(self, weights: dict) -> None:
        # Deserialized from object store — full copy
        self.model.load_state_dict(weights)

# Transfer: goes through Plasma object store
weights_ref = trainer.get_weights.remote()
ray.get(inference.load_weights.remote(weights_ref))

# Alternative: NCCL collectives (two-sided, blocking)
from ray.util.collective import allreduce
# Requires all participants to call collectively
```

Ray also has an experimental `tensor_transport` mechanism for NCCL/Gloo-backed method returns, but it's opt-in and method-level.

**Source links:**
- [`GroupManager`](https://github.com/ray-project/ray/blob/25da7108abbec7606e8e0ce057e3a5d22421850f/python/ray/util/collective/collective.py#L71) — collective group management
- [NCCL collective group](https://github.com/ray-project/ray/blob/25da7108abbec7606e8e0ce057e3a5d22421850f/python/ray/util/collective/collective_group/nccl_collective_group.py)
- [Tensor transport per-method config](https://github.com/ray-project/ray/blob/25da7108abbec7606e8e0ce057e3a5d22421850f/python/ray/actor.py#L1921)
- [C++ CoreWorker (object transfer)](https://github.com/ray-project/ray/blob/25da7108abbec7606e8e0ce057e3a5d22421850f/src/ray/core_worker/core_worker.h)

### Comparison

| Aspect | Monarch | Ray |
|--------|---------|-----|
| **Intra-node** | Shared process mesh, direct memory access | Plasma object store (shared memory) |
| **Cross-node bulk** | One-sided RDMA (`RDMABuffer.read_into`) | gRPC object transfer, or NCCL collectives |
| **GPU-GPU** | RDMA (bypasses CPU on remote side) | NCCL via `ray.util.collective` (two-sided) |
| **Weight sync pattern** | Handle message (control) + RDMA pull (data) | `ray.get(ref)` fetches through object store |
| **Blocking** | One-sided: receiver pulls when ready | Two-sided: collective requires all participants |
| **Handle size** | ~150 bytes (just metadata) | Full serialized object in Plasma |
| **Training interruption** | None — trainer CPU not involved in RDMA read | Trainer must serialize and send weights |

---

## 7. Distributed Tensors and Training

> **Notebook ref**: [`notebooks/04_distributed_tensors.py`](../notebooks/04_distributed_tensors.py) — `DeviceMesh.activate()`, distributed tensor ops, `reduce()`, `inspect()`.

### Monarch

Monarch provides **first-class distributed tensors**. When a `DeviceMesh` is activated, standard PyTorch operations automatically execute across all devices in the mesh. The controller sees a single logical tensor; the runtime manages distribution.

```python
# Monarch: distributed tensors with mesh activation
import torch
from monarch.actor import this_host
from monarch.common.device_mesh import DeviceMesh
from monarch.fetch import inspect

# Create mesh and activate it
mesh = this_host().spawn_procs({"gpu": 8})

with mesh.activate():
    # Standard PyTorch — but runs on all 8 GPUs
    t = torch.rand(3, 4, device="cuda")

    # Each GPU has its own copy — reduce across a dimension
    t.reduce("gpu", reduction="avg")

    # Inspect: fetch from a specific device to the controller
    local_copy = inspect(t, gpu=0)  # brings GPU 0's tensor to controller
    print(local_copy.shape)  # torch.Size([3, 4])

# Multi-dimensional meshes for parallelism strategies
mesh_2d = DeviceMesh(hosts=4, gpus_per_host=8, dims=("dp", "tp"))
with mesh_2d.activate():
    loss = model(X)
    loss.backward()
    # All-reduce gradients across data-parallel dimension only
    for p in model.parameters():
        p.grad.reduce_("dp", reduction="avg")
    optimizer.step()
```

**Source links:**
- [`Tensor` (distributed)](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/python/monarch/common/tensor.py#L78)
- [`DeviceMesh`](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/python/monarch/common/device_mesh.py#L154)
- [`DeviceMesh.activate()`](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/python/monarch/common/device_mesh.py#L310)
- [`reduce()`](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/python/monarch/common/tensor.py#L792)
- [`inspect()`](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/python/monarch/fetch.py#L50)
- [NCCL collective comms (Rust)](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/monarch_tensor_worker/src/comm.rs)

### Ray

Ray provides **no first-class distributed tensor**. Instead, Ray Train wraps PyTorch's `DistributedDataParallel` (DDP) or FSDP, managing the distributed setup but leaving tensor operations to PyTorch:

```python
# Ray Train: wraps PyTorch DDP
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
import ray.train.torch

def train_func():
    # Ray sets up the distributed environment
    model = build_model()
    model = ray.train.torch.prepare_model(model)  # wraps in DDP

    dataset = ray.train.get_dataset_shard("train")
    for batch in dataset.iter_torch_batches():
        loss = model(batch)
        loss.backward()  # DDP handles gradient sync
        optimizer.step()
        ray.train.report({"loss": loss.item()})

trainer = TorchTrainer(
    train_func,
    scaling_config=ScalingConfig(num_workers=8, use_gpu=True),
)
result = trainer.fit()
```

**Source links:**
- [`TorchTrainer`](https://github.com/ray-project/ray/blob/25da7108abbec7606e8e0ce057e3a5d22421850f/python/ray/train/torch/torch_trainer.py#L11)
- [`TrainingIterator`](https://github.com/ray-project/ray/blob/25da7108abbec7606e8e0ce057e3a5d22421850f/python/ray/train/trainer.py#L36)

### Comparison

| Aspect | Monarch | Ray |
|--------|---------|-----|
| **Distributed tensor** | First-class: `mesh.activate()`, standard PyTorch ops distributed | No — wraps DDP/FSDP via Ray Train |
| **Programming model** | Single controller writes normal PyTorch; runtime distributes | Training function runs on each worker (SPMD) |
| **Collective ops** | `tensor.reduce("dim", reduction="avg")` — mesh-dimension-aware | PyTorch DDP handles gradient sync automatically |
| **Mesh dimensions** | Named: `"dp"`, `"tp"`, `"pp"` — reduce/broadcast by name | N/A — DDP uses process groups |
| **Inspect/fetch** | `inspect(t, gpu=0)` — bring remote tensor to controller | Manual: `ray.get()` or `ray.train.report()` |
| **Parallelism strategies** | Compose via mesh dimensions (DP + TP + PP) | Configure via Ray Train (DDP, FSDP, DeepSpeed) |

---

## 8. Interactive DevX (Single Controller)

> **Notebook ref**: [`notebooks/02_interactive_devx.py`](../notebooks/02_interactive_devx.py) — SPMD actors, `serve()`, interactive endpoint calls from notebook.

### Monarch

Monarch's **single-controller paradigm** is designed for interactive development. One Python process (a notebook or script) orchestrates all distributed actors. The `SPMDActor` pattern lets you write SPMD-style code that's controlled from a single interactive session:

```python
# Monarch: interactive single-controller development
from monarch._src.spmd.actor import SPMDActor
from monarch._src.job.spmd import serve
from monarch.actor import Actor, endpoint, this_host

# Define an actor — same as always
class Researcher(Actor):
    @endpoint
    def train_step(self, lr: float) -> dict:
        # ... training logic ...
        return {"loss": 0.42, "step": self.step}

    @endpoint
    def get_state(self) -> dict:
        return {"weights_hash": hash(self.weights), "steps": self.step}

# In a notebook cell: spawn and interact
host = this_host()
procs = host.spawn_procs(per_host={"gpus": 4})
researchers = procs.spawn("researchers", Researcher)

# Interactive: call endpoints, inspect results, iterate
result = researchers.train_step.call(lr=1e-4).get()
print(result)  # see all 4 GPUs' results immediately

state = researchers.get_state.call().get()
print(state)  # inspect distributed state interactively

# Change hyperparameters on the fly — no restart needed
result2 = researchers.train_step.call(lr=5e-5).get()
```

The key insight: **the notebook IS the controller**. You don't submit jobs — you interactively drive distributed computation from your development environment.

**Source links:**
- [`SPMDActor`](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/python/monarch/_src/spmd/actor.py#L29)
- [`serve()` function](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/python/monarch/_src/job/spmd.py#L149)

### Ray

Ray uses a **job submission model**. You write a training script, submit it to a Ray cluster, and monitor it externally. Interactive development is possible via `ray.init()` in a notebook, but the programming model is still "submit remote tasks":

```python
# Ray: job submission model
import ray

ray.init()  # connect to cluster (or start local)

@ray.remote(num_gpus=1)
class Trainer:
    def train_step(self, lr: float) -> dict:
        return {"loss": 0.42}

# Create actors — but each is an independent remote process
trainers = [Trainer.remote() for _ in range(4)]

# Call: fan-out + gather pattern
refs = [t.train_step.remote(lr=1e-4) for t in trainers]
results = ray.get(refs)

# To change something: call new methods on existing handles
refs2 = [t.train_step.remote(lr=5e-5) for t in trainers]
results2 = ray.get(refs2)

# For production: Ray Jobs CLI
# $ ray job submit --working-dir . -- python train.py
# $ ray job status <job_id>
# $ ray job logs <job_id>
```

Ray also has **Ray Serve** for deploying models as HTTP endpoints, which is a different paradigm from Monarch's single-controller approach.

### Comparison

| Aspect | Monarch | Ray |
|--------|---------|-----|
| **Development model** | Notebook IS the controller (single controller) | Submit jobs to cluster; notebook as client |
| **Interaction** | Direct: `actors.method.call().get()` in notebook cell | Remote: `.remote()` + `ray.get()` |
| **State inspection** | `inspect(tensor, gpu=0)` — pull any distributed state | `ray.get(actor.get_state.remote())` |
| **Hyperparameter changes** | Instant: next `call()` uses new params | Same: next `.remote()` uses new params |
| **Production deployment** | Same code runs in notebook and production | Ray Jobs CLI, Ray Serve for serving |
| **SPMD support** | Native `SPMDActor`, `serve()` | Ray Train wraps DDP/FSDP |
| **Restart on code change** | Hot-reload: respawn actors on mesh | Restart workers or resubmit job |

---

## 9. Design Philosophy Summary

The comparison reveals a fundamental tension between two design philosophies:

### Monarch: Structured Topology-Aware Primitives

Monarch is a **structured, topology-aware actor system** built for the specific reality of large-scale GPU training. You explicitly build mesh topologies, spawn actors with awareness of their position, and get automatic tree-routing and supervision. The Erlang heritage is clear: message ordering guarantees, supervision trees, single-controller orchestration.

**Core bet**: At GPU-training scale (thousands of GPUs, failures every 3 hours), you need structured primitives that compose — meshes, tree-routing, supervision, RDMA handles. The cost of managing topology explicitly is repaid by the runtime's ability to route efficiently, handle failures systematically, and separate control plane from data plane.

```
Monarch: structured topology → efficient routing → systematic fault handling
         ┌─────────────────────────────────────────────────────────┐
         │  Controller (notebook/script)                           │
         │    ↓ spawn_procs()     ↓ spawn()     ↓ call()          │
         │  ┌─────────────┐    ┌──────────┐   ┌────────────────┐  │
         │  │  ProcMesh   │ →  │ActorMesh │ → │ Tree broadcast │  │
         │  │ {dp:4,tp:8} │    │ sliceable│   │ O(log N)       │  │
         │  └─────────────┘    └──────────┘   └────────────────┘  │
         │                                                         │
         │  Supervision: parent → __supervise__ → handle/escalate  │
         │  Data plane:  RDMABuffer → one-sided read (bypass CPU)  │
         └─────────────────────────────────────────────────────────┘
```

### Ray: General-Purpose Simplicity

Ray is a **general-purpose distributed computing framework** with a flat actor model layered on top of a sophisticated distributed object store. Scheduling is automatic, the programming model is simpler (just `@ray.remote` and `.remote()`), and it serves a broader range of workloads (ML training, serving, data processing, simulations).

**Core bet**: A simple, general-purpose abstraction (remote functions + actors + object store) is more valuable than specialized primitives, because higher-level patterns can be built on top (Ray Train, Ray Serve, Ray Data). The ecosystem wins over built-in structure.

```
Ray: simple abstraction → broad applicability → rich ecosystem
     ┌──────────────────────────────────────────────────────────┐
     │  Driver (script/notebook)                                │
     │    ↓ @ray.remote         ↓ .remote()    ↓ ray.get()     │
     │  ┌─────────────────┐  ┌────────────┐  ┌──────────────┐  │
     │  │  GCS + Raylet   │  │ ObjectRef  │  │ Plasma store │  │
     │  │  auto-schedule  │  │ ownership  │  │ shared mem   │  │
     │  └─────────────────┘  └────────────┘  └──────────────┘  │
     │                                                          │
     │  Fault tolerance: max_restarts + max_task_retries (flat) │
     │  Data plane: Plasma object store + optional NCCL         │
     └──────────────────────────────────────────────────────────┘
```

### When to Choose Each

| Criteria | Monarch | Ray |
|----------|---------|-----|
| **GPU training at scale** (1000+ GPUs) | Strong fit — meshes, RDMA, supervision | Possible via Ray Train, but less native |
| **Async RL** (training + inference overlap) | Purpose-built — RDMA weight sync, actor separation | Possible but requires manual coordination |
| **General distributed computing** | Can work, but mesh topology may be overkill | Strong fit — simple API, broad ecosystem |
| **ML serving** | Not the primary focus | Ray Serve is mature and production-ready |
| **Data processing pipelines** | Not designed for this | Ray Data handles this well |
| **Fault tolerance requirements** | Supervision trees for systematic handling | Flat restart policies for simple cases |
| **Interactive development** | Single controller in notebook — first-class | `ray.init()` in notebook — works but less native |
| **Team familiarity** | Requires learning mesh/actor concepts | `@ray.remote` is very approachable |

---

## Source Reference Index

### Monarch Sources (pytorch/monarch @ `febefda`)

| File | Key Items |
|------|-----------|
| [`python/monarch/_src/actor/actor_mesh.py`](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/python/monarch/_src/actor/actor_mesh.py) | `Actor` (L1569), `ActorMesh` (L1606) |
| [`python/monarch/_src/actor/endpoint.py`](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/python/monarch/_src/actor/endpoint.py) | `@endpoint` (L509), `call_one` (L241), `call` (L268), `stream` (L305), `broadcast` (L343) |
| [`python/monarch/_src/actor/proc_mesh.py`](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/python/monarch/_src/actor/proc_mesh.py) | `ProcMesh` (L234) |
| [`python/monarch/_src/actor/host_mesh.py`](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/python/monarch/_src/actor/host_mesh.py) | `this_host()` (L48), `spawn_procs()` (L145) |
| [`python/monarch/_src/actor/supervision.py`](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/python/monarch/_src/actor/supervision.py) | `unhandled_fault_hook()` (L17) |
| [`python/monarch/_src/rdma/rdma.py`](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/python/monarch/_src/rdma/rdma.py) | `RDMABuffer` (L210), `read_into()` (L263), `RdmaController` (L122) |
| [`python/monarch/_src/spmd/actor.py`](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/python/monarch/_src/spmd/actor.py) | `SPMDActor` (L29) |
| [`python/monarch/common/tensor.py`](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/python/monarch/common/tensor.py) | `Tensor` (L78), `reduce()` (L792) |
| [`python/monarch/common/device_mesh.py`](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/python/monarch/common/device_mesh.py) | `DeviceMesh` (L154), `activate()` (L310) |
| [`hyperactor/src/mailbox.rs`](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/hyperactor/src/mailbox.rs) | Rust mailbox engine (3977 lines) |
| [`hyperactor/src/ordering.rs`](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/hyperactor/src/ordering.rs) | `BufferState`, `OrderedSender` (L25-50) |
| [`hyperactor_mesh/src/proc_mesh.rs`](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/hyperactor_mesh/src/proc_mesh.rs) | Rust ProcMesh |
| [`hyperactor_mesh/src/actor_mesh.rs`](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/hyperactor_mesh/src/actor_mesh.rs) | Rust ActorMesh |
| [`hyperactor_mesh/src/comm/multicast.rs`](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/hyperactor_mesh/src/comm/multicast.rs) | Tree-based broadcast |
| [`monarch_rdma/src/rdma_manager_actor.rs`](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/monarch_rdma/src/rdma_manager_actor.rs) | Rust RDMA manager |
| [`monarch_rdma/src/rdma_components.rs`](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/monarch_rdma/src/rdma_components.rs) | `RdmaBuffer`, `QueuePair` |
| [`monarch_rdma/src/ibverbs_primitives.rs`](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/monarch_rdma/src/ibverbs_primitives.rs) | InfiniBand verbs |
| [`monarch_tensor_worker/src/comm.rs`](https://github.com/pytorch/monarch/blob/febefda69e1f9e9a696596b3de368ac557320b21/monarch_tensor_worker/src/comm.rs) | NCCL collective comms |

### Ray Sources (ray-project/ray @ `25da710`)

| File | Key Items |
|------|-----------|
| [`python/ray/actor.py`](https://github.com/ray-project/ray/blob/25da7108abbec7606e8e0ce057e3a5d22421850f/python/ray/actor.py) | `ActorClass` (L1201), `remote()` (L1356), `ActorHandle` (L1886), `_remote()` (L1511), `ActorMethod._remote()` (L802), `max_restarts` (L1394), `tensor_transport` (L1921), `_ray_allow_out_of_order_execution` (L1937) |
| [`python/ray/exceptions.py`](https://github.com/ray-project/ray/blob/25da7108abbec7606e8e0ce057e3a5d22421850f/python/ray/exceptions.py) | `RayActorError` (L332), `ActorDiedError` (L371) |
| [`python/ray/_private/worker.py`](https://github.com/ray-project/ray/blob/25da7108abbec7606e8e0ce057e3a5d22421850f/python/ray/_private/worker.py) | `ray.get()` (L2858), `ray.wait()` (L3074) |
| [`python/ray/includes/object_ref.pxi`](https://github.com/ray-project/ray/blob/25da7108abbec7606e8e0ce057e3a5d22421850f/python/ray/includes/object_ref.pxi) | `ObjectRef` (L37) |
| [`python/ray/util/placement_group.py`](https://github.com/ray-project/ray/blob/25da7108abbec7606e8e0ce057e3a5d22421850f/python/ray/util/placement_group.py) | `PlacementGroup` (L42), `placement_group()` (L146) |
| [`python/ray/util/collective/collective.py`](https://github.com/ray-project/ray/blob/25da7108abbec7606e8e0ce057e3a5d22421850f/python/ray/util/collective/collective.py) | `GroupManager` (L71) |
| [`python/ray/util/collective/collective_group/nccl_collective_group.py`](https://github.com/ray-project/ray/blob/25da7108abbec7606e8e0ce057e3a5d22421850f/python/ray/util/collective/collective_group/nccl_collective_group.py) | NCCL backend |
| [`python/ray/train/torch/torch_trainer.py`](https://github.com/ray-project/ray/blob/25da7108abbec7606e8e0ce057e3a5d22421850f/python/ray/train/torch/torch_trainer.py) | `TorchTrainer` (L11) |
| [`python/ray/train/trainer.py`](https://github.com/ray-project/ray/blob/25da7108abbec7606e8e0ce057e3a5d22421850f/python/ray/train/trainer.py) | `TrainingIterator` (L36) |
| [`src/ray/core_worker/core_worker.h`](https://github.com/ray-project/ray/blob/25da7108abbec7606e8e0ce057e3a5d22421850f/src/ray/core_worker/core_worker.h) | C++ CoreWorker (L167) |
| [`src/ray/core_worker/actor_handle.h`](https://github.com/ray-project/ray/blob/25da7108abbec7606e8e0ce057e3a5d22421850f/src/ray/core_worker/actor_handle.h) | C++ ActorHandle (L34), `allow_out_of_order_execution` (L51) |
| [`src/ray/gcs/gcs_server.h`](https://github.com/ray-project/ray/blob/25da7108abbec7606e8e0ce057e3a5d22421850f/src/ray/gcs/gcs_server.h) | GCS server (L98) |
| [`src/ray/gcs/gcs_actor_manager.h`](https://github.com/ray-project/ray/blob/25da7108abbec7606e8e0ce057e3a5d22421850f/src/ray/gcs/gcs_actor_manager.h) | GCS actor lifecycle (L93) |
| [`src/ray/gcs/gcs_actor_manager.cc`](https://github.com/ray-project/ray/blob/25da7108abbec7606e8e0ce057e3a5d22421850f/src/ray/gcs/gcs_actor_manager.cc) | Actor restart logic (2032 lines) |
| [`src/ray/raylet/node_manager.h`](https://github.com/ray-project/ray/blob/25da7108abbec7606e8e0ce057e3a5d22421850f/src/ray/raylet/node_manager.h) | Raylet NodeManager (L70) |
| [`src/ray/raylet/placement_group_resource_manager.h`](https://github.com/ray-project/ray/blob/25da7108abbec7606e8e0ce057e3a5d22421850f/src/ray/raylet/placement_group_resource_manager.h) | C++ PG resource manager (L49) |

### Notebook References (this repo)

| Notebook | Concepts |
|----------|----------|
| [`notebooks/01_history_and_vision.py`](../notebooks/01_history_and_vision.py) | Actor model, PingPong, meshes, Ports, message ordering |
| [`notebooks/02_interactive_devx.py`](../notebooks/02_interactive_devx.py) | Single controller, SPMD, `serve()`, interactive development |
| [`notebooks/03_fault_tolerance.py`](../notebooks/03_fault_tolerance.py) | Supervision trees, `__supervise__`, TorchFT |
| [`notebooks/04_distributed_tensors.py`](../notebooks/04_distributed_tensors.py) | Mesh reshape, distributed tensor ops, reduce, inspect |
| [`notebooks/07_rdma_weight_sync.py`](../notebooks/07_rdma_weight_sync.py) | RDMABuffer, magic pointer pattern |
| [`notebooks/07b_weight_sync_deep_dive.py`](../notebooks/07b_weight_sync_deep_dive.py) | ibverbs, RdmaController, queue pairs |
