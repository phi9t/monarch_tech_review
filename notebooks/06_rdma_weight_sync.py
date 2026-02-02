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
    # RDMA & Weight Synchronization

    This notebook explores efficient weight synchronization for async RL systems:

    1. **The Bandwidth Hierarchy** - NVLink, InfiniBand, PCIe
    2. **The Problem: Collectives Are Blocking** - Why RL needs something different
    3. **How RDMA Works** - ibverbs, one-sided operations
    4. **The Magic Pointer Pattern** - Control plane vs data plane separation
    5. **CPU Staging** - Decoupling trainer and generator timing
    6. **Circular Weight Buffers** - Versioning without memory churn
    7. **Weight Re-sharding** - Handling different tensor layouts
    8. **Putting It All Together** - The complete pattern
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The Bandwidth Hierarchy

    Modern HPC clusters have multiple interconnects with vastly different bandwidths:

    ```
    ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                              NODE A                                                      │
    │                                                                                                          │
    │    ┌───────────────────────────────────────────────────────────────────────────────────────────────┐     │
    │    │                              NVSwitch / NVLink Fabric                                         │     │
    │    │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                      │     │
    │    │  │GPU 0 │ │GPU 1 │ │GPU 2 │ │GPU 3 │ │GPU 4 │ │GPU 5 │ │GPU 6 │ │GPU 7 │                      │     │
    │    │  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘                      │     │
    │    │     ########################################################################  900 GB/s NVLink │     │
    │    └─────────────────────────────────────────┬─────────────────────────────────────────────────────┘     │
    │                                              │                                                           │
    │                                         ======  64 GB/s PCIe                                             │
    │                                              │                                                           │
    │    ┌─────────┐  ------ 48 GB/s ------ ┌──────┴──┐                  ┌───────┐          ┌───────┐          │
    │    │  CPU 0  │     CPU interconnect   │  CPU 1  │ ====== 64 GB/s ══│ NIC 0 │          │ NIC 1 │          │
    │    └────┬────┘                        └────┬────┘      PCIe        └───┬───┘          └───┬───┘          │
    │         │                                  │                           │                  │              │
    │         ══════════════════════ 64 GB/s PCIe ═══════════════════════════╪══════════════════╪              │
    │                                                                        │                  │              │
    └────────────────────────────────────────────────────────────────────────┼──────────────────┼──────────────┘
                                                                             │                  │
                                                                           ======  50 GB/s   ======
                                                                        IB NDR400         IB NDR400
                                                                             │                  │
                                                            ┌────────────────┴──────────────────┴────────────────┐
                                                            │                                                    │
                                                            │              InfiniBand Switch                     │
                                                            │                                                    │
                                                            └────────────────┬──────────────────┬────────────────┘
                                                                             │                  │
                                                                           ======  50 GB/s   ======
                                                                        IB NDR400         IB NDR400
                                                                             │                  │
    ┌────────────────────────────────────────────────────────────────────────┼──────────────────┼──────────────┐
    │                                                                        │                  │              │
    │         ══════════════════════ 64 GB/s PCIe ═══════════════════════════╪══════════════════╪              │
    │         │                                  │                           │                  │              │
    │    ┌────┴────┐                        ┌────┴────┐      PCIe        ┌───┴───┐          ┌───┴───┐          │
    │    │  CPU 0  │     CPU interconnect   │  CPU 1  │ ====== 64 GB/s ══│ NIC 0 │          │ NIC 1 │          │
    │    └─────────┘ ------ 48 GB/s ------  └─────────┘                  └───────┘          └───────┘          │
    │                                              │                                                           │
    │                                           ======  64 GB/s PCIe                                           │
    │                                              │                                                           │
    │    ┌─────────────────────────────────────────┴─────────────────────────────────────────────────────┐     │
    │    │     ########################################################################  900 GB/s NVLink │     │
    │    │  ┌──┴───┐ ┌──┴───┐ ┌──┴───┐ ┌──┴───┐ ┌──┴───┐ ┌──┴───┐ ┌──┴───┐ ┌──┴───┐                      │     │
    │    │  │GPU 0 │ │GPU 1 │ │GPU 2 │ │GPU 3 │ │GPU 4 │ │GPU 5 │ │GPU 6 │ │GPU 7 │                      │     │
    │    │  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘                      │     │
    │    │                              NVSwitch / NVLink Fabric                                         │     │
    │    └───────────────────────────────────────────────────────────────────────────────────────────────┘     │
    │                                              NODE B                                                      │
    └──────────────────────────────────────────────────────────────────────────────────────────────────────────┘

    Bandwidth encoding (line intensity):
      ########  NVLink/NVSwitch   900 GB/s   (GPU ↔ GPU, same node)
      ========  PCIe Gen5 / IB     50-64 GB/s (CPU↔GPU, CPU↔NIC, cross-node)
      --------  CPU interconnect   48 GB/s   (CPU ↔ CPU, same node)
    ```

    RDMA can transfer between any registered memory (CPU or GPU) via the NICs.

    | Interconnect | Bandwidth | Latency | Use Case |
    |--------------|-----------|---------|----------|
    | **NVLink/NVSwitch** | 900 GB/s | ~1 μs | Same-node GPU↔GPU |
    | **InfiniBand NDR400** | 50 GB/s | ~1-2 μs | Cross-node RDMA |
    | **PCIe Gen5 x16** | 64 GB/s | ~1-2 μs | CPU↔GPU, CPU↔NIC |
    | **CPU interconnect** | 48 GB/s | ~100 ns | CPU↔CPU (same node) |

    **Key observations:**

    1. **NVLink dominates** - 900 GB/s is ~18x faster than cross-node RDMA. Same-node GPU↔GPU
       communication is nearly free compared to crossing the network.

    2. **RDMA >> Ethernet** - InfiniBand/RoCE at 50 GB/s is ~4x faster than 100GbE (12.5 GB/s),
       plus kernel bypass and lower latency. Worth the complexity for HPC workloads.

    3. **PCIe is faster than you'd think** - At 64 GB/s, CPU↔GPU transfers aren't the bottleneck
       people often assume. The real cost is synchronization, not bandwidth.

    **Rule of thumb**: Place the most bandwidth-intensive, frequent operations on NVLink
    (gradients, activations). Use RDMA for cross-node communication (weight sync, sharding).
    PCIe is fine for occasional CPU↔GPU transfers.

    We'll focus primarily on **NVLink** and **RDMA** for this notebook. Most people use these
    via **collectives**, exposed through PyTorch distributed:

    ```python
    import torch.distributed as dist

    # Initialize process group - NCCL uses NVLink (same-node) and RDMA (cross-node)
    dist.init_process_group(backend="nccl")

    # All-reduce: average gradients across all GPUs
    dist.all_reduce(gradients, op=dist.ReduceOp.SUM)
    gradients /= world_size

    # All-gather: collect tensors from all ranks
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)

    # Broadcast: send from rank 0 to all others
    dist.broadcast(weights, src=0)
    ```

    This works great for training. But for RL weight sync, we need something different...
    """)
    return


@app.cell
def _():
    import torch
    return (torch,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. The Problem: Collectives Are Blocking

    Collectives work great for training - everyone computes gradients, then synchronizes.
    But async RL has a different access pattern.

    ### High Variance in Generation Times

    Generators have wildly different completion times:
    - Some prompts → 10 tokens (fast)
    - Other prompts → 1000 tokens (slow)

    With collectives, fast generators wait for slow ones:

    ```
    Generator 0: ├── gen (fast) ──┤  ⚠️ WAITING...
    Generator 1: ├────── gen (slow) ──────┤
    Generator 2: ├── gen (fast) ──┤  ⚠️ WAITING...
                                          ↓
                              all_gather(weights)  # Everyone waits!
    ```

    ### What About send/recv?

    PyTorch distributed does have point-to-point primitives:

    ```python
    # Sender side
    dist.send(tensor, dst=receiver_rank)

    # Receiver side
    dist.recv(tensor, src=sender_rank)
    ```

    But this is **two-sided** - both sender and receiver must coordinate:
    - Receiver must call `recv()` before sender's `send()` completes
    - Trainer would need to wait until generators are ready to receive
    - Still blocking on coordination!

    ### The One-Sided Solution: RDMA

    What if the sender could write directly to the receiver's memory without coordination?

    ```
    Two-sided (send/recv):
      Sender: "I have data"  ──────────►  Receiver: "I'm ready"
      Sender: sends data     ──────────►  Receiver: receives data
                             2 messages required

    One-sided (RDMA):
      Sender: writes directly to receiver's memory
                             No coordination needed!
    ```

    This is what RDMA enables: **one-sided memory operations**.
    """)
    return


# =============================================================================
# SECTION 3: How RDMA Works
# =============================================================================


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. How RDMA Works (ibverbs)

    RDMA (Remote Direct Memory Access) lets one machine read/write another machine's memory
    directly, bypassing the kernel and CPU on both sides.

    ### The ibverbs Stack

    ```
    ┌─────────────────────────────────────────────────────────┐
    │  Application (PyTorch, Monarch, etc.)                   │
    ├─────────────────────────────────────────────────────────┤
    │  libibverbs  (userspace RDMA API)                       │
    ├─────────────────────────────────────────────────────────┤
    │  Provider driver (mlx5, efa, rxe, etc.)                 │
    ├─────────────────────────────────────────────────────────┤
    │  Hardware (InfiniBand NIC, RoCE NIC, etc.)              │
    └─────────────────────────────────────────────────────────┘
    ```

    We'll focus on **InfiniBand** and **RoCE** (RDMA over Converged Ethernet).
    Other transports like AWS EFA exist but we won't cover them here.

    ### Key RDMA Operations

    | Operation | Description |
    |-----------|-------------|
    | `RDMA_WRITE` | Write to remote memory (one-sided) |
    | `RDMA_READ` | Read from remote memory (one-sided) |
    | `SEND/RECV` | Two-sided messaging (like TCP) |

    The magic is in `RDMA_WRITE` and `RDMA_READ` - they're **one-sided**:
    - Remote CPU is not involved
    - Remote application doesn't need to call anything
    - NIC handles everything in hardware

    ### Memory Registration

    Before RDMA, memory must be **registered** with the NIC:

    ```python
    # Conceptually (actual ibverbs API is in C)
    mr = rdma_register_memory(buffer, size)
    # Returns:
    #   - lkey: local access key (for local operations)
    #   - rkey: remote access key (share with remote peer)
    #   - addr: physical/virtual address
    ```

    The `(addr, rkey)` pair is a **remote-accessible pointer**. Share it with a peer,
    and they can read/write your memory directly.

    ### Queue Pair Setup

    Before any RDMA operations, you need to establish a **Queue Pair (QP)** between
    sender and receiver. This is a one-time connection setup:

    ```
    ┌─────────────┐                           ┌─────────────┐
    │   Sender    │                           │  Receiver   │
    │             │                           │             │
    │  Create QP  │ ─── exchange QP info ───► │  Create QP  │
    │  (qp_num,   │ ◄── (qp_num, lid, gid) ── │             │
    │   lid, gid) │                           │             │
    │             │                           │             │
    │  Move QP to │                           │  Move QP to │
    │  RTR → RTS  │                           │  RTR → RTS  │
    │             │                           │             │
    │  Now ready  │ ═══ RDMA operations ════► │  Now ready  │
    │  for RDMA!  │                           │  for RDMA!  │
    └─────────────┘                           └─────────────┘
    ```

    This is where **Monarch actors** shine. Because you can spawn arbitrary actors,
    you can create **RDMA Manager actors** that:
    - Initialize QPs on their respective hosts
    - Exchange QP info via actor messages
    - Manage the connection lifecycle

    ```python
    # Monarch pattern: RDMA managers as actors
    class RDMAManager(Actor):
        def __init__(self):
            self.qp = create_queue_pair()
            self.qp_info = get_qp_info(self.qp)  # (qp_num, lid, gid)

        @endpoint
        def get_qp_info(self) -> QpInfo:
            return self.qp_info

        @endpoint
        def connect(self, remote_qp_info: QpInfo):
            # Transition QP: INIT → RTR → RTS
            connect_qp(self.qp, remote_qp_info)

    # Setup: exchange QP info via actor messages, then RDMA is ready
    trainer_info = trainer_rdma.get_qp_info.call_one().get()
    generator_rdma.connect.call_one(trainer_info).get()
    ```

    The actor abstraction makes RDMA connection management natural and composable.

    ### Monarch Using Monarch: RdmaController

    Here's the cool part: **Monarch uses itself** to manage RDMA infrastructure. Looking at
    the actual Python code in `monarch/_src/rdma/rdma.py`:

    ```python
    # From Monarch's RDMA implementation
    from monarch._src.actor.proc_mesh import get_or_spawn_controller

    class RdmaController(Actor):
        '''Singleton controller that coordinates RDMA initialization.'''

        def __init__(self):
            # Track which proc meshes have RDMA initialized
            self._manager_futures: dict[ProcMesh, Future[RdmaManager]] = {}

        @endpoint
        async def init_rdma_on_mesh(self, proc_mesh: ProcMesh) -> None:
            '''Lazily initialize RDMA on a proc mesh.'''
            if proc_mesh not in self._manager_futures:
                self._manager_futures[proc_mesh] = Future(
                    coro=RdmaManager.create(proc_mesh)
                )
            await self._manager_futures[proc_mesh]

    # Cached initialization - only runs once per process
    @functools.cache
    def _ensure_init_rdma_manager():
        async def task():
            controller = await get_or_spawn_controller("rdma_controller", RdmaController)
            await controller.init_rdma_on_mesh.call_one(current_proc_mesh())
        return spawn_task(task())
    ```

    This is **Monarch building Monarch** - the RDMA subsystem uses the same patterns:

    - `get_or_spawn_controller("rdma_controller", RdmaController)` ensures one global controller
    - The controller lazily initializes RDMA managers per proc mesh
    - `@functools.cache` ensures we only bootstrap once per process
    - Under the hood, the actual RDMA operations are in Rust (`RdmaManagerActor`)

    It's actors all the way down.

    ### Why This Matters for Weight Sync

    Remember: CPU memory AND GPU memory can both be registered for RDMA.

    ```
    Trainer:
      1. Register weight buffer with RDMA NIC
      2. Get (addr, rkey) handle
      3. Share handle with generators (tiny message)

    Generator:
      1. Receive handle
      2. RDMA_READ directly from trainer's memory
      3. No coordination with trainer needed!
    ```

    The trainer doesn't even know when generators pull weights. True one-sided.
    """)
    return


# =============================================================================
# SECTION 4: The Magic Pointer Pattern
# =============================================================================


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. The Magic Pointer Pattern

    Now here's the key insight from our RDMA discussion: to represent remote data,
    we only need a **tiny handle** - the `(addr, rkey, size)` tuple.

    Monarch wraps this in `RDMABuffer`. Let's see how small it actually is:
    """)
    return


@app.cell
def _(torch):
    # Let's measure the actual size of an RDMABuffer handle
    # RDMABuffer requires actor context, so we spawn an actor to create them
    import pickle

    def show_fallback():
        """Fallback: show expected sizes based on RDMABuffer structure."""
        print("(RDMA not available - showing expected handle sizes)\n")
        print("RDMABuffer contains: addr (8B) + rkey (4B) + size (8B) + owner (~100B)")
        print("Total serialized size: ~150-200 bytes regardless of tensor size\n")

        sizes = [("1 KB", 1024), ("1 MB", 1024**2), ("1 GB", 1024**3)]
        handle_bytes = 150  # approximate

        for name, tensor_bytes in sizes:
            ratio = tensor_bytes / handle_bytes
            print(f"{name:<8} tensor → ~150 byte handle → {ratio:,.0f}x compression")

        print("\n→ Handle size is O(1) regardless of tensor size!")

    try:
        from monarch.rdma import is_rdma_available

        if not is_rdma_available():
            show_fallback()
        else:
            from monarch.actor import Actor, endpoint, this_host
            from monarch.rdma import RDMABuffer

            class BufferSizeDemo(Actor):
                """Actor that creates RDMABuffers and measures their size."""

                @endpoint
                def measure_buffer_sizes(self) -> list:
                    import pickle as _pickle
                    results = []
                    sizes = [
                        ("1 KB", 256),
                        ("1 MB", 256 * 1024),
                        ("10 MB", 256 * 1024 * 10),
                    ]

                    for name, numel in sizes:
                        tensor = torch.randn(numel)
                        tensor_bytes = tensor.numel() * tensor.element_size()

                        # Key: convert to byte view (1D contiguous uint8)
                        byte_tensor = tensor.view(torch.uint8).flatten()
                        buffer = RDMABuffer(byte_tensor)
                        handle_bytes = len(_pickle.dumps(buffer))

                        results.append((name, tensor_bytes, handle_bytes))

                    return results

            # Spawn actor and measure
            host = this_host()
            proc = host.spawn_procs({"procs": 1})
            demo = proc.spawn("buffer_demo", BufferSizeDemo)

            results = demo.measure_buffer_sizes.call_one().get()

            print("RDMABuffer handle size vs actual tensor size:\n")
            print(f"{'Tensor Size':<12} {'Actual Bytes':<15} {'Handle Size':<15} {'Ratio':<10}")
            print("-" * 55)

            for name, tensor_bytes, handle_bytes in results:
                ratio = tensor_bytes / handle_bytes
                print(f"{name:<12} {tensor_bytes:>12,} B   {handle_bytes:>6} B        {ratio:>8,.0f}x")

            print("\n→ Handle size is O(1) regardless of tensor size!")

    except Exception as e:
        print(f"(RDMA setup failed: {e})\n")
        show_fallback()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### The Magic Pointer

    This is the core pattern: **separate control plane from data plane**.

    - **Control plane** (actor messages): Send tiny handle (~100 bytes)
    - **Data plane** (RDMA): Bulk transfer of actual data (~10 GB)

    Think of `RDMABuffer` as a **magic pointer** - it's a pointer that works across machines:

    ```
    Trainer                              Generator
    ┌─────────────┐                     ┌─────────────┐
    │ weights     │                     │ local copy  │
    │ (10 GB)     │                     │ (empty)     │
    └──────┬──────┘                     └──────┬──────┘
           │                                   │
           │  1. Create RDMABuffer             │
           │     (register memory, get handle) │
           │                                   │
           ├─────── 2. Send handle ───────────►│  (~100 bytes via actor)
           │                                   │
           │◄────── 3. RDMA read ──────────────┤  (~10 GB via hardware)
           │        (no trainer involvement!)  │
    ```

    The trainer doesn't even know when generators pull weights. True one-sided.

    ### RDMABuffer in Action

    From `monarch.rdma`:

    ```python
    from monarch.rdma import RDMABuffer

    # Trainer side: register weights
    weights = torch.randn(1024, 1024, device="cuda")
    buffer = RDMABuffer(weights.view(torch.uint8).flatten())

    # Return buffer as part of an endpoint response
    # This is a TINY message - just the handle!
    @endpoint
    def get_weight_handle(self) -> RDMABuffer:
        return self.buffer

    # Generator side: receive handle, pull directly into GPU
    handle = trainer.get_weight_handle.call_one().get()  # Tiny message
    gpu_weights = model.weights.view(torch.uint8).flatten()
    handle.read_into(gpu_weights).get()                   # Bulk RDMA → GPU
    ```

    See the [GRPO Actor example](https://meta-pytorch.org/monarch/generated/examples/grpo_actor.html)
    for a minimal implementation showing RDMA data flow. We'll build a more complete
    version in the following sections.

    ### Batching with RDMAAction

    When you have multiple buffers to transfer, `RDMAAction` lets you batch them:

    ```python
    from monarch.rdma import RDMABuffer, RDMAAction

    # Instead of sequential transfers:
    # await buffer_a.read_into(local_a)
    # await buffer_b.read_into(local_b)
    # await buffer_c.read_into(local_c)

    # Batch them for concurrent execution:
    action = RDMAAction()
    action.read_into(buffer_a, local_a)
    action.read_into(buffer_b, local_b)
    action.read_into(buffer_c, local_c)
    await action.submit()  # Executes concurrently per owner
    ```

    This is especially useful when syncing multiple parameters - each can have
    its own RDMABuffer, and RDMAAction handles concurrent transfers.

    ### The Cost of Memory Registration

    RDMA memory registration is **expensive**:
    - Pins physical pages (prevents swapping)
    - Creates IOMMU/DMA mappings in the NIC
    - Can take milliseconds for large buffers

    **Don't register on the hot path!** Instead:

    ```python
    # Bad: Register per sync (slow!)
    def sync_weights(self):
        for param in model.parameters():
            buffer = RDMABuffer(param)  # Registration here = slow
            send(buffer)

    # Good: Register once at startup, reuse handles
    def __init__(self):
        # Allocate ONE big contiguous buffer
        total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        self.staging = torch.empty(total_bytes, dtype=torch.uint8)

        # Register ONCE
        self.handle = RDMABuffer(self.staging)

        # Track offsets for each param
        self.offsets = compute_param_offsets(model)

    def sync_weights(self):
        # Just pack data into already-registered buffer
        pack_params_into(self.staging, self.offsets)
        return self.handle  # Same handle, new data
    ```

    This pattern amortizes registration cost across all training iterations.

    ### Two Weight Sync Patterns

    With RDMABuffer as our building block, there are two main approaches:

    | Pattern | How it works | Trade-offs |
    |---------|--------------|------------|
    | **CPU Staging** | GPU → CPU buffer → RDMA → CPU → GPU | One MR, simple, but copies |
    | **Direct GPU** | GPU → RDMA → GPU (GPUDirect) | No copies, but one MR per param |

    **Pattern 1: CPU Staging (Contiguous Buffer)**

    Pack all parameters into one contiguous CPU buffer, register once:

    ```python
    class Trainer(Actor):
        def __init__(self):
            # Calculate total size for all parameters
            total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

            # Allocate ONE contiguous buffer, register ONCE
            self.staging_buffer = torch.empty(total_bytes, dtype=torch.uint8)
            self.handle = RDMABuffer(self.staging_buffer)

            # Track where each param lives in the buffer
            self.param_offsets = compute_offsets(model)

        def pack_weights(self):
            '''Copy all params into contiguous buffer.'''
            for name, param in model.named_parameters():
                offset = self.param_offsets[name]
                self.staging_buffer[offset:offset+size].copy_(param.view(torch.uint8))

        @endpoint
        def get_weight_handle(self) -> RDMABuffer:
            self.pack_weights()
            return self.handle  # Same handle, new data
    ```

    **Pattern 2: Direct GPU MRs**

    Register each GPU parameter directly, no CPU copies:

    ```python
    class Trainer(Actor):
        def __init__(self):
            # Register each param ONCE at startup
            self.handles = {}
            for name, param in model.named_parameters():
                byte_view = param.data.view(torch.uint8).flatten()
                self.handles[name] = RDMABuffer(byte_view)

        @endpoint
        def get_param_handles(self) -> dict[str, RDMABuffer]:
            # Handles are reused - data updates in place
            return self.handles
    ```

    Both patterns amortize MR registration cost across training iterations.
    Let's look at CPU staging in more detail (it's more common in async RL).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. CPU Staging Pattern

    ### GPU-Native RDMA Works!

    First, let's be clear: **GPU-native RDMA works** and is fast:
    - GPUDirect RDMA: NIC reads directly from GPU memory
    - No CPU copy needed (when hardware supports it)
    - Great for synchronous transfers

    ### Why CPU Staging for Async RL?

    The issue isn't bandwidth - it's **timing**:

    ```
    Direct GPU→GPU RDMA:
    ┌─────────────────────────────────────────────────────┐
    │ Generator GPU is mid-inference                       │
    │ ├── layer 1 ──┤ [RDMA arrives, needs sync!]         │
    │               ↓                                      │
    │         cudaDeviceSynchronize()  ← Blocks inference! │
    └─────────────────────────────────────────────────────┘
    ```

    With CPU staging, nothing on the critical path blocks:

    ```
    Trainer GPU ──► CPU staging buffer (RDMA registered)
                          │
                          │ [Sits here, ready anytime]
                          │
                          ▼
    Generator grabs when ready ──► Generator GPU
    ```

    The CPU buffer is a **temporal decoupling point**.
    """)
    return


@app.cell
def _(torch):
    def demonstrate_cpu_staging():
        """Demonstrate the CPU staging pattern."""
        if not torch.cuda.is_available():
            print("CUDA not available - showing conceptual flow")
            return

        # Trainer side: GPU weights → CPU staging buffer (RDMA registered)
        trainer_weights = torch.randn(1000, 1000, device="cuda:0")

        # Pin memory for efficient transfers and RDMA registration
        cpu_staging = torch.empty_like(trainer_weights, device="cpu").pin_memory()

        # D2H: Trainer dumps to CPU (async, non-blocking for trainer)
        cpu_staging.copy_(trainer_weights, non_blocking=True)
        torch.cuda.synchronize()  # Just for timing demo

        print("Trainer: Weights copied to CPU staging buffer (RDMA registered)")
        print(f"  GPU memory: {trainer_weights.device}")
        print(f"  CPU staging: pinned={cpu_staging.is_pinned()}")

        # Generator side: RDMA pulls from trainer's CPU → directly to generator's GPU
        # (In this demo we simulate the RDMA transfer with a local copy)
        generator_gpu_weights = torch.empty_like(cpu_staging, device="cuda:0")
        generator_gpu_weights.copy_(cpu_staging, non_blocking=True)  # Simulates RDMA → GPU
        torch.cuda.synchronize()

        print("Generator: Weights loaded directly to GPU (via RDMA)")
        print(f"  Weights match: {torch.allclose(trainer_weights, generator_gpu_weights)}")

    demonstrate_cpu_staging()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Circular Weight Buffers

    ### The Versioning Problem

    In async RL, trainer updates weights continuously. Generators need to:
    1. **Grab the latest** weights (not stale ones)
    2. **Not block** waiting for updates
    3. **Avoid memory churn** (re-registering RDMA buffers is expensive)

    ### Solution: Circular Buffer

    ```
    Trainer writes:     v0 → v1 → v2 → v3 → v4 → v5 → ...
                         ↓    ↓    ↓
    Buffer slots:      [slot0][slot1][slot2]  (circular, reused)
                         v3    v4    v5

    Generator reads: "Give me latest" → v5
    ```

    Benefits:
    - **Pre-registered RDMA buffers** - no memory registration on hot path
    - **Lock-free reads** - generators always get a consistent snapshot
    - **Bounded memory** - only N versions in flight
    """)
    return


@app.cell
def _(torch):
    from threading import Lock as _Lock
    from typing import Tuple as _Tuple, Optional as _Opt

    class CircularWeightBuffer:
        """Circular buffer for versioned weight storage."""

        def __init__(self, template_tensor: torch.Tensor, n_slots: int = 3):
            self.n_slots = n_slots
            self.slots = [
                torch.empty_like(template_tensor).pin_memory()
                if template_tensor.device.type == "cpu"
                else torch.empty_like(template_tensor, device="cpu").pin_memory()
                for _ in range(n_slots)
            ]
            self.latest_version = 0
            self._lock = _Lock()

            # In production: pre-register all slots with RDMA
            # self.rdma_handles = [RDMABuffer(slot) for slot in self.slots]

        def publish(self, weights: torch.Tensor) -> int:
            """Trainer publishes new weights. Returns version number."""
            with self._lock:
                slot_idx = self.latest_version % self.n_slots
                self.slots[slot_idx].copy_(weights)
                self.latest_version += 1
                return self.latest_version - 1

        def get_latest(self) -> _Tuple[torch.Tensor, int]:
            """Generator gets latest weights. Non-blocking."""
            with self._lock:
                if self.latest_version == 0:
                    raise RuntimeError("No weights published yet")
                slot_idx = (self.latest_version - 1) % self.n_slots
                version = self.latest_version - 1
                # Return a copy to avoid races
                return self.slots[slot_idx].clone(), version

        def get_version(self, version: int) -> _Opt[torch.Tensor]:
            """Get specific version if still available."""
            with self._lock:
                oldest_available = max(0, self.latest_version - self.n_slots)
                if version < oldest_available or version >= self.latest_version:
                    return None
                slot_idx = version % self.n_slots
                return self.slots[slot_idx].clone()

    # Demo
    _template = torch.randn(100, 100)
    weight_buffer = CircularWeightBuffer(_template, n_slots=3)

    # Trainer publishes versions
    for _v in range(5):
        _new_weights = torch.randn(100, 100) * (_v + 1)
        published_v = weight_buffer.publish(_new_weights)
        print(f"Published version {published_v}")

    # Generator grabs latest
    latest_weights, latest_version = weight_buffer.get_latest()
    print(f"\nGenerator got version {latest_version}, weights mean: {latest_weights.mean():.2f}")

    # Try to get old version (might be evicted)
    old_weights = weight_buffer.get_version(1)
    print(f"Version 1 available: {old_weights is not None}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Weight Re-sharding

    ### The Sharding Mismatch Problem

    Trainer and Generator often have **different tensor layouts**:

    | Role | Parallelism | Sharding |
    |------|-------------|----------|
    | Trainer | FSDP (8 GPUs) | `Shard(0)` - rows split across 8 GPUs |
    | Generator | TP (2 GPUs) | `Shard(1)` - columns split across 2 GPUs |

    Direct weight transfer doesn't work - we need **re-sharding**.

    ```
    Trainer (row-sharded):          Generator (column-sharded):
    ┌──────────────────┐            ┌─────────┬─────────┐
    │ GPU 0: rows 0-127│            │ GPU 0   │ GPU 1   │
    ├──────────────────┤     →      │ cols    │ cols    │
    │ GPU 1: rows 128+ │            │ 0-511   │ 512+    │
    └──────────────────┘            └─────────┴─────────┘
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Approach 1: Gather Then Slice

    Simple but inefficient:

    ```
    1. Each receiver gathers ALL sender shards → full tensor
    2. Each receiver slices out its portion
    ```

    **Pros**: Simple, works with any sharding
    **Cons**: 2x redundant data transfer (each receiver gets everything)

    ### Approach 2: Routed/Direct Transfer

    Optimal but complex:

    ```
    1. Pre-compute which sender chunks overlap with which receiver regions
    2. Send only the exact chunks needed
    3. Receivers place chunks directly in correct positions
    ```

    **Pros**: Minimal bandwidth (no redundancy)
    **Cons**: Complex overlap computation
    """)
    return


@app.cell
def _():
    from dataclasses import dataclass
    from typing import List, Tuple

    @dataclass
    class ShardMetadata:
        """Metadata describing a tensor shard."""
        rank: int
        global_shape: Tuple[int, ...]
        offset: Tuple[int, ...]  # Start position in global tensor
        local_shape: Tuple[int, ...]  # Shape of this shard

    def compute_shard_metadata(
        global_shape: Tuple[int, int],
        num_ranks: int,
        shard_dim: int,
    ) -> List[ShardMetadata]:
        """Compute shard metadata for a given sharding."""
        shards = []
        dim_size = global_shape[shard_dim]
        shard_size = dim_size // num_ranks

        for rank in range(num_ranks):
            offset = [0, 0]
            local_shape = list(global_shape)

            offset[shard_dim] = rank * shard_size
            local_shape[shard_dim] = shard_size

            shards.append(ShardMetadata(
                rank=rank,
                global_shape=global_shape,
                offset=tuple(offset),
                local_shape=tuple(local_shape),
            ))

        return shards

    # Demo: Trainer has row-sharding, Generator has column-sharding
    _global_shape = (1024, 1024)

    trainer_shards = compute_shard_metadata(_global_shape, num_ranks=4, shard_dim=0)
    generator_shards = compute_shard_metadata(_global_shape, num_ranks=2, shard_dim=1)

    print("Trainer shards (row-wise, 4 GPUs):")
    for s in trainer_shards:
        print(f"  Rank {s.rank}: offset={s.offset}, shape={s.local_shape}")

    print("\nGenerator shards (column-wise, 2 GPUs):")
    for s in generator_shards:
        print(f"  Rank {s.rank}: offset={s.offset}, shape={s.local_shape}")
    return (
        List,
        ShardMetadata,
        Tuple,
        dataclass,
        generator_shards,
        trainer_shards,
    )


@app.cell
def _(List, ShardMetadata, Tuple, dataclass, generator_shards, trainer_shards):
    @dataclass
    class TransferChunk:
        """A chunk to transfer from sender to receiver."""
        sender_rank: int
        receiver_rank: int
        sender_offset: Tuple[int, int]  # Where to read from sender
        receiver_offset: Tuple[int, int]  # Where to write in receiver
        shape: Tuple[int, int]  # Shape of the chunk

    def compute_overlap(
        sender: ShardMetadata,
        receiver: ShardMetadata,
    ) -> TransferChunk | None:
        """Compute overlap between sender and receiver shards."""
        # Find intersection in global coordinates
        s_start = sender.offset
        s_end = (s_start[0] + sender.local_shape[0], s_start[1] + sender.local_shape[1])

        r_start = receiver.offset
        r_end = (r_start[0] + receiver.local_shape[0], r_start[1] + receiver.local_shape[1])

        # Compute intersection
        inter_start = (max(s_start[0], r_start[0]), max(s_start[1], r_start[1]))
        inter_end = (min(s_end[0], r_end[0]), min(s_end[1], r_end[1]))

        # Check if there's actual overlap
        if inter_start[0] >= inter_end[0] or inter_start[1] >= inter_end[1]:
            return None

        shape = (inter_end[0] - inter_start[0], inter_end[1] - inter_start[1])

        # Convert to local coordinates
        sender_local = (inter_start[0] - s_start[0], inter_start[1] - s_start[1])
        receiver_local = (inter_start[0] - r_start[0], inter_start[1] - r_start[1])

        return TransferChunk(
            sender_rank=sender.rank,
            receiver_rank=receiver.rank,
            sender_offset=sender_local,
            receiver_offset=receiver_local,
            shape=shape,
        )

    def compute_transfer_plan(
        sender_shards: List[ShardMetadata],
        receiver_shards: List[ShardMetadata],
    ) -> List[TransferChunk]:
        """Compute all transfers needed for re-sharding."""
        transfers = []
        for sender in sender_shards:
            for receiver in receiver_shards:
                chunk = compute_overlap(sender, receiver)
                if chunk is not None:
                    transfers.append(chunk)
        return transfers

    # Compute transfer plan
    transfer_plan = compute_transfer_plan(trainer_shards, generator_shards)

    print(f"Transfer plan: {len(transfer_plan)} chunks needed\n")
    for chunk in transfer_plan:
        print(f"Sender {chunk.sender_rank} → Receiver {chunk.receiver_rank}")
        print(f"  Read from sender offset {chunk.sender_offset}, shape {chunk.shape}")
        print(f"  Write to receiver offset {chunk.receiver_offset}")
        print()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### The DTensor Dream (and Reality)

    If trainer and generator both used **DTensor** with the same sharding spec, re-sharding
    would be automatic - the framework handles overlap computation and transfers.

    ```python
    # Ideal world: DTensor handles it
    trainer_weights: DTensor  # Shard(0) across 8 GPUs
    generator_weights: DTensor  # Shard(1) across 2 GPUs

    # Re-sharding is just redistribution
    generator_weights = trainer_weights.redistribute(generator_placement)
    ```

    **In practice, it's harder**:
    - vLLM does its own sharding and weight fusing (not DTensor-compatible)
    - Training frameworks (FSDP, etc.) have different abstractions
    - You often need custom overlap computation like we showed above

    The routed approach (compute overlaps, send only needed chunks) is 2x faster than
    naive gather-then-slice, but requires this manual coordination.

    **For cross-node RDMA transfers**, the key insight remains: pre-compute the transfer
    plan once, then each sync just executes the planned RDMA operations with RDMAAction.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Putting It All Together

    The full async RL weight sync pattern:

    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │                         TRAINER                                  │
    │  1. Train step completes                                        │
    │  2. Copy weights to CPU staging buffer (non-blocking D2H)       │
    │  3. Publish to circular buffer with version tag                 │
    │  4. Continue training (no blocking!)                            │
    └─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │              CIRCULAR BUFFER (CPU, RDMA-registered)             │
    │  [slot 0: v3] [slot 1: v4] [slot 2: v5]                        │
    │                                 ↑ latest                        │
    └─────────────────────────────────────────────────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
    ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
    │   GENERATOR 0   │   │   GENERATOR 1   │   │   GENERATOR 2   │
    │                 │   │                 │   │                 │
    │ After gen done: │   │ After gen done: │   │ After gen done: │
    │ 1. Get latest   │   │ 1. Get latest   │   │ 1. Get latest   │
    │    version      │   │    version      │   │    version      │
    │ 2. RDMA read    │   │ 2. RDMA read    │   │ 2. RDMA read    │
    │    → GPU        │   │    → GPU        │   │    → GPU        │
    │ 3. Re-shard if  │   │ 3. Re-shard if  │   │ 3. Re-shard if  │
    │    needed       │   │    needed       │   │    needed       │
    └─────────────────┘   └─────────────────┘   └─────────────────┘
    ```

    **Key properties:**
    - Trainer never blocks waiting for generators
    - Generators pull directly to GPU when *they're* ready
    - Re-sharding happens locally on each generator
    - Circular buffer bounds memory, reuses RDMA registrations
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Code Pattern

    ```python
    # Trainer side
    class Trainer(Actor):
        def __init__(self):
            self.weight_buffer = CircularWeightBuffer(
                template=self.model.state_dict_tensor(),
                n_slots=3,
            )

        @endpoint
        def get_weight_handle(self) -> Tuple[RDMABuffer, int]:
            '''Return handle to latest weights and version.'''
            slot_idx = (self.weight_buffer.latest_version - 1) % 3
            handle = self.weight_buffer.rdma_handles[slot_idx]
            version = self.weight_buffer.latest_version - 1
            return handle, version

        async def train_loop(self):
            while True:
                loss = self.train_step()
                if self.step % sync_interval == 0:
                    # Non-blocking publish
                    self.weight_buffer.publish(self.model.weights)

    # Generator side
    class Generator(Actor):
        def __init__(self, trainer_ref):
            self.trainer = trainer_ref
            self.current_version = -1

        async def maybe_sync_weights(self):
            handle, version = await self.trainer.get_weight_handle.call_one().get()
            if version > self.current_version:
                # Pull via RDMA directly into GPU memory
                gpu_weights = self.model.weights.view(torch.uint8).flatten()
                await handle.read_into(gpu_weights)
                self.current_version = version

        async def generate_loop(self):
            while True:
                await self.maybe_sync_weights()
                output = self.generate(prompt)
                self.submit_to_buffer(output)
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    ### Key Takeaways

    1. **Bandwidth hierarchy matters**: NVLink (450 GB/s) >> InfiniBand (50-100 GB/s) >> PCIe
       - Know your hardware, optimize for the right interconnect

    2. **Collectives vs P2P**: Collectives are synchronized; RL needs async P2P
       - High variance in generation times makes blocking expensive

    3. **Magic pointer pattern**: Tiny handle over control plane, bulk data over data plane
       - ~100 bytes to describe 10 GB transfer

    4. **CPU staging**: Temporal decoupling for async RL
       - GPU-native RDMA works for sync cases
       - CPU staging ensures nothing blocks on the critical path

    5. **Circular buffers**: Version weights without memory churn
       - Pre-register RDMA buffers, reuse slots
       - Generators grab latest, never stale

    6. **Weight re-sharding**: Different layouts need overlap computation
       - Routed approach is 2x faster than gather
       - Pre-compute transfer plan, minimize redundant data

    ### Next Steps

    See **07_async_rl_e2e.py** for a complete async RL system that uses these patterns.
    """)
    return


if __name__ == "__main__":
    app.run()
