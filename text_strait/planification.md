# Exhaustive Architectural Specification: Bare-Metal C Transformer Engine & Python FFI Control Plane

## 1. System Philosophy & Architectural Boundaries

The system enforces a strict Control Plane vs. Data Plane separation:

-   **The Python Control Plane (Cold Path):**
    -   **Responsibilities:** Command-line parsing, JSON/YAML
        hyperparameter configuration, vocabulary/tokenizer string
        processing (BPE/WordPiece lookup tables), file path resolution,
        logging, and user-facing streaming output.
    -   **Strict Prohibition:** Python is forbidden from allocating,
        iterating over, or transforming floating-point weight or
        activation arrays during inference. Python never constructs
        native list objects of floats, never performs element-wise math,
        and never manages raw C pointers directly.
-   **The C Execution Engine (Hot Path):**
    -   **Responsibilities:** Physical RAM allocation, virtual memory
        alignment, CPU thread lifecycle management, atomic barrier
        synchronization, SIMD register packing, transcendental
        approximations, KV-cache management, and logit sampling.
    -   **Standalone Nature:** Compiled as a pure C11/C17 shared library
        (`libengine.so`) with zero third-party dependencies (linking
        only against libc, libm, and libpthread). Because the C core
        knows nothing about `PyObject*` or Python internals, it can
        simultaneously be linked into a standalone C CLI, Rust binary,
        or Go service without modification.

## 2. Mathematical Foundations & Data Layout Strategy

Every Transformer layer is composed of four fundamental linear and
non-linear primitives:

1.  **Inner Products (Dot Products):** Measuring directional alignment
    and magnitude projection ((`\mathbf{a}`{=tex}
    `\cdot `{=tex}`\mathbf{b}`{=tex} = `\sum `{=tex}a_i b_i)), bounded
    by the Cauchy--Schwarz inequality
    ((-`\Vert`{=tex}`\mathbf{a}`{=tex}`\Vert`{=tex}`\Vert`{=tex}`\mathbf{b}`{=tex}`\Vert `{=tex}`\le `{=tex}`\mathbf{a}`{=tex}
    `\cdot `{=tex}`\mathbf{b}`{=tex}
    `\le `{=tex}+`\Vert`{=tex}`\mathbf{a}`{=tex}`\Vert`{=tex}`\Vert`{=tex}`\mathbf{b}`{=tex}`\Vert`{=tex})).
2.  **Batched Dot Products (GEMV & GEMM):**
    -   **Autoregressive Decoding (1 token at a time):** Matrix-Vector
        Multiplication (GEMV), where a (M, K) weight matrix multiplies a
        (K, 1) activation vector to produce an (M, 1) output vector.
        This operation is memory-bandwidth bound.
    -   **Prompt Prefill (T tokens simultaneously):** Matrix-Matrix
        Multiplication (GEMM), where a (T, K) activation matrix
        multiplies a (K, M) weight matrix to produce a (T, M) output
        matrix. This operation is compute bound.
3.  **Outer Products & Low-Rank Expansions:** Expanding two vectors
    (N, 1) × (1, M) → (N, M) for gradient accumulation, LoRA adapter
    reconstruction, or rank-1 updates.
4.  **Row-Wise Reductions & Element-Wise Non-Linearities:** RMSNorm /
    LayerNorm, Softmax, Rotary Position Embeddings (RoPE), and gated
    activations (SwiGLU / GeGLU).

### Physical Memory Layout Rules

-   **Row-Major Contiguous Storage:** All multi-dimensional tensors are
    flattened into 1D contiguous float32 arrays in row-major order.
    Element (r, c) of a matrix with cols columns sits at linear index
    `r * stride_row + c * stride_col`.
-   **Zero-Copy Metadata Views (Strided Descriptors):**
    -   A tensor is never a raw pointer; internally, it is a descriptor
        containing `data_ptr`, `rows`, `cols`, `stride_row`,
        `stride_col`, and flags (read-only, view, transposed).
    -   **1D Vector Transposition:** Flipping a vector between row shape
        (1, N) and column shape (N, 1) executes in O(1) time with zero
        memory operations---only the descriptor's `rows` and `cols`
        fields swap.
    -   **Head Slicing:** Splitting a combined ((1,
        D\_{`\text{model}`{=tex}})) projection vector into H individual
        attention heads of size ((1, d\_{`\text{head}`{=tex}})) creates
        H lightweight view descriptors pointing at staggered offsets
        (`base_ptr + h * d_head`) inside the exact same buffer.
-   **Pre-Transposed Weight Ingestion:** Standard matrix multiplication
    multiplies rows of A by columns of B. Reading down columns of B in
    row-major memory causes massive stride jumps that defeat CPU cache
    lines. Therefore, all weight matrices ((W_Q, W_K, W_V, W_O,
    W\_{`\text{gate}`{=tex}}, W\_{`\text{up}`{=tex}},
    W\_{`\text{down}`{=tex}})) are stored pre-transposed in memory as
    ((N\_{`\text{out}`{=tex}}, N\_{`\text{in}`{=tex}})). Consequently,
    every dot product reads both the activation vector and the weight
    matrix row sequentially from ascending, contiguous memory addresses.

## 3. Memory Subsystem: Cache-Aligned Arena & Lifetime Management

Dynamic heap allocation (`malloc`/`free`) is strictly prohibited during
model execution due to non-deterministic latency, heap metadata
overhead, memory fragmentation, and Virtual Memory Page / TLB
(Translation Lookaside Buffer) thrashing.

### Three-Pool Arena Topology

At startup, the engine requests large, contiguous virtual memory blocks
aligned to 64-byte boundaries (matching the exact width of an x86
L1/L2/L3 cache line and a 512-bit AVX-512 ZMM register):

1.  **Arena A --- Static Weights Arena (Immutable):**
    -   **Allocation:** Allocated once via
        `aligned_alloc(64, total_weight_bytes)` or mapped directly from
        disk via `mmap(..., MAP_PRIVATE)` with
        `madvise(..., MADV_WILLNEED | MADV_HUGEPAGE)` to leverage 2 MB
        Linux Transparent Huge Pages (THP), drastically reducing page
        table walks.
    -   **Bump-Pointer Sub-Allocation:** Every weight tensor carved from
        this arena has its starting byte offset rounded up to the
        nearest multiple of 64 bytes (`(offset + 63) & ~63`). Even if a
        tensor has an odd number of floats, padding bytes are inserted
        after it so the next tensor always starts on a fresh 64-byte
        cache line.
2.  **Arena B --- Key-Value (KV) Cache Arena (Persistent Append-Only /
    Ring Buffer):**
    -   **Layout:** Pre-allocated at startup to hold
        `[num_layers][max_seq_len][num_kv_heads][head_dim]` for both
        Keys and Values.
    -   **Contiguous Token Ordering:** Laid out so that for a given
        layer and attention head, historical Key and Value vectors sit
        contiguously across sequence positions 0 .. pos, allowing the
        attention dot-product loop ((Q `\cdot `{=tex}K\^T)) to stream
        through past keys at maximum hardware prefetch speed.
3.  **Arena C --- Ephemeral Activation Scratchpad (Ping-Pong + Stack
    Rewind):**
    -   **Ping-Pong Residual Buffers:** Two permanent 64-byte aligned
        slots (`Slot_0` and `Slot_1`, each of size
        `D_model * sizeof(float)`) reside at offset 0 of the scratchpad.
        Layer L reads x from `Slot_0`, computes Attention into temporary
        buffers, adds the residual into `Slot_1`, computes the FFN/MLP,
        and writes the final layer output back into `Slot_0`.
    -   **Scoped Stack Frames:** Immediately above `Slot_0` and
        `Slot_1`, the scratchpad acts as a bump-pointer stack. Before
        entering Attention, the engine saves
        `watermark = arena->offset`. Temporary buffers (Q, K_curr,
        V_curr, attn_scores, attn_out) are bumped from this watermark.
        Immediately upon exiting Attention, `arena->offset = watermark`
        executes in a single CPU instruction.
    -   **Cache Pinning Effect:** Because every Transformer layer
        executes the exact same sequence of allocations and rewinds,
        Layer 1 through Layer N repeatedly read and write the exact same
        physical RAM addresses. Those addresses stay permanently hot in
        the CPU's L2/L3 cache throughout the entire forward pass.

## 4. Two-Tier Parallel Execution Architecture

### Tier 1: Inter-Core Thread Pool & Lock-Free Synchronization

-   **Persistent Worker Pool:** At `engine_create()`, T - 1 POSIX worker
    threads (`pthread_create`) are spawned and pinned (optionally via
    `pthread_setaffinity_np`) to physical CPU cores, avoiding
    hyperthread sibling contention on heavy FMA units. The calling
    thread serves as Worker 0.
-   **Dispatch Descriptor:** The engine maintains a single
    cache-line-padded `JobDescriptor` struct containing a function
    pointer (`kernel_fn`), pointers to input/output tensor descriptors,
    total work items N, and an atomic generation counter
    (`atomic_uint_fast64_t job_gen`).
-   **Zero-Contention Work Partitioning:**
    -   Work is partitioned along the output dimension (e.g., output
        rows in GEMV/GEMM, or independent attention heads in Multi-Head
        Attention).
    -   Chunk boundaries are strictly quantized to multiples of 16
        floats (64 bytes). This guarantees that Thread i's last written
        float and Thread i+1's first written float never share the same
        64-byte cache line---completely eliminating False Sharing
        (cache-line ping-ponging across core L1 caches).
    -   Because all inputs are read-only and all output slices are
        mathematically disjoint, zero mutexes, semaphores, or atomic
        read-modify-write operations occur inside the math kernels.
-   **Hybrid Spin-Wait / Futex Barrier:**
    -   Workers increment an `atomic_uint_fast32_t workers_done` counter
        with `memory_order_release` upon finishing their slice.
    -   For low-latency token generation, waiting threads spin briefly
        using `_mm_pause()` (reducing power consumption and memory-order
        violation pipeline flushes) before falling back to Linux futex
        sleep if idle between prompts.

### Tier 2: Intra-Core SIMD Vectorization (`<immintrin.h>`)

Inside every worker thread, inner loops operate on wide SIMD registers
(256-bit YMM registers on AVX2 holding 8 float32 lanes, or 512-bit ZMM
registers on AVX-512 holding 16 float32 lanes):

-   **Unrolled FMA Dot-Product Accumulators:**
    -   A single `VFMADD231PS` instruction has a latency of \~4--5
        cycles but a throughput of 2 instructions per cycle on modern
        x86 cores. If a loop accumulates into a single YMM register, the
        CPU stalls waiting on register dependencies.
    -   Therefore, inner dot-product loops use 4 independent YMM
        accumulator registers (`acc0`, `acc1`, `acc2`, `acc3`),
        processing 32 floats per loop iteration (4 × 8 floats). This
        saturates both FMA execution ports simultaneously and hides L1
        cache load latency.
-   **Horizontal Vector Reduction:**
    -   Once the main loop finishes, `acc0..acc3` are summed vertically
        into one YMM register (`_mm256_add_ps`), split into two 128-bit
        XMM halves (`_mm256_extractf128_ps` + `_mm256_castps256_ps128`),
        and horizontally collapsed via `_mm_hadd_ps` (or
        shuffle-and-add) into a single scalar float.
-   **Scalar Tail Epilogue:**
    -   Any remaining elements where `length % 32 != 0` are cleanly
        processed by an 8-wide SIMD cleanup loop followed by a standard
        1-wide scalar `for` loop (`i < length`), ensuring arbitrary
        tensor dimensions work without requiring zero-padding.
-   **Vectorized Transcendentals (`expf` for Softmax & SiLU):**
    -   Standard libm `expf()` cannot be auto-vectorized across SIMD
        lanes cleanly. The kernel implements a fast range-reduction +
        degree-5 minimax/Estrin polynomial approximation directly in
        AVX2 registers, computing 8 exponentials simultaneously with \<
        1 ULP relative error.

## 5. Numerical Stability & Transformer Kernel Specifications

Every C kernel enforces strict floating-point stability rules to prevent
NaN or Inf cascades in deep networks:

1.  **RMSNorm (`engine_rmsnorm`):**
    -   Computes sum of squares (`\sum `{=tex}x_i\^2) using 4-way
        unrolled AVX2 FMA accumulators, divides by
        (D\_{`\text{model}`{=tex}}), adds ε (e.g., 1e-5 or 1e-6),
        computes inverse square root via `1.0f / sqrtf(mean_sq + eps)`
        (or `_mm256_rsqrt_ps` refined with one Newton-Raphson step), and
        multiplies element-wise by learned scale vector γ.
2.  **Rotary Position Embeddings (`engine_rope`):**
    -   Applies complex-plane 2D rotations to pairs of coordinates
        ((x\_{2i}, x\_{2i+1})) in Q and K using precomputed cosine/sine
        frequency tables indexed by sequence position `pos`, avoiding
        costly `sinf()`/`cosf()` calls during inference.
3.  **Numerically Stable Softmax (`engine_softmax`):**
    -   **Pass 1 (Max Reduction):** Scans the logit/attention-score
        vector with `_mm256_max_ps` to find (m = `\max`{=tex}(x_i)).
    -   **Pass 2 (Exponentiation & Sum):** Subtracts m from every
        element ((x_i - m `\le 0`{=tex}), guaranteeing exp never
        overflows `FLT_MAX`), computes vectorized `exp(x_i - m)`, writes
        to output, and accumulates the scalar sum S.
    -   **Pass 3 (Normalization):** Multiplies all elements by the
        precomputed reciprocal `1.0f / S` using `_mm256_mul_ps`.
4.  **SwiGLU Feed-Forward Network:**
    -   Computes `gate = W_gate * x` and `up = W_up * x`, applies
        `SiLU(gate) = gate ⋅ σ(gate)` in SIMD registers, performs
        element-wise Hadamard multiplication `gate * up`, and projects
        back via `W_down`.

## 6. Entropy-Seeded PRNG & Zero-Copy File Ingestion

To guarantee Python never builds float arrays in user space:

-   **Lock-Free Per-Thread PRNG:** Uses Xorshift64\* or PCG32 state
    structs. When `seed == 0` is requested, the C engine invokes the
    Linux `getrandom()` syscall (`<sys/random.h>`) to pull 64 bits of
    cryptographic kernel entropy; when `seed != 0`, it runs
    deterministically for unit-test reproducibility.
-   **Gaussian Initialization (Box-Muller Transform):** Converts pairs
    of uniform \[0, 1) floats into standard normal
    (`\mathcal{N}`{=tex}(0, `\sigma`{=tex}\^2)) floats directly inside
    the C weights arena for Xavier/Kaiming initialization.
-   **Direct Binary Weight Loader:** Python parses model header metadata
    (tensor names, shapes, and byte offsets from formats like GGUF or
    Safetensors) and passes
    `(file_path, byte_offset, byte_length, target_tensor_id)` to C. C
    opens the file descriptor and transfers bytes directly into the
    64-byte aligned arena via `pread()` or `memcpy()` from an `mmap`
    region.

## 7. C ABI & Python FFI Contract (`ctypes` / `cffi`)

The binary interface (`engine.h`) obeys five strict ABI stability and
safety rules:

1.  **Opaque Context Pointer (`EngineCtx*`):** Python sees only an
    opaque `c_void_p`. All internal structs, thread handles, and arena
    pointers remain completely hidden inside `engine.c`, preventing ABI
    breakage when internal data structures evolve.
2.  **Validated Integer Handles (`TensorID`):** Tensors are referenced
    across the boundary solely as `uint32_t` indices into an internal C
    descriptor registry (`MAX_TENSORS`). Every C function validates
    `id < num_tensors`, checks shape compatibility (e.g.,
    `a.cols == b.rows` for matmul), and returns an explicit
    `EngineStatus` error code (`ENGINE_OK`, `ERR_OOM`,
    `ERR_SHAPE_MISMATCH`, `ERR_INVALID_ID`) rather than crashing with a
    segmentation fault.
3.  **Explicit C Type Widths:** Only fixed-width standard C types
    (`int32_t`, `uint32_t`, `uint64_t`, `size_t`, `float`) cross the
    boundary---never platform-ambiguous `long` or `int` where sizes can
    vary across OS ABIs.
4.  **Strict `argtypes` and `restype` Enforcement:** The Python wrapper
    module defines `argtypes` and `restype` for every exported symbol at
    import time, preventing 64-bit pointer truncation bugs on Linux
    x86-64.
5.  **Deterministic Resource Cleanup:** The Python `Engine` class
    implements `__enter__`, `__exit__`, and `__del__` hooks that invoke
    `engine_destroy(ctx)` to cleanly signal worker threads to exit,
    `pthread_join` all threads, and release arena memory back to the OS.

## 8. Phased Engineering & Verification Plan

  ----------------------------------------------------------------------------------------------------------
  Phase                   Milestone                  Verification & Acceptance Criteria
  ----------------------- -------------------------- -------------------------------------------------------
  **Phase 1: Core         Implement 64-byte aligned  Compile with
  Infrastructure**        Arena allocator,           `-Wall -Wextra -Werror -fsanitize=address,undefined`.
                          scratchpad watermark       Verify 64-byte pointer alignment
                          save/restore, TensorID     (`(uintptr_t)ptr % 64 == 0`) on every allocation and
                          registry, EngineStatus     zero leaks under Valgrind/ASan.
                          error codes, and           
                          `getrandom()`-seeded       
                          Xorshift/Box-Muller RNG.   

  **Phase 2: Scalar       Implement pure C scalar    Run side-by-side numerical parity tests comparing C
  Reference Kernels &     loops for dot, gemv, gemm, kernel outputs against pure Python/reference math
  Python FFI**            rmsnorm, softmax, rope,    across odd and even dimensions; assert maximum absolute
                          and swiglu. Expose via     error \< 1e-5.
                          `engine.h` and bind in     
                          Python via `ctypes`.       

  **Phase 3: AVX2/FMA     Replace scalar inner loops Re-run Phase 2's numerical parity test suite on
  SIMD Optimization**     with 4-way unrolled        non-aligned sizes (e.g., N = 1, 7, 8, 9, 31, 32, 33,
                          `<immintrin.h>` AVX2/FMA   4097). Measure single-core GFLOPS and bandwidth
                          kernels, horizontal        utilization.
                          reductions, vectorized     
                          exp, and scalar tail       
                          cleanup loops.             

  **Phase 4: Persistent   Implement persistent       Stress-test under ThreadSanitizer (`-fsanitize=thread`)
  Thread Pool &           pthread worker pool,       for 10,000+ iterations to prove zero data races or
  Barriers**              64-byte cache-line         deadlocks. Benchmark scaling across 1, 2, 4, 8+ cores.
                          chunking, and atomic       
                          generation barriers. Wire  
                          parallel dispatch into     
                          gemv, gemm, and Multi-Head 
                          Attention.                 

  **Phase 5: End-to-End C Implement the full         Verify end-to-end token generation matches reference
  Forward Pass & KV       multi-layer Transformer    model logits across multi-turn prompts; measure
  Cache**                 block loop, KV-cache       tokens/second throughput and peak RSS memory footprint.
                          updates, and               
                          temperature/top-k/argmax   
                          sampling inside a single   
                          `engine_forward_token()` C 
                          entry point.               
  ----------------------------------------------------------------------------------------------------------
