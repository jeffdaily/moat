# cudaKDTree notes

Upstream: ingowald/cudaKDTree (master, HEAD 0e174fe). Header-only CUDA k-d tree
(build + fcp/knn queries) plus sample/test `.cu` executables. Apache-2.0.
Strategy A port (pure CMake, colmap model): one `cukd/cuda_to_hip.h` compat
header, `.cu` marked `LANGUAGE HIP`, gated on `option(USE_HIP)`. CUDA build left
unchanged; every HIP divergence guarded by `USE_HIP`/`__HIP_PLATFORM_AMD__`.

## CUDA surface
- Host runtime: cudaMalloc/Free, cudaMallocManaged, cudaMallocAsync/FreeAsync,
  cudaMemcpy(+Async)/cudaMemsetAsync + memcpy-kind enums, cudaStream_t,
  cudaStream/DeviceSynchronize, cudaSetDevice, cudaError_t/Success,
  cudaGet{LastError,ErrorString}, cudaGetSymbolAddress (stats path). All aliased
  in the compat header (CUKD_CUDA_CALL(X) hides cuda##X behind macros, so symbols
  like cudaMemsetAsync/cudaMemcpyAsync/cudaSetDevice are easy to miss -- expand
  the macro to enumerate them).
- Thrust: builder_thrust.h (the DEFAULT builder) uses thrust::sort/device_vector/
  zip_iterator/etc. rocThrust (/opt/rocm/include/thrust) is a drop-in: same API,
  same header paths -- compiles unchanged, no source swap.
- CUB: spatial-kdtree.h uses cub::DeviceRadixSort::SortKeys. hipCUB mirrors the
  API; compat header does `#define cub hipcub` + `CUKD_CUB_INCLUDE`.
- Device intrinsics: __clzll, __float_as_int/__int_as_float/__uint_as_float/
  __float_as_uint, __int2float_rz, atomicAdd/atomicCAS/atomicMin/atomicMax. No
  __shfl/__ballot/__popc/__any/__activemask anywhere. No textures/surfaces.

## Warp size (the MOAT high-risk class): NO hazard
gfx90a is wave64. The only fixed-32 code is cukd/cubit/* (a shared-memory bitonic
sorter, compiled because builder.h always includes builder_bitonic.h, though the
DEFAULT builder is thrust). Its `threadIdx.x & -32`, `^(32-1)`, `l+32` are
bitonic-merge INDEX arithmetic over a __shared__ array with __syncthreads() at
every stage boundary >=32 -- not lane masks, no warp intrinsics. The sub-32
down-sweeps (seq 4/8/16) omit __syncthreads() and rely on warp-synchronous
execution of a <=32-thread sub-range; a 64-wide wavefront runs all 64 lanes in
lockstep, so a <=32-lane subgroup is a fortiori lockstep -- the assumption
strengthens, it does not break (the dangerous direction is assuming MORE than
warpSize lockstep, which never happens here). Query kernels (fcp.h, knn.h) are
per-thread traversals with per-thread candidate lists; no cross-lane comms.
Conclusion: correctness is wave-size independent. Verified on-GPU.

## Fault classes encountered and fixes
1. Device-attribute macro keyed on __CUDA_ARCH__. `__both__` (= __host__
   __device__) was defined via `#if defined(__CUDA_ARCH__)`, which is undefined
   during HIP device compilation (HIP uses __HIP_DEVICE_COMPILE__), so __both__
   functions became host-only -> "reference to __host__ function in __host__
   __device__ function". Fix: on HIP make __both__ always carry both attributes
   (common.h, cubit/common.h); introduce CUKD_DEVICE_CODE (compat header:
   __HIP_DEVICE_COMPILE__ on HIP, __CUDA_ARCH__ on CUDA) and convert every
   device-codepath `#ifdef __CUDA_ARCH__` (intrinsic-vs-host-fallback) to it.
2. clang two-phase lookup: unqualified `decode_dist2(...)` call into a dependent
   base (CandidateList<k>) -> "explicit qualification required". Fix: `this->`
   in HeapCandidateList/FixedCandidateList::maxRadius2 (knn.h). nvcc/MSVC lenient.
3. Function-template explicit specialization signature must match the primary on
   HIP/clang: `as_float_rz<int>` was `__device__` while the primary is __both__
   -> "no function template matches". Fix: make the (device-only-body)
   specialization __both__ (cukd-math.h).
4. cudaMallocAsync gated on `CUDART_VERSION >= 11020` (undefined on HIP). Fix:
   `|| defined(USE_HIP)` so HIP keeps the async allocator (hipMallocAsync).

## CDNA2/ROCm semantic bugs in the SPATIAL k-d tree (real, GPU-only)
The regular (left-balanced) tree -- the headline feature, default thrust builder,
all 3 traversals -- worked after the fault-class fixes. The SPATIAL k-d tree
(spatial-kdtree.h, used by the *-spatial-* tests) had three additional bugs, all
ROCm/CDNA-specific, found by validating its GPU queries against CPU brute force:

A. **Integer atomicMin/atomicMax are dropped on COARSE-GRAINED memory (gfx90a).**
   `::atomicMin/atomicMax(int*, int)` on a default hipMallocManaged buffer
   silently no-op (proven by micro-test: device memory PASS, managed memory leaves
   the value untouched). The spatial builder accumulates the point bounds box with
   int atomicMin/Max on encoded coords (atomic_grow), so the root/all bounds stayed
   empty (+/-FLT_MAX) -> every split plane = -FLT_MAX -> degenerate tree -> kNN
   reads OOB and faults. atomicAdd and atomicCAS are NOT affected (the regular
   builder's float bounds use atomicCAS, which is why it worked). Fix: emulate int
   min/max with an atomicCAS loop on HIP (atomicMinI32/atomicMaxI32 in
   spatial-kdtree.h); CUDA keeps the native intrinsics.
B. **Same for the leaf-offset unsigned atomicMin** (writePrimsAndLeafOffsets:
   `::atomicMin(&doneNode.offset, ...)`, sentinel (uint32_t)-1). Needs an UNSIGNED
   CAS min (atomicMinU32) -- a signed reinterpret would treat the sentinel as -1
   and never update.
C. **hipCUB DeviceRadixSort::SortKeys with a nonzero begin_bit is broken.** The
   build sorts the {nodeID:primID} 64-bit keys over bits [32,64) (nodeID only) to
   group prims per leaf. On ROCm (gfx90a) this does NOT sort by the high 32 bits
   (direct test: full-64-bit sort PASS, [32,64) sort BROKEN, 9974/10000
   misordered). Result: leaf offsets interleave (overlapping [off,off+count)
   ranges, 12301 uncovered + 4819 multi-hit prim slots) -> queries miss the true
   nearest (too-large distances). Fix: on HIP sort the full 64-bit key (begin_bit
   0); nodeID still dominates and the low primID bits just make per-leaf order
   deterministic. CUDA keeps [32,64).

After A+B+C the spatial tree partitions prims exactly (0 uncovered/0 multi-hit),
a host exhaustive tree traversal matches brute force, and device spatial fcp/knn
match both the regular tree and brute force to 1e-7.

## Build
    cmake -S . -B build-hip -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
      -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
      -DBUILD_ALL_TESTS=ON -DCUKD_ENABLE_STATS=ON
    cmake --build build-hip -j $(nproc)
ROCm 7.2.1, hipcc clang 22, gfx90a (MI250X). build-hip/ is gitignored via
src/.git/info/exclude.

Note: BUILD_ALL_TESTS passes -DCUKD_ENABLE_STATS=<0|1>. With stats=0 the macro
`defined(CUKD_ENABLE_STATS)` is still TRUE (the value is 0) so CUKD_STATS expands
and references cukd::g_traversalStats, which is declared only under `#if
CUKD_ENABLE_STATS` (value) -> undeclared-identifier. This latent upstream
contradiction (only reachable via BUILD_ALL_TESTS, which upstream CI does not
build) is sidestepped by configuring with CUKD_ENABLE_STATS=ON; not a port issue.

## Validation (gfx90a, HIP_VISIBLE_DEVICES pinned to a free GPU)
- Regular tree, in-tree `-v` brute-force verifier (verifyFCP = CPU nearest,
  verifyKNN = CPU std::priority_queue, rel err <= 1e-6, + recursive checkTree):
  FCP 17/17 configs PASS, kNN 51/51 configs PASS across dims {2,3,4,8} x
  {stackBased,stackFree,cct} x {regular,explicit-dim} x k in {4,8,50}, uniform and
  --clustered.
- Spatial tree: in-tree `-v` cannot run (its host checkTree dereferences
  tree.nodes, which the default async/device-pool allocator -- and CUDA's
  cudaMallocAsync alike -- does not make host-accessible; a test-harness
  assumption, not a port bug). Validated instead with standalone harnesses
  (agent_space): spatial fcp/knn vs the regular tree AND vs brute force =>
  mismatches=0, maxRel ~1e-7, for both the managed and the default resource.
- CTest builders: 14/15 pass (empty/simple/same-result-per-builder/issue5/
  compile). The 1 failure, cukdTestBuildersSameResult, is host-vs-device only:
  the 3 DEVICE builders (thrust/bitonic/inplace) agree; only the HOST builder
  hash differs, on the explicit-dim (+payload) case, i.e. a CPU-vs-GPU
  tie-break in widest-split-dim selection (multiple valid balanced k-d trees).
  builder_host.h is untouched by the port and this would differ identically on
  CUDA; the host builder's own unit tests pass. Pre-existing, not a port
  regression.
- Determinism: repeated query runs give identical CHECKSUM.
