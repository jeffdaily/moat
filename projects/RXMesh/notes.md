# RXMesh notes (linux-gfx90a lead port)

Upstream: owensgroup/RXMesh @ 30a4137. Strategy A (pure CMake; compat header +
`.cu` LANGUAGE HIP + USE_HIP option + arch from CMAKE_HIP_ARCHITECTURES).

## Build

Configure + build (GPU 0, gfx90a, ROCm 7.2.1):

```
HIP_VISIBLE_DEVICES=0 cmake -S projects/RXMesh/src -B projects/RXMesh/build -G Ninja \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DRX_USE_POLYSCOPE=OFF -DRX_BUILD_APPS=OFF -DRX_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release
HIP_VISIBLE_DEVICES=0 ninja -C projects/RXMesh/build RXMesh RXMesh_test
```

- `RX_USE_POLYSCOPE=OFF`: viz needs a display; all polyscope code is USE_POLYSCOPE-guarded.
- `RX_BUILD_APPS=OFF`: apps pull OpenMesh (CPU reference, fetched from a flaky
  non-standard-port git server) and the cusolverSp solver path; not needed to
  validate the core.
- Relocatable device code (`-fgpu-rdc` / HIP_SEPARABLE_COMPILATION) is REQUIRED:
  RXMesh declares `__device__` members in headers and defines them in separate
  `.cu` TUs, so the device linker must resolve them across TUs (the CUDA build
  uses CUDA_SEPARABLE_COMPILATION / -rdc=true for the same reason).

## Compat / build-system layer

- `include/rxmesh/util/cuda_to_hip.h`: runtime + cooperative-groups + cub alias,
  `__grid_constant__` -> empty on HIP, data-type enums. Includes <cstdlib>/<cstring>
  before hip_runtime.
- `include/rxmesh/util/cuda_to_hip_math.h`: hipBLAS/hipSOLVER/hipSPARSE handle +
  status + L1-BLAS + generic-sparse aliases + complex types. Included by macros.h.
- `include/rxmesh/hip_compat/`: HIP-only redirect headers (on the include path
  BEFORE the system path only under USE_HIP) so the CUDA include spellings
  (<cuda_runtime.h>, <cub/...>, <cooperative_groups.h>, <cuComplex.h>,
  <cublas_v2.h>, <cusparse.h>, <cuda_profiler_api.h>, ...) resolve to their HIP
  equivalents without editing every include site.
- CMake: root gates `project(... HIP)` vs CUDA on USE_HIP; RXMeshTarget marks the
  lib `.cu` LANGUAGE HIP, drops the unused cuBQL link + nvcc-only flags, swaps
  CUDA::cusparse/cusolver -> roc::hip*; RXMeshApp does the same for tests/apps.
  ThirdParty skips OpenMesh (apps-only) and cuBQL (CUDA-only) on HIP.

## Fault classes hit (the real porting work)

- WARP SIZE / wave64 (gfx90a is CDNA2, 64-lane): `WARP_SIZE` is now 64 on
  `__GFX9__`. The participation/deletion bitmasks are 32-bit words (one per 32
  mesh elements); a 64-lane `__ballot` spans TWO words, so a single uint32 ballot
  truncates lanes 32-63 and never writes the 2nd word. Added
  `ballot_sub_warp_32()` (kernels/util.cuh): 64-bit `__ballot`, each 32-lane
  sub-group extracts its own 32 bits and its lane-0 writes its word. Applied at
  `warp_update_mask` (dynamic deletion, dynamic_util.cuh) and `query_dispatcher`
  (participation bitmask).
- INLINE PTX: `lane_id()`/`warp_id()`/`dynamic_smem_size()`/`total_smem_size()`
  used `mov.u32 %0,%laneid|%warpid|%dynamic_smem_size|%total_smem_size` (CUDA-only
  asm). lane_id -> `__lane_id()`; warp_id -> `threadIdx.x/warpSize` (call sites
  only partition block-local work); the smem-size queries return 0xFFFFFFFF on HIP
  (only used by a debug over-allocation guard).
- `__CUDA_ARCH__` device/host split: NOT defined during HIP device compilation
  (HIP uses `__HIP_DEVICE_COMPILE__`). The pervasive `#ifdef __CUDA_ARCH__`
  device-vs-host data-pointer selection (e.g. Attribute::size/get_patch_info
  choosing m_d_* vs m_h_*) silently took the HOST branch on the GPU and
  dereferenced host pointers -> memory access fault in the query kernels. Rewrote
  every `#ifdef __CUDA_ARCH__` -> `#if defined(__CUDA_ARCH__) ||
  defined(__HIP_DEVICE_COMPILE__)` (and the `#ifndef` form) across include/. THE
  key correctness fix for the query path.
- `__CUDACC__` guards: HIP defines `__HIPCC__`, not `__CUDACC__`. lp_hashtable.h
  gated its `#include kernels/util.cuh` (and the in-shared-memory device path) on
  `#if defined(__CUDACC__)`, so on HIP `atomic_read` was undeclared and the device
  load path was dropped. Added `|| defined(__HIPCC__)` (lp_hashtable.h, attribute.h).
- clang strictness (vs nvcc/MSVC):
  - explicit instantiations must carry the same `__host__`/`__device__` target
    attributes as the primary (query.cu prologue/get_iterator -> __device__;
    patch_info.cu / attribute.cu glm/eigen -> __host__ __device__).
  - dependent member-template call needs `.template`: `query.dispatch<Op>` ->
    `query.template dispatch<Op>` (24 lib + 3 test sites).
  - dependent type as a ctor call needs `typename`: `HandleT::LocalT(x)` ->
    `typename HandleT::LocalT(x)` (cavity_manager_impl / context.h asserts).
  - a static constexpr member used as a non-type template arg must be qualified
    (`Attribute<T,H>::m_block_size`, not `attr.m_block_size`) -- accessing it via
    an instance is not a constant expression to clang (reduce_handle).
  - `.template` may only precede a template; `new_patch.template is_deleted(...)`
    (is_deleted is a plain overload) -> drop `.template`.
- Reference-member-to-temporary: `Iterator`'s `const LPHashTable&` member bound to
  a temporary `LPHashTable()` in the degenerate ctors (UB nvcc tolerates, clang
  rejects). Store it by value on HIP (it is a small handle, never reborrowed).
- `__shared__` of a non-trivially-constructible type is rejected by clang:
  `__shared__ LPPair s_stash[...]` (LPPair has a value-init ctor). Added
  RXMESH_SHARED_LPPAIR_ARRAY (lp_pair.cuh): on HIP back it with uninitialized
  aligned shared bytes (callers fill_n before use); CUDA path unchanged.
- `__nv_is_extended_*_lambda_closure_type` builtin (IS_HD_LAMBDA/IS_D_LAMBDA): no
  HIP/clang equivalent and no SFINAE-detectable substitute, so RXMeshStatic::
  for_each_*(location, lambda) cannot gate its device kernel launch on the closure
  type. On HIP the for_each(DEVICE, lambda) device branch is disabled (the lib
  itself only calls for_each(HOST, ...)); device iteration goes through the query
  path (Query::dispatch), which is what RXMesh's kernels actually use. Tests that
  exercise for_each(DEVICE,...) (test_attribute, test_indices) are excluded from
  the HIP test target.
- `cooperative_groups::memcpy_async`/`wait`: not in HIP cooperative groups.
  loader.cuh falls back to a synchronous block-strided copy on HIP; the deferred
  `cg::wait(block)` drains map to `block.sync()` (wait_for_copy).
- sm70 `#error` (ShmemMutex/ShmemMutexArray) and `#if __CUDA_ARCH__ >= 700`
  half-atomic histogram are CUDA-only ISA checks; gated to not fire / take the
  legacy uint32 path on HIP.
- hipProfilerStart/Stop return hipErrorNotSupported without a profiler tool; the
  CUDA_ERROR-wrapped profiler calls in the Queries benchmark were fatal (and the
  resulting exit() then hung HIP teardown) -> guarded out on HIP.

## Library scope

- cuBQL: linked in CMake but never #included in any source -> dropped on HIP.
- cuSolverSp (sparse direct Cholesky/QR: csrchol/csrqr/csrlsv*): NO hipSOLVER
  equivalent. The matrix/diff solver surface (test_solver, test_sparse_matrix,
  test_jacobian, test_hess, test_scalar, autodiff Newton) is excluded from the HIP
  test target. dense_matrix.h (pulled into the core via attribute.h) compiles via
  the hipBLAS L1 + hipSPARSE generic-API aliases. A full hipSOLVER/rocSOLVER port
  of the sparse direct solvers is follow-up work, not done here.

## Validation (GPU 0, gfx90a, HIP_VISIBLE_DEVICES=0) -- RXMesh_test

Run: `HIP_VISIBLE_DEVICES=0 ./projects/RXMesh/build/bin/RXMesh_test --gtest_filter=<...>`

PASS -- 21 tests (the static query + core mesh data-structure path, RXMesh's
contribution and the bulk of the warp-cooperative surface, incl. the wave64
ballot fixes):
- RXMeshStatic.* (9): Queries (all 8 ops VV/VE/VF/EV/EF/FV/FE/FF x 1000 iters
  vs CPU reference), Oriented_VV_Open, Oriented_VV_Closed, EVDiamond,
  MultiQueries, BoundaryVertex, ForEach, ForEachOnDevice, Export.
- RXMesh.* (3 active): Iterator, LPPair, LPHashTable (Benchmark disabled upstream).
- Util.* (6): Scan (hipCUB), AtomicMin, AtomicAdd, Align, BlockMatrixTranspose, Tet.
- RXMeshDynamic (3): PatchScheduler, PatchLock, Validate.
Determinism: Queries / Oriented_VV pass identically across repeated runs; the
participation bitmask (where the wave64 ballot bug would corrupt non-
deterministically) is stable.

BLOCKER -- dynamic mesh-EDITING tests hang:
- RXMeshDynamic.RandomFlips / RandomCollapse / TriangleRefinement / PatchSlicing
  HANG (infinite spin; never complete in 100s+ on a tiny mesh). rocgdb on the live
  hang: the `slice_patches` kernel waves are stuck in `ShmemMutex::lock`
  (kernels/shmem_mutex.cuh) spinning on `atomicCAS(m_mutex, 0, 1)` where `m_mutex`
  is a GARBAGE pointer (0x1, 0x10, 0x1000000000000, differing per lane), so the
  CAS on an invalid address never succeeds.
- Root: the ShmemMutex carries a function-local `__shared__ int` pointer set in
  `alloc()`. Pre-port this device code was dead on HIP (`#ifdef __CUDA_ARCH__`
  false) so the mutex was a silent no-op; enabling it correctly (via the
  __CUDA_ARCH__||__HIP_DEVICE_COMPILE__ rewrite) surfaced that the shared mutex
  pointer is not valid where it is dereferenced under HIP -fgpu-rdc.
- Tried and did NOT fix it: (1) `__syncthreads()` after the 0-init in alloc()
  (the pointer itself is bad, not a stale word); (2) making
  PatchStash::insert_patch(patch, ShmemMutex&) header-inline to remove the
  cross-TU device-function boundary (patch_stash.cu -> patch_stash.h). Both kept;
  neither stops the spin. Core 21 tests still pass after both (no regression).
- Candidate next steps (need deeper GPU debugging): build the dynamic-editing
  path without -fgpu-rdc; or replace the shared-pointer ShmemMutex with a scheme
  that recomputes the shared mutex address from a known shared base in each
  callee (avoid passing a raw __shared__ pointer through device calls); or audit
  whether the CavityManager `m_s_patch_stash_mutex` member's alloc() actually runs
  for the slice path.
