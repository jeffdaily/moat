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

EDITING-TEST BLOCKERS (RandomFlips/RandomCollapse/TriangleRefinement/PatchSlicing):

== (1) ShmemMutex spin-lock deadlock -- ROOT-CAUSED AND FIXED ==
- Symptom: slice_patches waves spin forever in ShmemMutex::lock on
  `atomicCAS(m_mutex, 0, 1)` (kernels/shmem_mutex.cuh).
- The prior "garbage / per-lane m_mutex pointer under -fgpu-rdc" hypothesis was
  WRONG. A standalone HIP repro (agent_space/rxmesh_shmem_mutex/) of exactly the
  pattern -- a __device__ helper that points a member at a function-local
  __shared__ int (mirroring alloc()), then an atomicCAS spin-lock, exercised by
  256 threads (4 wave64 wavefronts) -- swept the full matrix:
    {helper inlined vs separate-TU} x {-fgpu-rdc on/off} x {static vs dynamic
    extern __shared__}. The original split lock()/unlock() form HANGS in ALL
  configurations (incl. inlined + rdc-off + static shared -- no cross-TU, no rdc).
  The repro also showed m_mutex is a VALID, lane-uniform shared address
  (0x1000000000000 is just the generic-address-space tag; low32 == 0). So the
  bug is neither -fgpu-rdc codegen, nor cross-TU, nor a garbage pointer.
- TRUE root cause: a SIMT spin-lock deadlock. CUDA Volta+ has Independent Thread
  Scheduling so a lane that wins the CAS makes forward progress to unlock() even
  while sibling lanes of the same warp spin. AMD CDNA (gfx90a, wave64) has NO
  per-lane forward-progress guarantee within a wavefront: the winning lane is
  stuck at SIMT reconvergence waiting on the losing lanes, which spin forever ->
  classic lockstep spin-lock deadlock. This is exactly what ShmemMutex's own
  `#error ... requires sm70 ... Independent Thread Scheduling` warns about.
- Folding the critical section into the CAS-acquire loop ("body-in-loop")
  does NOT help: the compiler peels acquiring lanes off the exec mask
  (s_andn2_b64 exec,...) and keeps the lock held until the body runs for the
  reconverged wave -> still deadlocks (repro: 10/10 hang at -O3).
- FIX (USE_HIP/__HIP_DEVICE_COMPILE__-guarded; CUDA path unchanged): wave-serialize
  the lock. Added ShmemMutex::critical_section(func) and
  ShmemMutexArray::critical_section(loc, func): elect one active lane of the
  wavefront at a time via __ballot(1)/__ffsll, let that single lane fully
  acquire/run/release, then retire its bit and move to the next. With only one
  contending lane per wavefront there is no intra-wavefront spin-lock hazard
  (cross-wavefront contention is fine -- wavefronts schedule independently).
  Repro: wave-serial is 10/10 correct & deadlock-free at -O3, incl. the
  subset-of-lanes-from-a-loop shape that mirrors copy_to_hashtable.
  Call sites converted: PatchStash::insert_patch(patch, ShmemMutex&)
  (patch_stash.h) and CavityManager::add_edge_to_cavity_graph's two
  ShmemMutexArray critical sections (cavity_manager_impl.cuh). rocgdb confirms
  the hang moves PAST the mutex (slice now completes for the first patch with
  valid counts 151/417/267). The two earlier "fixes" (__syncthreads in alloc;
  header-inlining insert_patch) were neither necessary nor sufficient.

== (2) compute_vf (Op::VF) shared-memory overflow in validate() -- FIXED ==
- RXMeshDynamic::validate()'s check_ribbon launches the VF query (compute_vf),
  which here needs ~81920 B (80 KB) of dynamic shared memory. gfx90a offers only
  64 KB/block, so the launch fails with "invalid argument". A following
  cudaDeviceSynchronize() does NOT clear it (no kernel was queued), so the sticky
  error poisons the next unrelated CUDA call (it broke Util.Scan when the suite
  ran end-to-end, and slice_patches' pre-launch checkpoint). This is a hardware
  capability gap, not a logic bug (CUDA targets have >=96 KB optin).
- FIX: in check_ribbon, query cudaFuncGetAttributes(compute_vf).
  maxDynamicSharedSizeBytes and, if the required dyn smem exceeds it, skip the
  VF-based ribbon-faces sub-check (the EV-based ribbon-edges check still runs),
  log a warning, and return -- no failed launch, no sticky error.

== (3) Post-edit patch metadata = INVALID16 -- STILL BLOCKING (separate bug) ==
- With (1)+(2) fixed, slice_patches(patch 0) and cleanup() run to completion, but
  the next loop iteration reads patch 1 with num_v == num_e == num_f == 0xFFFF
  (INVALID16) at slice_patches entry (rxmesh_dynamic.h:166); the element-strided
  loops then run on a bogus huge bound and spin. rocgdb localized it to a
  `for(i=tid;i<num;...)` loop in BOTH the SLICE_GGP (line 273) and non-SLICE_GGP
  paths, so it is NOT block_mat_transpose/GGP (disabling SLICE_GGP did not help).
- Stage-instrumented (printf of pid + counts) slice_patches entry and
  remove_surplus_elements (RSE) entry pinned the transition:
    iter0: SLICE-ENTRY pid=0 v=163 e=447 f=285  (patch 0 sliced OK)
           RSE-ENTRY   pid=1 v=132 e=358 f=227  (patch 1 still VALID during
                                                 iter0 cleanup)
    iter1: SLICE-ENTRY pid=1 v=65535 e=65535 f=65535  (patch 1 now INVALID16)
  So a patch's stored element counts go valid->INVALID16 across one
  edit+cleanup cycle, on a NEIGHBOR patch (not the one being sliced).
- RSE recomputes and writes the counts at rxmesh_dynamic.cu:582-618:
  s_num_vertices = atomicMax(v+1) over s_vert_tag (the post-cleanup active
  bitmask from reevaluate_active_elements), then pi.num_vertices[0]=s_num_vertices.
  A spurious high tag bit (a wave64 Bitmask / __ballot artifact in the cleanup
  active-mask recompute) would make that max == 0xFFFF. This is the next thing to
  chase: audit reevaluate_active_elements + Bitmask set/store/ballot on wave64
  for an out-of-range bit, OR a patch-metadata aliasing bug when slicing adds a
  patch. It is a distinct CORRECTNESS bug from the ShmemMutex hang.
- The 3 cavity-based editing tests (RandomFlips/RandomCollapse/TriangleRefinement)
  no longer HANG after fix (1) -- the ShmemMutexArray wave-serialization removed
  the cavity-graph deadlock; they now fail with a CUDA error in update_host's
  cudaMemcpy (rxmesh_dynamic.cu:2877) whose size derives from the same corrupt
  num_edges[0] -- i.e. the same bug (3). PatchSlicing still hangs on bug (3).

Verdict: ShmemMutex hang == RXMesh logic bug (assumes Volta ITS), FIXED on HIP
(wave-serialization). Editing tests remain blocked on bug (3). Core 21 tests
pass TOGETHER (no regression) after (1)+(2) -- before (2) the suite aborted at
Util.Scan from the compute_vf sticky error. Determinism holds (PatchLock 10/10 on
a clean build; earlier 'flakiness' was a stale-binary artifact from an aborted
rebuild, not real).
Standalone repro + full matrix: agent_space/rxmesh_shmem_mutex/ (mutex.h,
main.hip, worker.hip, run.sh..run4.sh).

## 2026-05-30 -- sparse-solver dependency analysis (cuDSS / cusolverSp) + STRUMPACK dep

RXMesh's solver/autodiff path uses NVIDIA sparse solvers; the ROCm story per facility:
- High-level cusolverSp solve (csrlsvqr/csrlsvchol): AVAILABLE in hipSOLVER (hipsolverSp*, rocSPARSE+SuiteSparse backend, BUILD_WITH_SPARSE). Portable today.
- Low-level cusolverSp factorization (csrqr*/csrchol* Setup/BufferInfo/Factor/Solve/ZeroPivot), csrlsvluHost, csrsymrcm reordering: NOT in hipSOLVER. Feature request filed: https://github.com/ROCm/hipSOLVER/issues/443
- cuDSS (used heavily here): proprietary/closed-source -> NOT portable. Open ROCm-capable substitute: STRUMPACK (BSD; multifrontal sparse LU; already HIP/ROCm; ~1.9x faster than cuDSS single-GPU per a 2025 paper).

DEPENDENCY: RXMesh now depends_on STRUMPACK (pending). The solver path substitutes cuDSS -> STRUMPACK (and the high-level solve -> hipSOLVER csrlsvchol) once STRUMPACK is validated on ROCm in MOAT. This is SEPARATE from the ShmemMutex editing-path blocker; the core mesh-query + editing port does not need STRUMPACK.
