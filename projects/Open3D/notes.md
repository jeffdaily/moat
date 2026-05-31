# Open3D -- ROCm/HIP port notes (lead: linux-gfx90a, ROCm 7.2.1, MI250X)

Strategy A (pure CMake, USE_HIP gate + single compat header). Fork:
https://github.com/jeffdaily/Open3D branch `moat-port`. Upstream base
0333798fcff5a2fe95470e69291eca6a9efbae6c.

## How USE_HIP composes with BUILD_CUDA_MODULE
The GPU code is keyed on `BUILD_CUDA_MODULE` everywhere (CMake `if()` and
`#if defined(BUILD_CUDA_MODULE)` host guards). `USE_HIP=ON` forces
`BUILD_CUDA_MODULE=ON` and only swaps the language (enable_language(HIP)), the
math libs (hipBLAS/hipSOLVER/hipSPARSE), and retags `.cu` LANGUAGE HIP. The
NVIDIA path is untouched (`elseif(BUILD_CUDA_MODULE)` keeps the old block).

## The .cu -> LANGUAGE HIP mechanism (zero per-dir churn)
`open3d_set_global_properties(target)` (called by every GPU lib after its
`target_sources`) calls `open3d_set_hip_properties(target)` under USE_HIP. That
helper walks the target's sources, retags `*.cu` LANGUAGE HIP, sets
HIP_ARCHITECTURES from `${CMAKE_HIP_ARCHITECTURES}`, adds the `hip_compat` shim
include dir + the cpp root, and force-includes `cpp/open3d/core/hip/CUDAToHIP.h`
on the HIP TUs. The host linalg `.cpp` (AddMMCUDA.cpp ...) stay CXX; only the
math-lib link target swaps.

## Compat layer (the only files that know about HIP)
- `cpp/open3d/core/hip/CUDAToHIP.h` -- 45 cuda*->hip* runtime aliases, the
  `cub`->`hipcub` namespace alias, `OPEN3D_FULL_WARP_MASK (~0ull)`, an
  `__ffs(unsigned long long)` overload (for SlabHash's `__ffs(__ballot_sync)`),
  and `CUDA_VERSION 9000` to take the `>=9000` *_sync paths. `<cstring>`/
  `<cstdlib>` precede `<hip/hip_runtime.h>` (gpuRIR host-memcpy lesson).
- `cpp/open3d/core/hip/hip_compat/{cuda.h,cuda_runtime.h,cuda_runtime_api.h,
  cub/cub.cuh}` -- forwarding shims so the CUDA-spelled angle-bracket includes
  resolve under HIP without editing every include line (MPPI lesson). On
  include path only under USE_HIP, so the CUDA build uses the real toolkit
  headers.

## Fault-class fixes
1. FAISS warp-select (`core/nns/kernel/`) -- the highest-risk body.
   - `DeviceDefs.cuh`: kWarpSize STAYS 32. Decision: run the 64-lane wavefront
     as TWO independent 32-lane bitonic groups (popsift two-halves model), NOT
     kWarpSize=64. Reason: BlockSelectFloat*.cu instantiates NumWarpQ==32 (and
     ==1), and `kNumWarpQRegisters = NumWarpQ / kWarpSize` would be 0 if
     kWarpSize were 64 -- the warp queue collapses. With kWarpSize=32, every
     shfl uses width=32 (HIP confines exchange to the 32-lane group) and
     getLaneId() returns the in-group 0..31 lane, so each half behaves exactly
     like a 32-lane NVIDIA warp. Correct on wave32 AND wave64 (arch-unified).
   - `PtxUtils.cuh`: ALL inline PTX (bfe/bfi/mov %laneid/bar.sync) is replaced
     under USE_HIP with clang builtins; getLaneId() = threadIdx.x & 31 (the
     kernels launch 1-D blocks). The bitfield/lanemask/namedBarrier helpers are
     dead in the build but reimplemented for self-consistency.
   - `WarpShuffle.cuh`: __shfl_*_sync masks 0xffffffff -> OPEN3D_FULL_WARP_MASK
     (64-bit on HIP; HIP static_asserts sizeof(mask)==8).
   - `Select.cuh`: __any_sync(0xffffffff,..) -> OPEN3D_FULL_WARP_MASK. The
     full-wavefront __any over both halves is correct: a non-full half still
     merges its (sorted, initK-padded) thread queue before reset, a no-op.
   - `warpFence()` -> `__syncthreads()` on HIP (kernels are warp-convergent).
2. `core/kernel/ReductionCUDA.cu` (PyTorch-derived) already uses runtime
   `warpSize` (64) for its reduction tree; only the WARP_SHFL_DOWN default mask
   needed widening to 64-bit (OPEN3D_REDUCE_FULL_WARP_MASK).
3. SlabHash (`core/hashmap/CUDA/Slab*`) -- NON-default backend, SCOPED OUT of
   validation, made to COMPILE only: `kSyncLanesMask`/`kNodePtrLanesMask`
   widened to 64-bit under USE_HIP so the *_sync masks compile; `__ffs` of the
   64-bit ballot uses the compat `__ffs(ull)` overload. Its 32-lane lane
   election is NOT wave64-correct; a correct wave64 rewrite is deferred. Slab
   HashMap test cases are skipped.
4. Library swaps -- `LinalgHeadersCUDA.h` includes hipblas/hipsolver and aliases
   cublas*/cusolver* -> hipblas*/hipsolver*. The typed Dn{S,D} compat API
   (hipsolverDnSgetrf/gesvd/getrs/geqrf/ormqr + _bufferSize) is a clean drop-in
   (verified signatures match cuSOLVER, no extra lwork arg in the solve calls).
   The wider cuSOLVER status enum's orphan cases (INVALID_LICENSE, IRS_*,
   INVALID_WORKSPACE) are `#if !defined(USE_HIP)`-guarded in the fmt formatter.
5. stdgpu (DEFAULT hashmap backend) -- `3rdparty/stdgpu/stdgpu.cmake` builds the
   pinned commit's STDGPU_BACKEND_HIP with CMAKE_CXX_COMPILER=hipcc/clang (its
   host impl/*.cpp reach rocPRIM, clang-only) + CMAKE_HIP_ARCHITECTURES. The
   newer pinned commit already has a HIP-aware Findthrust, so the cupoch
   stdgpu-1.3.0 rot does not apply.
6. NPP image ops -- no ROCm equivalent. `NPPImage.cpp` body is
   `#if !defined(USE_HIP)`; under HIP it provides LogError stubs (so Image.cpp's
   CUDA branch links). NPP find + link guarded out. GPU image-filter ops are an
   explicitly-unsupported path on ROCm; CPU/IPP path unaffected.

## Scoped OUT of the lead (guarded, NOT failures)
- CUTLASS 1.3.3 ML conv ops (BUILD_PYTORCH_OPS/BUILD_TENSORFLOW_OPS OFF; CUTLASS
  does not port to ROCm).
- NPP GPU image filters (above).
- SlabHash backend wave64 correctness (non-default; compiles, tests skipped).
- BUILD_GUI / Filament / WebRTC / examples / python module (headless GPU core).

## Build (gfx90a)
Configure:
```
cmake -S projects/Open3D/src -B projects/Open3D/src/build \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DBUILD_PYTORCH_OPS=OFF -DBUILD_TENSORFLOW_OPS=OFF \
  -DBUILD_GUI=OFF -DBUILD_WEBRTC=OFF -DBUILD_EXAMPLES=OFF \
  -DBUILD_BENCHMARKS=OFF -DBUILD_PYTHON_MODULE=OFF \
  -DBUILD_UNIT_TESTS=ON -DBUILD_ISPC_MODULE=OFF \
  -DWITH_IPP=OFF \
  -DCMAKE_BUILD_TYPE=Release -DDEVELOPER_BUILD=ON
cmake --build projects/Open3D/src/build -j16 --target tests
```
Followers (gfx1100, gfx1151): same command, only CMAKE_HIP_ARCHITECTURES
changes -- no source edit by design.

## Test plan / validation
GPU gtest binary `build/bin/tests` (BUILD_UNIT_TESTS=ON). PermuteDevices
fixtures auto-run each parameterized test on BOTH CPU and the gfx90a device, so
CPU is the built-in oracle. Run serially on ONE GCD (HIP_VISIBLE_DEVICES). Gate
suites: *Tensor*, *MemoryManager*, *HashMap* (stdgpu), *NearestNeighborSearch*/
*KnnIndex*/*FixedRadiusIndex* (FAISS wave64), *Reduction*, t-geometry +
registration GPU tests.

## Additional build fixes found during bringup (beyond the plan)
- `__CUDACC__` cannot be faked globally: defining it forces rocThrust to its
  CUDA backend (compiler.h reads __CUDACC__ as "CUDA compiler"), dumping every
  thrust call into the "unimplemented for this system" generic fallback. Fix:
  do NOT define __CUDACC__; instead extend each `#ifdef __CUDACC__` device-vs-
  host guard across the kernel headers to `__CUDACC__ || __HIPCC__` (the
  cudaKDTree/GLM trap), force THRUST_DEVICE_SYSTEM=5 (HIP) via a compile def so
  rocThrust's auto-detect is bypassed, and provide OPEN3D_DEVICE_CODE
  (= __HIP_DEVICE_COMPILE__ on HIP) for the data-pointer-selection guards.
- `MiniVec.h` FN_SPECIFIERS keyed on __CUDACC__ -> +__HIPCC__ (else device
  kernels can't call MiniVec ctor/operator[]).
- `OPEN3D_ASSERT_HOST_DEVICE_LAMBDA` uses the nvcc-only intrinsic
  `__nv_is_extended_host_device_lambda_closure_type`; no-op it on HIP.
- `__ffs(__ballot_sync(...))` is ambiguous on HIP (ballot is unsigned long long,
  HIP has only __ffs(int)/__ffs(unsigned)); the compat header adds a
  __host__ __device__ __ffs(unsigned long long) (uses __ffsll device /
  __builtin_ffsll host).
- `GeometryMacros.h`: the `__CUDA_ARCH__ < 600` double-atomicAdd fallback would
  REDEFINE HIP's native atomicAdd(double*); guarded CUDA-only.
- `RegistrationImpl.h`: clang requires explicit template instantiations of
  __host__ __device__ function templates to carry the attributes -> prefixed
  each with OPEN3D_HOST_DEVICE (cudaKDTree attribute-match lesson).
- Host nodiscard: HIP marks the runtime API nodiscard (CUDA does not), tripping
  host .cpp under -Werror; relaxed -Werror=unused-result for CXX/HIP under
  USE_HIP (mirrors the CUDA build's NVCC 2809 suppression).
- gcc13 false `-Wmaybe-uninitialized` on Eigen SIMD in MinimumOBB/OBE.cpp
  (best_R IS Identity-initialized) -> -Wno-error=maybe-uninitialized for gcc.
- stdgpu install: the pinned commit's HIP backend dir lacks
  determine_thrust_paths.cmake (CUDA-only) but install() references it -> a
  PATCH_COMMAND copies the CUDA one in. stdgpu CXX must be hipcc/clang (its host
  impl/*.cpp reach rocPRIM); the compiler overrides go AFTER
  ExternalProject_CMAKE_ARGS_hidden so they win over the host g++ default.

## STATUS (2026-05-31): core BUILDS + broadly VALIDATES; stdgpu hashmap wall
The full `tests` binary builds for gfx90a (all ~31 HIP .cu TUs compile + link).
GPU validation on one MI250X GCD (HIP_VISIBLE_DEVICES, PermuteDevices CPU param
is the oracle, `/1` = the HIP device):
- *Tensor*: 398/398 functional PASS (1 SYCL skip; ReduceSum64bit2DCase0 once
  OOM'd on a contended GCD, PASSES on an idle one -- environmental).
- *Reduction*/*Reduce*: 34/34 PASS (warp reduction wave64-correct).
- *NearestNeighborSearch*/*KnnIndex*/*FixedRadiusIndex*/*Knn*/*Hybrid*: 16/16
  PASS -- the FAISS warp-select wave64 (highest risk) is CORRECT via the
  two-32-lane-subgroup model.
- Registration ICP (PointToPoint/PointToPlane/Colored), Feature (FPFH,
  correspondences), EvaluateRegistration: PASS (atomicAdd ICP within tolerance).
- t-geometry PointCloud/TriangleMesh/Transform (non-image): PASS.

EXPECTED scoped-out failures (NOT regressions, documented above): the NPP GPU
image-filter ops throw NppUnsupportedOnHIP, so the tests that exercise them fail
by design -- Image.{Filter,FilterGaussian,FilterBilateral,FilterSobel,Resize,
PyrDown,Dilate}, Odometry.RGBDOdometryMultiScale{PointToPlane,Intensity,Hybrid}
(image pyramids), PointCloud.CreateFromRGBDOrDepthImageWithNormals. These need
the CPU device on ROCm.

REAL remaining wall (why state stays `porting`, not `ported`):
- *HashMap* (the DEFAULT stdgpu backend): 17/20 PASS, but Reserve,
  InsertComplexKeys, MultivalueInsertion, HashSet FAIL on the GPU. Root cause is
  a concurrency bug in stdgpu's experimental HIP backend `unordered_base::
  try_insert`: with duplicate/colliding keys inserted concurrently, the
  per-bucket spinlock + `contains()` + occupied-bit dedup is not wave64-correct,
  so two same-key threads both report success (masks.Sum()=1024 vs 1023; and an
  off-by-one set after Reserve/rehash). Basic Insert/Find/Erase/Clear/IO/
  Multivalue-on-distinct-keys PASS, so the backend works for non-colliding
  workloads. This is an UPSTREAM stdgpu HIP-backend bug (its per-bucket lock /
  atomic memory ordering under a 64-lane wavefront), not an Open3D kernel bug.
  Fixing it means patching stdgpu's internal lock/atomic path -- deep and
  wave64-risky; deferred per the plan (Risk 6 / Open question 1).
  Next concrete step: either patch stdgpu's HIP atomic_ref/lock backend for
  wave64 dedup, or implement a correct wave64 SlabHash backend (also deferred)
  and select it for the HashMap tests. The hashmap is used by VoxelBlockGrid
  (TSDF integration), so VoxelBlockGrid GPU correctness is gated on this too.
