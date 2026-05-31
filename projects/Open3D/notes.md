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

## STATUS (2026-05-31): core BUILDS + fully VALIDATES incl. HashMap + VoxelBlockGrid -> ported
The full `tests` binary builds for gfx90a (all ~31 HIP .cu TUs compile + link).
GPU validation on one IDLE MI250X GCD (HIP_VISIBLE_DEVICES=0; this box has 4
GCDs 0-3 shared with other MOAT CLIs, so pick an idle one via `rocm-smi
--showuse`/`--showmemuse` and pin ONE -- two concurrent runs on one GCD livelock
the ICP tests. PermuteDevices CPU param is the per-test oracle, `/1` = the HIP
device):
- *Tensor*+*Reduction*+*MemoryManager*: 421/421 PASS (warp reduction
  wave64-correct; 1 SYCL skip).
- *NearestNeighborSearch*/*KnnIndex*/*FixedRadiusIndex*/*Knn*/*Hybrid*: PASS --
  the FAISS warp-select wave64 (highest risk) is CORRECT via the
  two-32-lane-subgroup model.
- *HashMap*: 20/20 PASS. The four dedup-heavy tests (Reserve, InsertComplexKeys,
  MultivalueInsertion, HashSet) are stable over 30 `--gtest_repeat` iterations
  with `--gtest_break_on_failure` (240 device runs, 0 fail).
- *VoxelBlockGrid* (TSDF integrate + ray casting + IO): 14/14 PASS, stable over
  5 reruns (it uses the StdGPU hashmap internally, so this is the same dedup
  surface end-to-end).
- Registration ICP (PointToPoint/PointToPlane/Colored), Feature (FPFH): PASS.

EXPECTED scoped-out failures (NOT regressions, documented above): the NPP GPU
image-filter ops throw NppUnsupportedOnHIP, so the tests that exercise them fail
by design -- Image.{Filter,FilterGaussian,FilterBilateral,FilterSobel,Resize,
PyrDown,Dilate}, Odometry.RGBDOdometryMultiScale{PointToPlane,Intensity,Hybrid}
(image pyramids), PointCloud.CreateFromRGBDOrDepthImageWithNormals. These need
the CPU device on ROCm.

## HashMap wall RESOLVED -- root cause was the Slab backend, NOT stdgpu
An earlier diagnosis (above, now corrected) blamed stdgpu `unordered_base::
try_insert` dedup for the 4 HashMap failures. That was WRONG. The HashMap/HashSet
tests loop over BOTH GPU backends: the test code (cpp/tests/core/HashMap.cpp)
pushes `Slab` then `StdGPU` for any CUDA device. The failing reduction
`masks.Sum()=1024 vs 1023` was on the FIRST backend iteration, which is **Slab**,
not the default StdGPU.

How it was pinned down (printf is unreliable -- the GPU FIFO drops output past
~1024 lines): a host recount split by backend iteration showed the FIRST line
(Slab) at host_bool=1024/1026 while the SECOND (StdGPU) was exactly 1023; a
`__device__` win-counter inside stdgpu's try_insert plus a host recount of
`impl_.size()` showed the StdGPU map ALWAYS produces exactly 1023 wins / size
1023, even on a run gtest then failed. Confirmation: with PRISTINE stdgpu and only
Slab excluded on ROCm, the 4 hard tests pass and the suite is 20/20. A false
trail avoided: stdgpu's HIP `atomic_load`/`atomic_store` are plain `volatile`,
which looks suspicious, but making them `__hip_atomic_*` changed no result --
StdGPU dedup was already correct; the issue was purely Slab.

The SlabHash backend (warp-cooperative, `tid>>5` 32-lane lane election, `__ffs`
of a ballot, `kSyncLanesMask`/`kNodePtrLanesMask`) is genuinely wave64-incorrect
on CDNA and is the NON-default backend explicitly scoped OUT of this port (a
correct wave64 rewrite is deferred). The DEFAULT, production-relevant StdGPU
backend is wave64-correct AS-IS; no stdgpu source patch is needed (stdgpu stays
the pinned tarball; 3rdparty/stdgpu/stdgpu.cmake is unchanged).

FIX (test-only, minimal footprint, committed on the fork): guard Slab out of the
HashMap and VoxelBlockGrid tests under `#if defined(USE_HIP)` -- HashMap.cpp 9
sites push only `StdGPU` on USE_HIP; VoxelBlockGrid.cpp's EnumerateBackends sets
`include_slab=false` on USE_HIP. `USE_HIP` reaches the C++ test TUs via
`open3d_set_global_properties(tests)` -> `target_compile_definitions(tests
PRIVATE USE_HIP)` (verified in build/cpp/tests/.../flags.make). With Slab
excluded, VoxelBlockGrid validates fully on GPU. Do NOT re-enable Slab on ROCm
without first porting its 32-lane lane election to wave64.

State: linux-gfx90a -> `ported` at fork sha 3311036 (single curated
[ROCm] commit, amended in place; the prior 2ae6d769 lacked these two test edits).

## Review 2026-05-31 (reviewer, /pr-review local-branch @ 3311036)
Verdict: review-passed. Strategy A is correct for this pure-CMake project; the diff is additive and USE_HIP-gated (778+/121-, single [ROCm] commit), and the NVIDIA path is byte-identical (the only host .cu/.cpp touched are CUDAUtils.cpp -- a behavior-neutral static_cast<void>(cudaGetLastError()) -- ReductionCUDA.cu whose non-HIP default mask stays unsigned int 0xffffffffu, and NPPImage.cpp wrapped #if USE_HIP/#else original/#endif). Commit hygiene clean: title 54 chars, Claude-disclosed, Test Plan present, no noreply/ghstack/AMD-internal refs.

Fault-class analysis independently verified (not taken on faith):
- StdGPU-vs-Slab root cause CONFIRMED. grep of StdGPUHashBackend.h returns zero warp/lane/ballot/shfl/tid>>5 patterns -- its dedup runs through stdgpu's open-addressing container (per-slot atomicCAS across all lanes), so it is wave64-correct as-is and needs no source patch (tarball + stdgpu.cmake content unchanged). Slab IS genuinely wave64-incorrect: SlabHashBackendImpl.h:106 lays out bucket_id*kWarpSize+lane_id, :203/:307 __ballot_sync(kSyncLanesMask,..), :206/:310 __ffs(work_queue)-1 lane election, :208 __shfl_sync(..,src_lane,kWarpSize) -- 32-lane election on a 64-lane wavefront.
- Test-only Slab guard is legitimate and reaches the TUs. 9 sites in HashMap.cpp push only StdGPU under #ifndef USE_HIP; VoxelBlockGrid.cpp EnumerateBackends sets include_slab=false under #ifdef USE_HIP. USE_HIP reaches the test binary via cpp/tests/CMakeLists.txt:53 open3d_set_global_properties(tests) -> Open3DSetGlobalProperties.cmake:97 target_compile_definitions(tests PRIVATE USE_HIP).
- FAISS two-32-lane-halves model is internally consistent. kWarpSize stays 32 (DeviceDefs.cuh:53); getLaneId()=threadIdx.x&31 (PtxUtils.cuh:70); all shfl default width=kWarpSize=32 so exchanges stay within a 32-lane half; masks widened to OPEN3D_FULL_WARP_MASK (~0ull). The UNCHANGED Reduction.cuh blockReduceAll was checked for a wave64 hazard and is correct under this model: warpId=threadIdx.x/32 and the final combine (warpId==0, smem[laneId], laneId 0..31, width-32 warpReduceAll) decomposes the block into logical 32-lane warps exactly like NVIDIA, valid for blockDim.x<=1024. L2Select.cuh exercises this path -- covered by the KNN GPU tests.
- Slab masks: kSyncLanesMask->64-bit (correct ballot/shfl mask so *_sync compiles); kNodePtrLanesMask=0x7FFFFFFFull is used BOTH as a ballot mask and as a (1<<lane_id)& lane bitmask and is wave-incorrect, but that is inside the scoped-out compile-only Slab backend with tests excluded, so it does not gate validation.
- Library swaps clean: LinalgHeadersCUDA.h aliases the typed cublas/cusolverDn{S,D} API to hipblas/hipsolver, orphan wider cuSOLVER status enum cases #if !defined(USE_HIP)-guarded. NPP/SlabHash/CUTLASS/GUI scope-outs are clean guarded deferrals.

Finding (non-blocking, comment-only -- fold into the validator's SHA-update amend, do NOT churn HEAD on its own):
- cmake/Open3DSetGlobalProperties.cmake:106 comment claims "The compat header defines __CUDACC__ (Open3D's device-guard requirement)". This is factually wrong and contradicts CUDAToHIP.h:29 ("We do NOT fake __CUDACC__ for HIP"); grep confirms __CUDACC__ is never #defined anywhere. The real mechanism is extending each __CUDACC__ guard to ||__HIPCC__, and THRUST_DEVICE_SYSTEM=5 (line 113) is set precisely BECAUSE __CUDACC__ is absent and rocThrust's auto-detect must be pinned. The code is correct; only the comment's premise is wrong and will mislead. Reword to: rocThrust's compiler.h would otherwise auto-detect the wrong backend, so pin THRUST_DEVICE_SYSTEM=5 (HIP).

Safe to proceed to GPU validation. The missing GPU run at review time is expected (validator stage runs the real gfx90a tests next).
