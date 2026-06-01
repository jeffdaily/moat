# rmcl notes

rmcl's portable GPU compute IS rmagine's CUDA backend, so the substantive ROCm
port lives on a **jeffdaily/rmagine** fork and rmcl stays the named MOAT project
tracking it. rmcl wants rmagine 2.4-2.5.0; the fork is branched off upstream
rmagine main (v2.4.2), which is in range.

## Fork / dependency

- Fork: https://github.com/jeffdaily/rmagine  (branch `moat-port`, off upstream main 6b93e86)
- Validated rmagine commit (Stage 1): pin rmcl's source_dependencies.yaml to the
  moat-port HEAD recorded in status.json `head_sha`.
- Actions disabled on the fork.
- rmcl itself was NOT modified this run (its rmcl_ros .cu + ROS 2 layer is a
  separate milestone, see "Deferred").

## Stage 1 delivered: rmagine::cuda HIP compute backend (gfx90a, validated)

Strategy A (compat header + LANGUAGE HIP) behind a new top-level `USE_HIP`
option. NVIDIA build byte-identical when USE_HIP=OFF.

Key pieces:
- NEW `src/rmagine_cuda/include/rmagine/util/cuda/cuda_to_hip.h` -- single compat
  shim mapping the ~30 cuda runtime + cuCtx driver + curand device symbols used
  to hip/hiprand spellings. Included in place of `<cuda_runtime.h>`/`<cuda.h>`/
  `<curand*.h>` at every rmagine_cuda TU (11 redirected include lines + the
  math_svd test). Force-included on HIP TUs via `CMAKE_HIP_FLAGS -include`.
- `src/rmagine_cuda/CMakeLists.txt` -- USE_HIP branch: enable_language(HIP),
  LANGUAGE HIP on RMAGINE_CUDA_SRCS, `-fgpu-rdc` + `--hip-link` for the cross-TU
  `__device__` svd/umeyama_transform calls (CUDA used CUDA_SEPARABLE_COMPILATION),
  link hip::host + hip::hiprand (hip::device PRIVATE so the --offload-arch flag
  does not leak to plain-C++ consumers), drop the cublas/cusolver link refs (no
  call sites), default CMAKE_HIP_ARCHITECTURES=gfx90a only when unset.

### Root cause: wave64 warp-synchronous reduction tail (the decisive fix)
`statistics.cu` sum_kernel and the four `math_batched.cu` chunk_sums kernels ran
the `__syncthreads` tree only to s>32, then a 32-lane `volatile` warpReduce tail
with no `__syncwarp`. On gfx90a (wave64) the low 32 lanes of a 64-lane wavefront
are not lockstep across those unsynced +32..+1 steps -> wrong, run-to-run
nondeterministic sums (NaN/garbage covariance). Fix (USE_HIP-guarded, CUDA
unchanged): drop the warp tail, run the full block-wide `__syncthreads` tree to
s>0. The statistics_p2p/p2l/objectwise_p2l kernels already ran the full tree
(s>0), so they needed no change.

### Two more AMD-surfaced bugs (also UB on CUDA, fixed unconditionally)
- `memory_math.cu` multNx1(Quaternion,Quaternion) called multNxN_kernel, which
  reads `b[id]` for id<N off a size-1 buffer; the Transform/Matrix3x3 siblings
  already use multNx1_kernel (reads `b[0]`). CUDA tolerated the OOB read; AMD
  faults (cuda_math memory access fault). Switched to multNx1_kernel.
- `shared_functions.h` keyed RMAGINE_FUNCTION on `__CUDA_ARCH__` (device pass
  only). HIP/clang then saw the shared math/memory helpers as host-only when
  device code called them and rejected the decl/def attribute mismatch. Re-keyed
  on `__CUDACC__`/`__HIPCC__` (defined in BOTH passes on each toolchain); a plain
  g++ build defines neither, so the CPU path is unchanged. `MemoryView::raw`
  definitions in Memory.tcc got the matching RMAGINE_FUNCTION.

### CudaContext
`cuCtxSetSharedMemConfig`/`cuCtxGetSharedMemConfig`/`CU_SHARED_MEM_CONFIG_*`
USE_HIP-guarded out (CDNA has no configurable LDS bank width). The rest of the
driver context API maps 1:1 to hipCtx*; `hipCtxCreate(&ctx, 0, dev)` matches the
pre-CUDA-13 `cuCtxCreate` form taken when CUDA_VERSION is undefined under HIP.

## Build recipe (gfx90a; standalone, no ROS/Embree/Vulkan/OptiX)

```
cd projects/rmcl/rmagine_src   # jeffdaily/rmagine @ moat-port
export HIP_VISIBLE_DEVICES=0
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DRMAGINE_EMBREE_DISABLE=ON -DRMAGINE_OPTIX_DISABLE=ON \
  -DRMAGINE_VULKAN_DISABLE=ON -DRMAGINE_VULKAN_CUDA_INTEROP_DISABLE=ON \
  -DRMAGINE_OUSTER_DISABLE=ON -DRMAGINE_BUILD_TESTS=ON -DRMAGINE_BUILD_TOOLS=OFF
cmake --build build -j
```
Followers: same command, only change `-DCMAKE_HIP_ARCHITECTURES=gfx1100` (or
gfx1151). No source change -- the wave64 fix (full __syncthreads tree) is
wave-agnostic and correct on wave32 too. Deps: ROCm 7.2, Eigen3, TBB, Boost,
assimp (apt: libboost-all-dev libassimp-dev).

## Validation (real gfx90a / MI250X, GCD 0) -- PASS

```
cd build && ctest --output-on-failure -R '^cuda_'   # 6/6 PASS
ctest --output-on-failure -R '^core_'               # 12/12 PASS (host unchanged)
```
- cuda_math, cuda_memory, cuda_memory_slicing, cuda_math_svd,
  cuda_math_statistics, cuda_math_reduction all PASS.
- cuda_math_reduction and cuda_math_statistics dispatch the fixed sum_kernel /
  statistics kernels (confirmed AMD_LOG_LEVEL=3) and assert no NaN in the
  cross-statistics mean/covariance.
- Determinism: two runs of cuda_math_statistics are bit-identical; reduction
  values identical run-to-run (only concurrent printf interleave differs).
- hipRAND is not bitwise-identical to cuRAND (expected); the noise/random paths
  are validated statistically, not bitwise.

### Gotchas (for followers / future passes)
- The compat header comment must not contain a literal end-of-comment marker
  inside a block comment (`cuda*` then `/curand*` closed the comment early ->
  bogus parse errors). Spell symbol globs out in prose.
- `-fgpu-rdc` is required: svd/umeyama_transform are cross-TU `__device__`
  functions; without it the device link fails with undefined hidden symbols.
- hip::device must be PRIVATE on the lib or `--offload-arch=gfx90a` leaks onto
  the plain-C++ test compiles (g++ rejects it).
- hiprand cmake target is `hip::hiprand`, not bare `hiprand` (-> `-lhiprand` not
  found).
- `timeit.sh` cd's to the MOAT repo root, so pass an ABSOLUTE build dir to the
  wrapped cmake/ctest command.

## Deferred (this host / scope)

- rmcl-layer milestone (rmcl_ros particle_motion.cu + resampling.cu + the MICP
  CUDA sensors + ROS 2 nodes): needs ROS 2 jazzy + Embree, not on this host. The
  two .cu are trivial (curand->hiprand; resampling's reduction already runs the
  full __syncthreads tree, wave-safe). Build them through rmagine's HIP toolchain
  in a colcon workspace once ROS 2 + Embree are provisioned. Not a blocker for
  the rmagine_cuda deliverable.

## Stage 2 (OptiX->HIPRT MCL backend)

The Monte-Carlo / global-localization GPU path is NVIDIA-OptiX-gated and is a
separate HIPRT reimplementation stage, to be done AFTER EnvGS Stage 2 lands and
proves the HIPRT pattern on this host. Scope:
- rmagine ships an OptiX ray-mesh-intersection backend (rmagine_optix:
  optixAccelBuild / optixTrace / module+SBT+pipeline, ~43 OptiX symbols across
  ~60 files). rmcl's RCCOptix + rmcl_ros optix BeamEvaluateProgram
  (__raygen__/__closesthit__/__miss__ PTX) + PCDSensorUpdaterOptix +
  rmcl_localization build only `if(RMCL_OPTIX AND RMCL_EMBREE)`.
- Plan: reimplement rmagine's OptiX ray-mesh backend against AMD HIPRT (the
  EnvGS Stage 2 path + agent_space/hiprt_probe/ are the reference), so the
  rmcl_localization Monte-Carlo node gets an AMD GPU ray-casting backend. Build
  the BVH with HIPRT, port the per-beam likelihood raygen/closesthit/miss to a
  HIPRT trace kernel, and feed the result into the existing rmcl correspondence /
  statistics compute (already HIP-validated in Stage 1).
- HIPRT is NOT in ROCm 7.2 (only hiprtc); needs the HIPRT SDK. Greenlit by jeff
  (OptiX->HIPRT) but explicitly NOT in scope this run.
- The Vulkan RT backend (cross-vendor) is a separate deferral: no Vulkan SDK on
  this host.
