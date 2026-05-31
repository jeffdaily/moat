# arrayfire notes

ArrayFire 3.10.0, pure-CMake multi-backend array library. Port shape: a NEW sibling
backend `src/backend/hip` cloned from `src/backend/cuda` (additive; the CUDA/NVIDIA path
stays byte-for-byte). The HIP backend reports `AF_BACKEND_CUDA` so the unified dispatcher and
the 139 gtest binaries treat it as the "cuda" backend (no public ABI/enum change). Lead
platform linux-gfx90a (MI250X, ROCm 7.2.1).

Fork: https://github.com/jeffdaily/arrayfire (branch `moat-port`; `master` stays a clean
upstream mirror). Actions disabled on the fork. base_sha 492718b.

## Install as a dependency
N/A -- arrayfire is a leaf (no other MOAT project depends on it). It vendors only
`extern/half` and finds host/GPU libraries externally; no inter-project MOAT deps.

## Phase 0 (GO/NO-GO): hipRTC runtime-JIT -- RESULT: GO

The make-or-break risk was arrayfire's bespoke NVRTC + CUDA-Driver-API runtime-JIT engine
(`src/backend/cuda/compile_module.cpp`). On NVIDIA it does
`nvrtcCompileProgram(--gpu-architecture=compute_XX)` -> `nvrtcGetPTX` ->
`cuLinkCreate`/`cuLinkAddData(CU_JIT_INPUT_PTX)`/`cuLinkComplete` -> `cuModuleLoadData`. The
question was whether the hipRTC analogue (direct code object, no PTX, no cuLink, `--offload-arch`)
works for arrayfire's TWO JIT paths.

Standalone repro `agent_space/af_hiprtc_poc/af_poc.cpp` (built with
`/opt/rocm/bin/hipcc -O2 -std=c++17 af_poc.cpp -o af_poc -lhiprtc`, run
`HIP_VISIBLE_DEVICES=0 ./af_poc`) reproduces both arrayfire JIT paths with arrayfire's actual
constructs and runs them end to end on gfx90a. RESULT: PASS.

- JIT path (`sourceIsJIT=true`): an `extern "C" __global__` element-wise kernel carrying the
  `kernel/jit.cuh` preamble -- `typedef float2 cuFloatComplex;` / `double2`, `#include <cuda_fp16.h>`,
  `__device__` complex helpers (`__caddf`/`__cmulf`...), and an integer-round intrinsic
  (`__double2ll_rn`). hipRTC compiles it to a 5848-byte code object; `hipModuleLoadData` +
  `hipModuleGetFunction("af_jit_kernel")` + `hipModuleLaunchKernel` + host verify all pass.
- Templated path (`sourceIsJIT=false`): a templated kernel instantiated by name expression
  via `hiprtcAddNameExpression` / `hiprtcGetLoweredName` (lowered name
  `_Z15af_templ_kernelIfLi3EEviPT_PKS0_S3_`), loaded and launched, host verify passes. (This
  matches the already-proven cudf hipRTC PoC, re-confirmed in arrayfire's shape.)

The flow `hiprtcCompileProgram(--offload-arch=gfx90a)` -> `hiprtcGetCodeSize`/`hiprtcGetCode`
-> `hipModuleLoadData` -> `hipModuleGetFunction` -> `hipModuleLaunchKernel` is fully functional.
The arch flag is derived from `hipGetDeviceProperties(...).gcnArchName` with the
`:sramecc+:xnack-` feature suffix stripped (`gfx90a`).

### Three concrete arrayfire-specific hipRTC deltas (all fixed in the repro; none a blocker)

1. **Empty-string ("") header source is REJECTED by `hiprtcCreateProgram` (INVALID_INPUT)**,
   where NVRTC accepts it. `compile_module.cpp` passes several `string("")` dummy header
   bodies ("DUMMY ENTRY TO SATISFY ..." for math.h/stdbool.h/stdlib.h/vector_types.h/utility).
   On HIP each dummy header source must be NON-EMPTY (a `"/* empty */\n"` placeholder works;
   `min4.cpp` mode 1 vs mode 2 isolates this exactly). A NULL header pointer aborts in the
   std::string ctor.
2. **`--device-as-default-execution-space` is REJECTED by hipRTC** ("unknown argument").
   arrayfire passes it unconditionally (`compile_module.cpp:275`). Drop it on HIP -- HIP makes
   device the default execution space for `__global__`/`__device__` via `__HIP_DEVICE_COMPILE__`
   anyway. Same for `--gpu-architecture=compute_XX` -> `--offload-arch=gfx90a`, and the
   `--device-debug`/`--generate-line-info` NVRTC debug flags (drop or map).
3. **An embedded header whose body does `#include <hip/...>` needs `-I/opt/rocm/include`** as a
   runtime compile option (hipRTC does not implicitly know the ROCm include path). This is the
   cudf lesson; the cleanest model is to inject `cuda_fp16.h` (the name arrayfire's JIT source
   `#include`s) as a body of `#include <hip/hip_fp16.h>` and pass `-I/opt/rocm/include`. NOTE:
   hipRTC also provides `__half`/`__float2half`/`__half2float` AND `float2`/`double2` as
   builtins with NO header at all (`min2.cpp` modes 2,3), so the fp16/vector-type headers may
   even be droppable; the `-I/opt/rocm/include` + injected-header model is the conservative
   choice that keeps the source strings unchanged.

Repro files: `agent_space/af_hiprtc_poc/{af_poc.cpp, RESULT.log}` (full repro) and the
isolation probes `min.cpp`/`min2.cpp`/`min3.cpp`/`min4.cpp`. ROCm 7.2.1, hiprtc version 9.0,
AMD clang (roc-7.2.1), 4x MI250X (GCDs 0-3), pinned to GCD 0.

## JIT engine restructure (compile_module.cpp on HIP)

Given Phase 0 GO, the port of `compile_module.cpp` is:
- `nvrtc.h` -> `hip/hiprtc.h`; `nvrtc*` -> `hiprtc*`; `NVRTC_SUCCESS` -> `HIPRTC_SUCCESS`.
- Drop the entire PTX + cuLink block (`nvrtcGetPTXSize`/`nvrtcGetPTX`, `cuLinkCreate`/
  `cuLinkAddData(CU_JIT_INPUT_PTX)`/`cuLinkComplete`, `cuLinkDestroy`). Replace with
  `hiprtcGetCodeSize`/`hiprtcGetCode` into a `vector<char>` and `hipModuleLoadData(&mod, code)`.
- Arch: replace `--gpu-architecture=compute_%d%d` with `--offload-arch=<gcnArchName-stripped>`.
- Drop `--device-as-default-execution-space` and the NVRTC `--device-debug`/`--generate-line-info`.
- Empty dummy header bodies -> non-empty placeholder; add `-I/opt/rocm/include` to the JIT
  compile options (or to the non-JIT options) so injected `hip/...` includes resolve.
- The disk cache stores the code-object blob instead of cubin (key on gcnArchName instead of
  compute capability). Keep the deterministicHash integrity check.
- `cuModuleGetFunction`/`cuLaunchKernel` -> `hipModuleGetFunction`/`hipModuleLaunchKernel` (these
  are in Kernel.hpp/jit.cpp, the runtime side).

## Build commands (gfx90a, headless)
See plan.md "Build commands". Cap `-j 16`. CPU backend stays ON as the in-process reference for
the gtest CPU-vs-GPU diffs.

## Validatable core (first `ported`)
arith/JIT, reduce/scan/ireduce, sort/sort_by_key/set, blas/dot/matmul, fft/fft_real,
random/rng_quality, transpose/reorder/join/moddims, index/assign/lookup. Deferred (documented):
cudnn-gated convolveNN (MIOpen), graphics/imageio (Forge/FreeImage), nonfree sift/gloh, the CV
kernels (fast/harris/orb/sift).

## Session 1 progress (porter) -- what landed, what remains

Fork HEAD pushed: jeffdaily/arrayfire @ moat-port, commit 260285364a006 ([ROCm] Add
HIP backend scaffold; port runtime-JIT engine to hipRTC). 372 files (the cloned backend +
two CMake edits), 996 lines of genuine port diff on top of the byte-identical clone.

LANDED and VERIFIED (GPU on gfx90a, GCD 0):
- Phase 0 hipRTC go/no-go (above) -- GO. Both arrayfire JIT paths proven end to end.
- compile_module.cpp engine restructure (NVRTC+PTX+cuLink -> hipRTC+code-object). The exact
  ported call sequence re-verified standalone (agent_space/af_hiprtc_poc/engine_check.cpp:
  hiprtcCreateProgram with non-empty headers -> --offload-arch=<stripped gcnArchName> ->
  hiprtcGetCode -> hipModuleLoadData, with the ROCM_PATH fallback). PASS.
- kernel/shfl_intrinsics.hpp wave64 fix (64-bit FULL_MASK + _sync intrinsics). Compiled and
  launched on gfx90a (agent_space/af_hiprtc_poc/shfl_check.hip). PASS.
- nvrtc_shims/ (cuComplex.h/cuda_fp16.h(.hpp)/math_constants.h/vector_types.h/
  vector_functions.h HIP substitutes); hip_compat.h (driver/runtime/complex aliasing);
  Module.hpp include swap; kernel/config.hpp kWarpSize abstraction.
- Top-level CMake AF_BUILD_HIP wiring + test harness "cuda"-tag; src/backend/hip/CMakeLists.txt
  wires the JIT-header codegen with the HIP shims.

NOT YET DONE (the remaining work before the backend compiles + the gtest core GPU-validates).
The honest next-wall, in dependency order:

1. device_manager.cpp (~60 CUDA refs) -- the heaviest single file. NVML device enumeration ->
   rocm-smi / hipDeviceGetAttribute; the cublas/cusolver/cufft VERSION queries and the
   compute-capability->cores table (compute2cores) need a gfx-arch analogue or stub; cudnn
   refs are AF_WITH_CUDNN-gated (OFF). platform.cpp (~20 refs) getComputeCapability /
   getDeviceProp(cudaDeviceProp->hipDeviceProp_t, has .gcnArchName/.warpSize/.major/.minor).
2. The CUDA-library swap files (~26), each a focused per-file translation with enum/signature
   deltas (PORTING_GUIDE library-swap lessons apply directly):
   - blas.cu/cublas.cpp/cublas.hpp -> hipBLAS (+ hipBLASLt for the Lt path; watch hipBLAS v2
     enums). solve.cu/cusolverDn.cpp/.hpp + cholesky/lu/qr/svd/inverse.cpp -> hipSOLVER.
   - cufft.cu/cufft.hpp/fft.cu/fftconvolve -> hipFFT (MPPI: status-enum non-1:1, guard orphan
     case labels in any cufftGetErrorString-style switch).
   - sparse*.cu/cusparse*.cpp/cusparse_descriptor_helpers.hpp -> hipSPARSE generic api (amgcl:
     prefer hipSPARSE over rocSPARSE for the persistent-descriptor generic api; watch the
     several-void*-typedef descriptor aliasing gotcha in cusparse_descriptor_helpers.hpp).
   - sort*.cu/set.cu/ThrustAllocator.cuh/ThrustArrayFirePolicy.hpp -> rocThrust/hipCUB. rmm:
     swap <thrust/system/cuda/...> -> <thrust/system/hip/...>, thrust::cuda::par ->
     thrust::hip::par, thrust::cuda_cub -> thrust::hip_rocprim in the custom policy.
   - random_engine.cu is mostly own device-code (Philox/Threefry/MT), not a curand swap.
3. Finish src/backend/hip/CMakeLists.txt: the afhip add_library with the full source list
   (mirror src/backend/cuda minus cuDNN/Forge/static-CUDA), set_source_files_properties(
   <.cu/.cuh kernel TUs> LANGUAGE HIP), HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}",
   force-include hip_compat.h on every HIP TU (target_compile_options -include
   .../hip_compat.h), link hip::host/hipblas/hipfft/hipsolver/hipsparse + ${HIPRTC_LIBRARY},
   and install/export as afcuda. Build cmd in plan.md; cap -j 16.
4. reduce.hpp / reduce_by_key.hpp wave64 staging (DEFERRED ON PURPOSE -- do with GPU
   validation, not blind). The kernels use cub::WarpReduce (hipCUB's is wave-width-aware) but
   surround it with hardcoded `tidx<32`, `s_ptr[tidx+32]`, `nwarps=THREADS/32`, `tid/32`,
   `tid%32`, `tid<32`, `==31`. Rework to kWarpSize (now in config.hpp) so the staging is
   correct on wave64; the s_val[nwarps] sizing and the second-stage `if(tid<32)` fold are the
   hazards. Validate with the `reduce`/`scan`/`ireduce` gtests AND a fixed-seed run-to-run
   determinism check (MPPI lesson) once the backend builds -- applying it unvalidated risks a
   per-arch hack that ping-pongs platforms.
5. textures: LookupTable1D.hpp + interp paths -- rule-of-five on hipTextureObject_t, and the
   popsift/gpuRIR float-element-read linear-filter rejection (manual lerp) if any bind uses
   cudaFilterModeLinear. Enumerate exactly when reaching the index/lookup/approx ops.
6. Deferred features (documented, not core): convolveNN/cudnn -> MIOpen (AF_WITH_CUDNN=OFF for
   the first build), graphics/imageio (Forge/FreeImage), nonfree sift/gloh, CV kernels
   (fast/harris/orb/sift/regions/homography/nearest_neighbour).

Then: build for gfx90a headless (plan.md build cmd, -j 16), iterate compile errors file by
file, and GPU-validate the core gtest subset (jit/arith/reduce/scan/sort/set/blas/fft/random/
transpose/index) on one isolated GCD via `ctest -R '^CUDA\.' --output-on-failure` run serially.
Only then -> `ported`.

## Gotchas log
- hipRTC hiprtcCreateProgram REJECTS an empty-string ("") header source body (INVALID_INPUT)
  where NVRTC accepts it; a NULL header pointer aborts in std::string ctor. arrayfire's
  compile_module.cpp passes several string("") dummy header entries -- they must be a
  non-empty placeholder on HIP. (agent_space/af_hiprtc_poc/min4.cpp isolates: "" mode FAILS,
  non-empty mode PASSES.)
- hipRTC REJECTS the NVRTC flag --device-as-default-execution-space ("unknown argument");
  drop it (HIP makes device the default execution space). Likewise map
  --gpu-architecture=compute_XX -> --offload-arch=<gcnArchName stripped of :sramecc+:xnack->,
  and drop NVRTC's --device-debug/--generate-line-info.
- hipRTC does NOT implicitly know /opt/rocm/include: an injected header whose body does
  `#include <hip/...>` fails to find it; pass -I<rocm>/include (ROCM_PATH or /opt/rocm) as a
  runtime-compile option. hipRTC DOES provide __half/__float2half/__half2float and float2/
  double2 as builtins with no header (min2.cpp), so the fp16/vector shims are belt-and-braces.
- The HIP backend keeps `namespace cuda` and builds as library afcuda (reports
  AF_BACKEND_CUDA): do NOT rename the namespace to `hip` (that would churn 354 files and the
  unified/api/test layers). The directory is hip/, the identity stays cuda. AF_BUILD_HIP is
  mutually exclusive with AF_BUILD_CUDA.
