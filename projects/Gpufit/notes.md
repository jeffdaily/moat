# Gpufit notes

## Porting 2026-05-30

Strategy A (colmap model), validated on gfx90a (MI250X, wave64), ROCm 7.2.1.

### Build classification
Pure CMake. NOT a PyTorch extension. One wrinkle vs. the CudaSift/popsift
templates: Gpufit drives the device build through the **legacy `FindCUDA`
module** (`find_package(CUDA)`, `cuda_add_library`, `cuda_add_executable`,
`CUDA_SELECT_NVCC_ARCH_FLAGS`), not modern `enable_language(CUDA)`. The legacy
machinery can't drive HIP, so the `USE_HIP` branch supplies a parallel modern
path (`enable_language(HIP)` + `add_library`/`add_executable` + `LANGUAGE HIP` +
`HIP_ARCHITECTURES`); the CUDA branch is left byte-for-byte under
`if(NOT USE_HIP)`.

### Two solver backends; ported the default (GJ)
`USE_CUBLAS` selects the per-fit linear solver:
- ON: cuBLAS batched LU (`cublasX{getrf,getrs}Batched`).
- OFF (**upstream default**): self-contained Gauss-Jordan kernel
  (`cuda_gaussjordan.cu`), no GPU math library.
Ported the default GJ path: no library swap, and -- decisively -- it has no warp
intrinsics. (hipBLAS IS installed here, so a `USE_CUBLAS=ON` HIP build via
cuBLAS->hipBLAS is a viable follow-up, but it's not the upstream default and was
out of scope for the correctness-first port.)

### Wave64 fault class: NOT EXPOSED on the default path
- No `__shfl*`/`__ballot`/`__any`/`__all`/`__activemask`/`__syncwarp` anywhere.
- No hardcoded 32 / lane masks.
- Only `warpSize` use is `info.cu:20` `warp_size_ = devProp.warpSize;` (runtime
  query, = 64 on gfx90a). Its sole consumer is `lm_fit_cuda.cu:306`
  `while ((n_hessians_per_block + 1) * n_unique_values < info_.warp_size_)`, a
  block-PACKING heuristic (more hessians/block on a wider wavefront), not a
  32-lane reduction. No popsift-class bug.
- GJ reductions are `__shared__` + `__syncthreads()`/`__threadfence()` tree
  reductions over the (small) parameter dimension -- wave-size-agnostic.
No textures/surfaces/cudaArray, no `__constant__` memory.

### Files (all HIP-guarded; CUDA path unchanged)
New (HIP build only):
- `Gpufit/cuda_to_hip.h` -- compat header; aliases only the cuda runtime
  symbols actually used (malloc/free/memcpy/memset/memGetInfo, device
  count/props, error+version queries, memcpy-kind enums). Force-included into
  HIP TUs. No warp/texture/lib mappings (none are used).
- `Gpufit/hip_compat/cuda_runtime.h`, `.../device_launch_parameters.h` -- shims
  (`#include "../cuda_to_hip.h"`) so the direct includes of those CUDA headers
  resolve on ROCm. The `../` is required: host `.cpp` TUs do NOT get the
  force-include, so the shim must locate the compat header on its own.

Edited:
- `CMakeLists.txt` (top) -- `option(USE_HIP)` declared early; raise
  `cmake_minimum_required` to 3.21 only on the HIP path; add `HIP` language to
  `project()` on HIP.
- `Gpufit/CMakeLists.txt` -- guard the FindCUDA arch block and the cuBLAS block
  with `if(NOT USE_HIP)`; HIP branch of the library does `add_library` +
  `LANGUAGE HIP` + `HIP_ARCHITECTURES gfx90a` + `USE_HIP` define + force-include
  + `hip_compat` include. Also `find_package(hip)` + link `hip::host` so the
  host `.cpp` TUs (which include `gpu_data.cuh` -> the shim ->
  `<hip/hip_runtime.h>`) get the ROCm include dirs and `__HIP_PLATFORM_AMD__`.
- `examples/c++/CMakeLists.txt` -- `add_cuda_example` builds the device-API
  `.cu` example as a HIP TU under `USE_HIP`; the 3 pure-`.cpp` examples are
  untouched.
No source-logic edits were needed.

GOTCHA (build-system, cost ~1 iteration): host `.cpp` TUs include `gpu_data.cuh`
(the `Device_Array` template body uses `cudaMalloc`/`cudaError_t`). They are
compiled as CXX (gcc), so they need the HIP runtime headers + symbol aliases
too. Fixed by linking `hip::host` (carries `/opt/rocm/include` + the AMD
platform define) AND by making the shim include `../cuda_to_hip.h` (the
force-include only fires for `COMPILE_LANGUAGE:HIP`, never for the host TUs).
libGpufit.so confirmed: `.hip_fatbin` with `gfx90a` code objects, NEEDED
`libamdhip64.so.7`, undefined `hip*` symbols.

### Validation (gfx90a, HIP_VISIBLE_DEVICES=3)
Build: clean (100%), `-DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a
-DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_BUILD_TYPE=Release
-DUSE_CUBLAS=OFF`.
All 9 Boost known-answer suites PASS ("No errors detected", exit 0):
Error_Handling, Linear_Fit_1D, Gauss_Fit_1D, Gauss_Fit_2D,
Gauss_Fit_2D_Elliptic, Gauss_Fit_2D_Rotated, Cauchy_Fit_2D_Elliptic,
Fletcher_Powell_Helix_Fit, Brown_Dennis_Fit. These assert the LM fit recovers
the true params (e.g. Gauss_1D `{4,2,0.5,1}` to 1e-6) with chi^2 < 1e-6 -- i.e.
it CONVERGES correctly on AMD, not NaN/divergent.
Cpufit_Gpufit_Test_Consistency PASS (HIP vs CPU agree). Simple_Example and the
HIP-compiled CUDA_Interface_Example run end-to-end (10000-fit MC 2D Gaussian:
99.99% converged, fitted means match truth, ~1.66 mean iters).
Determinism: known-answer suites pass on every repeat; CUDA_Interface_Example
prints byte-identical means/chi^2 across runs.

Result: linux-gfx90a -> ported. No blockers.
