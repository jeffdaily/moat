# DEM-Engine notes

## Port attempt 2026-06-05

### Summary

DEM-Engine uses NVIDIA's Jitify library for runtime kernel compilation (NVRTC). The project compiles 43 kernel files at runtime using Jitify, making this the defining architectural feature.

### ROCm Jitify compatibility issue

ROCm provides a Jitify fork at https://github.com/ROCm/jitify that is marked as "early-access technology preview". Investigation revealed:

1. The ROCm fork's `jitify.hpp` (v1 API used by DEM-Engine) is unchanged from NVIDIA's version and still requires `cuda.h` (CUDA Driver API header)
2. HIP support is only in `jitify2.hpp` which has a different API incompatible with v1
3. ROCm does not provide a `cuda.h` shim header
4. Windows is explicitly unsupported by ROCm jitify

### Options considered

1. **Migrate to jitify2 API**: Would require rewriting JitHelper.cpp and all kernel launch sites. The jitify2 API is different -- uses `jitify::Program` -> `jitify::ProgramCache` pattern with different configure/launch syntax.

2. **AOT kernel compilation**: Could pre-compile kernels at build time instead of runtime. However, DEM-Engine's architecture relies on runtime compilation for user-customizable force models and particle definitions (see src/kernel/DEMUserScripts/*.cu). AOT would break this key feature.

3. **cuda.h shim for Driver API**: Would need to alias ~50+ driver API symbols (cuModule*, cuFunction*, cuLaunch*, etc.). This is complex and may hit other missing functionality.

### Files modified

- CMakeLists.txt: Added USE_HIP option, FetchContent for ROCm jitify
- src/core/CMakeLists.txt: Changed to use JitifyPath variable
- src/algorithms/CMakeLists.txt: Added HIP language support, hipcub linking
- src/DEM/CMakeLists.txt: Removed CUB dependency (only algorithms needs it)
- src/core/utils/cuda_to_hip.h: New compat header (partial)
- src/core/utils/GpuError.h, GpuManager.h, CudaAllocator.hpp, ManagedMemory.hpp: Use compat header
- src/DEM/Defines.h, src/kernel/CUDAMathHelpers.cuh: Use compat header
- src/algorithms/*.cu: Added HIP/hipcub includes

### Build status

Configuration succeeds but build fails on:
1. CXX targets need `__HIP_PLATFORM_AMD__` defined when including hip_runtime.h
2. jitify.hpp requires `cuda.h` which is not available on ROCm

### Recommendation

DEM-Engine port is blocked on ROCm Jitify maturity. The jitify v1 API is not supported for HIP. Options:
- Wait for ROCm to port jitify v1 API to HIP
- Upstream port request: ask DEM-Engine maintainers to migrate to jitify2 API
- Consider AOT compilation mode for HIP (breaks customizable force models)

This is a genuine architectural mismatch, not a mechanical porting issue.
