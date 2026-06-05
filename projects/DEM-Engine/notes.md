# DEM-Engine notes

## Port status

Build succeeded with JitKernel abstraction for HIP runtime compilation.

### Summary

DEM-Engine uses NVIDIA's Jitify library for runtime kernel compilation (NVRTC). The project compiles 43 kernel files at runtime using Jitify. ROCm's Jitify fork only supports the incompatible v2 API, so a unified abstraction layer was implemented.

### JitKernel Abstraction

**New files:**
- `src/core/utils/JitKernel.h` - Unified API header
- `src/core/utils/JitKernel.inl` - Template inline implementations  
- `src/core/utils/JitKernel_cuda.cpp` - CUDA backend wrapping jitify v1
- `src/core/utils/JitKernel_hip.cpp` - HIP backend using hiprtc directly

**Key classes:**
- `deme::jit::ProgramCache` - replaces `jitify::JitCache`
- `deme::jit::Program` - replaces `jitify::Program`
- `deme::jit::Kernel` - kernel handle with `instantiate()` method
- `deme::jit::KernelLauncher` - fluent launch builder with `configure()` and `launch()`

**API compatibility:**
The abstraction preserves the existing call pattern:
```cpp
program->kernel("name").instantiate().configure(...).launch(args...)
```

### Additional HIP fixes

1. **CUDAMathHelpers.cuh** - Added `#ifndef __HIP_PLATFORM_AMD__` guards around vector operators that conflict with HIP's built-in `HIP_vector_type` operators

2. **DEMCubWrappers.cu** - Added `CUB_RUNTIME_FUNCTION` macro definition for HIP (not provided by hipCUB)

3. **dT.h** - Added missing `#include <unordered_set>`

4. **CMakeLists.txt** - Added `__HIP_PLATFORM_AMD__` compile definition for CXX files including HIP headers

### Build commands

```bash
mkdir build && cd build
cmake .. -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

### Build status

- Configuration: SUCCESS
- Compilation: SUCCESS (with warnings)
- Linking: SUCCESS
- All 26 demo executables built

### Validation

Validation requires running demos on GPU. Suggested validation demos:
- `DEMdemo_SingleSphereCollide` - Basic contact detection
- `DEMdemo_BallDrop` - Gravity and collision
- `DEMdemo_ContactChain` - Force propagation

## Review 2026-06-05

### Summary

The port introduces a JitKernel abstraction layer to bridge jitify v1 (CUDA) and hiprtc (HIP). The architecture is sound for isolating platform differences, but there are critical correctness issues in both the HIP hiprtc integration and the caching design that will cause runtime failures.

### Port Correctness

1. **hiprtc templated kernel name mangling (JitKernel_hip.cpp:40-57)**: The `findKernelName()` function attempts to resolve kernel names by trying the plain name, then `name<type>`. This will NOT work for C++ kernels with hiprtc because hiprtc mangles names (the kernels are NOT `extern "C"`). The correct approach requires `hiprtcAddNameExpression()` before compilation and `hiprtcGetLoweredName()` after. At least two kernel launch sites use templated instantiation: `kT.cpp:373` with `"deme::DEMDataKT"` and `dT.cpp` with `"deme::DEMDataDT"`. These will fail at runtime with `hipModuleGetFunction` unable to find the mangled kernel name.

2. **ProgramCache::program() returns reference, buildProgram() moves from it (JitHelper.cpp:155)**: The new `ProgramCache::program()` returns `Program&` (a reference to a cached Program), but `JitHelper::buildProgram()` does `return std::move(kcache->program(...))`. This moves from the cached object, leaving it in a moved-from state. Subsequent calls for the same kernel name will return a broken Program. The original jitify's `JitCache::program()` returned by value, so this semantic changed. Fix: Either change `ProgramCache::program()` to return by value (removing caching benefit), or change `buildProgram()` to return `Program&` and not move.

### Fault Classes

3. **Hardcoded warp size 32 (src/DEM/Defines.h:32)**: `#define DEME_CUDA_WARP_SIZE 32` is used to compute `NUM_ACTIVE_TEMPLATE_LOADING_THREADS` (line 65-66), which is substituted into JIT-compiled kernels via `_nActiveLoadingThreads_`. On AMD CDNA (gfx90a), wavefront size is 64. Using 32 may cause correctness issues if the value is used for warp-level synchronization or if it affects shared memory sizing assumptions. This macro needs to be made platform-aware: 64 for CDNA, 32 for RDNA and CUDA.

### Build System

The CMake changes are correct: `enable_language(HIP)` gated properly, `CMAKE_HIP_ARCHITECTURES` defaults to gfx90a only when unset (allowing follower archs), and hipCUB/hipcub linking is correct.

### Testing

Build succeeded with all 26 demo executables. No GPU validation has run yet. The hiprtc kernel name resolution issue (finding #1) will cause runtime failures.

### Commit Hygiene

- Title format correct: `[ROCm]` prefix, under 72 chars
- No `Co-Authored-By: noreply` trailer
- Claude attribution present in body

### Recommendation

**Request Changes**

The port will fail at runtime due to the hiprtc kernel name mangling issue. The two templated kernel instantiations (`deme::DEMDataKT`, `deme::DEMDataDT`) cannot be found by `hipModuleGetFunction` without using `hiprtcAddNameExpression`/`hiprtcGetLoweredName`. Additionally, the caching design has a use-after-move bug that will cause failures when the same kernel is recompiled. The warp size hardcode is a correctness risk on gfx90a.

Required fixes:
1. Implement hiprtc name expression API for templated kernels
2. Fix the return-by-value vs return-by-reference mismatch in buildProgram/ProgramCache
3. Make DEME_CUDA_WARP_SIZE platform-aware (64 for CDNA, 32 for RDNA/CUDA)
