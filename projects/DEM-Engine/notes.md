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
