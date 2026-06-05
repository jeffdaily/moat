# CubbyFlow notes

## Port attempt 2026-06-05

### Summary

The port is blocked due to a fundamental incompatibility between CubbyFlow's CUDA code design and HIP's two-pass compilation model.

### Root cause

CubbyFlow uses `#ifdef __CUDA_ARCH__` to select between different return types for template methods in CUDAArrayBase:

```cpp
#ifdef __CUDA_ARCH__
    CUBBYFLOW_CUDA_DEVICE Reference operator[](size_t i);  // Returns T&
#else
    CUBBYFLOW_CUDA_HOST HostReference operator[](size_t i);  // Returns wrapper class
#endif
```

This works with CUDA's single-pass compilation model where `__CUDA_ARCH__` is defined only during device code generation within a single translation unit.

HIP uses a two-pass compilation model:
1. Host pass: `__HIP_DEVICE_COMPILE__` is 0 (or undefined), so host declarations are visible
2. Device pass: `__HIP_DEVICE_COMPILE__` is 1, so device declarations are visible

When templates are instantiated across these passes, the compiler sees only one set of declarations depending on which pass is active. This causes host code calling `operator[]` to find only device-attributed functions (declared during the device pass's template instantiation), resulting in "call to __device__ function from __host__ function" errors.

### Attempted fixes

1. Defined `CUBBYFLOW_DEVICE_CODE` macro that works for both CUDA (`__CUDA_ARCH__`) and HIP (`__HIP_DEVICE_COMPILE__`)
2. Updated all `#ifdef __CUDA_ARCH__` checks to use `#if CUBBYFLOW_DEVICE_CODE`
3. Added compat header includes to ensure macro is defined before use
4. Various attempts to fix attribute matching between declarations and definitions

None of these fixes the fundamental issue that HIP's two-pass compilation sees the template differently in each pass.

### Required fix (out of scope for automated porting)

The proper fix requires restructuring CUDAArrayBase to either:
1. Use a unified API that works for both host and device (same return type, `__host__ __device__` attributes)
2. Use explicitly named methods like `deviceAt()` and `hostAt()` instead of overloads
3. Use SFINAE or if-constexpr based on calling context rather than preprocessor conditionals

This is a significant refactor of the template hierarchy and goes beyond the scope of a mechanical CUDA-to-HIP port.

### Files modified (not functional)

The following changes were made but do not result in a working build:
- CMakeLists.txt: Added USE_HIP option with HIP language support
- Sources/Core/CMakeLists.txt: Added HIP target handling
- Tests/CUDATests/CMakeLists.txt: Added HIP target handling
- Examples/CUDASPHSim/CMakeLists.txt: Added HIP target handling
- Includes/Core/CUDA/cuda_to_hip.h: Created compat header (CUDA-to-HIP aliases)
- Includes/Core/Utils/Macros.hpp: Updated for HIP compiler detection
- Includes/Core/CUDA/CUDAUtils.hpp: Added HIP include, guarded vector operators
- Various -Impl.hpp files: Added `CUBBYFLOW_CUDA_HOST_DEVICE` attributes to definitions
- Multiple headers: Updated to use CUBBYFLOW_DEVICE_CODE macro
