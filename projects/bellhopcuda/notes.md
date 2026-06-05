# bellhopcuda notes

## HIP Port Summary

### Strategy
Strategy A: CUDA-to-HIP compatibility header with minimal source changes.
- Added `src/util/cuda_to_hip.h` with CUDA->HIP symbol aliases
- Added `config/hip/` with HIP CMake configuration
- Updated `__CUDA_ARCH__` guards to also handle `__HIP_DEVICE_COMPILE__`
- Used HIP atomics for device code (std::atomic not callable from HIP device)

### Build Commands
```bash
mkdir build-hip && cd build-hip
cmake .. -DBHC_ENABLE_CUDA=OFF -DBHC_ENABLE_HIP=ON \
         -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

### Validation (linux-gfx90a, MI250X)
- Ray tracing: CPU and GPU outputs identical (MunkB_ray.ray)
- TL computation: 99.9% of values within 1e-4 relative tolerance
  - Max absolute diff: 2e-8
  - This is expected floating-point variation between implementations

### Key Changes
1. `CMakeLists.txt`: Added `BHC_ENABLE_HIP` option, enable HIP language
2. `config/CMakeLists.txt`: Route to hip/ subdirectory when HIP enabled
3. `config/SetupCommon.cmake`: Mark .cu sources as LANGUAGE HIP
4. `config/hip/CMakeLists.txt`: HIP project configuration
5. `config/hip/SetupHIP.cmake`: HIP compiler flags
6. `src/util/cuda_to_hip.h`: CUDA->HIP compatibility header
7. `src/common.hpp`: Define `__HIP_PLATFORM_AMD__`, include HIP runtime
8. `src/common_setup.hpp`: Handle isinf/isnan intrinsics for HIP
9. `src/util/UtilsCUDA.cuh`: Add HIP guards for error checking
10. `src/util/atomics.hpp`: Use `__HIP_DEVICE_COMPILE__` guard
11. `src/util/errors.hpp`: Use HIP atomics in device code

## Review 2026-06-05

Strategy A port reviewed against PORTING_GUIDE patterns and AMD fault classes.

### Verified clean
- No warp intrinsics -- code stores warpSize but never uses it for kernel logic
- No textures/surfaces
- atomicAdd/atomicOr on managed memory (ErrState) -- NOT affected by gfx90a coarse-grained bug (only atomicMin/atomicMax are)
- No hardcoded warp size 32 in kernel logic
- No rule-of-five issues

### Build system
- CMAKE_HIP_ARCHITECTURES defaults only when unset; followers can override via -DCMAKE_HIP_ARCHITECTURES=<arch>
- CUDA path preserved; .cu sources marked LANGUAGE HIP only when BHC_ENABLE_HIP=ON

### Commit hygiene
- Proper [ROCm] prefix, no noreply trailer, test plan present

**Approved for validation.**
