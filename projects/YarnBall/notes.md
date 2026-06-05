# YarnBall notes

## Porting attempt 2026-06-05 (gfx90a)

### Files modified
- `CMakeLists.txt` - created from scratch for HIP/CUDA dual build
- `KittenEngine/cuda_to_hip.h` - CUDA-to-HIP symbol aliases
- `KittenEngine/YarnBall/YarnBall.cu` - HIP include guards
- `KittenEngine/YarnBall/sim/cosserat.cu` - HIP include guards
- `KittenEngine/YarnBall/sim/collision.cu` - HIP include guards
- `KittenEngine/YarnBall/sim/iteration.cu` - HIP include guards
- `KittenEngine/KittenEngine/KittenGpuLBVH/lbvh.cu` - HIP include guards, deviceClamp helper
- `KittenEngine/KittenEngine/KittenGpuLBVH/lbvh.cuh` - HIP include guards
- `KittenEngine/KittenEngine/includes/modules/Common.h` - conditional GPU includes
- `KittenEngine/KittenEngine/includes/modules/ComputeBuffer.h` - conditional GPU includes
- `KittenEngine/KittenEngine/includes/modules/Bound.h` - device-safe boundMin/boundMax helpers
- `KittenEngine/YarnBall/YarnBall.h` - type forward declarations
- All .cu files: fixed kernel launch syntax `<< <` -> `<<<`

### Dependencies installed
- libembree-dev (embree4, project expects embree3)
- libimgui-dev
- libstb-dev
- libcli11-dev
- libjsoncpp-dev
- Generated glad headers via `glad` Python package

### Blocking issue: GLM device code incompatibility

Ubuntu's GLM 0.9.9.8 has device code annotation issues when used with HIP:
- `glm::min/max/clamp` vectors select SIMD paths via `detail::is_aligned<Q>::value` 
- Those paths use template specializations not marked `__device__`
- Same issue affects `glm::length`, `glm::dot`, `glm::cross`, `glm::mix`, `glm::inverse`, etc.

The project uses GLM heavily in device kernels (Bound.h, Common.h, Rotor.h, SymMat.h, all simulation .cu files).

Workaround attempts:
- `GLM_FORCE_CUDA` - did not help
- `GLM_FORCE_PURE` - did not help
- `GLM_FORCE_DEFAULT_ALIGNED_GENTYPES` - did not help
- Custom `boundMin/boundMax` in Bound.h - partial fix, but glm used pervasively elsewhere

### Required to unblock
1. Use a newer GLM version (1.0.x) that has proper HIP/CUDA device annotations
2. OR wrap all GLM vector math with device-safe equivalents in device code
3. Address CUDA Graph API aliases (hipGraph*, hipStream*)

### Other issues observed
- `cuda.h` include fails in some .cpp files (hip_runtime.h requires `__HIP_PLATFORM_AMD__`)
- `fopen_s` is MSVC-only, Linux uses `fopen`
- `std::chrono::system_clock` vs `steady_clock` type mismatch in Timer/StopWatch
