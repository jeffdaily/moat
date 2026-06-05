# fdtd3d notes

## Build

### HIP/ROCm build (gfx90a)
```bash
cd /var/lib/jenkins/moat/projects/fdtd3d/src
mkdir build && cd build
cmake .. \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DSOLVER_DIM_MODES=DIM3 \
  -DVALUE_TYPE=d \
  -DCOMPLEX_FIELD_VALUES=ON \
  -DPRINT_MESSAGE=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
cmake --build . -j$(nproc)
```

### Test commands
```bash
# Unit test (requires device ID argument)
./Source/UnitTests/unit-test-cuda-grid 0

# 3D simulation
./Source/fdtd3d --3d --size x:20,y:20,z:20 --use-cuda --cuda-gpus 0 --time-steps 100
```

## Port notes

### HIP-specific changes
1. **Compat header**: `Source/Helpers/cuda_to_hip.h` aliases CUDA runtime APIs to HIP
2. **PAssert.h host/device handling**: HIP validates `__device__` function bodies during host compilation, unlike NVCC. We provide a unified `__host__ __device__` wrapper for `program_fail` and simplified DPRINTF that skips SOLVER_SETTINGS checks to avoid host/device symbol visibility issues.
3. **Template keyword**: Added `template` keyword before dependent template member function calls in `InternalScheme.inc.h` -- required by HIP/clang but NVCC was lenient.
4. **Separable compilation**: All HIP libraries and executables use `HIP_SEPARABLE_COMPILATION ON` for cross-TU device code linking.
5. **Test exclusions**: Some CPU unit tests require all DIM modes or MPI; they are excluded from HIP builds when those conditions are not met.

### Verified on
- AMD Instinct MI250X (gfx90a)
- ROCm 7.2.1

## Install as a dependency

N/A - fdtd3d is an end-user application, not a library.

## Review 2026-06-05

### Summary

This port adds HIP support for AMD GPUs using Strategy A (compat header + LANGUAGE HIP). It is a clean, minimal-footprint port of a pure CMake project with no warp-level primitives, no textures, and no external CUDA libraries. The port follows MOAT best practices.

### Port Correctness

No issues. The port correctly:
- Uses a single `cuda_to_hip.h` compat header with CUDA->HIP symbol aliases
- Marks `.cu` files as `LANGUAGE HIP` instead of renaming
- Adds `template` keyword for dependent template member calls (required by clang, valid C++ that nvcc also accepts)
- Handles HIP's stricter host/device code separation in PAssert.h via a three-way `#if __HIPCC__ / #elif __CUDACC__ / #else` pattern that preserves the original CUDA path
- Sets `__CUDA_ARCH__` to 600 on HIP so native `atomicAdd(double*)` is used

### Fault Classes

No issues. This project has no fault class exposure:
- No warp-level primitives (`__shfl*`, `__ballot`, etc.) -- no wave64/wave32 hazard
- No hardcoded 32 for warp size (the `32` in unit-test-cuda-grid.cu is a grid dimension)
- No textures or surfaces -- no rule-of-five or pitch alignment concerns
- No external CUDA libraries -- no library swap concerns
- No OOB neighbor reads -- the FDTD stencils are within grid bounds

### Minimal Footprint

No issues. Host C++ is untouched. Changes are confined to:
- CMakeLists.txt files (CMake USE_HIP gating)
- cuda_to_hip.h (new compat header)
- PAssert.h (HIP-specific host/device handling)
- InternalScheme.inc.h (template keyword -- valid C++, identical behavior on CUDA)
- CudaInclude.h (one line to include compat header)

### Build System

No issues. Uses `enable_language(HIP)` gated by `USE_HIP` option (default OFF). `CMAKE_HIP_ARCHITECTURES` defaults to gfx90a when unset but accepts any arch via cache variable.

### Commit Hygiene

No issues:
- Title `[ROCm] Add HIP support for AMD GPUs` is 35 chars with correct prefix
- Body explains the port, mentions Claude, has a Test Plan section
- No `Co-Authored-By: noreply` trailer
- No MOAT jargon in upstream-visible text
- No AMD-internal account references

### Backward Compatibility

No issues. The CUDA path is preserved via `elseif ("${CUDA_ENABLED}")` in CMake and `#elif defined(__CUDACC__)` in PAssert.h.

### Recommendation

**Approve** -- The port is correct, minimal, and ready for validation.

## Validation 2026-06-05

### Platform: linux-gfx90a (AMD Instinct MI250X)

Built from scratch at commit baae8b3c91db20e5a45c332500c8da2200941fe0.

### Build
```bash
cd /var/lib/jenkins/moat/projects/fdtd3d/src/build
HIP_VISIBLE_DEVICES=3 cmake .. \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DSOLVER_DIM_MODES=DIM3 \
  -DVALUE_TYPE=d \
  -DCOMPLEX_FIELD_VALUES=ON \
  -DPRINT_MESSAGE=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
cmake --build . -j$(nproc)
```
Build time: 312.8s (5.2 minutes)

### Test Results
All tests passed on real GPU hardware.

1. **GPU unit test**: `unit-test-cuda-grid 0`
   - Status: PASS
   - Duration: 0.255s
   - Tests grid operations on device

2. **3D electromagnetic simulation**: 
   ```
   ./Source/fdtd3d --3d --size x:20,y:20,z:20 --use-cuda --cuda-gpus 0 --time-steps 100
   ```
   - Status: PASS
   - Duration: 0.587s (simulation runtime: 0.277s)
   - Completed all 100 time steps
   - Grid size: 20x20x20
   - Device: AMD Instinct MI250X
   - No numerical errors

### Summary
Port validated successfully on gfx90a. Both GPU tests pass with correct numerical output.
