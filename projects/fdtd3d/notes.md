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
