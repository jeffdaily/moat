# GOMC notes

## Build

### HIP/ROCm (AMD GPUs)

```bash
mkdir build_hip && cd build_hip
cmake .. -DUSE_HIP=ON -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_HIP_ARCHITECTURES=gfx90a
cmake --build . -j$(nproc)
```

For other AMD architectures, change `-DCMAKE_HIP_ARCHITECTURES` (e.g., gfx1100 for RDNA3).

### CUDA (NVIDIA GPUs)

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Test

Run a quick validation with the AR_KR test system:
```bash
cd test/input/Systems/AR_KR/SingleRun
HIP_VISIBLE_DEVICES=0 ./GOMC_GPU_GEMC in.conf
```

## Porting notes

### Random123 library

The bundled Random123 library needed several defines for HIP device code:
- `R123_CUDA_DEVICE __device__` -- mark array methods as device functions
- `R123_THROW(x) abort()` -- disable exceptions in device code
- `R123_USE_SSE 0` etc. -- disable x86 SSE intrinsics on GPU
- Use `r123::double2` explicitly to avoid collision with HIP's `double2` type

These defines are placed in `TransformParticlesCUDAKernel.{cu,cuh}` before including Random123 headers.

### hipCUB flag pollution

The `hip::hipcub` CMake target propagates `-x hip --offload-arch=<arch>` flags to all compilation units via `hip::device`. This causes g++ (the CXX compiler) to fail on host .cpp files. The workaround is to manually include hipCUB/rocprim directories and link only `amdhip64`, avoiding the imported target's flag pollution.

### HIP separable compilation

The port uses `HIP_SEPARABLE_COMPILATION ON` and `-fgpu-rdc` to allow `__device__` functions defined in one translation unit to be called from another (e.g., `CalcEnGPU` defined in CalculateEnergyCUDAKernel.cu and called from CalculateForceCUDAKernel.cu).
