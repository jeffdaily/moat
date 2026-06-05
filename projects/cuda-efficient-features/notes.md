# cuda-efficient-features notes

## Port Status (2026-06-05)

Port in progress. Compiles and runs on GPU but tests hang (likely kernel execution issue).

### What Works
- HIP compilation of all .cu files (Strategy A: cuda_to_hip.h compat header)
- cuBLAS -> hipBLAS swap in cuda_hash_sift.cpp
- Thrust -> rocThrust swap (thrust::cuda::par -> thrust::hip::par)
- Custom HIP kernels for OpenCV CUDA replacements (Gaussian blur, resize, integral image)
- Custom HIP-native GpuMat class that replaces OpenCV's GpuMat for memory management
- HIP-native Stream and HostMem wrappers
- All 20 source files compile successfully

### What Blocks

Tests hang with GPU at 100% utilization. This indicates a kernel stuck in an infinite loop or deadlock. Root cause investigation needed:
- Possible warp size assumptions (WARP_SIZE=32 vs wave64 on gfx90a) - reviewed and looks OK
- Possible atomic operation or synchronization issues
- Need to isolate which kernel is hanging

### Progress Made (from previous blocked state)

The OpenCV CUDA runtime dependency has been resolved by creating a complete HIP-native GpuMat replacement:
- `hip_compat/opencv2/core/cuda.hpp` - Full GpuMat class with upload/download/create/setTo/copyTo using HIP runtime
- `hip_compat/opencv2/core/cuda_stream_accessor.hpp` - StreamAccessor shim for HIP
- Modified all source files to use HIP-native types under USE_HIP guard
- Converted InputArray/OutputArray handling to avoid OpenCV CUDA backend calls

### Build Commands

```bash
cd src
cmake -B build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -DBUILD_TESTS=ON
cmake --build build -j8
```

### Test Commands

```bash
HIP_VISIBLE_DEVICES=2 ./build/tests/tests
```

### Next Steps

1. Debug kernel hang: add prints before/after each kernel launch to identify which kernel is stuck
2. Check for race conditions in shared memory usage
3. Verify PtrStepSz conversions are correct for all kernel arguments
