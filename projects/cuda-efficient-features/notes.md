# cuda-efficient-features notes

## Port Status (2026-06-05)

The port compiles successfully for HIP/gfx90a, but fails at runtime due to OpenCV's lack of HIP support.

### What Works
- HIP compilation of all .cu files (Strategy A: cuda_to_hip.h compat header)
- cuBLAS -> hipBLAS swap in cuda_hash_sift.cpp
- Thrust -> rocThrust swap (thrust::cuda::par -> thrust::hip::par)
- Custom HIP kernels for OpenCV CUDA replacements (Gaussian blur, resize, integral image)
- Custom cuda_types.hpp with HIP-aware __host__ __device__ attributes

### What Blocks

The project uses OpenCV's `GpuMat`, `Stream`, and `HostMem` types for GPU memory management. While these types are defined in OpenCV's core headers, their runtime operations (`upload()`, `download()`, `setTo()`, `copyTo()`) call into OpenCV's CUDA backend library, which:

1. Is NOT compiled with HIP support in upstream OpenCV
2. Calls `throw_no_cuda()` at runtime when CUDA is not available

The backtrace shows `cv::cuda::GpuMat::upload()` fails in HashSIFTImpl constructor when uploading the bMatrix constant data.

### Potential Solutions

1. **Replace all OpenCV CUDA operations with raw HIP** (~500 LOC)
   - Replace `GpuMat` with raw device pointers managed by hipMalloc/hipFree
   - Replace `upload()`/`download()` with `hipMemcpy`
   - Replace `setTo()` with `hipMemset` or a custom kernel
   - This is the cleanest approach but requires significant refactoring

2. **Build OpenCV with HIP support**
   - Requires non-upstream patches
   - AMD has no official HIP-enabled OpenCV
   - rocCV (preview in ROCm 7.0) has limited modules

3. **Use AMD alternatives (RPP, MIVisionX)**
   - Different APIs, significant refactoring required

### Files Created for HIP Port

- `src/modules/cuda_efficient_features/src/cuda_to_hip.h` - CUDA/HIP compat header
- `src/modules/cuda_efficient_features/src/hip_kernels.h` and `.cu` - replacement kernels
- `src/modules/cuda_efficient_features/src/hip_compat/` - shim headers for OpenCV

### Decision Needed

Recommend Option 1 (replace OpenCV CUDA ops with raw HIP) if the port value justifies ~500 LOC of refactoring. The kernel-level code is already ported; only the host-side memory management needs replacement.
