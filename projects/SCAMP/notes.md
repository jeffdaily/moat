# SCAMP notes

## Port Summary (linux-gfx90a)

Strategy A port with cuda_to_hip.h compatibility header.

### Key Changes

1. **cuda_to_hip.h**: Main compat header with:
   - kWarpSize: 64 for __GFX9__ (CDNA), 32 otherwise (RDNA)
   - SCAMP_FULL_WARP_MASK: 64-bit (0xffffffffffffffffULL) for HIP
   - CUDA->HIP runtime/FFT/CUB symbol aliases
   - hipCUB/hipFFT includes guarded with `#if defined(__HIPCC__)`

2. **Wave64 warp reduction fix** (kernels_compute.h): SUM_THRESH profile's warp reduction extended to cover 64 lanes with strides 32,16,8,4,2,1 on __GFX9__, using 0x3f lane mask for lane0 check.

3. **Shfl variants skipped on HIP**: The cov-shuffle (shfl) kernel variants (ur==0) are fundamentally tied to 32-lane warps. CMake skips them on HIP builds via `continue()` in the variant foreach.

4. **Library swaps**: cuFFT -> hipFFT, CUB (DeviceMergeSort) -> hipCUB/rocPRIM.

5. **Host-side CUDA->HIP mappings**: Added to multiple .cpp files (tile.cpp, scamp_interface.cpp, autotune_bench.cpp, main.cpp, qt_helper.cpp, common.cpp) for runtime calls like cudaSetDevice, cudaMemsetAsync, cudaGetDeviceCount, etc.

### Build

```bash
cmake -B build -DUSE_HIP=ON
cmake --build build -j$(nproc)
```

### Test

```bash
cd test && bash run_tests.sh ../build/SCAMP /tmp/results.txt ""
# All Tests Passed!
```

### Gotchas

- hipCUB/rocPRIM headers contain device intrinsics that fail in host C++ compilation; they must be included only in HIP-compiled translation units (guarded with `#if defined(__HIPCC__)`)
- hip::hipcub must be linked PRIVATE to avoid propagating HIP compile options to downstream C++ targets
- The shfl kernel variants would need significant redesign for wave64 (they use 32-bit shuffle masks and assume 32-lane warps); easier to skip them and rely on the sliding-window variants which work on both wave32 and wave64
