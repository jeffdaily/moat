# prismatic notes

## Port summary

Prismatic is a C++/CUDA STEM simulation package that uses legacy FindCUDA (cuda_add_executable, cuda_add_library). The port follows Strategy A with a FindCUDA shim approach.

### Key changes

1. **cuda_to_hip.h compat header**: Maps CUDA runtime, cuComplex, and cuFFT symbols to HIP equivalents. Placed in `include/cuda_to_hip.h`.

2. **HIPShim.cmake**: Provides `cuda_add_executable`, `cuda_add_library`, and `cuda_add_cufft_to_target` shim macros that compile .cu sources with HIP language when USE_HIP=ON. Placed in `cmake/HIPShim.cmake`.

3. **Warp-synchronous reduction fix**: The warpReduce_cx functions in PRISM03_calcOutput.cu used NVIDIA's implicit warp lockstep assumption with volatile shared memory. On AMD wave64, this races. Fixed by adding explicit `__syncthreads()` barriers in the HIP code path (guarded with `#if defined(USE_HIP)`).

4. **Kernel launch syntax**: Fixed `<< <` and `>> >` (spaces in launch syntax) to correct `<<<` and `>>>` form.

5. **Test code fixes**: Fixed void pointer arithmetic in ioTests.cpp and added -Wno-c++11-narrowing for clang strictness.

### Build command (gfx90a)

```bash
mkdir build && cd build
cmake .. \
    -DUSE_HIP=ON \
    -DCMAKE_HIP_ARCHITECTURES=gfx90a \
    -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
    -DPRISMATIC_ENABLE_GPU=ON \
    -DPRISMATIC_ENABLE_CLI=ON \
    -DPRISMATIC_TESTS=ON \
    -DCMAKE_PREFIX_PATH=/opt/rocm \
    -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

### Validation

Both Multislice and PRISM algorithms execute successfully on gfx90a:
```bash
./prismatic -i ../SI100.XYZ -o test_ms.h5 -a m -g 1
./prismatic -i ../SI100.XYZ -o test_prism.h5 -a p -g 1
```

### Notes for followers

- Set `-DCMAKE_HIP_ARCHITECTURES=gfx1100` (or gfx1101/gfx1201) when configuring
- The warp reduction fix is arch-unified (correct on both wave64 and wave32)

## Review 2026-06-05

### Summary
Reviewed moat-port branch (dd8d6df4) against upstream master. Port uses Strategy A (compat header + HIPShim.cmake for legacy FindCUDA). The warp-synchronous reduction fix is correct and arch-unified.

### Findings

**Advisory (does not block validation):**
- Commit message contains "Strategy A (colmap model)" -- MOAT-internal jargon that must be removed before the upstream PR. This can be handled during PR-prep squash after all platforms validate.

### Verification completed

1. **Port correctness**: Strategy A correctly implemented with cuda_to_hip.h compat header and HIPShim.cmake shim macros for legacy FindCUDA
2. **Fault classes**: Wave64 reduction fixed with USE_HIP-guarded __syncthreads(); no hardcoded 32 outside of properly-guarded code; cuFFT->hipFFT mappings correct (CUFFT_INVERSE->HIPFFT_BACKWARD)
3. **Arch-unified**: The __syncthreads() fix works on both wave64 (gfx90a) and wave32 (gfx1100/gfx1101) -- extra syncs are harmless on wave32
4. **Minimal footprint**: Changes confined to new files (cuda_to_hip.h, HIPShim.cmake) and USE_HIP guards; CUDA path preserved
5. **Build system**: HIPShim.cmake correctly gates HIP language and shim macros behind USE_HIP; CMAKE_HIP_ARCHITECTURES defaults to gfx90a when unset
6. **Commit hygiene**: Title [ROCm] prefixed, <= 72 chars, body explains changes with Test Plan, Claude mentioned, no noreply trailer, jeffdaily account

### Outcome
**review-passed** -- proceed to validation
