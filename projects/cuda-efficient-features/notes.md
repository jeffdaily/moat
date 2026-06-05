# cuda-efficient-features notes

## Port Status (2026-06-05)

Port complete. All 44 tests pass on linux-gfx90a.

### What Works
- HIP compilation of all .cu files (Strategy A: cuda_to_hip.h compat header)
- cuBLAS -> hipBLAS swap in cuda_hash_sift.cpp
- Thrust -> rocThrust swap (thrust::cuda::par -> thrust::hip::par)
- Custom HIP kernels for OpenCV CUDA replacements (Gaussian blur, resize, integral image)
- Custom HIP-native GpuMat class that replaces OpenCV's GpuMat for memory management
- HIP-native Stream and HostMem wrappers
- All 20 source files compile successfully
- BAD descriptor tests (22 cases): PASS
- HashSIFT descriptor tests (22 cases): PASS

### Key Fixes (from previous blocked state)

Two root causes for test hangs and incorrect results:

1. **Wave64 shuffle divergence**: AMD gfx90a uses 64-lane waves, and `__shfl_xor_sync` requires ALL threads in the wave to participate. CUDA's 32-lane warp allows threads to skip shuffles via early return, but wave64 hangs if any lanes exit before the shuffle. Fixed by moving shuffles outside conditionals so all threads participate, with inactive threads contributing zero and ignoring the result.

2. **Column vector stride**: GpuMat::create aligned step to 256 bytes even for single-column matrices. This broke array indexing when kernel code treats keypoints as a flat float4 array (keypoints[i]) rather than using row-indexed access. Now single-column matrices use natural stride.

Affected kernels:
- computeBADKernel: Move keypoint load inside bounds check to avoid OOB read, move shuffles outside the kpIdx < nkeypoints guard
- normalizeDescriptors (HashSIFT): Keep computation under threadIdx.y == 0 guard, but all threads participate in the reduction shuffle

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

## Review 2026-06-05

### Summary
Clean port implementing HIP/ROCm support for cuda-efficient-features, a CUDA keypoint detection and descriptor extraction library. The port correctly uses Strategy A (compat header + LANGUAGE HIP), fixes wave64 shuffle divergence, and adds a HIP-native GpuMat to replace OpenCV CUDA dependency.

### Findings
No blocking issues found.

### Verified
1. **Port Strategy**: Strategy A (compat header) correctly applied; CMake properly gates USE_HIP, marks .cu files as LANGUAGE HIP, swaps cuBLAS -> hipBLAS
2. **Wave64 shuffle fix (cuda_bad.cu:321-325, cuda_hash_sift.cu:199-201)**: Correct -- shuffles moved outside conditionals so all 64 lanes participate; inactive threads contribute zero
3. **FULL_WARP_MASK (cuda_to_hip.h:55)**: Correctly defined as 64-bit `0xffffffffffffffffULL` for HIP
4. **WARP_SIZE=32 in cuda_hash_sift.cu:46**: Safe -- used as loop stride over 128-element descriptor, not physical wave width; shuffle widths (1-16) work on both wave64 and wave32
5. **GpuMat column vector stride fix (cuda.hpp:220)**: Single-column matrices use natural stride (no 256B alignment) so flat array indexing works
6. **CUDA build preserved**: All else() branches maintain original CUDA path
7. **Commit messages**: Properly prefixed [ROCm], under 72 chars, mention Claude, no noreply trailers
8. **No MOAT jargon**: Code and commits clean of internal vocabulary
9. **Library swaps**: cuBLAS -> hipBLAS via abstraction layer in cuda_hash_sift.cpp; Thrust execution policy via USE_HIP guard

### Recommendation
**Approve** -- ready for validator to run GPU tests.
