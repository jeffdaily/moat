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

## Validation 2026-06-05

### Platform: linux-gfx90a
**GPU arch**: gfx90a (HIP_VISIBLE_DEVICES=2)
**Validated commit**: 0611e58c81772564f732d0a87c80570e8ec98619

### Build Commands
```bash
cd /var/lib/jenkins/moat/projects/cuda-efficient-features/src
git submodule update --init --recursive
cmake -B build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -DBUILD_TESTS=ON
cmake --build build -j8
```

### Test Commands
```bash
HIP_VISIBLE_DEVICES=2 ./build/tests/tests
```

### Test Results
**PASS**: 44/44 tests passed on real GPU

- BAD descriptor tests: 22/22 PASS
- HashSIFT descriptor tests: 22/22 PASS

Total runtime: 26270 ms

All tests pass on gfx90a. Wave64 shuffle divergence fixes confirmed working correctly on AMD hardware.

### Platform: linux-gfx1100
**GPU arch**: gfx1100 (AMD Radeon Pro W7800 48GB)
**Validated commit**: 0611e58c81772564f732d0a87c80570e8ec98619

### Build Commands
```bash
cd /var/lib/jenkins/moat/projects/cuda-efficient-features/src
git submodule update --init --recursive
cmake -B build-gfx1100 -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DBUILD_TESTS=ON
cmake --build build-gfx1100 -j16
```

### Test Commands
```bash
HIP_VISIBLE_DEVICES=0 ./build-gfx1100/tests/tests
```

### Test Results
**PASS**: 44/44 tests passed on real GPU

- BAD descriptor tests: 22/22 PASS
- HashSIFT descriptor tests: 22/22 PASS

Total runtime: 15784 ms

All tests pass on gfx1100 (wave32 RDNA3 arch). The port correctly handles both wave64 (gfx90a) and wave32 (gfx1100) architectures with the same code.

## Validation 2026-06-08

### Platform: windows-gfx1201
**GPU arch**: gfx1201 (AMD Radeon RX 9070 XT, RDNA4, wave32)
**HIP_VISIBLE_DEVICES**: 0 (only GPU present; gfx1101 V710 offline this session)
**Validated commit**: ebf2595 (0611e58 + Windows -fPIE guard)
**ROCm**: TheRock 7.14.0a20260604 via PyTorch venv

### Windows Source Fix Required
The porter's CMakeLists.txt used `target_compile_options($<COMPILE_LANGUAGE:HIP>:-fPIE>)` unconditionally.
clang targeting x86_64-pc-windows-msvc rejects -fPIE. Guarded it with `if(NOT WIN32)` --
Linux builds are unchanged (the flag is still applied on Linux).
Committed as ebf2595 to fork moat-port.

### Build Commands
```
VENV=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages
ROCM_DEVEL=$VENV/_rocm_sdk_devel
CLANG=$ROCM_DEVEL/lib/llvm/bin/clang++.exe
CLANG_C=$ROCM_DEVEL/lib/llvm/bin/clang.exe
OPENCV_DIR=B:/develop/opencv-install/extracted/opencv/build/x64/vc16/lib

cmake -S src -B src/build-gfx1201 -GNinja \
  -DUSE_HIP=ON \
  -DCMAKE_C_COMPILER=$CLANG_C \
  -DCMAKE_CXX_COMPILER=$CLANG \
  -DCMAKE_HIP_COMPILER=$CLANG \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DCMAKE_PREFIX_PATH=$ROCM_DEVEL \
  -DOpenCV_DIR=$OPENCV_DIR \
  -DBUILD_TESTS=ON \
  -DBUILD_SAMPLES=OFF \
  -DCMAKE_BUILD_TYPE=Release

cmake --build src/build-gfx1201 -j24
```

Build: 25/25 targets, exit 0. Warnings only (pre-existing -Wunused-* and -Winconsistent-missing-override).

### DLL Setup (runtime)
Copy into `src/build-gfx1201/tests/` before running:
- amdhip64_7.dll, amd_comgr.dll, rocm_kpack.dll, hiprtc0714.dll, hiprtc-builtins0714.dll (from _rocm_sdk_core/bin)
- hipblas.dll, rocblas.dll, rocsolver.dll, libhipblaslt.dll (from _rocm_sdk_devel/bin)
- opencv_world4110.dll (from opencv build/x64/vc16/bin)

### Test Commands
```
cd src/build-gfx1201/tests
HIP_VISIBLE_DEVICES=0 \
ROCBLAS_TENSILE_LIBPATH=<venv>/_rocm_sdk_libraries/bin/rocblas/library \
./tests.exe
```

### Test Results
**PASS**: 44/44 tests passed on real GPU

- BAD descriptor tests: 22/22 PASS
- HashSIFT descriptor tests: 22/22 PASS

Total runtime: 15467 ms

Note: rocblaslt "Cannot read TensileLibrary_lazy_gfx1201.dat" messages are printed at HashSIFT/0 startup
but are benign -- hipBLASLt lazy loading falls back to plain hipblas GEMM (hipblasGemmEx), which works
correctly. All 22 HashSIFT tests pass.

All tests pass on gfx1201 (wave32 RDNA4 arch).
