# cuPDLPx notes

## Build (linux-gfx90a)

```bash
cd projects/cuPDLPx/src
cmake -B build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCUPDLPX_BUILD_CLI=ON -DCUPDLPX_BUILD_TESTS=OFF -DCUPDLPX_BUILD_PYTHON=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

For other architectures:
```bash
cmake -B build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 ...  # RDNA3
cmake -B build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1101 ...  # Windows
```

## Test

```bash
# Download test LP instance
wget https://miplib.zib.de/WebData/instances/2club200v15p5scn.mps.gz

# Run solver (gfx90a MI250X)
HIP_VISIBLE_DEVICES=0 ./build/cupdlpx 2club200v15p5scn.mps.gz . -v

# Expected output:
# Status: OPTIMAL
# Primal objective: ~-121.22
# Relative residuals < 1e-4
```

## Port details

- Strategy A: cuda_to_hip.h compat header, LANGUAGE HIP in CMake
- cuBLAS -> hipBLAS, cuSPARSE -> hipSPARSE, CUB -> hipCUB
- CUDA Graph API maps cleanly to HIP Graph
- No warp intrinsics, no textures: no warp-size issues

## Gotchas

- hipCUB header is C++ only (includes `<algorithm>`); the compat header guards it
  with `#ifdef __cplusplus` so C files compile cleanly
- cusparseSpMVOp is not available in hipSPARSE; the port forces `CUPDLPX_HAS_SPMVOP=0`
  so the standard cusparseSpMV path is used (this is already the default path in upstream
  cuPDLPx unless you have CUDA 13.1+)
- cublasDnrm2_v2_64 maps to hipblasDnrm2 (32-bit size_t); LP problem sizes fit comfortably

## Review 2026-06-05

### ROCm Fault Classes

- Warp size: PASS. No warp intrinsics (`__shfl*`, `__ballot`, `__activemask`) or hardcoded 32 found. All kernels use simple linear thread indexing (`blockIdx.x * blockDim.x + threadIdx.x`).
- Rule-of-five on handles: PASS. cusparse/cublas handle creation/destruction properly paired; no texture/surface handles.
- OOB reads: N/A. No stencil/neighbor kernels; linear array indexing is bounds-checked by problem dimensions.
- Texture pitch: N/A. No texture/surface usage.
- Library swaps: PASS. cuBLAS -> hipBLAS, cuSPARSE -> hipSPARSE, CUB -> hipCUB all mapped correctly. `hipsparseSpMV_preprocess` exists in ROCm 7.x (verified in /opt/rocm/include/hipsparse).

### Strategy A Correctness

- PASS. `cuda_to_hip.h` compat header with `#if defined(USE_HIP)` guards. LANGUAGE HIP set on .cu files.
- Include order correct: utils.h includes cuda_to_hip.h first, then cusparse_compat.h. CUDA headers guarded with `#if !defined(USE_HIP)`.

### Build System

- PASS. CMake uses `CMAKE_HIP_ARCHITECTURES` with default-only-when-unset pattern (lines 45-47). Followers can override with `-DCMAKE_HIP_ARCHITECTURES=gfx1100`.
- PASS. Proper find_package for hip, hipblas, hipsparse, hipcub, rocprim.

### Commit Hygiene

- Title: PASS. `[ROCm] Add HIP/ROCm support for AMD GPUs` (40 chars, under 72).
- Co-Authored-By trailer: PASS. None present.
- Author email: PASS. jeff.daily@amd.com (not AMD-internal noreply).
- **PROBLEM**: Body contains MOAT jargon "Strategy A (compat header)". Upstream-visible text must not use MOAT vocabulary.

### Action Required

Porter must amend commit message to remove "Strategy A (compat header)" and describe the approach in plain language (e.g., "The port keeps sources in CUDA spelling; a compatibility header maps CUDA symbols to HIP at compile time.").

### Resolution (2026-06-05)

Amended commit message: replaced "Strategy A (compat header)" with "a compatibility header approach". Pushed db252232c95948f61825cf568c8a673c4e87850d to fork.

## Review 2026-06-05 (re-review)

Previous review requested removal of "Strategy A" jargon from commit message. Porter amended commit message correctly.

However, **residual jargon found in code**:

- `internal/cuda_to_hip.h:7`: Comment says "Strategy A port: keep all source files in CUDA spelling; this header handles the translation to HIP." This is upstream-visible code and must not contain MOAT vocabulary.

Porter must amend the file comment to remove "Strategy A port" and describe the approach in plain language (e.g., "Compatibility header port: sources remain in CUDA spelling; this header handles the translation to HIP.").

## Review 2026-06-05 (re-review #2)

Verified fix at b114e2dcec9f9d35165b2f9355b13016c648fbc0:

- `internal/cuda_to_hip.h:7`: Now reads "Compatibility header port: sources remain in CUDA spelling; this header handles the translation to HIP." -- MOAT jargon removed.

Full jargon search across all source files: PASS (no Strategy A/B, lead, follower, head_sha, validated_sha, moat-port, curated commit).

Commit messages: PASS (both db25223 and b114e2d use plain language, no noreply trailer).

Ready for validation.

## Validation 2026-06-05 (linux-gfx90a)

Platform: MI250X gfx90a (HIP_VISIBLE_DEVICES=0)
Commit: b114e2dcec9f9d35165b2f9355b13016c648fbc0

Build command:
```bash
cd projects/cuPDLPx/src
cmake -B build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCUPDLPX_BUILD_CLI=ON -DCUPDLPX_BUILD_TESTS=OFF -DCUPDLPX_BUILD_PYTHON=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Build result: SUCCESS (warnings only, no errors)

Test cases:
1. MIPLIB instance 2club200v15p5scn.mps.gz (17013 rows, 200 cols):
   - Status: OPTIMAL
   - Iterations: 3000
   - Primal objective: -121.2216698
   - Dual objective: -121.2221271
   - Objective gap: 1.879e-06
   - Primal infeas: 4.889e-06 (< 1e-4 threshold)
   - Dual infeas: 2.399e-05 (< 1e-4 threshold)

2. MIPLIB instance datt256.mps.gz (9873 rows, 196503 cols after presolve):
   - Status: OPTIMAL
   - Iterations: 400
   - Primal objective: 256.0024269
   - Dual objective: 256.0067475
   - Objective gap: 8.422e-06
   - Primal infeas: 4.338e-05 (< 1e-4 threshold)
   - Dual infeas: 3.397e-05 (< 1e-4 threshold)

3. MIPLIB instance 30n20b8.mps.gz (larger problem):
   - Status: OPTIMAL
   - Iterations: 60400
   - Primal objective: 1.565716771
   - Dual objective: 1.565960731
   - Objective gap: 5.905e-05
   - Primal infeas: 6.280e-07 (< 1e-4 threshold)
   - Dual infeas: 5.946e-05 (< 1e-4 threshold)

4. Infeasible problem (synthetic test):
   - Status: PRIMAL_INFEASIBLE (detected by PSLP presolver)
   - Correctly handles infeasibility detection

GPU validation:
- All tests run with HIP_VISIBLE_DEVICES=0 on MI250X gfx90a
- CUDA Graph API -> HIP Graph API: PASS (no graph-related errors)
- cuBLAS -> hipBLAS: PASS (all BLAS operations correct)
- cuSPARSE -> hipSPARSE: PASS (SpMV operations correct)
- CUB -> hipCUB: PASS (device reductions correct)
- No HIP errors (verified with ROCM_LOG_LEVEL=4)
- Presolve (CPU) functionality: PASS (no regression)

Result: PASS - All LP instances converge to OPTIMAL with correct residuals within tolerance (1e-4). GPU kernels execute correctly. No regressions in non-GPU functionality.

## Validation 2026-06-05 (linux-gfx1100)

Platform: gfx1100 RDNA3
Commit: b114e2dcec9f9d35165b2f9355b13016c648fbc0

Build command:
```bash
cd projects/cuPDLPx/src
cmake -B build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCUPDLPX_BUILD_CLI=ON -DCUPDLPX_BUILD_TESTS=OFF -DCUPDLPX_BUILD_PYTHON=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j32
```

Build result: SUCCESS (warnings only, same as gfx90a)

Test cases:
1. MIPLIB instance 2club200v15p5scn.mps.gz:
   - Status: OPTIMAL
   - Primal objective: -121.2216698
   - Dual objective: -121.2221271
   - Objective gap: 1.879e-06
   - Primal infeas: 4.889e-06 (< 1e-4 threshold)
   - Dual infeas: 2.399e-05 (< 1e-4 threshold)

2. MIPLIB instance datt256.mps.gz:
   - Status: OPTIMAL
   - Primal objective: 256.0024269
   - Dual objective: 256.0067475
   - Objective gap: 8.422e-06
   - Primal infeas: 4.338e-05 (< 1e-4 threshold)
   - Dual infeas: 3.397e-05 (< 1e-4 threshold)

GPU validation:
- All tests run with HIP_VISIBLE_DEVICES=0 on RDNA3 gfx1100
- CUDA Graph API -> HIP Graph API: PASS
- cuBLAS -> hipBLAS: PASS
- cuSPARSE -> hipSPARSE: PASS
- CUB -> hipCUB: PASS
- No HIP errors (verified with ROCM_LOG_LEVEL=4)
- Numerical results match gfx90a exactly

Result: PASS - All LP instances converge to OPTIMAL with identical numerical results to gfx90a validation. GPU kernels execute correctly on RDNA3.

## Validation 2026-06-08 (windows-gfx1201)

Platform: AMD Radeon RX 9070 XT, gfx1201 (RDNA4, wave32), Windows 11 Pro for Workstations
GPU index: HIP_VISIBLE_DEVICES=0 (only GPU present after gfx1101 V710 went offline)
Commit: 98ce76664d227a9c634e963ea928b340e189d749 (Windows build fixes on top of b114e2d)
ROCm: 7.14.0a20260604 (TheRock nightly venv)

### Windows-specific build fixes (committed as 98ce766 on moat-port)

Two issues found and fixed:

1. **mps_parser.c strtok_r error**: The CMake glob pulls `mps_parser.c` into the
   static/shared library build. This file uses `strtok_r` (POSIX, absent from
   Windows CRT). Since the CLI is already disabled on Windows (getopt.h missing),
   `mps_parser.c` is dead code there. Fix: add `if(WIN32)` guard in CMakeLists.txt
   to remove it from the C_SOURCES list on Windows.

2. **test_interface.c stale API**: Four assignments of `matrix_desc_t.zero_tolerance`
   which was removed from the struct in upstream commit 9709dfe. Caused compilation
   errors. Also added Test 9 (presolve=false) to force GPU PDLP solver execution.

### Build

```
ROCM=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel
cmake -S projects/cuPDLPx/src -B agent_space/cupdlpx_gfx1201_build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=$ROCM/lib/llvm/bin/amdclang.exe \
  -DCMAKE_CXX_COMPILER=$ROCM/lib/llvm/bin/amdclang++.exe \
  -DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/amdclang++.exe \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DCMAKE_PREFIX_PATH="$ROCM;_rocm_sdk_core" \
  -DCUPDLPX_BUILD_CLI=ON -DCUPDLPX_BUILD_TESTS=ON -DCUPDLPX_BUILD_PYTHON=OFF
# Post-configure: strip -fuse-ld=lld-link from build.ninja (7 occurrences)
sed -i 's/-fuse-ld=lld-link//g' agent_space/cupdlpx_gfx1201_build/build.ninja
cmake --build agent_space/cupdlpx_gfx1201_build --target cupdlpx_core --target cupdlpx_shared --target test_interface -j24
```

Build result: SUCCESS (cupdlpx_core.lib, cupdlpx.dll, tests/test_interface.exe built)
Note: CLI disabled on Windows (getopt.h unavailable); zlib example/minigzip skipped (internal zlib tests, irrelevant).
Note: fuse-ld=lld-link stripped from build.ninja (CMake 4.3 Windows-Clang injects it; rejects in HIP device-link mode).

### Runtime DLL setup

Copied to tests/ dir (exe-dir search beats System32):
- amdhip64_7.dll, amd_comgr.dll, rocm_kpack.dll (ROCm runtime)
- hiprtc0714.dll, hiprtc-builtins0714.dll
- hipblas.dll, hipsparse.dll, libhipblaslt.dll
- rocblas.dll, rocsparse.dll, rocsolver.dll
- PSLP.dll (from _deps/pslp-build/), zlib1.dll (from _deps/zlib-build/)
- cupdlpx.dll (from build root)

ROCBLAS_TENSILE_LIBPATH=_rocm_sdk_libraries/bin/rocblas/library set at runtime.

### Test results

```
HIP_VISIBLE_DEVICES=0 ROCBLAS_TENSILE_LIBPATH=.../rocblas/library ./tests/test_interface.exe
```

9/9 PASS (RC=0, 0.37s):
- Tests 1-4: LP solve via Dense/CSR/CSC/COO matrix formats -> OPTIMAL (primal obj=3, presolve reduces to 0 rows)
- Tests 5-8: same with warm start -> OPTIMAL (presolve reduces; warm start silently ignored as documented)
- Test 9: CSR with presolve=false -> GPU PDLP solver invoked:
  - 400 iterations on GPU
  - Primal objective: 3.000539009 (true optimum 3.0)
  - Status: OPTIMAL, objective gap 7.338e-05 (< 1e-4 threshold)
  - Primal infeas: 3.700e-05, dual infeas: 2.954e-05 (both < 1e-4)

GPU execution confirmed:
- AMD_LOG_LEVEL=3 shows hipGetDevice (device 0), hipMemcpy HostToDevice, hipMemcpyAsync HostToDevice
- hipBLAS/hipSPARSE loaded from theRock nightly (gfx1201 kernels)

Result: PASS - GPU PDLP solver runs correctly on AMD Radeon RX 9070 XT (gfx1201, RDNA4).
All matrix format variants produce consistent OPTIMAL solutions. hipBLAS SpMV,
BLAS-1 operations, and CUDA Graphs -> HIP Graphs work correctly on gfx1201.

## Revalidation 2026-06-08 (linux-gfx90a)

Platform: MI250X gfx90a (HIP_VISIBLE_DEVICES=3), ROCm 7.2.1
Commit: 98ce76664d227a9c634e963ea928b340e189d749 (head)
Previous validated: b114e2dcec9f9d35165b2f9355b13016c648fbc0

### Delta classification

Single commit b114e2dc..98ce7666 (Windows build fixes):
- CMakeLists.txt: `if(WIN32)` guard excluding mps_parser.c on Windows (no effect on Linux)
- test/test_interface.c: removes four stale `zero_tolerance` field assignments + adds Test 9 (GPU solver path)

moatlib classify: `mixed` (not arch-independent -- test changes affect Linux too). codeobj_diff on
the main library: `libcupdlpx.so: identical` (151 exports, device ISA identical). PSLP dep:
`indeterminate` (extraction failed) but byte-for-byte identical between builds (same MD5).
Per CLAUDE.md policy, `indeterminate` triggers full GPU revalidation.

### Build

```bash
cmake -S projects/cuPDLPx/src -B agent_space/cupdlpx_test_build \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCUPDLPX_BUILD_CLI=ON -DCUPDLPX_BUILD_TESTS=ON -DCUPDLPX_BUILD_PYTHON=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build agent_space/cupdlpx_test_build -j$(nproc)
```

Build result: SUCCESS (warnings only, no errors)

### Test results (HIP_VISIBLE_DEVICES=3)

test_interface (9/9 PASS):
- Tests 1-4: LP solve via Dense/CSR/CSC/COO matrix formats -> OPTIMAL (primal obj=3, presolve reduces to 0 rows)
- Tests 5-8: same with warm start -> OPTIMAL (presolve reduces; warm start silently ignored as documented)
- Test 9: CSR with presolve=false -> GPU PDLP solver invoked, 400 iterations, primal obj=3.000539 (gap 7.338e-05 < 1e-4)

CLI LP test (2club200v15p5scn.mps.gz, 17013 rows, 200 cols):
- Status: OPTIMAL
- Primal objective: -121.2216698 (matches original validation exactly)
- Dual objective: -121.2221271
- Objective gap: 1.879e-06
- Primal infeas: 4.889e-06 (< 1e-4)
- Dual infeas: 2.399e-05 (< 1e-4)
- Iterations: 3000

Result: PASS - All tests pass on gfx90a. Numerical results match original validation exactly.
No regression from the Windows-specific build fixes.
