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
