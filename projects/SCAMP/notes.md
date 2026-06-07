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

## Review 2026-06-05

### Summary
Strategy A port enabling AMD GPU support via HIP with cuda_to_hip.h compat header, wave64 warp reduction fix for SUM_THRESH profile, and CMake-based shfl variant skipping. The port is sound with two minor issues to address.

### Findings

1. **MOAT vocabulary in commit message and CMake comment** (must fix)
   - Commit message uses "Strategy A approach" -- remove MOAT-internal term
   - `src/core/gpu_kernel/CMakeLists.txt:93`: "follower platforms pass their own arch" -- reword to "other platforms pass their own arch via CMAKE_HIP_ARCHITECTURES"

2. **Duplicated symbol mappings in host .cpp files** (acceptable, document)
   - Several host .cpp files (`qt_helper.cpp`, `tile.cpp`, `common.cpp`, `main.cpp`, `scamp_interface.cpp`, `autotune_bench.cpp`, `device_props.cpp`, `kernel_config.cpp`) define their own CUDA->HIP macros rather than including cuda_to_hip.h
   - This is necessary because cuda_to_hip.h includes hipFFT/hipCUB headers guarded by `__HIPCC__` which fail host compilation
   - Current approach works but creates maintenance burden; consider documenting this pattern in cuda_to_hip.h

### Verified Correct

- Wave64 warp reduction: strides 32,16,8,4,2,1 with `__GFX9__` guard, 0x3f lane mask
- Shfl variant skip via CMake `continue()` is valid given fundamental 32-lane assumptions
- kWarpSize abstraction follows PORTING_GUIDE (`__GFX9__` device, upper bound for host)
- Multi-arch compatible CMAKE_HIP_ARCHITECTURES
- Library swaps: cuFFT->hipFFT, CUB->hipCUB
- hip::hipcub linked PRIVATE (correct)
- CUDA path preserved, USE_HIP defaults OFF

### Recommendation
changes-requested: fix MOAT vocabulary, then ready for validation.

## Fixes (2026-06-05)

Addressed reviewer findings:
1. Removed "Strategy A" jargon from commit message -- reworded to describe the approach directly
2. Changed CMakeLists.txt comment from "follower platforms pass their own arch" to "other platforms pass their own arch via CMAKE_HIP_ARCHITECTURES"

Amended commit and force-pushed to 58f2e7edac7f1a7f9a7c08ede18dc6e0cf714466.

## Validation 2026-06-05 (linux-gfx90a)

SHA: 58f2e7edac7f1a7f9a7c08ede18dc6e0cf714466
GPU: AMD Instinct MI210 (gfx90a)

### Build
```bash
cd /var/lib/jenkins/moat/projects/SCAMP/src
cmake -B build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DBUILD_SCAMP_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```
Build: PASS (4.6MB binary)

### Test
```bash
cd test && bash run_tests.sh ../build/SCAMP /tmp/scamp-results.txt ""
```

Results: All 52 tests PASSED
- Self-join tests: 19 (randomwalk 8K/16K/32K/64K/64K_nan with various tile sizes)
- Aligned AB-join tests: 19 (same datasets, aligned mode)
- AB-join tests: 14 (cross-dataset joins 16K vs 32K, with keep_rows variants)

Matrix profile accuracy:
- Max MP value difference: 2.24e-06 (across all tests)
- MP index differences: 0-1 indices per test (acceptable for floating-point)

All tests completed successfully with "All Tests Passed!" confirmation.

## Validation 2026-06-05 (linux-gfx1100)

SHA: 58f2e7edac7f1a7f9a7c08ede18dc6e0cf714466
GPU: AMD Radeon Pro W7800 (gfx1100)

### Build
```bash
cd /var/lib/jenkins/moat/projects/SCAMP/src
git submodule update --init --recursive
cmake -B build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DBUILD_SCAMP_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```
Build: PASS (4.6 MB binary)

### Test
```bash
cd test && bash run_tests.sh ../build/SCAMP /tmp/scamp-gfx1100-results.txt ""
```

Results: All 52 tests PASSED
- Self-join tests: 19 (randomwalk 8K/16K/32K/64K/64K_nan with various tile sizes)
- Aligned AB-join tests: 19 (same datasets, aligned mode)
- AB-join tests: 14 (cross-dataset joins 16K vs 32K, with keep_rows variants)

Matrix profile accuracy:
- Max MP value difference: 2.24e-06 (across all tests)
- MP index differences: 0-1 indices per test (acceptable for floating-point)

All tests completed successfully with "All Tests Passed!" confirmation.

## Validation 2026-06-07 (linux-gfx90a, SUM_THRESH coverage)

SHA: 58f2e7edac7f1a7f9a7c08ede18dc6e0cf714466
GPU: AMD Instinct MI250X (gfx90a), HIP_VISIBLE_DEVICES=2

Closes deferred item `scamp-sumthresh-validation`: the initial validation ran only the default 1NN_INDEX profile via run_tests.sh; the wave64 SUM_THRESH warp reduction fix in kernels_compute.h was not exercised on real GPU.

### Test: SUM_THRESH GPU vs CPU oracle

```bash
SCAMP=/var/lib/jenkins/moat/projects/SCAMP/src/build/SCAMP
INPUT=/var/lib/jenkins/moat/projects/SCAMP/src/test/SampleInput/randomwalk8K.txt

# GPU run (gfx90a wave64 SUM_THRESH reduction path)
HIP_VISIBLE_DEVICES=2 $SCAMP \
  --window=100 --input_a_file_name=$INPUT \
  --profile_type=SUM_THRESH --threshold=0.5 \
  --max_tile_size=2000000

# CPU reference (same parameters, --no_gpu)
$SCAMP \
  --window=100 --input_a_file_name=$INPUT \
  --profile_type=SUM_THRESH --threshold=0.5 \
  --max_tile_size=2000000 --no_gpu --num_cpu_workers=1
```

Tested thresholds: 0.0, 0.125, 0.5 (all three thresholds from the Python test suite).

Results (8093-element SUM_THRESH output, randomwalk8K window=100):

| threshold | GPU non-zero | CPU non-zero | max |GPU-CPU| | verdict |
|-----------|-------------|-------------|-----------------|---------|
| 0.0       | 8093        | 8093        | 0.00e+00        | PASS    |
| 0.125     | 8093        | 8093        | 0.00e+00        | PASS    |
| 0.5       | 8082        | 8082        | 0.00e+00        | PASS    |

GPU output is bit-identical to the CPU reference across all thresholds. The wave64 SUM_THRESH warp reduction (strides 32,16,8,4,2,1 under `__GFX9__` with SCAMP_FULL_WARP_MASK=0xffffffffffffffff) produces correct results on gfx90a.
