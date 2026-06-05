# arbor notes

## Port Summary

Arbor had existing HIP support (PR #1007, April 2020) but with several issues that needed fixing for modern AMD architectures and ROCm 7.x.

## Fixes Applied

### 1. Wave-size abstraction (gpu_common.hpp)

The original code returned 64 for ALL HIP builds. This breaks RDNA devices (gfx10xx/gfx11xx) which use 32-wide wavefronts.

Fixed `threads_per_warp()` to use the `__GFX9__` predefined macro to return the correct value per-arch:
- CDNA (gfx9xx): 64-wide wavefront
- RDNA (gfx10xx/gfx11xx): 32-wide wavefront

### 2. 64-bit lane masks (hip_api.hpp, reduce_by_key.hpp, cuda_api.hpp)

HIP's `__ballot()` returns `unsigned long long` (64-bit) to support 64-wide wavefronts. The original code used 32-bit `unsigned` which truncates the upper half.

- Added `lane_mask_type` typedef (64-bit on HIP, 32-bit on CUDA)
- Updated `ballot()`, `active_mask()`, `shfl_up()`, `shfl_down()` to use 64-bit masks
- Updated `key_set_pos::key_mask` to use `lane_mask_type`
- Use `__clzll()`/`__ffsll()` for 64-bit bit operations on HIP

### 3. Double shuffle bit conversion (hip_api.hpp)

Critical bug: `shfl(double)` used `static_cast<uint64_t>(x)` which truncates the double to an integer (0.5 becomes 0), losing the value entirely.

Fixed to use `__double_as_longlong()` / `__longlong_as_double()` for bit-preserving conversion.

### 4. Masked any() emulation (hip_api.hpp)

HIP's `__any()` operates on the full wavefront regardless of active mask. The original code ignored the mask parameter.

Fixed to emulate CUDA's `__any_sync(mask, pred)` using `(__ballot(pred) & mask) != 0`.

### 5. Default architecture update (CMakeLists.txt)

Updated default `ARB_HIP_ARCHITECTURES` from `gfx906 gfx900` (Vega-era) to `gfx90a` (MI200-class).

### 6. CMake version (CMakeLists.txt)

Lowered `cmake_minimum_required` from 4.0.0 to 3.19 for broader compatibility with existing ROCm toolchains.

## Build Instructions

```bash
# In projects/arbor/src/build
cmake .. \
  -DARB_GPU=hip \
  -DARB_HIP_ARCHITECTURES="gfx90a" \
  -DARB_WITH_PYTHON=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=hipcc \
  -DCMAKE_C_COMPILER=clang

cmake --build . -j$(nproc)

# Build and run tests
cmake --build . --target unit
./bin/unit
```

## Test Results (gfx90a)

All 1182 unit tests pass, including:
- reduce_by_key tests (4 tests) - exercises warp-level reduction with various block sizes
- GPU-specific tests (63 tests) - event_stream, stack, vector, spikes, intrinsics

## Follower Platform Notes

The `threads_per_warp()` fix correctly returns 32 for RDNA (gfx10xx/gfx11xx), and the 64-bit lane mask operations are backward compatible. Followers should build and validate with their arch:

```bash
cmake .. -DARB_HIP_ARCHITECTURES="gfx1100" ...  # Linux gfx1100
cmake .. -DARB_HIP_ARCHITECTURES="gfx1101" ...  # Windows gfx1101
cmake .. -DARB_HIP_ARCHITECTURES="gfx1201" ...  # Windows gfx1201
```

## Review 2026-06-05

### changes-requested (FIXED)

**DEFECT: Generated mechanism code uses 32-bit lane mask (gpuprinter.cpp:394)**

The port correctly fixes the hand-written warp primitives in hip_api.hpp and reduce_by_key.hpp, but misses the CODE GENERATOR that emits mechanism GPU code. In modcc/printer/gpuprinter.cpp line 394:

```cpp
out << "unsigned lane_mask_ = arb::gpu::ballot(0xffffffff, tid_<n_);\n";
```

This generates code with:
1. A 32-bit `unsigned` type for `lane_mask_` (should be `arb::gpu::lane_mask_type`)
2. A 32-bit full-mask literal `0xffffffff` (should be `arb::gpu::lane_mask_type(-1)` or `0xffffffffffffffffULL`)

On wave64 devices, `arb::gpu::ballot()` returns a 64-bit value, but this generated code truncates it to 32 bits. The generated `lane_mask_` then feeds `reduce_by_key(..., lane_mask_)` which expects a `lane_mask_type` (64-bit on HIP). The upper 32 lanes of the wavefront are masked out, causing incorrect reductions in generated mechanism kernels.

**Fix applied**: In modcc/printer/gpuprinter.cpp, changed line 394 to:
```cpp
out << "arb::gpu::lane_mask_type lane_mask_ = arb::gpu::ballot(arb::gpu::lane_mask_type(-1), tid_<n_);\n";
```

This ensures generated mechanism code uses the same 64-bit mask type and full-mask pattern as the test code.

Note: The hand-written test file (test_reduce_by_key.cu) was correctly updated to use `gpu::lane_mask_type(-1)` but the code generator was not updated to match.

### Post-fix verification (gfx90a)

All 1182 unit tests pass after the fix. Generated mechanism code now uses the correct 64-bit lane mask type.

## Review 2026-06-05 (post-fix)

### Summary
The gpuprinter.cpp fix is correct. The port fixes HIP warp primitives (lane masks, wave-size abstraction, double shuffle) for CDNA/RDNA architectures. All 1182 unit tests pass on gfx90a.

### Verification

**gpuprinter.cpp fix (line 394)**:
- Changed `unsigned lane_mask_` to `arb::gpu::lane_mask_type lane_mask_` -- correct, matches the typedef in hip_api.hpp (64-bit) and cuda_api.hpp (32-bit)
- Changed `0xffffffff` to `arb::gpu::lane_mask_type(-1)` -- correct, produces full-mask for any lane_mask_type width

The generated code now matches the hand-written test code pattern in test_reduce_by_key.cu (lines 17, 117).

**Fault class checks**:
- Wave-size: `threads_per_warp()` correctly uses `__GFX9__` for CDNA (64) vs default (32) for RDNA
- Lane masks: 64-bit `unsigned long long` on HIP, 32-bit `unsigned` on CUDA
- No hardcoded 32/64 in GPU code paths (remaining 0xffffffff in codebase are in _deps/ or SIMD/CPU code)
- `__clzll`/`__ffsll` for 64-bit bit ops on HIP, `__clz`/`__ffs` on CUDA

**Build system**: Correct Strategy A (existing native HIP backend, not a from-scratch port). CMake arch default updated to gfx90a; cache variable preserved so followers can override.

**Backward compatibility**: CUDA code path unchanged except for additive `lane_mask_type` typedef.

**Commit hygiene**: [ROCm] prefix, under 72 chars, Claude disclosure, no noreply trailer, jeffdaily account.

### Recommendation
**Approve** -- ready for GPU validation.

## Validation 2026-06-05

### Platform: linux-gfx90a
**GPU**: AMD Instinct MI250X / MI250 (gfx90a)
**ROCm**: 7.x

### Build Configuration
```bash
cd /var/lib/jenkins/moat/projects/arbor/src/build
# Already configured with:
# -DARB_GPU=hip
# -DARB_HIP_ARCHITECTURES=gfx90a
# -DARB_WITH_PYTHON=OFF
# -DCMAKE_BUILD_TYPE=Release
# -DCMAKE_CXX_COMPILER=hipcc
# -DCMAKE_C_COMPILER=clang
```

### Test Results
**Command**: `HIP_VISIBLE_DEVICES=1 ./bin/unit`

**Result**: PASS
- Total tests: 1182 from 182 test suites
- Passed: 1182
- Failed: 0
- Total time: 11.754 seconds

### GPU-Specific Tests Validated
All GPU tests passed, including critical warp-level operations:

1. **reduce_by_key** (4 tests) - Exercises warp-level reductions with 64-bit lane masks
   - no_repetitions: PASS
   - single_repeated_index: PASS
   - scatter: PASS
   - scatter_twice: PASS

2. **GPU initialization** (2 tests)
   - gpu_initialisation: PASS (160ms)
   - gpu_null: PASS

3. **event_stream_gpu** (2 tests)
   - single_step: PASS
   - multi_step: PASS (7ms)

4. **spikes_gpu** (2 tests)
   - threshold_watcher: PASS (1ms)
   - threshold_watcher_interpolation: PASS (26ms)

### Port Validation
The fixes to HIP warp primitives are validated on real gfx90a hardware:
- 64-bit lane masks (`lane_mask_type`) work correctly
- Wave-size abstraction (`threads_per_warp()`) returns 64 for CDNA/gfx90a
- Double shuffle bit conversion (`__double_as_longlong()`) preserves values
- Masked any() emulation works correctly
- Generated mechanism code (gpuprinter.cpp) uses correct 64-bit masks

### Conclusion
All 1182 unit tests pass on gfx90a, including GPU-specific tests exercising warp primitives, reductions, and event handling. The port correctly handles wave64 CDNA architecture.
