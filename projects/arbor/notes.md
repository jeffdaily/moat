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
