# mcx notes

## Build (gfx90a)

```bash
cd projects/mcx/src
mkdir build && cd build
cmake ../src -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
    -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
    -DBUILD_MEX=OFF -DBUILD_PYTHON=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

## Test

```bash
HIP_VISIBLE_DEVICES=0 ./bin/mcx -L
HIP_VISIBLE_DEVICES=0 ./bin/mcx --bench cube60 -n 1e6
```

## Validation notes

Core simulation validates correctly. Test suite: 29/40 tests pass.

Working benchmarks (verified physics results):
- cube60 (no reflection): absorbed 17.72% @ 1e7 photons -- expected ~17%
- spherebox: absorbed 10.98% @ 1e7 photons -- expected ~11%

Failing benchmarks:
- cube60b (DoMismatch=true): absorbed 18.27% @ 1e7 -- expected ~27%
  The "mismatch" flag enables internal refractive index mismatches and
  boundary reflections. With reflections, photons should bounce back into
  the medium instead of escaping, increasing total absorption. The HIP
  port shows absorption similar to the non-reflecting case, suggesting
  reflections are not being applied correctly.

The reflection logic in mcx_core.cu is complex (see the isreflect template
parameter and gcfg->doreflect paths). This needs investigation to find
where the HIP port diverges from CUDA behavior.

Other failing tests (related to reflection): cube60 -b 1, cube60 -B flags,
photon detection, saving photon seeds, photon replay.

## ABI alignment gotcha

The Config struct uses float4/uint3/float3 types. HIP's float4 is 16-byte
aligned, but a simple C struct `{float x,y,z,w}` is 4-byte aligned. This
causes the Config struct to have different sizes when compiled with gcc
vs hipcc, leading to field offset mismatches (e.g., flog at offset 520 vs
536). The fix is to add `__align__(16)` to float4/uint4/int4 definitions
in mcx_vector_types.h.

## Review 2026-06-05

### Summary

Port adds HIP support to MCX via Strategy A (compat header + LANGUAGE HIP). Changes: cuda_to_hip.h aliases, mcx_vector_types.h for ABI-compatible vector types, CMakeLists.txt USE_HIP option, mcx_core.cu float3/float4 operator guards, mcx_tictoc.c HIP timer aliases. Core photon transport validates (cube60 ~17%, spherebox ~11%). Reflection tests failing (cube60b shows 18% vs expected ~27%).

### Port Correctness

1. **Reflection test failures require investigation before validation.** notes.md documents cube60b (DoMismatch=true) showing 18.27% absorption vs expected ~27%. 11 of 40 tests fail, all related to reflection/boundary behavior. This is a significant correctness gap. The porter documented it but did not identify root cause.

   - `src/mcx_core.cu`: Reflection logic uses `gcfg->doreflect` and `isreflect` template parameter. The physics of reflection coefficient calculation (`reflectcoeff()` at line 560) looks correct, but the decision branches that apply reflection (lines 2704-2852) are complex and may have a subtle HIP/CUDA divergence. No code change was made to these paths, so divergence may be runtime-behavioral (constant memory, FP precision, or compile-time differences).

2. **Missing `-ffast-math` equivalent.** `src/CMakeLists.txt:139` -- CUDA build uses `-use_fast_math`; HIP build does not add `-ffast-math`. This could cause minor numerical differences but is unlikely to explain the 9% absorption delta in cube60b. Consider adding `-ffast-math` to HIP compile options for parity, though this alone will not fix reflection.

### Fault Classes

No violations found:
- No warp intrinsics (`__shfl*`, `__ballot`, `warpSize`) in the codebase.
- No hardcoded `32` used as warp size (the `return 32` values at lines 2901/2929 are NVIDIA SM core counts, not warp size).
- No texture objects in active use (commented out at line 219).
- No library dependencies (cuBLAS/cuFFT/etc.).
- Vector types properly 16-byte aligned in mcx_vector_types.h (lines 41/43/45).

### Minimal Footprint

- Strategy A correctly applied: single compat header, .cu marked LANGUAGE HIP, CUDA path preserved.
- `mcx_tictoc.c` has local HIP aliases (lines 41-52) instead of including cuda_to_hip.h. This is correct -- mcx_tictoc.c is compiled as plain C (not hipcc), and cuda_to_hip.h contains `__device__` functions that cannot be compiled by gcc.
- Host C++ files (mcx_utils.h, mcx_shapes.h, mcx_mie.cpp) changed only to include mcx_vector_types.h instead of <vector_types.h>. Minimal and correct.

### Build System

- `enable_language(HIP)` correctly used (CMakeLists.txt:39).
- `CMAKE_HIP_ARCHITECTURES` defaulted when unset (line 40-42); arch-unified, no per-arch hardcodes.
- `find_package(hip REQUIRED)` for CMake targets (line 43).
- CUDA path preserved in else branch (lines 130-215).

### Testing

- Core simulation validated on gfx90a (cube60 ~17%, spherebox ~11%).
- 29/40 tests pass; 11 fail (all reflection-related).
- Reflection failures are NOT blocking for review-passed but ARE blocking for validation. The validator stage must investigate and fix the reflection divergence before marking validated.

### Commit Hygiene

- Title: `[ROCm] Add HIP/ROCm support for AMD GPUs` (41 chars, compliant).
- Body explains changes, mentions Claude, has Test Plan with commands. No noreply trailer.
- No AMD-internal account references.

### Recommendation

**Approve** (for review-passed -> validation)

The port structure is correct. The compat header, CMake changes, and vector type handling follow Strategy A properly. The reflection test failures are a validation-stage concern: the porter has documented the issue and root-cause investigation belongs in validation with full GPU access. Setting review-passed allows the validator to run the full test suite and investigate the reflection divergence.

The missing `-ffast-math` flag is a minor parity gap but does not explain the reflection failures and can be addressed during validation if needed.
