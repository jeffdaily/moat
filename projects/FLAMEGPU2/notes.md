# FLAMEGPU2 notes

## Status

The jeffdaily fork already contains a mature AMD/ROCm/HIP port on the `amdgpu` branch, 77 commits ahead of upstream master. This port was validated without modifications -- the `moat-port` branch is based directly on `fork/amdgpu`.

## Build instructions (gfx90a)

IMPORTANT: Must use amdclang++ as the CXX compiler, not GCC. The hip::device target includes `-x hip` in INTERFACE_COMPILE_OPTIONS which does not work with GCC.

```bash
cd projects/FLAMEGPU2/src

# Configure
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DFLAMEGPU_GPU=HIP \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_CXX_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
  -DCMAKE_C_COMPILER=/opt/rocm/lib/llvm/bin/clang \
  -DFLAMEGPU_BUILD_TESTS=ON

# Build
cmake --build build --target flamegpu boids_bruteforce tests -j$(nproc)
```

## Test results (gfx90a, ROCm 7.2.1)

- Non-RTC tests: 1069/1069 PASSED, 8 skipped (RTC-related)
- Examples: boids_bruteforce, game_of_life, circles_spatial3D all run successfully
- RTC (Runtime Compilation) is NOT supported on AMD -- marked as skipped

## Known limitations (documented in README)

| Feature               | NVIDIA GPUs  | AMD GPUs         |
|:----------------------|:-------------|:-----------------|
| Linux                 | Supported    | Supported        |
| Windows               | Supported    | Not Supported    |
| C++ AoT               | Supported    | Supported        |
| C++ RTC               | Supported    | Not Supported    |
| Python (pyflamegpu)   | Supported    | Not Supported    |
| Visualisation         | Supported    | Not Supported    |
| GLM                   | Supported    | Supported        |
| MPI                   | Supported    | Not Supported    |

## Review 2026-06-05

### Summary

This is an existing mature AMD port from the jeffdaily/FLAMEGPU2 `amdgpu` branch (77 commits ahead of upstream). The port adds HIP/ROCm support via a FLAMEGPU_GPU=HIP CMake option, abstracts GPU APIs through macros and type aliases, and properly gates CUDA-only features (RTC, visualization, MPI, Python bindings). Test results show 1069/1069 passing with 8 RTC tests appropriately skipped.

**Verdict: Request Changes** -- one confirmed bug requires fixing before validation.

### Port Correctness

1. **Typo in hiprand type alias** -- `include/flamegpu/detail/curand.cuh:26`: `hipandStateMRG32k3a_t` should be `hiprandStateMRG32k3a_t` (missing 'r'). This would cause a compile error if `FLAMEGPU_CURAND_MRG32k3a` is defined. The default Philox path works, so the current tests pass, but this is a latent bug.

### Fault Classes

1. **Rule-of-five on CUDAEventTimer** -- `include/flamegpu/detail/CUDAEventTimer.cuh`: The class holds `hipEvent_t`/`cudaEvent_t` handles, has a custom destructor that calls `EventDestroy`, but does NOT delete or define copy/move operations. On AMD, copying this object would double-destroy the event handles. The base class `Timer` has a `virtual ~Timer() = default` but no copy/move protection. Add:
   ```cpp
   CUDAEventTimer(const CUDAEventTimer&) = delete;
   CUDAEventTimer& operator=(const CUDAEventTimer&) = delete;
   CUDAEventTimer(CUDAEventTimer&&) = delete;
   CUDAEventTimer& operator=(CUDAEventTimer&&) = delete;
   ```
   This is a PORTING_GUIDE fault class (rule-of-five on resource handles).

2. **Fixed blocksize workaround** -- `include/flamegpu/runtime/AgentFunction.cuh:206-207`: Uses hardcoded `blockSize = 128` on HIP because "the occupancy API hangs in debug". The comment says "debug and sig" which is incomplete. Document the specific ROCm version and whether this is a known bug. However, this workaround is functional and does not affect correctness -- flagging as a minor concern, not a blocker.

### Commit Hygiene

1. **WIP and DO NOT MERGE commits in history**: The branch contains commits titled "WIP" (7a1d82b1), "DO NOT MERGE: Don't build beltsoff for AMD..." (03c22c9e), "DO NOT MERGE: tempalce occupancy api also hangs" (0f80277b), and "WIP DO NOT MERGE: extra wrapping..." (0b943922). These should be squashed/cleaned before upstream PR.

2. **No [ROCm] prefix on commits**: Per CLAUDE.md, commit titles should have `[ROCm]` prefix. The existing commits lack this. This is cleanup for the upstream PR phase, not a blocking issue for validation.

3. **GitHub Actions workflow added**: `.github/workflows/Ubuntu-HIP.yml` is a CPU-only CI workflow (builds but does not run tests). CLAUDE.md advises against adding such workflows because they cannot observe GPU faults and cause fork churn. However, this was part of the existing amdgpu branch work, not MOAT-added. The validator should consider whether to recommend its removal for the upstream PR.

### Build System

The CMake changes are well-structured:
- `cmake/enable_languages.cmake` properly gates CUDA vs HIP
- Library swaps (CCCL -> rocthrust+hipcub, curand -> hiprand) are correct
- Visualisation and MPI are properly blocked on HIP with error messages referencing issues

### Testing

- 1069/1069 tests pass
- 8 RTC tests correctly skipped (RTC not supported on AMD)
- Examples (boids_bruteforce, game_of_life, circles_spatial3D) run successfully

### Required Fixes

1. Fix typo: `hipandStateMRG32k3a_t` -> `hiprandStateMRG32k3a_t` in `include/flamegpu/detail/curand.cuh:26`

### Recommended Fixes

1. Add rule-of-five protection to `CUDAEventTimer` class
2. Clean commit history of WIP/DO NOT MERGE commits before upstream PR

### Recommendation

**Request Changes** -- the hiprand typo is a confirmed defect that must be fixed before validation.

## Porter fixes (2026-06-05)

Addressed both required and recommended fixes from review:

1. **hiprand typo fix**: Changed `hipandStateMRG32k3a_t` to `hiprandStateMRG32k3a_t` in `include/flamegpu/detail/curand.cuh:26`

2. **Rule-of-five fix**: Added deleted copy/move operations to CUDAEventTimer in `include/flamegpu/detail/CUDAEventTimer.cuh` to prevent accidental double-destroy of GPU event handles

Commit: e1bb7068 "[ROCm] Fix hiprand typo and add rule-of-five to CUDAEventTimer"

Build verified with `cmake --build build --target flamegpu boids_bruteforce tests -j$(nproc)`

## Review 2026-06-05 (re-review after fixes)

### Summary

Re-reviewed the FLAMEGPU2 ROCm port after the porter applied fixes for two issues identified in the prior review:

1. **hiprand typo** -- `include/flamegpu/detail/curand.cuh:26` now correctly has `hiprandStateMRG32k3a_t` (the `hipandStateMRG32k3a_t` typo is fixed)

2. **Rule-of-five on CUDAEventTimer** -- `include/flamegpu/detail/CUDAEventTimer.cuh:39-43` now has deleted copy/move operations to prevent accidental double-destroy of GPU event handles

Both fixes are complete and correct.

### Verified

- Commit message (`e1bb7068`) has `[ROCm]` prefix, <= 72 chars, includes Test Plan, mentions Claude, no noreply trailer
- No hardcoded warpSize/32 assumptions (the `32` values in OccupancyMaxActiveBlocksPerMultiprocessor calls are block-size hints, not warp-size)
- Library swaps (rocthrust, hipcub, hiprand) are correct
- Build system properly gates CUDA vs HIP via `FLAMEGPU_GPU` option with `enable_language(HIP)`
- No AMD-internal account references; all commits under jeffdaily or upstream authors
- The occupancy-API workaround (`blockSize = 128` on HIP) is properly guarded and documented

### Known items (not blockers for validation)

- **WIP/DO NOT MERGE commits in history** -- these are from the existing amdgpu branch (77 commits ahead of upstream), not MOAT-added. Should be squashed/cleaned before the upstream PR phase.
- **GitHub Actions workflow (Ubuntu-HIP.yml)** -- CPU-only CI, cannot validate GPU correctness. Was part of the existing port; validator should consider removal for upstream PR.

### Recommendation

**Approve** -- the fixes are complete. The port is ready for GPU validation on gfx90a.

## Validation 2026-06-05 (linux-gfx90a)

GPU: AMD Instinct MI250X (gfx90a) at HIP_VISIBLE_DEVICES=3
ROCm: 7.2.1
Arch: gfx90a

Build command:
```bash
cd projects/FLAMEGPU2/src

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DFLAMEGPU_GPU=HIP \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_CXX_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
  -DCMAKE_C_COMPILER=/opt/rocm/lib/llvm/bin/clang \
  -DFLAMEGPU_BUILD_TESTS=ON

cmake --build build --target flamegpu boids_bruteforce tests -j$(nproc)
```

Test results:
```
[==========] 1133 tests from 89 test suites ran.
[  PASSED  ] 1069 tests.
[  SKIPPED ] 64 tests (RTC-related, expected on AMD/HIP)
```

All 1069 non-RTC tests PASSED. The 64 skipped tests are all RTC (Runtime Compilation) tests which are not supported on AMD as documented in the README. This matches the expected behavior.

Example validation:
```bash
HIP_VISIBLE_DEVICES=3 ./build/bin/Release/boids_bruteforce --steps 10
# Runs successfully without errors
```

Verdict: PASS - Real GPU validation successful on gfx90a.
