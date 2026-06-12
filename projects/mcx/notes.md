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

## Validation 2026-06-05 (linux-gfx90a)

### Build

```bash
cd /var/lib/jenkins/moat/projects/mcx/src
rm -rf build && mkdir build && cd build
cmake ../src -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
    -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
    -DBUILD_MEX=OFF -DBUILD_PYTHON=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

Build time: ~40 seconds on 128-core gfx90a host.

### Test Results

GPU: AMD Instinct MI250X (gfx90a)
Command: `HIP_VISIBLE_DEVICES=0 ./test/testmcx.sh`
Result: **11 of 40 tests FAIL**

Passing tests (29/40):
- Binary/libraries/execution/version/help/info tests
- Default options
- Homogeneous domain simulation
- Cyclic boundary condition
- Isotropic/cone beam sources
- Boundary detector flags
- 2D simulation, unitinmm
- Heterogeneous domain
- Detect photon data -w flag
- Progress bar, RNG tests
- Trajectory feature -D M
- Memory access (valgrind)

Failing tests (11/40) -- ALL reflection-related:
1. cube60b benchmark (DoMismatch=true): absorbed 18.27% vs expected ~27%
2. cube60 -b 1 (manual reflection flag): absorbed 18.27% vs expected ~27%
3. cube60 -B flag (facet boundary condition): expected ~27%
4. cube60planar benchmark: fails
5. Photon detection in cube60b: fails
6. Fourier source: fails
7. Pencil array source: fails
8. Saving photon seeds: fails
9. Photon replay -E flag: fails
10. Photon replay: fails
11. JSON dump with volume from builtin example: fails

### Root Cause Investigation

Verified the following are CORRECT:
1. Template parameter `isreflect=1` for reflection-enabled kernels (switch case 1000 at line 3782)
2. Runtime constant `gcfg->doreflect=1` read correctly from constant memory (confirmed via printf)
3. Constant memory transfer via `hipMemcpyToSymbol` works (aliased correctly in cuda_to_hip.h)
4. Struct layout: `MCXParam` has correct alignment, `float3` is 12 bytes in both host and device code
5. Host-side configuration: `cfg->isreflect=1` for cube60b, correctly copied to `param.doreflect`

The configuration flags are correct. The bug is in the PHYSICS: reflection/transmission decisions or Fresnel coefficient calculations produce wrong results on HIP compared to CUDA.

Possible causes (not yet isolated):
1. Subtle floating-point precision difference in Fresnel equation (lines 2748-2762)
2. RNG stream divergence causing different reflection/transmission decisions (line 2775: `rand_next_reflect(t) > Rtotal`)
3. Unidentified HIP/CUDA behavior difference in `fabsf`, `sqrtf`, or division operations
4. Miscompilation of complex conditional at lines 2736-2740

### Added -ffast-math

Modified `src/CMakeLists.txt` line 49 to add `-ffast-math` to HIP flags, matching CUDA's `-use_fast_math` (line 139). This is a correctness improvement for parity but did NOT fix the reflection failures (absorption still 18.27%).

### Verdict

**VALIDATION FAILED: correctness bug in reflection physics**

Core photon transport validates, but boundary reflection gives numerically wrong results (~9% absorption error). This is not a minor numerical difference -- a Monte Carlo photon simulator with broken reflection cannot be upstreamed. The porter needs to isolate whether this is:
- A HIP compiler miscompilation (compare generated code objects)
- A floating-point math library difference (compare intermediate Rtotal values)
- An RNG divergence (compare random streams with fixed seed)
- A logic error introduced in porting (diff the reflection code path against upstream)

Recommend: debug cube60b reflection step-by-step with printf/logging to find where HIP diverges from CUDA. The reflection coefficient calculation (lines 2748-2762) and the reflection/transmission decision (line 2775) are the prime suspects.

### Platform Stats

Arch: gfx90a (MI250X)
ROCm: 7.2
Compile time: ~40s
Test time: ~3 minutes (full test suite)
Passing core tests: 29/40
Blocking failures: 11/40 (all reflection-related)

## Porter Investigation 2026-06-05

### Deep Dive into Reflection Bug

Investigated the cube60b benchmark failure (18% absorption vs expected 27%). The 9% absorption gap suggests reflection is barely helping despite being enabled.

### Findings

1. **Reflection IS happening**: Added atomic counters confirming:
   - TIR (Total Internal Reflection): ~390K events per 1M photons
   - Partial reflection (rand <= Rtotal): ~40K events per 1M photons
   - Total reflections: ~430K per 1M photons
   
2. **Fresnel coefficients are correct**: Debug output shows Rtotal ~2.5% at near-normal incidence (n1=1.37 to n2=1.0), matching theoretical ((1.37-1)/(1.37+1))^2 = 2.44%.

3. **TIR threshold is correct**: Critical angle is ~47 degrees from normal. Photons at steeper angles (cphi < 0.73) correctly trigger TIR.

4. **Forced reflection produces 99.6% absorption**: When transmission path is disabled (`if (false && ...)`), absorption jumps to 99.6%, confirming reflection mechanics work.

5. **No early exits or Russian Roulette**: Counter showed 0 early exits, 0 Russian Roulette exits. Photons escape mainly through the transmission decision after Fresnel check.

### Hypothesis

The reflected photons are not propagating correctly after reflection. Possible causes:
- Position adjustment after reflection (`mcx_nextafterf`) may place photon incorrectly
- `idx1dold` restoration may not fully revert state
- Some subtle HIP/CUDA difference in how the next iteration processes the reflected photon

### What works

- Core photon transport (cube60 ~17% matches expected)
- Fresnel coefficient calculation
- TIR detection
- Reflection velocity flip
- Transmission path and escape

### What fails

- Reflected photons don't contribute to absorption as expected
- Expected ~10% more absorption from reflection; getting ~0.6%

### Next steps

1. Compare assembly output between CUDA and HIP builds for the reflection code path
2. Add verbose logging of a single reflected photon's full trajectory
3. Check if `__float2int_rn` or `mcx_nextafterf` behave differently on AMD GPUs

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

## Reflection bug ROOT CAUSE FOUND + FIXED 2026-06-11 (linux-gfx90a)

RESOLVED. cube60b now absorbs 27.26% (was 18.46%); all reflection/boundary/
source tests pass. Fork HEAD 0803c7c, state ported.

### Root cause: AMDGPU backend miscompiles branch-selected in-place float negate

The reflection branch in mcx_main_loop (mcx_core.cu ~line 2807) flips the one
velocity component normal to the struck face, selected at runtime by flipdir[3]:

    (flipdir[3]==0)?(v.x=-v.x):((flipdir[3]==1)?(v.y=-v.y):(v.z=-v.z));

The ROCm clang AMDGPU backend MISCOMPILES this at -O1/-O2/-O3 (correct only at
-O0): the generated control flow stores the UNMODIFIED component and drops the
negation. So a reflected photon kept its outward velocity and escaped on the
next step instead of bouncing back in -- reflection added almost no absorption.

This is NOT UB and NOT aliasing: -fno-strict-aliasing does not help; a minimal
standalone reproducer (a plain {float x,y,z,nscat} __align__(16) struct, one
__global__, one runtime branch arg) reproduces it with zero aliasing/uninit.
It reproduces for the ternary, if/else, temp-variable, and *=-1 spellings --
ANY form where the negated component is chosen INSIDE a runtime branch. An
UNCONDITIONAL whole-vector negate (v.z=-v.z with no enclosing branch) compiles
correctly, as does a BRANCHLESS per-component sign multiply.

Minimal repro (gfx90a, ROCm 7.2.1, hipcc -O3): k = {if(fd==0)v.x=-v.x; else
if(fd==1)v.y=-v.y; else v.z=-v.z;} returns the input z unchanged for fd==2.

### Fix (mcx_core.cu, branchless, arch-unified, CUDA-bit-identical)

    v.x *= (flipdir[3]==0)?-1.f:1.f;
    v.y *= (flipdir[3]==1)?-1.f:1.f;
    v.z *= (flipdir[3]==2)?-1.f:1.f;

All three multiplies execute unconditionally; each picks its own sign. Verified
correct for fd=0,1,2 in isolation and in the full sim. Arithmetically identical
to the original on CUDA, so the shared code path is safe on both backends.

### Diagnostic method (how it was localized)

Instrumented atomic counters showed reflections DID happen (~1.79M/1M photons)
but step count barely rose (198M no-reflect -> 208M with-reflect, only ~5 extra
steps per reflection). A single-photon pre/post printf showed v.z IDENTICAL
before and after the flip statement -> isolated to the negation -> minimal
standalone repro -> assembly (-S) confirmed the backend emits a store of the
unmodified component for the fd==2 path.

### Validation 2026-06-11 (AMD Instinct MI250X, gfx90a, ROCm 7.2.1)

Physics benchmarks (all match expected):
- cube60 (no reflect): 17.72% (~17%)
- cube60b (reflect):    27.26% (~27%)  [was 18.46%]
- cube60 -b 1:          27.26% (~27%)
- cube60planar:         25.52% (~25%)
- spherebox:            10.98% (~11%)
- skinvessel:           39.80% (~39%)
- Determinism: same seed -> identical 27.39688% across two runs

Test suite (test/testmcx.sh): 36/40 pass (was 29/40). ALL reflection/boundary/
source tests now pass. The 4 remaining failures are HOST-side/test-staleness,
NOT port or GPU bugs:
1. "dump json input with volume": test greps an EXACT zlib base64 string
   (eAHs3YuCo7iSBNC); our zlib emits a valid but different header (eJzs...).
   Host zmat/zlib compression artifact, arch-independent.
2. "saving photon seeds": greps for "after encoding: 13x.x%"; we get 129.1%.
   Same host-zlib compression-ratio brittleness.
3+4. "photon replay -E" / "photon replay": the test invokes
   replaytest_detp.JDAT but the fork's own upstream commit 6d7a81a renamed the
   output to .JDT, so the file is replaytest_detp.jdt. Run with the correct
   .jdt extension, replay is CORRECT: simulated==detected (3002==3002),
   absorbed 35.63% (in the expected 30-38% band). A stale upstream test, not a
   port issue. Left testmcx.sh unmodified (it is upstream's).

## Review 2026-06-12 (linux-gfx90a, reviewer)

Reviewed fork moat-port @ 0803c7c vs base 6d7a81a (2 commits: 0a54e6d port,
0803c7c reflection fix). Strategy A correctly applied (one cuda_to_hip.h compat
header, .cu marked LANGUAGE HIP, CUDA path preserved in the else branch). Fault
classes clean: no warpSize/32 hardcodes, no warp intrinsics, no texture/RAII
handles, no library swaps, no OOB neighbor reads in the diff. Vector-type ABI
handled (float4/uint4/int4 __align__(16); float3 stays 12B to match HIP).
Commit hygiene compliant ([ROCm] titles <=72 chars, Claude named, no noreply
trailer, no AMD-internal accounts, ASCII). Verdict: review-passed; the items
below are non-blocking and can be addressed at validation.

Findings (non-blocking):

1. Missing fast-math parity. CMakeLists.txt:49 sets CMAKE_HIP_FLAGS without
   -ffast-math, while the CUDA path (CMakeLists.txt:139) uses -use_fast_math.
   The 2026-06-05 validation note claims -ffast-math was added "to line 49 for
   parity" but it is NOT in HEAD -- it was reverted or never landed. This is a
   numerical-parity gap (transcendentals/division contraction differ between the
   two GPU builds). It did not block the physics benchmarks (all match
   expected), so it is not a correctness blocker, but the notes and the code
   disagree; either add the flag for parity or remove the stale note.

2. Root-cause framing slightly over-broad vs the fix. The 0803c7c message states
   the AMDGPU miscompile hits "any form where the negated component is chosen
   inside a runtime branch" (ternary/if-else/temp/*=-1). Yet the position-update
   ternaries immediately below the fix (mcx_core.cu:2819-2823 p.x/p.y/p.z =
   mcx_nextafterf(...) and :2824 flipdir[N] = floorf(...)) are the SAME
   runtime-branch-selected in-place-store pattern, left unchanged, and all 36
   passing tests exercise that path correctly. So the actual miscompile is
   narrower than the message implies -- specific to the in-place NEGATION, not
   branch-selected stores in general (branch-selected stores are pervasive in
   this kernel, e.g. :2384-2392, and work). The fix is correct and sufficient;
   the message's generalization is just imprecise. No code change required.

3. add_compile_definitions(USE_HIP) (CMakeLists.txt:52) is directory-scoped and
   sits before add_subdirectory(zmat) (:59), so USE_HIP leaks into the
   third-party zmat compile. Harmless today (zmat does not include the mcx
   vector/compat headers) but broader than necessary; prefer
   target_compile_definitions on the mcx targets.

4. pmcx (Python) and mcxlab (MEX) HIP targets are wired in CMake but were not
   built/validated (plan open question 1). Out of scope for review; validator
   need not gate on them, but the build-ability of those targets under USE_HIP
   is unverified.
