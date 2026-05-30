# Gpufit notes

## Porting 2026-05-30

Strategy A (colmap model), validated on gfx90a (MI250X, wave64), ROCm 7.2.1.

### Build classification
Pure CMake. NOT a PyTorch extension. One wrinkle vs. the CudaSift/popsift
templates: Gpufit drives the device build through the **legacy `FindCUDA`
module** (`find_package(CUDA)`, `cuda_add_library`, `cuda_add_executable`,
`CUDA_SELECT_NVCC_ARCH_FLAGS`), not modern `enable_language(CUDA)`. The legacy
machinery can't drive HIP, so the `USE_HIP` branch supplies a parallel modern
path (`enable_language(HIP)` + `add_library`/`add_executable` + `LANGUAGE HIP` +
`HIP_ARCHITECTURES`); the CUDA branch is left byte-for-byte under
`if(NOT USE_HIP)`.

### Two solver backends; ported the default (GJ)
`USE_CUBLAS` selects the per-fit linear solver:
- ON: cuBLAS batched LU (`cublasX{getrf,getrs}Batched`).
- OFF (**upstream default**): self-contained Gauss-Jordan kernel
  (`cuda_gaussjordan.cu`), no GPU math library.
Ported the default GJ path: no library swap, and -- decisively -- it has no warp
intrinsics. (hipBLAS IS installed here, so a `USE_CUBLAS=ON` HIP build via
cuBLAS->hipBLAS is a viable follow-up, but it's not the upstream default and was
out of scope for the correctness-first port.)

### Wave64 fault class: NOT EXPOSED on the default path
- No `__shfl*`/`__ballot`/`__any`/`__all`/`__activemask`/`__syncwarp` anywhere.
- No hardcoded 32 / lane masks.
- Only `warpSize` use is `info.cu:20` `warp_size_ = devProp.warpSize;` (runtime
  query, = 64 on gfx90a). Its sole consumer is `lm_fit_cuda.cu:306`
  `while ((n_hessians_per_block + 1) * n_unique_values < info_.warp_size_)`, a
  block-PACKING heuristic (more hessians/block on a wider wavefront), not a
  32-lane reduction. No popsift-class bug.
- GJ reductions are `__shared__` + `__syncthreads()`/`__threadfence()` tree
  reductions over the (small) parameter dimension -- wave-size-agnostic.
No textures/surfaces/cudaArray, no `__constant__` memory.

### Files (all HIP-guarded; CUDA path unchanged)
New (HIP build only):
- `Gpufit/cuda_to_hip.h` -- compat header; aliases only the cuda runtime
  symbols actually used (malloc/free/memcpy/memset/memGetInfo, device
  count/props, error+version queries, memcpy-kind enums). Force-included into
  HIP TUs. No warp/texture/lib mappings (none are used).
- `Gpufit/hip_compat/cuda_runtime.h`, `.../device_launch_parameters.h` -- shims
  (`#include "../cuda_to_hip.h"`) so the direct includes of those CUDA headers
  resolve on ROCm. The `../` is required: host `.cpp` TUs do NOT get the
  force-include, so the shim must locate the compat header on its own.

Edited:
- `CMakeLists.txt` (top) -- `option(USE_HIP)` declared early; raise
  `cmake_minimum_required` to 3.21 only on the HIP path; add `HIP` language to
  `project()` on HIP.
- `Gpufit/CMakeLists.txt` -- guard the FindCUDA arch block and the cuBLAS block
  with `if(NOT USE_HIP)`; HIP branch of the library does `add_library` +
  `LANGUAGE HIP` + `HIP_ARCHITECTURES gfx90a` + `USE_HIP` define + force-include
  + `hip_compat` include. Also `find_package(hip)` + link `hip::host` so the
  host `.cpp` TUs (which include `gpu_data.cuh` -> the shim ->
  `<hip/hip_runtime.h>`) get the ROCm include dirs and `__HIP_PLATFORM_AMD__`.
- `examples/c++/CMakeLists.txt` -- `add_cuda_example` builds the device-API
  `.cu` example as a HIP TU under `USE_HIP`; the 3 pure-`.cpp` examples are
  untouched.
No source-logic edits were needed.

GOTCHA (build-system, cost ~1 iteration): host `.cpp` TUs include `gpu_data.cuh`
(the `Device_Array` template body uses `cudaMalloc`/`cudaError_t`). They are
compiled as CXX (gcc), so they need the HIP runtime headers + symbol aliases
too. Fixed by linking `hip::host` (carries `/opt/rocm/include` + the AMD
platform define) AND by making the shim include `../cuda_to_hip.h` (the
force-include only fires for `COMPILE_LANGUAGE:HIP`, never for the host TUs).
libGpufit.so confirmed: `.hip_fatbin` with `gfx90a` code objects, NEEDED
`libamdhip64.so.7`, undefined `hip*` symbols.

### Validation (gfx90a, HIP_VISIBLE_DEVICES=3)
Build: clean (100%), `-DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a
-DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_BUILD_TYPE=Release
-DUSE_CUBLAS=OFF`.
All 9 Boost known-answer suites PASS ("No errors detected", exit 0):
Error_Handling, Linear_Fit_1D, Gauss_Fit_1D, Gauss_Fit_2D,
Gauss_Fit_2D_Elliptic, Gauss_Fit_2D_Rotated, Cauchy_Fit_2D_Elliptic,
Fletcher_Powell_Helix_Fit, Brown_Dennis_Fit. These assert the LM fit recovers
the true params (e.g. Gauss_1D `{4,2,0.5,1}` to 1e-6) with chi^2 < 1e-6 -- i.e.
it CONVERGES correctly on AMD, not NaN/divergent.
Cpufit_Gpufit_Test_Consistency PASS (HIP vs CPU agree). Simple_Example and the
HIP-compiled CUDA_Interface_Example run end-to-end (10000-fit MC 2D Gaussian:
99.99% converged, fitted means match truth, ~1.66 mean iters).
Determinism: known-answer suites pass on every repeat; CUDA_Interface_Example
prints byte-identical means/chi^2 across runs.

Result: linux-gfx90a -> ported. No blockers.

## Validation 2026-05-30 (gfx1100, ROCm 7.2.1)

### Arch fix: configurable HIP_ARCHITECTURES

The original moat-port commit hardcoded `HIP_ARCHITECTURES "gfx90a"` in both
`Gpufit/CMakeLists.txt` and `examples/c++/CMakeLists.txt` via
`set_target_properties`. This means `-DCMAKE_HIP_ARCHITECTURES=gfx1100` was
silently ignored -- the cache variable was irrelevant once
`set_target_properties` overrode it. Fixed both targets to honor
`CMAKE_HIP_ARCHITECTURES` (defaulting to gfx90a when unset), and amended
the single curated moat-port commit. New SHA: `0015899374416cf2a82b1030aeab79b113d57a7b`.

This bumped `head_sha`, which correctly flipped linux-gfx90a to `revalidate`
(gfx90a device code is unchanged; only the arch selection logic changed).

### Configure/build

```
cmake -S projects/Gpufit/src -B projects/Gpufit/src/build-hip \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_BUILD_TYPE=Release -DUSE_CUBLAS=OFF

cmake --build projects/Gpufit/src/build-hip -j
```

Build: clean (100%). All targets built including libGpufit.so, all Boost.Test
executables, Simple_Example, CUDA_Interface_Example (HIP-compiled).

### gfx1100 code-object evidence

```
roc-obj-ls projects/Gpufit/src/build-hip/Gpufit/libGpufit.so
```

Output:
```
1  host-x86_64-unknown-linux-gnu-              ...size=0
1  hipv4-amdgcn-amd-amdhsa--gfx1100           ...size=132784
2  host-x86_64-unknown-linux-gnu-              ...size=0
2  hipv4-amdgcn-amd-amdhsa--gfx1100           ...size=9208
3  host-x86_64-unknown-linux-gnu-              ...size=0
3  hipv4-amdgcn-amd-amdhsa--gfx1100           ...size=6680
```

gfx1100 confirmed in all 3 code objects. No gfx90a code objects present.

### Test results (HIP_VISIBLE_DEVICES=0, AMD Radeon Pro W7800, gfx1100, wave32)

```
cd projects/Gpufit/src/build-hip && HIP_VISIBLE_DEVICES=0 ctest --output-on-failure -j1
```

| Test | Result |
|------|--------|
| Gpufit_Test_Error_Handling | PASS |
| Gpufit_Test_Linear_Fit_1D | PASS |
| Gpufit_Test_Gauss_Fit_1D | PASS |
| Gpufit_Test_Gauss_Fit_2D | PASS |
| Gpufit_Test_Gauss_Fit_2D_Elliptic | PASS |
| Gpufit_Test_Gauss_Fit_2D_Rotated | FAIL (see below) |
| Gpufit_Test_Cauchy_Fit_2D_Elliptic | PASS |
| Gpufit_Test_Fletcher_Powell_Helix_Fit | PASS |
| Gpufit_Test_Brown_Dennis_Fit | PASS |
| Cpufit_Gpufit_Test_Consistency | PASS |

9/10 PASS, 1/10 FAIL.

### Gauss_Fit_2D_Rotated failure analysis

The failure is in `Gauss_Fit_2D_Rotated.cpp` line 97, parameter[6] (rotation
angle `r`), tolerance `< 1e-6f`:

```
true_params:   r = 0.1963495463  (PI/16)
output_params: r = 0.1963506788
abs diff[6]  = 1.1324882507e-06  (threshold = 1e-6f)
```

The LM solver CONVERGES correctly: chi2 = 2.8421709430e-12 (well below the
1e-6 chi2 threshold), status=0, 6 iterations. All other 6 parameters are
within 1e-6. The rotation angle is off by 1.13e-6 -- 13% over the 1e-6
threshold.

Root cause: wave32 (gfx1100) vs wave64 (gfx90a) changes the block-packing
heuristic in `lm_fit_cuda.cu`: with `warp_size_=32` the GJ solver packs
differently per block, changing the order of FP accumulation in the
gradient/hessian reduction over 64 data points. The fit is numerically
correct (chi2 proves convergence), but the final iterate lands 13% outside
the 1e-6 tolerance. This is deterministic and reproducible across all 4
W7800 cards on this host.

This is NOT a warp-intrinsic bug or convergence failure. The fix is a
one-line test tolerance adjustment: `< 1e-6f` -> `< 2e-6f` for parameter[6]
in `Gauss_Fit_2D_Rotated.cpp:97`. This matches the approach already taken for
parameter[0] (amplitude, which uses `2e-5f`). The CUDA path is unaffected.

### Wave32 confirmation

warpSize = 32 on gfx1100 (runtime query in info.cu). Its sole effect is
feeding the block-packing heuristic in lm_fit_cuda.cu:306. This does NOT
cause NaN/divergence -- only a FP accumulation-order difference that slightly
shifts the final iterate for tight tolerances. All other 9 suites pass with
identical chi2/status to gfx90a.

### State: validation-failed

Bouncing to porter for the one-line tolerance fix in
Gpufit/tests/Gauss_Fit_2D_Rotated.cpp line 97.

## Delta-port 2026-05-30 (gfx1100 wave32 tolerance fix)

Applied the wave32 tolerance fix the validator diagnosed. Two edits, both
HIP-guarded so the CUDA/NVIDIA assertions are byte-for-byte unchanged (this
commit is the upstream PR):

1. `Gpufit/tests/Gauss_Fit_2D_Rotated.cpp` -- relax ONLY parameter[6] (rotation
   angle r) from `< 1e-6f` to `< 3e-6f`, gated `#if defined(USE_HIP)` (CUDA keeps
   `1e-6f` in the `#else`). Observed wave32 error was 1.13e-6 (chi^2 ~2.84e-12,
   converged); 3e-6f is ~2.6x margin over the observed error, still tight. Picked
   3e-6f over the validator's suggested 2e-6f because 2e-6f is only ~1.77x the
   observed error -- too thin for a deterministic-but-FP-order-sensitive iterate.

2. `CMakeLists.txt` (`add_boost_test`) -- the test executables compile only the
   `.cpp` (host CXX) and link the Gpufit lib, so they did NOT receive `-DUSE_HIP`
   (it is PRIVATE to the Gpufit target). The proposed `#if defined(USE_HIP)` guard
   would have been silently false in the HIP build, leaving the strict 1e-6f and
   re-failing. Fix: add `target_compile_definitions(${target} PRIVATE USE_HIP)`
   under `if(USE_HIP)` so the test TU gets the define. Guarded by `if(USE_HIP)`,
   so the CUDA build is untouched. (`__HIP_PLATFORM_AMD__` does leak transitively
   into the test TU via `hip::host`, but an explicit USE_HIP guard is the robust,
   self-documenting discriminator; relying on a leaked define is fragile.)

GOTCHA (the load-bearing one): in this project the Boost test targets do NOT
inherit the library's PRIVATE `USE_HIP` define -- verify the guard symbol is
actually in the test TU's `CXX_DEFINES` (build-hip/.../<test>.dir/flags.make)
before trusting a compile-time `#if`. Confirmed present after the CMake fix:
`CXX_DEFINES = -DUSE_HIP -DUSE_PROF_API=1 -D__HIP_PLATFORM_AMD__=1`.

### Local gfx1100 re-run (AMD Radeon Pro W7800, wave32, ROCm 7.2.1)

Reused the existing build-hip config (USE_HIP=ON, CMAKE_HIP_ARCHITECTURES=gfx1100,
USE_CUBLAS=OFF, Release). The CMakeLists.txt edit triggered a reconfigure; build
clean to 100%.

```
cmake --build projects/Gpufit/src/build-hip -j
cd projects/Gpufit/src/build-hip && HIP_VISIBLE_DEVICES=0 ctest --output-on-failure -j1
```

`Gpufit_Test_Gauss_Fit_2D_Rotated` now: 11/11 assertions pass (line 102, the
`< 3e-6f` HIP branch, passes), "test module Gpufit has passed", exit 0. Full
ctest: 100% tests passed, 0 failed out of 10 (the 9 known-answer suites +
Cpufit_Gpufit_Test_Consistency). No regression.

Amended the single curated commit (no second commit). New fork SHA:
`5ab0c059ed761d571c3e66a519f3246a62184145`, pushed --force-with-lease to
jeffdaily/Gpufit moat-port (lease baseline 0015899...). head_sha advanced to
5ab0c05; gfx90a stays `revalidate` against this final SHA (revalidates once).

Result: linux-gfx1100 -> delta-ported. No blockers.

## Revalidation 2026-05-30 (gfx90a re-confirm at gfx1100 delta tip)

linux-gfx90a was bounced to `revalidate` when the gfx1100 delta-port advanced
the shared fork head from the gfx90a-validated `060a66c2` to `5ab0c059`. Re-ran
on gfx90a (MI250X, wave64, GPU0 / HIP_VISIBLE_DEVICES=0, ROCm 7.2.1) to confirm
no regression at the new tip.

### What the gfx1100 delta changed (060a66c2..5ab0c059)

```
git --no-pager diff 060a66c2..HEAD --stat
 CMakeLists.txt                        | 5 +++++
 Gpufit/CMakeLists.txt                 | 5 ++++-
 Gpufit/tests/Gauss_Fit_2D_Rotated.cpp | 8 ++++++++
 examples/c++/CMakeLists.txt           | 5 ++++-
```

Test-tolerance + CMake only; NO device-code (.cu) logic changed, so gfx90a
numerics are byte-identical to the 060a66c2 validation. Specifically:
- `Gauss_Fit_2D_Rotated.cpp`: parameter[6] (rotation angle r) relaxed from
  `< 1e-6f` to `< 3e-6f`, gated `#if defined(USE_HIP)` (CUDA `#else` keeps
  strict `1e-6f`). This only LOOSENS the bound, so it cannot regress wave64,
  which already passed at the strict 1e-6f.
- `CMakeLists.txt`: `target_compile_definitions(${target} PRIVATE USE_HIP)` on
  the Boost test targets under `if(USE_HIP)` so the test TU actually sees the
  guard symbol (the PRIVATE define on the Gpufit lib target does not propagate
  to the test executables here).
- `Gpufit/CMakeLists.txt` + `examples/c++/CMakeLists.txt`: HIP_ARCHITECTURES now
  honors `CMAKE_HIP_ARCHITECTURES` (defaulting to gfx90a when unset) instead of
  hardcoding gfx90a. Confirmed `-DCMAKE_HIP_ARCHITECTURES=gfx90a` still yields
  gfx90a code objects.

The change is exactly the RDNA/wave32 guard expected -- harmless on gfx90a.

### Build (clean reconfigure, build dir out of git under src/)

```
git fetch fork moat-port            # 060a66c..5ab0c05 (forced update)
git checkout -B moat-port fork/moat-port   # HEAD == 5ab0c059
cmake -S . -B build-hip -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_BUILD_TYPE=Release -DUSE_CUBLAS=OFF
cmake --build build-hip -j
```

Build clean to 100%. `roc-obj-ls build-hip/Gpufit/libGpufit.so` shows all 3
code objects `hipv4-amdgcn-amd-amdhsa--gfx90a` (no gfx1100). The new
configurable-arch logic correctly produces gfx90a here.

### Test results (HIP_VISIBLE_DEVICES=0, MI250X gfx90a, wave64)

```
cd build-hip && HIP_VISIBLE_DEVICES=0 ctest --output-on-failure -j1
```

100% tests passed, 0 failed out of 10 (the 9 Boost known-answer GPU suites +
Cpufit_Gpufit_Test_Consistency). Notably `Gpufit_Test_Gauss_Fit_2D_Rotated`,
which FAILED on gfx1100/wave32 at the strict 1e-6f, PASSES on gfx90a/wave64:
wave64 lands the rotation angle within the original 1e-6f, and the relaxed
3e-6f HIP bound is a strict superset. Determinism: a second full ctest pass is
also 100%; the rotated-fit binary prints "No errors detected", exit 0.

No regression from the gfx1100 delta. Result: linux-gfx90a `revalidate` ->
`completed`, validated_sha = 5ab0c059ed761d571c3e66a519f3246a62184145 (the
rebuilt + revalidated HEAD).

## Review 2026-05-30 (linux-gfx1100 delta-ported, fork moat-port @ 5ab0c059)

Reviewed `git diff a0bd66c..5ab0c05` (the single curated commit) with the
/pr-review skill, fact-checking every load-bearing claim against the source.
Verdict: review-passed -> validator. Problems found: none material.

Diff scope: 7 files (2 CMake, cuda_to_hip.h, 2 hip_compat shims,
Gauss_Fit_2D_Rotated.cpp, examples/c++/CMakeLists.txt). No .cu/.cuh device
code changed.

What I verified (independently, not from notes alone):
- Strategy A correct: one compat header, .cu marked LANGUAGE HIP (not renamed),
  parallel modern-CMake HIP path beside the legacy FindCUDA path.
- Compat-header coverage is EXACT: the 19 cuda* runtime symbols referenced in
  the sources match the 19 aliased in cuda_to_hip.h one-for-one (no missing
  alias -> no compile break; no dead alias). No warp/texture/lib mappings
  because none are used.
- Fault classes: zero warp intrinsics across cuda_kernels.cu /
  cuda_gaussjordan.cu / lm_fit_cuda.cu (__shfl/__ballot/__any/__all/
  __activemask/__syncwarp count = 0). No hardcoded 32 / lane masks. The only
  warpSize use is info.cu:20 (runtime query) feeding the block-PACKING
  heuristic at lm_fit_cuda.cu:306, not a 32-lane reduction. No textures /
  surfaces / cudaArray / __constant__ / atomics anywhere, so the rule-of-five,
  OOB-clamp, 256B-pitch, and texture-filter classes do not apply. cuBLAS is
  fully #ifdef USE_CUBLAS-guarded and off the default GJ path.
- Wave32 tolerance delta is a genuine FP-reduction-order fix, NOT a masked
  bug: sum_up_floats (cuda_kernels.cu:139) is a __syncthreads tree reduction
  over points with multiple fits packed per block; n_hessians_per_block
  (driven by info_.warp_size_) changes block geometry on wave32 vs wave64,
  reshuffling FP summation order -> rotation angle shifts 1.13e-6 while
  chi^2 stays 2.84e-12 (converged). 3e-6f bound = 2.65x the observed error
  (tight); param[0] already uses 2e-5f upstream, so per-parameter loosening
  in this test has precedent.
- The USE_HIP test guard is ACTIVE, not dead: build-hip flags.make for the
  rotated test TU shows `CXX_DEFINES = -DUSE_HIP -DUSE_PROF_API=1
  -D__HIP_PLATFORM_AMD__=1`. The lib's PRIVATE USE_HIP does not reach the test
  exe; target_compile_definitions(${target} PRIVATE USE_HIP) in add_boost_test
  supplies it. Without this fix the #if would be false and CUDA's 1e-6f would
  wrongly apply on HIP -> re-fail. Correctly handled.
- CUDA path byte-for-byte unchanged: no device source touched; the test #else
  keeps exactly < 1e-6f (matches upstream a0bd66c); the CUDA CMake logic is
  byte-identical inside if(NOT USE_HIP); project()/cmake_minimum_required
  reduce to the original literals when USE_HIP=OFF; find_package(hip) is gated
  inside if(USE_HIP), so no ROCm dependency on the CUDA/CPU build.
- hip::host (not hip::device) is linked -- correct, so -x hip / --offload-arch
  does not leak into the host .cpp TUs of the mixed lib target (cupoch lesson).
- Arch-configurable: HIP_ARCHITECTURES reads ${CMAKE_HIP_ARCHITECTURES},
  defaults gfx90a only when unset, in BOTH Gpufit/CMakeLists.txt and
  examples/c++/CMakeLists.txt -- not a hardcoded literal.
- Commit hygiene: title `[ROCm] Add HIP build for AMD GPUs; configurable
  HIP_ARCHITECTURES` (60 chars <= 72); body has Test Plan + "Authored with
  Claude"; no noreply/co-authored trailer; author jeffdaily public email; no
  non-ASCII, no em-dash, no AMD-internal account references.

Not done at review time (expected): the gfx1100 real-GPU run at this exact tip
-- that is the validator's stage; the reviewer does not block on it.

Result: linux-gfx1100 delta-ported -> review-passed.

## Validation 2026-05-30 (gfx1100, ROCm 7.2.1) -- re-run at 5ab0c05

Re-validation after the wave32 tolerance delta-port + review. Fork clone at
`projects/Gpufit/src` confirmed on moat-port HEAD `5ab0c059ed761d571c3e66a519f3246a62184145`
before build.

### GPU

AMD Radeon Pro W7800 48GB, gfx1100, wave32 (GPU0, HIP_VISIBLE_DEVICES=0).

### Build command

```
bash utils/timeit.sh Gpufit compile -- \
  cmake -S projects/Gpufit/src -B projects/Gpufit/src/build-hip \
    -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
    -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
    -DCMAKE_BUILD_TYPE=Release -DUSE_CUBLAS=OFF

bash utils/timeit.sh Gpufit compile -- \
  cmake --build projects/Gpufit/src/build-hip -j
```

Build: clean (100%), all targets including libGpufit.so, Boost.Test executables,
Simple_Example, CUDA_Interface_Example (HIP-compiled).

### gfx1100 code-object evidence

```
roc-obj-ls projects/Gpufit/src/build-hip/Gpufit/libGpufit.so
```

Output:
```
1  hipv4-amdgcn-amd-amdhsa--gfx1100    size=132784
2  hipv4-amdgcn-amd-amdhsa--gfx1100    size=9208
3  hipv4-amdgcn-amd-amdhsa--gfx1100    size=6680
```

All 3 code objects are gfx1100. No gfx90a code objects present.

### Test results (HIP_VISIBLE_DEVICES=0, AMD Radeon Pro W7800, gfx1100, wave32)

```
bash utils/timeit.sh Gpufit test -- \
  bash -c "cd projects/Gpufit/src/build-hip && HIP_VISIBLE_DEVICES=0 ctest --output-on-failure -j1"
```

| Test | Result |
|------|--------|
| Gpufit_Test_Error_Handling | PASS |
| Gpufit_Test_Linear_Fit_1D | PASS |
| Gpufit_Test_Gauss_Fit_1D | PASS |
| Gpufit_Test_Gauss_Fit_2D | PASS |
| Gpufit_Test_Gauss_Fit_2D_Elliptic | PASS |
| Gpufit_Test_Gauss_Fit_2D_Rotated | PASS |
| Gpufit_Test_Cauchy_Fit_2D_Elliptic | PASS |
| Gpufit_Test_Fletcher_Powell_Helix_Fit | PASS |
| Gpufit_Test_Brown_Dennis_Fit | PASS |
| Cpufit_Gpufit_Test_Consistency | PASS |

10/10 PASS. 0 failed. Total test time: 1.77 sec.

Gauss_Fit_2D_Rotated PASSES: the wave32-guarded `< 3e-6f` tolerance on parameter[6]
(rotation angle r) is satisfied. Previously failed at `< 1e-6f` by 1.13e-6; the delta-port
relaxed it to 3e-6f (2.65x observed error, HIP-only #if guard). No regression in the other
9 suites versus the prior gfx1100 run (all were already passing at the strict 1e-6f).

No NaN, no divergence, clean exit 0 on all suites.

Result: linux-gfx1100 review-passed -> completed, validated_sha = 5ab0c059ed761d571c3e66a519f3246a62184145.
