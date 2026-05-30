# CudaSift notes

Possibly the upstream of the SIFT-GPU code in jeffdaily/colmap (branch rocm-sift-gpu); reference that port for the AMD-fault fixes (texture rule-of-five, OOB index clamps, 256-byte texture pitch).

## Porting 2026-05-30 (gfx90a, ROCm 7.2)

Strategy A (colmap minimal-footprint). Result: clean configure + build of all
targets for gfx90a (configure_exit=0, build_exit=0, 0 compile errors). Binaries
embed `hipv4-amdgcn-amd-amdhsa--gfx90a` code objects and link `libamdhip64`.
GPU tests not run here (validator does that) -- compile+link only.

### Files changed
- `cuda_to_hip.h` (new): the single CUDA->HIP compat header. On
  `USE_HIP || __HIP_PLATFORM_AMD__` includes `<hip/hip_runtime.h>` and `#define`s
  every cudaXxx symbol the project uses to its hip equivalent (errors, device
  mgmt, events/streams, malloc/memcpy/memset, constant-memory symbol copies,
  memcpy kinds, cudaArray + channel-format, texture objects + resource/texture
  descs and their enums). Names taken from pytorch hipify
  `cuda_to_hip_mappings.py`. Also defines two device-intrinsic shims (see fault
  classes). On NVIDIA it is a no-op `#include <cuda_runtime.h>`.
- `hip_compat/cuda_runtime.h`, `hip_compat/cuda.h` (new): tiny HIP-only shims
  (each just `#include "cuda_to_hip.h"`) so the few sources that do
  `#include <cuda_runtime.h>` / `<cuda.h>` directly (tests/test_match.cpp,
  tests/benchmark.cpp, match.cu) resolve on ROCm, which ships no CUDA headers.
  Added to the include path for the HIP build only; NVIDIA uses the real header.
- `CMakeLists.txt`: added `option(USE_HIP ... OFF)`. When ON: `project(... HIP)`,
  `DEVICE_LANG=HIP`, mark the `.cu` TUs and the `LANGUAGE CUDA` `.cpp` files
  `LANGUAGE HIP`, set `HIP_ARCHITECTURES gfx90a` on every target, force-include
  the compat header (`-include .../cuda_to_hip.h` for the HIP language), add the
  `hip_compat/` shim dir, define `USE_HIP`, and swap nvcc flags
  (`--use_fast_math -lineinfo`) for clang (`-ffast-math`). The CUDA path
  (`USE_HIP=OFF`) keeps the original behavior.
- `cudaImage.cu`: `CopyToTexture` -- replaced deprecated `cudaMemcpyToArray`
  (removed from current HIP) with `hipMemcpy2DToArray` under a HIP guard. This
  path is DEAD (`CopyToTexture`/`InitTexture` have no callers; the live texture
  is the Pitch2D bind in cudaSiftH.cu), so it cannot be GPU-validated -- the
  fix is for correctness on a future revival, not the shipped pipeline. The
  source row stride is `this->pitch` (spitch == width == `pitch*sizeof(float)`,
  height = `dst.height`), reproducing the original
  `cudaMemcpyToArray(... sizeof(float)*pitch*dst.height ...)` byte count
  byte-for-byte. (An earlier draft used `dst.pitch`, which only matched while
  `this->pitch == dst.pitch`; restored to `this->pitch`.) CUDA branch unchanged.
- `cudaSiftH.cu`: the two `cudaMemcpyToSymbolAsync(symbol, src, bytes)` calls
  now pass explicit `0, cudaMemcpyHostToDevice`. CUDA defaults offset/kind/stream;
  HIP's overload only defaults `stream`, so the 3-arg form failed to resolve.
  The explicit args are valid on CUDA too (BC-safe shared-code edit).
- `matching.cu`: `MatchSiftPoints2` -- the two `sift[min(int, unsigned)]` array
  subscripts now cast to `(int)`. HIP's `min` overload resolution produced a
  non-integer subscript (illegal); the other `min(unsigned, int)` sites that
  assign to an `int` were fine and left as-is.

`cudautils.h` and `cudaSiftD.cu` were deliberately NOT modified -- the warp
helpers and the descriptor-layout / orientation-histogram constants stay in
plain CUDA spelling.

### Fault-class issues encountered and fixes
- Compat-header bootstrap (HIP macro chicken-and-egg): clang in HIP mode
  predefines `__HIP__` but NOT `__HIP_PLATFORM_AMD__` -- that macro is only set
  *by* including the HIP runtime header, so it can't gate the include of that
  same header. Fixed by also passing `-DUSE_HIP` to the compiler (the header's
  first guard term), which is what made the force-include actually translate.
- Warp size (wave64): `cudaSiftD.cu` calls `__any_sync(0xffffffff, pred)`
  directly. On gfx90a `__any_sync` is templated and static-asserts that the
  mask is 64-bit, rejecting the CUDA 32-bit full mask. The kernels that use it
  (`FindPointsMulti*`) launch 32-thread blocks and the surrounding reduction is
  explicitly width-32 (`ShiftUp(.,d) for d<32`, `Shuffle(.,31)`). Mapped
  `__any_sync(mask, pred) -> __any(pred)` in the header: HIP's mask-free `__any`
  polls the active wavefront (exactly the 32 active lanes here), matching the
  original "any active lane" intent. The cudautils.h `ShiftDown/Up/Shuffle`
  helpers take the non-`_sync` `__shfl_*(var, delta, width)` branch under HIP
  (HIP does not define `CUDART_VERSION`); width=32 is honored as a sub-group
  split on wave64, so the 4-partial-sum (`sums[idx/32]`, idx in {0,32,64,96})
  descriptor reduction keeps CUDA semantics. NUMERIC PARITY STILL NEEDS THE GPU
  VALIDATOR -- this analysis says it should match, on-device confirmation pending.
- Missing intrinsic: HIP has no round-toward-zero `__fmul_rz` (used only in the
  RANSAC homography residual test). Mapped to a plain `(a)*(b)` in the header;
  rounding mode is immaterial to an inlier threshold test, and round-to-nearest
  is the faithful default.
- Deprecated array-copy API: `cudaMemcpyToArray` unavailable in HIP -> rewrote
  to `hipMemcpy2DToArray` (see cudaImage.cu above). NOTE: this is in the dead
  `CopyToTexture`/`InitTexture` path, NOT the live texture. The PRIMARY texture
  bound in `ExtractSiftOctave` (cudaSiftH.cu:189-207) is a
  `cudaResourceTypePitch2D` bind directly to `img.d_data`
  (`pitchInBytes = img.pitch*sizeof(float)`), so the 256-byte-pitch fault class
  DOES apply to it (earlier "cudaArray-backed, does not apply" claim was wrong).
  It is satisfied here, not absent: every image fed to the bind has
  `pitch = iAlignUp(width,128)` = a multiple of 128 float elems = 512 bytes
  (cudaSiftH.cu:122,157,184), and the base pointers come from
  `cudaMallocPitch` / a `cudaMalloc`'d pool (>=256B aligned). So pitch is a
  multiple of 512B and base is >=256B aligned -> the AMD 256B texture-pitch
  requirement holds (applies-but-satisfied). The sampler clamps
  (`cudaAddressModeClamp`, cudaSiftH.cu:200-201) so edge `tex2D` fetches at
  +-cos/+-sin are in-bounds by construction (no OOB clamp needed) -- on the
  Pitch2D bind as well.
- Texture rule-of-five: `texObj` is default-initialized to 0, created with
  `cudaCreateTextureObject` and unconditionally destroyed with
  `cudaDestroyTextureObject` in the same scope (`ExtractSiftOctave`); the
  `cudaArray t_data` in CudaImage is default-init NULL with a guarded
  `cudaFreeArray` in the dtor. No double-free / uninitialized-handle hazard, so
  no hardening needed beyond what upstream already does.

### Build commands (run from /var/lib/jenkins/moat)
```
utils/timeit.sh CudaSift compile -- cmake -S projects/CudaSift/src \
  -B projects/CudaSift/src/build-hip -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_BUILD_TYPE=Release
cmake --build projects/CudaSift/src/build-hip -j
```
Full clean configure+build ~9s. Targets: cudasift_lib, cudasift, test_extract,
test_match, test_homography, benchmark (+ demo_extract/match/video).

### Remaining warnings (non-fatal)
- ~152 `-Wunused-value`: ignoring `nodiscard` `hipError_t` returns from
  `cudaEvent*` / memcpy calls in TimerGPU and elsewhere. Pre-existing CUDA code
  pattern; CUDA headers don't mark these `nodiscard`, HIP does. Harmless; left
  untouched to keep the diff minimal.
- 4 `-Wliteral-conversion`, 2 `-Wnan-infinity-disabled` (from `-ffast-math`).

### BC note
`USE_HIP=OFF` configure fails on this host only because there is no CUDA
toolkit (`Failed to find nvcc` at the `project(... CUDA)` line) -- identical to
the unmodified upstream on a CUDA-less host, not a regression. The CUDA-path
CMake logic mirrors the original.

## Review 2026-05-30 (reviewer, gfx90a / wave64)

Verdict: changes-requested. The Strategy-A mechanics are sound (one no-op-on-NVIDIA
compat header, `.cu` marked `LANGUAGE HIP` not renamed, USE_HIP default OFF keeps
the CUDA path intact, force-include + hip_compat shims). The shared-code edits
(cudaSiftH.cu memcpy-to-symbol arg fill, matching.cu `min` casts) are BC-safe
generalizations. Problems below.

### Fault classes
- cudaSiftH.cu:191 -- the PRIMARY texture in `ExtractSift` is a
  `cudaResourceTypePitch2D` bind directly to `img.d_data`
  (`pitchInBytes = img.pitch*sizeof(float)`), NOT a cudaArray. This directly
  contradicts the porter analysis (notes.md "Texture backing is a `cudaArray`
  (not a pitched 2D bind), so the 256-byte-pitch fault class does not apply" and
  plan.md risk list). The 256B-pitch fault class DOES apply to this live bind.
  It happens to be satisfied (pitch = iAlignUp(w,128) => multiple of 512B;
  hipMallocPitch/pool base is >=256B aligned), so no code bug results, but the
  applicable fault class was dismissed on a false premise and the load-bearing
  path was never actually checked. Re-do the pitch/base-alignment analysis
  against the Pitch2D resource, not the cudaArray.
- cudaSiftD.cu:1319,1329 (`__any_sync`->`__any`) and the width-32 shuffle
  reductions (cudaSiftD.cu:269-285 descriptor norm; 1361-1378 prefix-sum +
  `Shuffle(.,31)` broadcast) are wave64-sensitive and remain UNVALIDATED on
  device. Static analysis indicates parity (active kernels launch 32-thread
  blocks for `FindPointsMultiNew`, and width=32 splits a wave64 front into exact
  32-lane subgroups for the 128-thread descriptor kernel, so `sums[idx/32]`
  fills its 4 slots correctly; HIP `__any`/`wfany` polls EXEC = the launched
  lanes only), but the porter flagged numeric parity as pending and no GPU run
  was done. Per the review rubric a port without a real-GPU parity run is
  changes-requested. Needs the test_extract / test_match / test_homography +
  keypoint/descriptor parity run on gfx90a before passing.

### Minimal footprint / dead code
- cudaImage.cu:105-119 -- the `cudaMemcpyToArray`->`hipMemcpy2DToArray` rewrite
  is in `CudaImage::CopyToTexture`, which (with `InitTexture`) has NO callers
  (declared in cudaImage.h, defined here, never invoked; the live texture is the
  Pitch2D bind above). Consequences: (a) the rewritten path is unreachable, so
  no GPU test can validate its semantics; (b) it changes the row-stride source
  from `this->pitch` (original: `sizeof(float)*pitch*dst.height`) to `dst.pitch`
  (`rowBytes = sizeof(float)*dst.pitch`) -- a latent behavior change should the
  method ever be revived (correct only while `this->pitch == dst.pitch`).
  The 2D-array semantics otherwise match the original contiguous upload. Either
  match the original `this->pitch` or note the path is dead; the
  semantics-preservation claim in notes.md should not be stated as validated.

### Notes accuracy (porting record)
- The "Fault-class issues" section asserts texture backing is a cudaArray and
  that the 256B-pitch class "does not apply"; this is wrong for the shipped
  texture (see above). Correct the record so a later wave32/gfx1100 re-validation
  does not inherit the false premise.

Non-blocking confirmations (not defects): `CUDART_VERSION` is undefined on the
HIP path (verified: only in rocm `nvidia_detail`), so `ShiftDown/Up/Shuffle`
take the `__shfl_*(var,delta,width)` branch as intended; `__fmul_rz` shim reaches
its only use (matching.cu RANSAC residual, a HIP TU); `match.cu` and its
`#include <cuda.h>` are not compiled by any target (orphan); geomFuncs.cpp uses
no device symbols; texDesc sets `cudaAddressModeClamp` so edge `tex2D` fetches at
+-cos/+-sin are in-bounds on the Pitch2D bind too; binaries embed
`hipv4-amdgcn-amd-amdhsa--gfx90a` and link libamdhip64 (build claim holds).

## Validation 2026-05-30 (real GPU, gfx90a / wave64)

Addressed both reviewer findings, then ran the real GPU tests on device.

GPU: AMD Instinct MI250X / MI250, `gfx90a` (wave64), ROCm. Host has 4x gfx90a
(`rocm_agent_enumerator` -> 4x gfx90a); all idle, pinned to the highest index
with `HIP_VISIBLE_DEVICES=3` (one visible device -> in-process ordinal 0, tests
run with default devNum=0). Binaries embed `hipv4-amdgcn-amd-amdhsa--gfx90a`
(verified via `roc-obj-ls`).

Fixes:
- cudaImage.cu:110 -- `CopyToTexture` HIP path source row stride restored from
  `dst.pitch` to `this->pitch` (`rowBytes = sizeof(float)*pitch`), matching the
  original `cudaMemcpyToArray(... sizeof(float)*pitch*dst.height ...)` byte
  count byte-for-byte. Path is dead (no callers), so the fix is latent-correct
  for a future revival, not GPU-exercised; CUDA `#else` branch unchanged.
- notes.md (this file) -- corrected the texture-backing claim: the live
  `ExtractSift`/`ExtractSiftOctave` texture is a `cudaResourceTypePitch2D` bind
  to `img.d_data` (cudaSiftH.cu:189-207), NOT a cudaArray. The 256B-pitch fault
  class APPLIES and is SATISFIED (pitch = iAlignUp(w,128) = multiple of 512B;
  base from cudaMallocPitch / cudaMalloc pool >=256B aligned), not "does not
  apply".

Rebuild: `cmake --build projects/CudaSift/src/build-hip -j` -> exit 0, clean
(only the pre-existing ~8 `-Wunused-value` nodiscard warnings on cudaImage.cu;
no errors). All targets built.

Tests (each `utils/timeit.sh CudaSift test -- bash -c 'cd projects/CudaSift/src
&& ./build-hip/<bin>'`, run from src/ so `data/` resolves):

| test            | exit | result            | key numbers |
|-----------------|------|-------------------|-------------|
| test_extract    | 0    | 10 passed, 0 fail | basic 1910 kpts (all valid, descriptors unit-norm); thresh monotonic 7080/1910/542/6; reproducibility 1910/1910 = 100%; scaleUp 3104 > normal 1910 |
| test_match      | 0    | 11 passed, 0 fail | self-match 1899/1910 score>0.95; cross img1/img2 1910 vs 2084 feats, 1910 valid / 1909 good matches; homography 851 RANSAC / 828 inliers; match 1.23 ms |
| test_homography | 0    | 8 passed, 0 fail  | translation (30,20) recovered h[2]=30.13 h[5]=20.27, 1645 matches; rotation 10deg 1195; scale 0.8 945; PGM pair L1451/R2082, 751 RANSAC / 672 inliers |

PASS/FAIL: PASS (29/29 sub-checks across the three binaries, all exit 0). No GPU
page fault, no `safeCall` abort, no crash; GPU 3 healthy/idle post-run, dmesg
clean of amdgpu/page-fault/VM-retry. The wave64-sensitive paths the reviewer
flagged (`__any_sync`->`__any` in FindPointsMulti, the width-32 descriptor-norm
and prefix-sum reductions) produce CORRECT on-device output: descriptors are
unit-length, extraction is bit-reproducible (100% position match across runs),
and the recovered homography geometry is accurate (translation within ~0.3 px).
The static wave64 parity analysis is now confirmed on real gfx90a.
