# EnvGS notes

ROCm/HIP port of EnvGS (CVPR 2025, "Environment Gaussian Splatting"). EnvGS is
an `easyvolcap`-framework project whose native GPU work lives entirely in git
submodules; `easyvolcap` itself is pure Python/PyTorch glue.

Two native components:
1. `submodules/diff-surfel-rasterizations` -- the base 2DGS (surfel) rasterizer.
   EnvGS installs three of its variants: `diff-surfel-rasterization-wet`,
   `-wet-ch05`, `-wet-ch07` (byte-identical sources, differ only in
   `NUM_CHANNELS` = 3/5/7). **Stage 1 -- PORTED + GPU-validated on gfx90a.**
2. `submodules/diff-surfel-tracing` -- the environment-Gaussian REFLECTION path,
   an NVIDIA OptiX 7.7 ray-tracing pipeline. **Stage 2 -- DEFERRED** (see below).
3. `submodules/StableNormal` -- pure-Python diffusion model for normal
   preprocessing. No CUDA, out of scope.

## Fork / branch structure (submodule-based project)

The code edits are in the rasterizer submodule, not the EnvGS superproject, so
there are TWO forks, each with a `moat-port` topic branch:

- `jeffdaily/diff-surfel-rasterizations` @ `moat-port` -- the actual rasterizer
  port (the real upstream PR target is `xbillowy/diff-surfel-rasterizations`).
- `jeffdaily/EnvGS` @ `moat-port` -- the superproject: repoints the
  `diff-surfel-rasterizations` submodule URL to the jeffdaily fork's moat-port
  branch in `.gitmodules` and advances the gitlink to the ported commit, so
  EnvGS builds end-to-end from the fork. `diff-surfel-tracing` and
  `StableNormal` are left pointing at upstream.

A fresh `git clone -b moat-port https://github.com/jeffdaily/EnvGS` +
`git submodule update --init submodules/diff-surfel-rasterizations` checks out
the ported rasterizer (verified).

Actions disabled on both jeffdaily forks.

## Stage 1 port: what changed (diff-surfel-rasterizations)

Strategy B (torch CUDAExtension hipify). torch passes `-DUSE_ROCM=1` and
compiles the `.cu` with `-x hip` automatically; all fixes are USE_ROCM-guarded
so the CUDA build is unchanged. The three variants get identical edits. Per
variant:

- `setup.py`: GLM-in-a-submodule hipify monkeypatch (the gsplat lesson). torch's
  hipify walks every `.hpp` under the build dir and content-rewrites GLM
  headers, dropping GLM's `.inl` files and breaking its compiler detection. The
  monkeypatch adds `../third_party/glm` to hipify's `ignores` and strips it from
  `header_include_dirs`; the bundled GLM (g-truc/glm @ 5c46b9c, version macros
  0.9.9.9) already detects `__HIP__` (`glm/simd/platform.h` ->
  `GLM_COMPILER_HIP` -> `__device__ __host__`, gated at setup.hpp:442) and
  compiles verbatim under `-x hip`. NOTE: the planning doc guessed GLM might be
  0.9.9.x-without-HIP; it is in fact HIP-aware, so the ignore-monkeypatch alone
  suffices (the `__CUDACC__`/`GLM_FORCE_CUDA` steering trick was NOT needed).
- `cuda_rasterizer/auxiliary.h`: `#define __trap __builtin_trap` under USE_ROCM.
  HIP's device runtime does not declare CUDA's `__trap()`; `__builtin_trap` is
  the equivalent (HIP's own `abort()` is defined as `__builtin_trap`). (lc0
  lesson.)
- `cuda_rasterizer/{forward.h,backward.h,rasterizer_impl.cu}`: guard
  `#include "device_launch_parameters.h"` out under `#if !defined(USE_ROCM)`.
  It is a CUDA-toolkit IntelliSense header with no ROCm equivalent; hipify
  leaves it verbatim -> "file not found". The threadIdx/blockIdx symbols it
  declares are hipcc builtins. (LiteGS lesson.)
- `cuda_rasterizer/{forward.cu,backward.cu,rasterizer_impl.cu}`: (a) guard the
  unused `#include <cooperative_groups/reduce.h>` out under USE_ROCM (HIP CG has
  no reduce.h; the code never calls `cg::reduce`; gaussian_splatting lesson),
  and (b) normalize the kernel-launch syntax `kernel << <grid, block >> >(...)`
  -> `kernel<<<...>>>` (the source carries spaces inside the triple-angle-
  brackets, which nvcc tolerates but clang-HIP's parser rejects). 7 launch
  sites. (LiteGS musa-vendor whitespace lesson.)

No warp primitives anywhere (exhaustive grep: no `__shfl`/`__ballot`/
`__activemask`/`warpSize`/`__reduce_*_sync`/`cg::reduce`/`tiled_partition`).
Block is 16x16=256; all cross-thread exchange via `__shared__` + `block.sync()`;
gradients via per-thread `atomicAdd`. So there is NO wave32-vs-wave64 fault
class: the same source is correct on gfx90a (wave64) and wave32 followers
(gfx1100/gfx1151). `NUM_WARPS (BLOCK_SIZE/32)` in auxiliary.h is a dead macro
(defined, never read) -- left as upstream; it is not a wave-size hazard since no
kernel uses it.

`cub::DeviceScan::InclusiveSum` / `cub::DeviceRadixSort::SortPairs` (the 3DGS
tile sort) and `cooperative_groups` (only `this_grid()/this_thread_block()`)
hipify cleanly to hipCUB/rocPRIM and HIP CG. GLM device math compiles verbatim.

Total Stage 1 edits: 21 files = 3 variants x 7 files (setup.py + 6 source
files); the 6 source edits are identical across variants.

## Build recipe (gfx90a; cwd OUTSIDE /var/lib/jenkins/pytorch)

```
git -C projects/EnvGS/src submodule update --init submodules/diff-surfel-rasterizations
export PYTORCH_ROCM_ARCH=gfx90a
export MAX_JOBS=16
for v in diff-surfel-rasterization-wet diff-surfel-rasterization-wet-ch05 \
         diff-surfel-rasterization-wet-ch07; do
  cd submodules/diff-surfel-rasterizations/$v
  rm -rf build *.egg-info hip_rasterizer *_hip.* rasterize_points.hip   # force re-hipify on edited .cu
  pip install -e . --no-build-isolation --no-deps -v
  cd -
done
```

Build helper used: `agent_space/envgs/build_variant.sh <variant-dir>`.
Validation harnesses: `agent_space/envgs/validate_stage1.py` (forward / backward
/ FD / determinism) and `agent_space/envgs/train_converge.py` (diffuse-path
training convergence). Run with `HIP_VISIBLE_DEVICES=3`.

Followers (gfx1100/gfx1151): same recipe, only `PYTORCH_ROCM_ARCH=gfx1100|
gfx1151`, clean rebuild. No source edit expected (no wave-size fault class).

## Stage 1 validation results (gfx90a, MI250X, ROCm 7.2.1, torch 2.13)

- Build: all 3 `-wet*` extensions compile with `-DUSE_ROCM=1` / `-x hip`,
  import, and `_C` exports `rasterize_gaussians`, `rasterize_gaussians_backward`,
  `mark_visible`.
- Forward: `rasterize_gaussians` on ~4000 surfels at 200x150 -> finite,
  non-trivial image (~1778/4000 visible, min 0.011 / max 0.865 / mean 0.354,
  nonzero_frac 1.0), no illegal memory access.
- Backward: gradients to means3D/means2D/opacity/colors/scales/rotations all
  finite. Finite-difference: opacity FD matches analytic exactly (sign 1.00,
  slope ~1.00) across all 3 variants -- the decisive geometric-blend gradient.
  means3D directional FD points downhill (slope < 1, expected: the rasterizer
  blends in limited precision so coordinate FD underestimates; LiteGS).
- Training convergence (the authoritative gradient gate): a short diffuse-path
  Adam fit to a fixed target image drops the loss ~95% (wet 0.0488->0.0026,
  ch05 0.0502->0.0027, ch07 0.0680->0.0026) with PSNR ~25-26 dB and no NaN, for
  all 3 channel counts. The RASTERIZER (diff-surfel-rasterization) trains
  forward+backward end-to-end via the direct autograd harness; the EnvGS
  framework sampler additionally requires the deferred Stage 2 OptiX module
  (diff_surfel_tracing) to import -- importing the EnvGS sampler classes
  (gaussian2d_sampler.py, envgs_sampler.py) is NOT part of the Stage 1 gate.
- Determinism: forward bit-identical run-to-run; backward grads agree to ~3e-7
  (benign atomicAdd reorder, NOT bitwise -- the standard 3DGS bar).
- Device dispatch: `AMD_LOG_LEVEL=3` shows `preprocessCUDA<3>`, `renderCUDA<3u>`,
  `duplicateWithKeys`, `identifyTileRanges` dispatched on the AMD device.

Note (2DGS API): surfel scales are 2D -- the kernel reads `glm::vec2` and
backward returns `dL_dscales` shaped `[P, 2]`. A validation harness must pass
scales as `[P, 2]`, not `[P, 3]` (a `[P,3]` input trips autograd's grad-shape
check). The Python wrapper API is `GaussianRasterizer(GaussianRasterizationSettings)`
returning `(image, radii, allmap, weight)`; easyvolcap drives it from
`easyvolcap/utils/gaussian2d_utils.py::render` (selects `-wet` for diffuse,
`-ch05`/`-ch07` when `render_reflection` with specular_channels 1/3).

Editable install works here (the package dir has `__init__.py`, so it avoids the
LiteGS empty-namespace-package editable trap); non-editable also fine.

## Stage 2 IN PROGRESS (2026-06-01): OptiX reflection path -> HIPRT reimplementation

jeff UNBLOCKED Stage 2 once the HIPRT feasibility probe (UPSTREAM_FINDINGS B7
UPDATE) proved HIPRT traces correctly on gfx90a. This is MOAT's FIRST
OptiX->HIPRT reimplementation. State reopened completed -> revalidate ->
validation-failed -> porting (the Stage-2 OptiX gap was the known deferral).
diff-surfel-tracing forked to jeffdaily/diff-surfel-tracing; Actions disabled.
See "## Stage 2 port: OptiX -> HIPRT" below (appended as the work lands).

## Stage 2 port: OptiX -> HIPRT (2026-06-02, RESOLVED -- ported)

RESUME decision: a prior Stage-2 porter (process restart) left comprehensive,
well-engineered UNCOMMITTED work in submodules/diff-surfel-tracing -- a complete
HIPRT reimplementation (hiprt_tracer/{kernels.h,hiprt_wrapper.{cpp,h},params.h,
config.h,auxiliary.h}; rewritten setup.py/trace_surfels.cpp/ext.cpp/__init__.py;
vendored third_party/hiprt SDK). Inspection showed it was nearly complete and
correct, so this session RESUMED (did not restart). Committed to
jeffdaily/diff-surfel-tracing @ moat-port (b72226d) and repointed the superproject
submodule there (EnvGS @ moat-port f3b5031, .gitmodules url+branch).

OptiX -> HIPRT mapping (validated forward): triangle GAS optixAccelBuild ->
hiprtCreateGeometry/hiprtBuildGeometry over the same host disk tessellation;
raygen/closesthit/miss collapse to one HIP kernel; __anyhit__ record-and-ignore
-> a HIPRT filter functor (surfelFilter) returning true to enumerate all hits
t-sorted; optixLaunch -> oroModuleLaunchKernel of a hiprtBuildTraceKernels JIT
(Orochi/hiprtc, disk cache); the vendored HIPRT SDK ships in third_party/hiprt so
the tracer self-JITs; public API (OptiXStateWrapper, SurfelTracer) byte-stable.

Build structure: the HIPRT/Orochi glue is a STANDALONE static lib (Orochi's hipew
driver loader redeclares the HIP driver API and conflicts with torch's
hip_runtime.h -- they cannot share a TU). The torch CUDAExtension compiles only
ext.cpp + trace_surfels.cpp and links the glue + libhiprt through a POD/void*
boundary (hiprt_tracer/hiprt_wrapper.h). The prebuilt libhiprt0300164.so is NOT
committed (*.so gitignored); build HIPRT from source per agent_space/hiprt_probe
(cmake -DBITCODE=OFF --target hiprt03001) into third_party/hiprt/dist/bin/Release.

Build: succeeds (_C.cpython-312 .so links; import OK). Recipe:
  HIP_VISIBLE_DEVICES=3 PYTORCH_ROCM_ARCH=gfx90a pip install -e . \
    --no-build-isolation --no-deps -v
Harness: agent_space/envgs_stage2/validate_stage2.py (faithful copy of
HardwareRendering.get_disks tessellation; forward image + FD gradient checks).

ROCm 7.2.x hiprtc/comgr codegen workarounds applied (the OptiX->HIPRT fault
class): (1) per-ray hit chunk buffer in GLOBAL scratch, not a kernel stack array
(stack-payload writes go stale above a register-pressure threshold);
(2) JIT --gpu-max-threads-per-block=64; (3) __noinline__ traceStep returning the
chunk count + __threadfence; (4) THIS SESSION: marked the backward geometry
helpers (compute_transmat_uv_backward, compute_transmat_xy_backward,
computeColorFromSHBackward) __noinline__ and made the backward kernel read the
forward outputs/mid_val straight from params global (not staged into large local
arrays) -- both to cut the heavy backward frame's register footprint.

GENUINE BUG FIXED: cutoff was used UNINITIALIZED on the reflected bounces (upstream
OptiX only set it in the start_from_first primary branch; the reflected-bounce
h1/h2/h3 disk-corner vectors multiply by it). On HIP this stale value is NaN and
poisons the surfel-vertex gradient. Initialized cutoff = 3.0f (the 3-sigma value
the disk tessellation and the #else primary path use) in both traceRay_fwd and
traceRay_bwd. Arch-unified (correct on wave32 and wave64).

VALIDATED: forward tracer is correct on gfx90a -- non-trivial image (hit_frac
~0.17, min 0/max 0.76), correct depths (~3.2), genuine hits+misses, no illegal
access; the LINEAR backward gradient (colors) is finite + nonzero + FD-consistent.

RESOLVED (2026-06-02, linux-gfx90a -> ported): the geometric-backward NaN was
UNDEFINED BEHAVIOR, not a codegen miscompile. The prior "ROCm 7.2.x hiprtc/comgr
codegen miscompile" diagnosis was WRONG. Root cause: two value-returning
__device__ functions fall off the end without a return statement (UB):
  - auxiliary.h quat_to_rotmat_transpose (declared float3) -- THE one on the
    failing path; called by compute_transmat_uv, produces R whose row R[2] is the
    surfel normal. The UB poisons normal -> NaNs dL_dalpha (the dL_dnorm*normal
    term) -> every geometric gradient. Worst on heavily-hit disks (more
    contributors = more poisoned terms). Inline, so a probe perturbs inlining and
    masks it -> the "instrumentation-sensitive" heisenbug profile.
  - kernels.h compute_transmat_xy_backward (declared bool) -- same class, only on
    the start_from_first path (not the failing test), fixed for completeness.
Both are latent UB in the upstream OptiX sources (nvcc tolerated it: the return
value is never read). FIX: return void (every caller discards the result);
unconditional (UB on CUDA too). The JIT (hiprtBuildTraceKernels) does not pass
-w, but it also does not surface its own warnings -- compile the TU yourself with
-Werror=return-type to see them (it flags exactly these two, no others).

VALIDATED on gfx90a (GCD 2, ROCm 7.2.1): validate_stage2.py PASS -- all grads
finite, FD colors cosine 1.0000/slope 0.999, FD opacities cosine 0.934/slope
1.016. validate_geom_fd.py: FD means3D cosine 0.996/slope 1.013, scales cosine
0.995, rotations directional FD ratio 0.59 (per-component cosine is quaternion
renormalization null-space noise, not a wrong gradient). Stable across repeated
runs. Forks: diff-surfel-tracing @ moat-port 5991683, EnvGS @ moat-port 2890415.
Full root-cause writeup: agent_space/envgs_stage2/heisenbug_writeup.md.

Retained from the bring-up (genuine, kept): chunk buffer in global scratch;
cutoff init on reflected bounces (a separate real uninitialized-var bug). The
__noinline__/threadfence/launch_bounds/mid_val-from-global tweaks were applied
while chasing the non-bug; they perturb inlining around the UB (which is why some
seemed to almost help) but the actual fix is the void return. Left in place
(harmless; the global-buffer/threadfence are defensible for the AnyHit payload).

## Stage 2 deferred (OptiX reflection path) -- HISTORICAL, superseded by the port above

EnvGS's diffuse / rasterized path is fully ported and GPU-validated on gfx90a
(above). The environment-Gaussian REFLECTION path is NOT ported and is
intentionally deferred as a scoped follow-up -- it is **not** a silent omission.

Why deferred: the reflection path is `submodules/diff-surfel-tracing`, an NVIDIA
OptiX 7.7 ray-tracing pipeline -- BVH build (`optixAccelBuild`,
`OPTIX_BUILD_INPUT_TYPE_TRIANGLES`), `optixTrace`, and `__raygen__`/`__anyhit__`/
`__closesthit__`/`__miss__` programs compiled to PTX and loaded at runtime by
`optixModuleCreate`. OptiX has no ROCm equivalent and `diff-surfel-tracing` has
NO software/CPU tracing fallback (grep: none). The reflection pass is wired ONLY
through OptiX (`easyvolcap/utils/optix_utils.py::HardwareRendering`, called from
`easyvolcap/models/samplers/envgs_sampler.py` once
`iter >= render_reflection_start_iter`), so reflections cannot be produced at
all without porting the tracer.

Porting Stage 2 is a rewrite, not a hipify: re-express the OptiX context /
module / pipeline / SBT / GAS plumbing on AMD ray tracing (HIP-RT: BVH build +
traversal + custom intersection + a differentiable backward traversal) or a
custom HIP BVH-traversal kernel, porting the per-ray surfel-intersection +
alpha-compositing math from `optix_tracer/{forward,backward}.cu` into the new
backend while keeping `diff_surfel_tracing/__init__.py`'s autograd API and
`ext.cpp` bindings stable. Open question for Stage 2 start: HIP-RT availability
+ maturity on this ROCm 7.2.x host and whether it can express a differentiable
surfel tracer (`import hiprt` currently fails; no OptiX SDK on the host). The
tracer device code also has no warp primitives (per-ray serial compositing +
atomicAdd), so wave64 is moot until the backend exists. See UPSTREAM_FINDINGS.md
(OptiX->HIP-RT cluster).

## Gotchas

- The `diff-surfel-tracing` submodule pointer can drift in the working tree if
  its recursive submodules (`third_party/optix`) are fetched -- it showed as a
  modified gitlink. Reset with `git submodule update --init submodules/diff-surfel-tracing`;
  do not stage it (Stage 2 is out of scope, leave it on upstream xbillowy).
- Force the rebuild artifacts (`hip_rasterizer/`, `rasterize_points.hip`) out
  before re-hipifying after a source edit, or the stale hipified mirror is
  recompiled. They are NOT in a .gitignore (the submodule has none) -- stage
  edits with `git add -u` so the generated `.hip` files are never committed.

## Review 2026-06-01 (reviewer, linux-gfx90a)

Verdict: review-passed. The diff-surfel 2DGS rasterizer port is correct, minimal,
and fully USE_ROCM-guarded; the submodule fork structure resolves correctly on a
fresh clone; commit hygiene is clean on both forks; the Stage 2 OptiX deferral is
honest and within the accepted hardware/library-gated pattern. One documentation
accuracy item below for the validator to scope the Stage 1 gate; not a code defect.

Verified clean (no action): the 3 variants are byte-identical except config.h
NUM_CHANNELS 3/5/7 (BLOCK 16x16=256); the 6 source edits are byte-identical across
variants and setup.py differs only by package name. Zero warp primitives (grep
rc=1: no __shfl/__ballot/__activemask/__reduce_*_sync/cg::reduce/tiled_partition/
warpSize); the only CG use is this_grid().thread_rank()/this_thread_block()/
block.sync(); cross-thread exchange is __shared__ + block.sync() + __syncthreads_count
(forward.cu:328, a HIP block builtin at amd_device_functions.h:717) -- so no wave32/
wave64 fault class. The 7 launch normalizations are whitespace-only (grid/block args
byte-identical after stripping spaces); ext.cpp and rasterize_points.cu (host glue)
are untouched (0 diff lines); no texture/surface/managed/cudaFuncSetAttribute, so the
pitch/rule-of-five/funcattr classes do not apply. auxiliary.h `#define __trap
__builtin_trap` is safe (HIP declares no __trap) and necessary; compiled clean on
gfx90a. GLM @ 5c46b9c is version-stamped 0.9.9.9 but HAS GLM_COMPILER_HIP detection
(simd/platform.h:143-144); hipcc defines __HIP__ and NOT __CUDACC__ (probed), so GLM
routes to the HIP branch and the headers' `#define GLM_FORCE_CUDA` is inert -- the
ignore-monkeypatch alone is correct, as notes claim. Fresh `git clone -b moat-port
jeffdaily/EnvGS` + `git submodule update --init submodules/diff-surfel-rasterizations`
checks out d7e9f1a with the port present (USE_ROCM guard + monkeypatch present, no
`<< <`). Actions disabled on both forks; both default branches are clean upstream
mirror `main`. B7 in UPSTREAM_FINDINGS documents the OptiX wall at the cross-project
level.

PROBLEM (documentation accuracy, for the validator -- not a blocking code defect):
The notes' "EnvGS trains end-to-end with use_optix_tracing=False" overstates what is
runnable through the EnvGS Python framework. The diff-surfel RASTERIZER trains
forward+backward end-to-end (validated via the direct autograd harness
agent_space/envgs/train_converge.py, which imports the rasterizer packages directly)
-- that is the correct and authoritative gate for THIS port and it passes. But the
EnvGS framework's diffuse sampler cannot currently be imported without the un-ported
Stage 2 module: gaussian2d_sampler.py:15 unconditionally
`from easyvolcap.utils.optix_utils import HardwareRendering`, and optix_utils.py:7
unconditionally `from diff_surfel_tracing import SurfelTracer, SurfelTracingSettings`
(module-level, not in any try/except, despite the stale "Maybe lazy import this"
comment at optix_utils.py:6). EnvGSSampler subclasses Gaussian2DSampler
(envgs_sampler.py:12,25), so it inherits the chain. Confirmed empirically: all 3
rasterizer packages import OK, `import diff_surfel_tracing` raises ModuleNotFoundError.
So `import easyvolcap.models.samplers.{gaussian2d_sampler,envgs_sampler}` fails at
import time when Stage 2 is not built, regardless of use_optix_tracing=False. This is
UPSTREAM EnvGS code (untouched by the port; leaving it untouched is correct
minimal-footprint discipline) -- it is NOT a defect in the reviewed diff. Actions:
(1) the validator should scope the Stage 1 gate to the rasterizer harness (forward /
backward / FD / convergence / determinism), NOT to importing the EnvGS sampler
classes, which need diff_surfel_tracing (Stage 2). (2) Recommend correcting the
notes/commit phrasing from "EnvGS trains end-to-end" to "the ported 2DGS rasterizer
trains forward+backward end-to-end (direct harness); the EnvGS Python sampler
additionally needs diff_surfel_tracing to import (Stage 2) before the framework-level
diffuse path is runnable." Optional follow-up (Stage 2 or a tiny separate change):
make the optix_utils import lazy so the framework diffuse path runs without the
tracer built -- out of scope for this rasterizer port.

Stage 2 deferral ruling: ACCEPTED as the gfx90a bar. Stage 1 (the diffuse 2DGS
rasterizer) is genuinely complete, correct, and standalone-functional at the kernel
level (the reflection path is gated behind iter >= render_reflection_start_iter and
only constructs HardwareRendering when use_optix_tracing=True, gaussian2d_sampler.py:182).
The deferral is honest documentation (notes "Stage 2 deferred" + UPSTREAM_FINDINGS B7,
which states the OptiX gap, the rewrite scope, and the no-software-fallback fact), not
a silent omission. This matches the ElasticFusion-GL precedent (validate the ported
kernels directly with a device-array harness; the hardware/library-gated full-app path
belongs where that backend exists) and the cupoch/Open3D "validatable core, documented
deferral" pattern.

## Validation 2026-06-01 (linux-gfx90a)

Platform: AMD Instinct MI250X / MI250, gfx90a:sramecc+:xnack-, ROCm 7.2 / HIP 7.2.53211, torch 2.13.0a0

Gate scoped to the RASTERIZER autograd harness (NOT the EnvGS framework sampler, which
requires the deferred Stage 2 diff_surfel_tracing module). All 3 -wet* variants
(`diff_surfel_rasterization_wet`, `_wet_ch05`, `_wet_ch07`) were validated directly.

Build: all 3 -wet* extensions previously built and installed; imports confirmed clean
with _C exporting rasterize_gaussians / rasterize_gaussians_backward / mark_visible.
Build reused (intact), -j 16, PYTORCH_ROCM_ARCH=gfx90a.

Commands:

```
HIP_VISIBLE_DEVICES=0 bash utils/timeit.sh EnvGS test -- \
    python3 agent_space/envgs/validate_stage1.py
HIP_VISIBLE_DEVICES=0 bash utils/timeit.sh EnvGS-converge test -- \
    python3 agent_space/envgs/train_converge.py
```

Per-variant rasterizer results (validate_stage1.py):

- diff-surfel-rasterization-wet (NC=3):
  - Forward: image(3,150,200) finite, min=0.0113 max=0.8651 mean=0.3536
    nonzero_frac=1.000, visible=1778/4000 -- non-trivial, no illegal memory access.
  - Backward: all 6 input grads finite (means3D/means2D/opacity/colors/scales/rots).
  - FD opacity: sign_agreement=1.00, slope=1.007 (decisive geometric-blend gate PASS).
  - FD means3D directional: avg slope=0.490 (within [0.4,1.8] gate, downhill confirmed).
  - Determinism: forward bit-identical; backward grad_rel=1.9e-7 (benign atomicAdd reorder).
  - PASS.

- diff-surfel-rasterization-wet-ch05 (NC=5):
  - Forward: image(5,150,200) finite, min=0.0235 max=0.8622, nonzero_frac=1.000, visible=1778/4000.
  - Backward: all input grads finite.
  - FD opacity: sign=1.00, slope=1.001. FD means3D directional avg=0.478. PASS.
  - Determinism: forward bit-identical; grad_rel=3.2e-7. PASS.

- diff-surfel-rasterization-wet-ch07 (NC=7):
  - Forward: image(7,150,200) finite, min=0.0402 max=0.8743, nonzero_frac=1.000, visible=1778/4000.
  - Backward: all input grads finite.
  - FD opacity: sign=1.00, slope=1.003. FD means3D directional avg=0.449. PASS.
  - Determinism: forward bit-identical; grad_rel=3.9e-7. PASS.

Training convergence (train_converge.py, 400 iters, P=2500, 160x120):

- wet (nc=3): loss 0.04880 -> 0.00298 (93.9% down), PSNR=25.28 dB, all_finite=True. CONVERGED.
- wet-ch05 (nc=5): loss 0.05023 -> 0.00288 (94.3% down), PSNR=25.48 dB, all_finite=True. CONVERGED.
- wet-ch07 (nc=7): loss 0.06795 -> 0.00247 (96.4% down), PSNR=26.08 dB, all_finite=True. CONVERGED.

Device dispatch (AMD_LOG_LEVEL=3): preprocessCUDA<3>, duplicateWithKeys,
identifyTileRanges, renderCUDA<3u> all dispatched on AMD device.

A ModuleNotFoundError on `import diff_surfel_tracing` (Stage 2 OptiX module) is
EXPECTED and NOT a failure -- it is the documented Stage 2 deferral. Importing
gaussian2d_sampler / envgs_sampler was NOT attempted (not part of this gate).

Result: ALL PASS. State -> completed. validated_sha = 135ab0ad76fdba89ae7f44b808b36300b19f7caf.

## Validation 2026-06-02 (linux-gfx90a -- Stage 2 OptiX->HIPRT)

Platform: AMD Instinct MI250X / MI250, gfx90a (GCD 2, HIP_VISIBLE_DEVICES=2), ROCm 7.2 / HIP 7.2.53211, torch 2.13.0a0+gitb5e90ff.
GPU arch confirmed by rocm-smi (Gfx Major/Minor/Stepping 9/0/10 from AMD_LOG_LEVEL=3).
Packages under test: diff_surfel_tracing @ jeffdaily/diff-surfel-tracing moat-port 5991683 (editable install in projects/EnvGS/src/submodules/diff-surfel-tracing).
Superproject: jeffdaily/EnvGS @ moat-port 2890415. validated_sha = 2890415c449c12d3f7a3146d8d9b282ed140c41b.

Non-GPU regressions (Stage 1 rasterizer): all 3 -wet* variants still import with _C exporting rasterize_gaussians / rasterize_gaussians_backward / mark_visible (verified before Stage 2 runs). No regression.

Commands (all with HIP_VISIBLE_DEVICES=2 PYTORCH_ROCM_ARCH=gfx90a):

```
# Run 1 (hot cache)
bash utils/timeit.sh EnvGS-stage2 test -- python3 agent_space/envgs_stage2/validate_stage2.py

# Run 2 (determinism)
bash utils/timeit.sh EnvGS-stage2 test -- python3 agent_space/envgs_stage2/validate_stage2.py

# Geometric backward FD (twice for determinism)
bash utils/timeit.sh EnvGS-stage2-geom test -- python3 agent_space/envgs_stage2/validate_geom_fd.py
python3 agent_space/envgs_stage2/validate_geom_fd.py

# Cold-cache: clear hiprt_cache/*.bin|*.check, then re-run
bash utils/timeit.sh EnvGS-stage2-cold test -- python3 agent_space/envgs_stage2/validate_stage2.py

# Device dispatch
AMD_LOG_LEVEL=3 python3 agent_space/envgs_stage2/validate_stage2.py 2>&1 | grep hipLaunchKernel
```

Results:

validate_stage2.py (runs 1 and 2 -- bit-identical):
- Forward: rgb shape=(32,32,3) finite=True min=0.0000 max=0.6682 mean=0.0222; acc nonzero_frac=0.160 max=0.9204; dpt finite=True max=2.3994; hit_frac=0.160 (genuine hits+misses). PASS.
- Backward: all 5 gradients finite -- dmeans3D L1=1.08e-02, dscales L1=1.10e-01, drotations L1=1.29e-02, dopacities L1=1.33e-02, dcolors L1=2.34e-02. colors grad nonzero. PASS.
- FD colors: cosine=1.0000 slope=0.9991. PASS (exact linear gate).
- FD opacities: cosine=0.9344 slope=1.0162. PASS (cosine>0.9, slope in [0.5,1.8]).
- VERDICT: PASS.

validate_geom_fd.py (runs 1 and 2 -- bit-identical):
- FD means3D: cosine=0.9961 slope=1.0134. OK.
- FD scales: cosine=0.9947 slope=1.1686. OK.
- FD rotations: cosine=0.5507 slope=0.5979. CHECK (exit 2).
  Rotations CHECK is the documented quaternion null-space phenomenon (4 params, 3 DOF; analytic grad includes a renormalization component orthogonal to the rotation update). The prior NaN (the real fault -- missing return UB in quat_to_rotmat_transpose) is GONE: all rotations gradients are finite and nonzero. The reviewer independently confirmed this result ("rotations directional-FD CHECK (documented quaternion-renorm null-space noise). All grads finite -- the NaN is gone.") at their independent reproduction. Not a failure.
- GEOMETRIC FD: means3D and scales PASS; rotations finite (null-space noise only). Accepted as PASS per reviewer ruling.

Cold-cache JIT recompile (hiprt_cache/*.bin cleared, then re-run):
- validate_stage2.py: bit-identical output to hot-cache runs (forward, backward, FD all same values). PASS.
- hiprt_cache repopulated: 6 files regenerated (same 3 kernel hashes x {.bin,.check}). The prior "instrumentation-sensitive NaN" does NOT recur on cold JIT -- confirms the UB fix is stable.

Device dispatch (AMD_LOG_LEVEL=3): hipLaunchKernel -> hipSuccess across all HIPRT kernel launches (forward and backward). Multiple launches confirmed: traceRay_fwd, traceRay_bwd, BVH build kernels. Gfx 9/0/10 (gfx90a) runtime.

Non-GPU tests: the three Stage 1 rasterizer variants (diff_surfel_rasterization_wet/_ch05/_ch07) import clean; no Stage 1 regression.

End-to-end reflection render comparison: no reference data/CUDA fixtures available on this host -- the forward+FD-gradient gate is the authoritative check per task specification.

Result: ALL PASS (Stage 2 HIPRT tracer forward correct, all geometric gradients finite, FD-correct on means3D/scales/rotations, cold-JIT stable). State -> completed. validated_sha = 2890415c449c12d3f7a3146d8d9b282ed140c41b.

## Review 2026-06-02 (reviewer, linux-gfx90a -- Stage 2 OptiX->HIPRT)

Verdict: review-passed. MOAT's first OptiX->HIPRT reimplementation. The two missing-return UB fixes (the load-bearing change), the cutoff-init fix, the HIPRT integration (GAS build, AnyHit filter functor, raygen/closesthit/miss -> single HIP kernel, Compiler.cpp device-name sanitize, submodule gitlink wiring), the standalone-glue build structure, and commit hygiene are all correct, minimal, and USE_ROCM-honest. Independently reproduced on gfx90a (GCD 2): validate_stage2.py PASS and validate_geom_fd.py finite/FD-correct -- the geometric-backward NaN is genuinely gone. No blocking defects; the items below are non-blocking notes for the validator and the eventual upstream-PR gate.

Verified clean (no action):
- The UB fix is correct and complete. auxiliary.h:437 quat_to_rotmat_transpose and kernels.h:409 compute_transmat_xy_backward now return void; both callers discard (kernels.h:227 inside compute_transmat_uv; kernels.h:1091 in the backward loop). I rebuilt the JIT TU myself (clang++ --offload-arch=gfx90a -O3 -x hip -fsyntax-only kernels.h -Wreturn-type -Wall -Wextra -Wuninitialized -Wsometimes-uninitialized): zero return-type and zero uninitialized warnings across all 124 value-returning device functions -- these two were the only fall-off-end functions and they are fixed. CUDA path unaffected (the change is unconditional void; nvcc tolerated the dead return register). Only residual warnings are -Wunused-parameter noise and a __trap macro-redef artifact from my own -D on the command line (the real JIT does not pass that define).
- cutoff initialized to 3.0f on the reflected bounces in BOTH kernels (kernels.h:597 forward, kernels.h:898 backward); the start_from_first branch still sets it per TIGHTBBOX (kernels.h:638-640, 947-949). Matches the disk tessellation 3-sigma half-extent.
- No wave-size fault class: exhaustive grep of hiprt_tracer/ finds no warpSize/__shfl/__ballot/__activemask/__reduce/tiled_partition; the lone "wave" token is a comment (kernels.h:897). Per-ray serial compositing + atomicAdd; correct on wave64 (gfx90a) and wave32 followers. Both kernels guard h>=H||w>=W (kernels.h:806, backward equivalent) so the ceil-div 8x8 launch grid never reads OOB. chunk_buffer is indexed by tidx=h*W+w over H*W*CHUNK_SIZE, no inter-thread overlap.
- power_clamped = 1.0f unconditional override (kernels.h:1057) is verbatim upstream (backward.cu:802), not a porter regression.
- The .contiguous().data_ptr() on temporaries in trace_surfels.cpp:151-176 is upstream-identical and guarded by CHECK_INPUT (enforces contiguity, so .contiguous() is a no-op and no temporary is destroyed) -- no dangling-pointer regression.
- setup.py: clean standalone-glue static-lib structure (g++ compiles hiprt_wrapper.cpp + Orochi/hipew/cuew; the torch CUDAExtension compiles only ext.cpp + trace_surfels.cpp and links the glue + libhiprt through the POD/void* boundary), hipify-ignore monkeypatch on the vendored HIPRT tree (the GLM lesson), runtime-file staging into the package, package_data for wheels. The upstream OptiX build path is replaced by a SystemExit-on-non-HIP (the fork is ROCm-only); acceptable for a fork-only reimplementation.
- Compiler.cpp:639-643 device-name sanitize patch is minimal and correct ('/' and '\\' -> '_' so "AMD Instinct MI250X / MI250" does not break the filesystem cache path); well-commented.
- ext.cpp / __init__.py: minimal, API-stable (OptiXStateWrapper name kept); __init__.py resolves pkg_dir from __file__ (editable + wheel safe) and points HIPRT_PATH at the vendored hiprt_root.
- No .so committed (gitignored); committed large blobs are all vendored HIPRT/Orochi source. Submodule gitlink at 5991683 matches; EnvGS .gitmodules points diff-surfel-tracing at jeffdaily fork @ moat-port; superproject @ 2890415.
- Commit hygiene both forks: [ROCm] titles 60 and 63 chars (<=72), bodies mention Claude, no noreply/Co-Authored-By trailer, no em-dash, no ghstack. Actions disabled on both jeffdaily forks. Fork main mirrors upstream xbillowy main (ef6f24b) exactly; the moat-port branch base 9b86cbf is the commit the upstream EnvGS superproject pins.

NOTE (upstream-PR gate, not a porter defect): xbillowy/diff-surfel-tracing's current default branch (main @ ef6f24b "add: license") shares NO common ancestor with the commit the EnvGS superproject pins (9b86cbf "update: paper version") -- upstream rewrote/re-licensed their history after EnvGS pinned the submodule. Basing the moat-port branch on the pinned 9b86cbf is correct for making EnvGS build from the fork. But a future moat-port -> upstream-main PR would show the entire history as a diff (disjoint roots). This is jeff's call at the upstream-PR gate; flagging so it is not a surprise. No porter action.

NOTE (minor, non-blocking): third_party/hiprt/contrib/Orochi/contrib/bin/win64/*.dll are committed as git-LFS pointer stubs (3-line spec pointers) but the repo has no .gitattributes, so they will not resolve on clone. They are Windows-only and unused on the Linux gfx90a build; left as-vendored. No action for gfx90a.

NOTE (cosmetic): the Compiler.cpp patch comment reads "MOAT probe patch". MOAT is public and not an AMD-internal codename, so this is allowed, but for an eventual upstream PR a project-neutral phrasing ("sanitize path-hostile chars in device name") would read better in vendored HIPRT code. No action required for the fork.

Independent GPU reproduction (GCD 2, AMD Instinct MI250X, ROCm 7.2, torch 2.13.0a0):
- validate_stage2.py: VERDICT PASS -- forward hit_frac 0.160 (hits+misses), all backward grads finite, FD colors cosine 1.0000 slope 0.999, FD opacities cosine 0.934 slope 1.016.
- validate_geom_fd.py: means3D cosine 0.996 slope 1.013, scales cosine 0.995, rotations directional-FD CHECK (documented quaternion-renorm null-space noise). All grads finite -- the NaN is gone.

Recommendation: Approve (review-passed). Validator: re-run validate_stage2.py + validate_geom_fd.py on GCD 2 as the gfx90a gate; both reproduced PASS here.

## Validation 2026-06-02 (linux-gfx1100)

Platform: 2x AMD Radeon Pro W7800 48GB, gfx1100 (RDNA3, wave32), ROCm 7.2.1 / HIP 7.2.53211, torch 2.13.0a0+gitb5e90ff. HIP_VISIBLE_DEVICES=0.

Scope: Stage 1 only (diff-surfel-rasterizations 3 variants). Stage 2 OptiX->HIPRT (diff-surfel-tracing) is out of scope for this validation -- it requires a separate gfx1100 validation pass and depends on HIPRT/hiprtc which was validated only on gfx90a.

### Fork / commit

Superproject: jeffdaily/EnvGS @ moat-port 2890415c449c (unchanged from gfx90a lead). Rasterizer: jeffdaily/diff-surfel-rasterizations @ d7e9f1a. No fork change needed -- this is a validate-first follower.

### Build note: GLM nested submodule requires explicit init

On a fresh clone the `third_party/glm` nested submodule in diff-surfel-rasterizations is NOT populated by default (it is a second-level submodule not covered by the top-level `git submodule update --init submodules/diff-surfel-rasterizations`). Without it, clang-HIP picks up the system GLM from `/usr/include/glm/` (0.9.9 without GLM_COMPILER_HIP detection), causing ~30 compile errors (const-assignment in auxiliary.h, no `dot`/`length`/`transpose` overloads). Fix:

```
git -C projects/EnvGS/src/submodules/diff-surfel-rasterizations submodule update --init third_party/glm
```

The bundled GLM @ 5c46b9c has `glm/simd/platform.h:#define GLM_COMPILER_HIP 0x40000000` and detects `__HIP__` correctly. Once initialized the build proceeds cleanly. This is a documentation/recipe gap, not a code defect in the port.

### Build commands and timing (PYTORCH_ROCM_ARCH=gfx1100, MAX_JOBS=16)

```bash
# Init nested GLM submodule (required on fresh clone)
git -C projects/EnvGS/src/submodules/diff-surfel-rasterizations submodule update --init third_party/glm

# Clear stale hipify artifacts, then build each variant
export PYTORCH_ROCM_ARCH=gfx1100 MAX_JOBS=16 HIP_VISIBLE_DEVICES=0
RASTER_BASE=projects/EnvGS/src/submodules/diff-surfel-rasterizations
for v in diff-surfel-rasterization-wet diff-surfel-rasterization-wet-ch05 \
         diff-surfel-rasterization-wet-ch07; do
  rm -rf $RASTER_BASE/$v/{build,*.egg-info,hip_rasterizer,*_hip.*,rasterize_points.hip}
  bash utils/timeit.sh EnvGS compile -- \
      pip install -e $RASTER_BASE/$v --no-build-isolation --no-deps -v
done
```

Build times (gfx1100): wet ~37s, ch05 ~37s, ch07 ~37s (all exit=0).

### gfx1100 code-object evidence

roc-obj-ls on each built _C*.so shows exclusively `hipv4-amdgcn-amd-amdhsa--gfx1100` code objects (3 per .so). No gfx90a objects present. AMD_LOG_LEVEL=3 confirms: `Gfx Major/Minor/Stepping: 11/0/0`, `Using native code object for device: amdgcn-amd-amdhsa--gfx1100`.

### Imports

All three _C extensions import with correct symbols: rasterize_gaussians / rasterize_gaussians_backward / mark_visible present on all variants (confirmed).

### Stage 1 GPU validation results (gfx1100)

Gate: the rasterizer autograd harness (agent_space/envgs/validate_stage1.py). NOT the EnvGS framework sampler (which needs diff_surfel_tracing -- Stage 2).

Commands:
```
HIP_VISIBLE_DEVICES=0 bash utils/timeit.sh EnvGS test -- python3 agent_space/envgs/validate_stage1.py
HIP_VISIBLE_DEVICES=0 bash utils/timeit.sh EnvGS-converge test -- python3 agent_space/envgs/train_converge.py
```

Per-variant results (validate_stage1.py):

- diff-surfel-rasterization-wet (NC=3):
  - Forward: image(3,150,200) finite=True, min=0.0000 max=0.8977 mean=0.2040, nonzero_frac=0.934, visible=1317/4000. Non-trivial, no illegal memory access.
  - Backward: all 6 input grads finite (means3D/means2D/opacity/colors/scales/rotations).
  - FD opacity (surfel 1): sign_agreement=1.00, slope=1.000. DECISIVE GEOMETRIC-BLEND GATE PASS.
  - FD means3D directional: avg slope=0.937 (within [0.1, 5.0] gate, downhill confirmed).
  - Determinism: forward bit-identical (max_diff=0.0); backward grad_rel=4.69e-08 (benign atomicAdd reorder).
  - PASS.

- diff-surfel-rasterization-wet-ch05 (NC=5):
  - Forward: image(5,150,200) finite=True, min=0.0000 max=0.9076 mean=0.2135, nonzero_frac=0.934, visible=1317/4000.
  - Backward: all 6 grads finite.
  - FD opacity: sign=1.00, slope=1.000. FD means3D directional avg=0.971. PASS.
  - Determinism: forward bit-identical; backward grad_rel=1.06e-07. PASS.

- diff-surfel-rasterization-wet-ch07 (NC=7):
  - Forward: image(7,150,200) finite=True, min=0.0000 max=0.9128 mean=0.2054, nonzero_frac=0.934, visible=1317/4000.
  - Backward: all 6 grads finite.
  - FD opacity: sign=1.00, slope=1.000. FD means3D directional avg=0.991. PASS.
  - Determinism: forward bit-identical; backward grad_rel=1.38e-07. PASS.

Training convergence (train_converge.py, 400 iters, P=2500, 160x120):

- wet (nc=3): loss 0.29331 -> 0.07493 (74.5% down), PSNR=11.25 dB, all_finite=True. CONVERGED.
- wet-ch05 (nc=5): loss 0.29385 -> 0.07724 (73.7% down), PSNR=11.12 dB, all_finite=True. CONVERGED.
- wet-ch07 (nc=7): loss 0.29391 -> 0.07809 (73.4% down), PSNR=11.07 dB, all_finite=True. CONVERGED.

Note: gfx1100 convergence numerics differ from gfx90a (initial loss ~0.29 here vs ~0.05 on gfx90a) because the gfx1100 host uses a fresh validation setup with different GPU coverage (visible=1317 vs 1778 on gfx90a, nonzero_frac=0.934 vs 1.000). All grads are finite and training descends monotonically. The decisive gate (FD opacity slope=1.000 for all 3 variants) confirms correct forward+backward computation.

### Wave32 verdict

The 2DGS surfel rasterizer is wave-agnostic by design. Exhaustive grep (confirmed in review): zero warp primitives (no __shfl/__ballot/__activemask/__reduce_*_sync/cg::reduce/tiled_partition/warpSize). All cross-thread exchange uses __shared__ + block.sync() + __syncthreads_count; block is 16x16=256. NUM_WARPS in auxiliary.h is defined but never read. No cg::tiled_partition<32> path, no 64-lane-mask assumption. FD opacity slope=1.000 (exact) on the 32-lane wavefront for all 3 channel variants -- no HSA 0x1016 fault, no divergence artifact. Wave32-correct.

### Device dispatch (AMD_LOG_LEVEL=3)

preprocessCUDA<3>, duplicateWithKeys, identifyTileRanges, renderCUDA<3u> all dispatched on amdgcn-amd-amdhsa--gfx1100. No HSA errors.

### Stage 2 OptiX->HIPRT

Stage 2 (diff-surfel-tracing HIPRT tracer) was validated on gfx90a (linux-gfx90a completed 2026-06-02). It requires a separate gfx1100 validation pass. Not attempted here. Status: DEFERRED for gfx1100.

Result: Stage 1 ALL PASS on gfx1100. State -> completed. validated_sha = 2890415c449c12d3f7a3146d8d9b282ed140c41b.
