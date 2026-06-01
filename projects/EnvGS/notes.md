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
  all 3 channel counts.
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

## Stage 2 deferred (OptiX reflection path)

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
