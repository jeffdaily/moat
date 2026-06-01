# op43dgs notes (ROCm/HIP port)

LetianHuang/op43dgs (ECCV'24 "Optimal Projection" 3DGS). Strategy B (torch
CUDAExtension auto-hipify). Lead platform linux-gfx90a (MI250X, ROCm 7.2.1,
torch 2.13.0a0 / HIP 7.2.5). Fork: jeffdaily/op43dgs, port on `moat-port`
(default branch `main` is a clean upstream mirror). Base sha 728de13.

This is the canonical Inria diff-gaussian-rasterization + simple-knn in THREE
camera variants (pinhole / fisheye / panorama), wave-AGNOSTIC: zero warp
intrinsics, zero PTX, zero half2, zero cg::reduce, zero textures/managed memory.
The only Cooperative Groups use is block/grid scaffolding (this_grid/thread_rank,
this_thread_block/sync). Block is 16x16=256 = 4 wavefronts on wave64; all
cross-thread communication is __shared__ + block.sync(). No wave64 lane-math
rework was needed (confirmed: deterministic forward + passing FD backward).

## Three variants, one package name (key build fact)
submodules/diff-gaussian-rasterization-{pinhole,fisheye,panorama} ALL declare
the module `diff_gaussian_rasterization._C`. They are MUTUALLY EXCLUSIVE installs
-- only one resolves at a time. The build script `pip uninstall -y
diff_gaussian_rasterization` before each variant install. The three share ~95%
of the code; the only substantive divergence is the projection math
(computeCov2D in forward.cu + its analytic backward in backward.cu). The HIP
edits are IDENTICAL across all three (applied by agent_space/op43dgs/
apply_variant_edits.py for fisheye/panorama, by hand for pinhole). pinhole is
the primary gate (environment.yml default + the trainer's import).

## Fixes applied (all build-time, USE_ROCM-guarded or setup.py-only; CUDA byte-identical)
1. Guard `#include <cooperative_groups/reduce.h>` under `#if !defined(USE_ROCM)`
   (absent in ROCm 7.2.1; include-only -- cg::reduce is NEVER called here, unlike
   gsplat/joeyan, so nothing to replace). 3 sites per rasterizer + 1 in simple-knn.
2. Normalize the 27 MUSA/Inria-style spaced kernel launches `<< <` -> `<<<` and
   `>> >` -> `>>>` (clang-HIP's parser rejects `name << <...>` "expected
   expression"; nvcc tolerates it). 8 per rasterizer (24) + 3 in simple-knn = 27.
   ZERO standard `<<<` in the stock tree, so the whole repo uses the spaced form.
3. Bundled-GLM hipify-protection monkeypatch in each rasterizer setup.py (x3).
   GLM 0.9.9.9 (GLM_VERSION 999), 139 .inl files each. torch's AOT hipify walks
   every header reachable via header_include_dirs and content-rewrites it,
   copying only .hpp/.h (dropping the .inl files) and corrupting GLM detection.
   Fix (gsplat precedent): monkeypatch torch.utils.hipify.hipify_python.hipify to
   (a) add the variant's third_party/glm dir to `ignores` and (b) strip it from
   `header_include_dirs`; GLM then resolves pristine from source via the existing
   -I. Gated on `PYTORCH_ROCM_ARCH or os.path.exists('/opt/rocm')` so the CUDA
   build is untouched. simple-knn vendors no GLM -> its setup.py is unchanged.
   NOTE: GLM 0.9.9.9 HAS a GLM_COMPILER_HIP path (simd/platform.h: it checks
   __CUDACC__ first, else __HIP__), so under hipcc (which defines __HIP__ not
   __CUDACC__) GLM correctly emits __device__ __host__ with NO __CUDACC__ hack.
   The stray `#define GLM_FORCE_CUDA` in rasterizer_impl.cu/forward.h/backward.h
   is INERT on HIP (it only matters inside GLM's __CUDACC__ branch) and was left
   as-is. The cub:: namespace was NOT a problem (torch hipify maps both the
   include and the namespace here) -- no `namespace cub = hipcub;` alias needed.

### Two extra walls the plan did not list (both trivial, both from guarding out device_launch_parameters.h)
4. `#include "device_launch_parameters.h"` is a CUDA-only header (threadIdx/
   blockIdx/blockDim decls); ROCm ships no shim and torch hipify has no mapping.
   On HIP these symbols come from <hip/hip_runtime.h> (which hipify injects), so
   the include is redundant -- guard it `#if !defined(USE_ROCM)`. 1 .cu + 2 .h
   per rasterizer; 1 in simple-knn.
5. `FLT_MAX` (simple_knn.cu, device code) and `__trap()` (auxiliary.h in_frustum,
   each rasterizer) were reaching their decls transitively through the CUDA
   toolkit headers that device_launch_parameters.h pulled in. Once that include
   is guarded out on HIP they go undeclared:
   - simple_knn.cu: add `#include <cfloat>` (standard, portable, idempotent on
     CUDA) for FLT_MAX.
   - auxiliary.h: `#if defined(USE_ROCM) #define __trap __builtin_trap #endif`
     (HIP has no __trap intrinsic; __builtin_trap() is the clang device-callable
     equivalent -- illegal-instruction trap, like CUDA __trap()). Call site
     unchanged.
6. simple_knn.cu also had `#define __CUDACC__` right before <cooperative_groups.h>
   (a stock-Inria trick to force CG device qualifiers). Guarded out under
   `#if !defined(USE_ROCM)`: HIP's hip_cooperative_groups.h needs no such hint,
   and a stray __CUDACC__ under hipcc risks steering other headers (incl. cub/
   thrust) down a CUDA path. The rasterizers do NOT define __CUDACC__ (only
   simple-knn did), so GLM in the rasterizers detects __HIP__.

## Build recipe (gfx90a)
Build cwd must be OUTSIDE /var/lib/jenkins/pytorch (that source tree shadows the
installed torch and breaks CUDAExtension hipify). Cap `-j 16` (shared host).
```
export HIP_VISIBLE_DEVICES=2          # this host: GCD 2 free (0=raft,1=FAISS,3=EnvGS)
export PYTORCH_ROCM_ARCH=gfx90a       # follower: gfx1100 / gfx1151, NO source change
export MAX_JOBS=16
P=/opt/conda/envs/py_3.12/bin/python
SRC=/var/lib/jenkins/moat/projects/op43dgs/src

# simple-knn (shared by all variants)
rm -rf $SRC/submodules/simple-knn/{build,*.egg-info}; find $SRC/submodules/simple-knn -name '*.hip' -delete
$P -m pip install $SRC/submodules/simple-knn --no-build-isolation --no-deps

# rasterizer -- ONE variant at a time (shared module name); uninstall the prior first
$P -m pip uninstall -y diff_gaussian_rasterization
rm -rf $SRC/submodules/diff-gaussian-rasterization-pinhole/{build,*.egg-info,hip_rasterizer}
$P -m pip install $SRC/submodules/diff-gaussian-rasterization-pinhole --no-build-isolation --no-deps
#   then repeat for -panorama and -fisheye (uninstall between)
```
Helper scripts: agent_space/op43dgs/build_simpleknn.sh, build_raster.sh <variant>.
INCREMENTAL gotcha (Strategy B): after editing a .cu, delete that extension's
build/ AND the hipified mirror (hip_rasterizer/, *.hip) before rebuilding or
torch recompiles the STALE hipified copy. The .gitignore excludes build/,
*.egg-info, *.hip, hip_rasterizer/, __pycache__, *.so.
Torch auto-defines USE_ROCM for the HIP compile (confirmed: the include guards
fired with no -DUSE_ROCM in setup.py).
Non-torch python deps the trainer needs: `plyfile tqdm` (+ opencv-python
torchvision for the optional eval), installed `--no-deps` so the host ROCm torch
wins; do NOT install environment.yml's torch==1.12.1.

## Validation (real gfx90a, HIP_VISIBLE_DEVICES=2) -- no formal test suite; built from the public ops
Harnesses in agent_space/op43dgs/: raster_common.py (scene/camera using the
project's own getWorld2View2/getProjectionMatrix conventions), tier1_forward.py,
tier2_backward.py, tier3_train.py, fish_fit.py (single-cam fit), val_simpleknn.py.
- simple-knn distCUDA2: finite + nonneg + bitwise-deterministic across 2 runs.
- pinhole (PRIMARY): Tier1 forward finite + 24% coverage + bitwise-deterministic;
  Tier2 FD backward opac/scales/sh slope ~1.0, means sign-agreement 98% (see
  below), grad-sums stable ~1e-9; Tier3 synthetic 5-view fit loss 0.0115->0.0005,
  PSNR 25.7->49.9 dB, no NaN. AMD_LOG_LEVEL=3 confirms preprocessCUDA<3>,
  renderCUDA<3u>, duplicateWithKeys, identifyTileRanges dispatch on GPU.
- fisheye: Tier1 finite+deterministic; Tier2 opac/sh slope ~1.0, scales+means
  sign-gated; single-cam fit PSNR 22.3->29.8 dB.
- panorama: Tier1 finite+deterministic; Tier2 opac/sh slope ~1.0, scales 0.99,
  means slope 1.77 (sign 100%); single-cam fit PSNR 32.7->45.3 dB.
- stock trainer-path imports (gaussian_renderer.render, scene.GaussianModel, both
  compiled modules) all OK -> no Python regression.

### Tier-2 gating rationale (important -- not a bug)
The means3D gradient FD-checks at a SCALED slope (pinhole ~0.16, panorama ~1.77)
but with ~100% SIGN agreement, eps-INDEPENDENT (so intrinsic, not FD curvature).
This is op43dgs's optimal-projection design, NOT the port: backward.cu:765-766
replaces the stock Inria `dL_dmean2D += dL_dG*dG_ddelx*ddelx_dx` with cross-terms
times tan_fovx/tan_fovy, i.e. the analytic screen-space gradient is an
APPROXIMATE local-affine descent direction (the paper's contribution), not the
exact finite-difference derivative. The fisheye scales gradient is likewise a
scaled descent direction (slope ~0.71). The port does not touch this math (only
includes/launch/__trap/GLM-packaging changed), so these slopes are CUDA-identical.
Gate: slope~1.0 on the camera-independent quantities (opacity always ~0.98, sh
~1.0); sign-agreement + grad-sum stability on the cov2D-dependent ones (means
all variants; scales on the curved variants). Tier-3 / single-cam fits are the
end-to-end proof the gradients optimize correctly (all converge). float-atomicAdd
run-to-run grad variation is ~1e-9 here (well under float-atomic noise), benign.

## Follower notes (gfx1100 / gfx1151)
- No source change expected: the port is wave-agnostic. A follower needs only
  `PYTORCH_ROCM_ARCH=gfx1100` (or gfx1151) + a clean rebuild of all 4 extensions.
  Validate-first on the moat-port branch; delta-port only on failure.
- gfx1151 (Windows) ext.cpp uses PYBIND11_MODULE -- watch the c10 inherited-ctor
  dllexport gap (the fused-ssim blocker; gsplat dodged it via TORCH_LIBRARY).
  Note only; do not act on the lead.

## Review 2026-06-01 (reviewer, linux-gfx90a, /pr-review local-branch mode)
Branch moat-port @ 9430d42 vs base 728de13. Verdict: APPROVE -> review-passed.
No problems found (per skill philosophy this section lists problems only; none to
list). 23 files: new root .gitignore (additive) + 3 setup.py (build-only) + 18
source files (6 per rasterizer variant) + simple_knn.cu, every source edit
USE_ROCM-guarded. No CMake (Strategy B correct). No host/CPU C++ touched.

Fact-checked (all VALID):
- Variant payload byte-identical: stripping hunk headers/context, the +/- payload
  of fisheye and panorama equals pinhole; the 8x3+3=27 launch normalizations are
  1:1 and (token-level proof: strip <>/ws, every removed line pairs with an added
  line) preserve kernel name, template/grid/block args verbatim -- only `<< <`->
  `<<<` / `>> >`->`>>>` spacing changed. Correctness-neutral on both backends.
- cooperative_groups/reduce.h guard is a PURE include guard: exhaustive grep finds
  ZERO cg::reduce / tiled_partition / thread_block_tile / __shfl / __ballot / __any
  / __popc / __reduce_*_sync / warpSize in the device tree (excl GLM). Nothing to
  replace. Wave-agnostic confirmed.
- NUM_WARPS (BLOCK_SIZE/32) in auxiliary.h is DEAD (only its own #define matches;
  zero references) -- stock Inria leftover, not a wave64 hazard. All __shared__
  arrays are sized by BLOCK_SIZE (256), never by warp count.
- GLM hipify monkeypatch VERIFIED EFFECTIVE on this torch: it wraps the exact
  hipify_python.hipify object that CUDAExtension invokes; both torch call sites
  (cpp_extension.py:1552 setup-path, :2319) pass header_include_dirs as a KEYWORD
  (so kwargs.get sees it; the load-bearing strip works); hipify matches `ignores`
  via fnmatch (line 155/191) so os.path.join(glm_dir,'*') is a valid glob; gated on
  PYTORCH_ROCM_ARCH or /opt/rocm so the CUDA build is untouched (inert, byte-for-byte).
- Build fixes correct + necessary (all fallout from guarding CUDA-only
  device_launch_parameters.h): `#include <cfloat>` for FLT_MAX (6 device sites in
  simple_knn.cu; standard/idempotent, safe unconditionally); `#define __trap
  __builtin_trap` USE_ROCM-guarded (object-like macro cleanly rewrites the bare
  `__trap();` call at auxiliary.h:165 in in_frustum's prefiltered branch; clang
  device builtin); `#define __CUDACC__` guarded out on ROCm (only simple-knn had it;
  rasterizers never defined it -- confirmed by grep).
- cub begin_bit=0 EVERYWHERE: rasterizer SortPairs `..., num_rendered, 0, 32+bit`
  (begin_bit literal 0); simple_knn SortPairs/Reduce use defaulted begin_bit (0).
  NOT the cudaKDTree nonzero-begin_bit hipCUB bug. CustomMin/CustomMax return
  float3 BY VALUE (not a ref to a param) -> cudf dangling-ref UB class N/A. cub::
  used unaliased; namespace resolution is a build-time matter the validator settles.

means3D / curved-variant-scales FD-slope RULING: the porter's dismissal is SOUND.
The complete changed-line set in forward.cu and backward.cu (all 3 variants) is
exactly (a) the reduce.h include guard and (b) the launch-spacing normalization --
ZERO lines touch computeCov2D / computeCov2DCUDA / the analytic backward / any
projection math, and the launch grid/block args are unchanged. The scaled-but-
sign-correct slope (pinhole ~0.16, panorama ~1.77, fisheye scales ~0.71) is
therefore intrinsic to op43dgs's optimal-projection analytic backward (an
approximate local-affine descent direction, the paper's contribution), identical
on CUDA. Validating it by ~100% sign-agreement + eps-independence + grad-sum
stability + training convergence (PSNR up) rather than FD magnitude is the correct
gate, and it holds because the diff provably does not alter the math.

Commit hygiene clean: title 66 chars, [ROCm] prefix; Claude disclosed; Test Plan
with literal commands; no Co-Authored-By/noreply trailer; ASCII, no em-dash; no
AMD-internal account refs; fork main == origin/main == 728de13 (clean mirror), port
only on moat-port; build artifacts (hip_rasterizer/, *.hip, build/) gitignored and
untracked. GPU validation is the validator's next stage (not a review blocker).

Minor (non-blocking, not a change-request): each setup.py drops one trailing space
from the upstream LICENSE comment line -- cosmetic, zero behavioral effect, in a
file already edited for the monkeypatch.
