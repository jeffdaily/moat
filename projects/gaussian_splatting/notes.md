# gaussian_splatting (joeyan) -- ROCm/HIP port notes

Upstream: joeyan/gaussian_splatting @ 8ddaa799846ab75810305f457daf36668b68bddf (branch main).
Fork: jeffdaily/gaussian_splatting, port on `moat-port` (master/main is a clean upstream mirror).
A from-scratch MIT 3DGS reimpl: ONE PyTorch CUDAExtension `splat_cuda` built from
`src/*.cu` + `src/bindings.cpp`. No submodules, no simple-knn, no cub (per-tile depth sort
is `torch::sort`). Strategy B (torch auto-hipify). Wave-agnostic (gsplat/fused-ssim family).

## The one required fix (gfx90a lead)
`src/render_backward.cu` only. ROCm 7.2.1 ships `<hip/hip_cooperative_groups.h>` (which
hipify maps `<cooperative_groups.h>` to) but has NO `<cooperative_groups/reduce.h>`, no
`cg::reduce`, no `cg::plus`. The backward kernel uses `cg::tiled_partition<32>` plus six
`cg::reduce(warp, grad, cg::plus<T>())` then a rank-0 `atomicAdd` per 32-tile.

Fix (all `#if defined(USE_ROCM)`-guarded; CUDA path byte-identical):
1. Guard out `#include <cooperative_groups/reduce.h>` under `#if !defined(USE_ROCM)`.
   hipify does NOT rewrite this include (it leaves it verbatim), so it must be guarded.
2. Add a templated butterfly and route the six sites through a macro:
   ```
   template <typename T>
   __device__ __forceinline__ T warpReduceSum(const cg::thread_block_tile<32>& warp, T val) {
       #pragma unroll
       for (int offset = warp.size() / 2; offset > 0; offset >>= 1)
           val += warp.shfl_xor(val, offset);
       return val;
   }
   #define WARP_REDUCE_SUM(warp, val) warpReduceSum(warp, val)             // ROCm
   #define WARP_REDUCE_SUM(warp, val) cg::reduce(warp, val, cg::plus<T>()) // CUDA
   ```
   MUST template over T: the backward is instantiated for BOTH float AND double (the
   float64 gradchecks exercise the double path; HIP tile shfl_xor supports double).

Wave64 correctness: a `thread_block_tile<32>` `shfl_xor` uses the tile width (32) as the
shuffle width, so it stays inside the tile's 32 lanes of a 64-wide wavefront. The block is
16x16 = 256 threads -> 4 wavefronts, each holding 2 independent 32-lane tiles; each tile's
rank-0 does its own atomicAdd, matching the NVIDIA warp granularity. No lane-math rework.
This is the direct gsplat fault #4 template (see PORTING_GUIDE changelog 2026-05-30).

NO other source edits. NO compat header, NO symbol renames, NO setup.py change: torch's
HIP CUDAExtension compile already passes `-DUSE_ROCM=1` (confirmed in the build log), so the
guards fire without help. NO `-ffp-contract` pin was needed (see validation below).

## Build recipe (gfx90a; cwd must be OUTSIDE /var/lib/jenkins/pytorch)
```
export HIP_VISIBLE_DEVICES=3            # free GCD; cap concurrent GPU agents at 4
export PYTORCH_ROCM_ARCH=gfx90a         # follower: gfx1100, clean rebuild, NO source edit
export MAX_JOBS=16
cd projects/gaussian_splatting/src
rm -rf build *.egg-info src/*.hip       # Strategy B: delete build/ so edited .cu re-hipifies
/opt/conda/envs/py_3.12/bin/python -m pip install -e . --no-build-isolation --no-deps -v
```
Reusable script: `agent_space/gsplat_build.sh`. torch 2.13.0a0 (ROCm 7.2.5), device MI250X.
hipify writes `src/*.hip` + `src/matrix_hip.cuh` mirrors next to the sources; these are
gitignored (`*.hip`, `*_hip.cuh`) and must NOT be committed.

Test deps (install with --no-deps so the host ROCm torch is not clobbered; do NOT install
requirements.txt torch==2.1.2): numpy, opencv-python (cv2), scipy, yaml already present;
added `torchmetrics lightning_utilities` (test_utils SSIM) and `tyro docstring_parser
typeguard shtab` (test_dataloader import). torchvision (trainer/colmap_splat only) not installed.

## Test invocation gotcha
`python -m unittest discover test` MUST run with cwd = the repo src dir (it puts both the
repo root, for `splat_py`, and `test/`, for `gaussian_test_data`, on sys.path). Do NOT run
it through `utils/timeit.sh` directly: timeit.sh cd's to the MOAT root, so `discover test`
then resolves Python's stdlib `test` package instead. Use `agent_space/gsplat_test.sh`
(cd's to src first) and wrap THAT with timeit. Single modules: `cd test && PYTHONPATH=<src>
python -m unittest <module>`.

## Validation on gfx90a (HIP_VISIBLE_DEVICES=3) -- DECISIVE GATE PASSES
Tier 1 (decisive for the reduce fix): all 16 float64 gradchecks PASS, deterministically
across two independent runs:
- `test_rasterize_autograd` 8/8 (gradcheck on RenderImage, SH {0,4,9,16} x {bg,no-bg}).
- `test_cuda_autograd_functions` 8/8 (projection/sigma/jacobian/conic/SH precompute).
These prove the warpReduceSum butterfly is numerically exact in float (and double).

Tier 2/3: `test_rasterize.test_rasterize_no_sh` PASSES (forward bitwise to golden);
the full forward render is bitwise-deterministic across runs (verified).

### Pre-existing upstream test failures -- NOT caused by the port (all reproduce off-GPU)
Proven platform-independent, do not block the gate:
- `test_rasterize` SH paths (use_precompute, use_per_pixel_viewdir): the recorded goldens
  are inconsistent with the CURRENT source's SH math. At pixel [340,348] exactly ONE splat
  contributes; a pure float64 CPU recompute of that single-splat composite (replicating
  render.cu arithmetic) reproduces the HIP value [0.6331,0.1562,0.1562] EXACTLY, not the
  golden [0.5363,0.0593,0.0593]. The `splat_cuda` precompute kernel itself matches a
  hand-computed numpy SH reference to 7 decimals in both f32 and f64. Setting the higher-order
  SH coeffs to 0 makes the SH path reproduce the no-SH golden exactly (DC path is exact).
- `test_projection` (uv[0,1], jacobian[0,1,2]) at places=4 and `test_utils`
  test_compute_rays_world_frame at the default places=7: goldens recorded tighter than
  float32 epsilon. The EXACT float64 math gives uv[0,1]=573.98641 (rounds to .9864) while the
  golden recorded .9863; HIP's float32 573.98645 is actually CLOSER to the true value than
  the golden. `-ffp-contract=on` cannot fix a wrong golden, so it was NOT applied.
- `test_dataloader` (3 tests): hardcodes the upstream author's local path
  `/home/joe/Downloads/garden/sparse/0/points3D.bin`. CPU-only; cannot pass without that
  dataset. Unrelated to the port.

Upstream runs NO CI (no workflows), so these stale/fragile goldens were never gated.

## Followers (delta-plan, not lead scope)
- gfx1100: expect a clean rebuild with `PYTORCH_ROCM_ARCH=gfx1100`, NO source edit (the fix
  is wave-agnostic; tile<32> shfl_xor is correct on wave32 too).
- gfx1151 (Windows): `bindings.cpp` uses PYBIND11_MODULE which can hit the c10 inherited-ctor
  dllexport blocker (the fused-ssim gfx1151 wall). Note for the delta-plan; gsplat dodged it
  via TORCH_LIBRARY.
