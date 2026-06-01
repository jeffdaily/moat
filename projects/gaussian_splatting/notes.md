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

## Review 2026-06-01 (reviewer, linux-gfx90a)
Verdict: APPROVE -> review-passed. Reviewed `git diff 8ddaa79...HEAD` on jeffdaily/gaussian_splatting@moat-port (HEAD a4d8b9a) with the /pr-review skill (ROCm-fault-class aware). The delta is exactly 2 files (.gitignore +4 hipify-artifact ignores; src/render_backward.cu +25/-5), additive, all `#if defined(USE_ROCM)`-guarded, CUDA path byte-identical. No problems found. The validator runs the real-GPU gates next (a missing GPU run is expected at review time and is not a blocker).

warpReduceSum equivalence + wave64-tile correctness (verified against the ROCm 7.2.1 HIP CG header /opt/rocm-7.2.1/include/hip/amd_detail/amd_hip_cooperative_groups.h):
- `cg::thread_block_tile<32>::size()` returns the tile size 32 (numThreads=tileSize, header :811/:824), so warpReduceSum (render_backward.cu:19-26) folds offsets 16,8,4,2,1 = a complete XOR butterfly over 2^5=32 lanes; every lane ends with the tile-wide sum, the same all-reduce cg::reduce(plus) provides.
- `thread_block_tile<32>::shfl_xor` forwards to `__shfl_xor(var, laneMask, numThreads)` with numThreads=32 as the WIDTH (header :863-864). Width-32 partitions the 64-lane wavefront into two independent 32-lane segments, so each tile's butterfly stays inside its own 32 lanes -- no cross-tile contamination on wave64. is_valid_tile_size<32> passes (pow2 and <= wavefront 64). No lane-math rework needed; the claim is correct.
- T=double: shfl_xor<T> is templated, double is supported on HIP (two 32-bit shuffles); the float64 gradchecks exercise the double instantiations.
- Leader election: `thread_rank()` is tile-relative (`workgroup::thread_rank() & (numThreads-1)`, header :816), so rank-0 fires once per 32-tile; one atomicAdd per tile == NVIDIA warp granularity, no fan-out/double-count.
- All 6 upstream cg::reduce sites (orig :244-246,253,259) route through WARP_REDUCE_SUM; zero cg::reduce survives on the ROCm path. CUDA-path macro expands to the original `cg::reduce(warp,val,cg::plus<T>())` (T from the enclosing kernel template) -> byte-identical preprocessed CUDA.

Stale-golden vs port-bug ruling (anchored on the gradchecks): the 8 non-passing tests are PORT-INDEPENDENT, not ROCm regressions. Decisive: the edited file (render_backward.cu) is backward-only; the forward rasterizer (render.cu) is shared-mem + __syncthreads with NO cg::reduce, NO warp ops, NO output atomics, and does not include the edited file -- so the failing FORWARD-pixel tests (test_rasterize SH x2, test_projection, test_utils ray) physically cannot be affected by the warpReduceSum change. The 16/16 float64 gradchecks (test_rasterize_autograd 8/8, test_cuda_autograd_functions 8/8) are the real correctness gate: torch.autograd.gradcheck compares the analytic HIP backward against central finite differences of the HIP forward and never reads any stored golden, so a pass proves the reduce (incl. double) is self-consistent regardless of golden staleness; SH_16 at atol=3e-5 exercises the full 16-coeff SH backward in double. The 2 SH-golden failures are explained by the source's SH math (sh_to_rgb DC term = SH_0 * (0.5/SH_0) = 0.5 exact, matching the passing DC-only no-SH golden; the higher-order goldens were recorded against older SH math). test_projection (places=4 on ~574 magnitude) and test_utils ray (default places=7 on ~0.5-0.7) are recorded tighter than float32 epsilon -- fragile on any float32 backend including CUDA. test_dataloader (3) ERRORs on a hardcoded missing path TEST_DATASET_PATH="/home/joe/Downloads/garden" (test_dataloader.py:7), CPU-only, unrelated.

Fault-class sweep (whole src/ tree, all clean): no raw warp intrinsics / warpSize / lane masks / PTX / half2 / textures / surfaces / cudaArray / managed memory / atomicMin/Max / cub / thrust / cuBLAS / __CUDA_ARCH__ / std::min-max / GLM / math_constants.h. Only atomicAdd (float/double leader-only in render_backward.cu; one int counter tile_culling.cu:171) on plain torch device tensors (not managed) -> atomicMin/Max-silent-drop class N/A. setup.py is a torch CUDAExtension with only `-O3 -std=c++17` (no --use_fast_math) and is unedited -> Strategy B correct; the `__expf` (depth.cu:94, render.cu:138, render_backward.cu:181) is a runtime fast-exp kernel arg, not a compiler flag. render_backward.cu is genuinely the only file needing a source fix.

Commit hygiene: clean. Title 54 chars, `[ROCm]` prefix; body discloses Claude ("Authored with assistance from the Claude agent (MOAT porter)"); no Co-Authored-By/noreply trailer; ASCII-only (-> arrows are ASCII); Test Plan with fenced commands. `.hip`/`_hip.cuh` artifacts gitignored and untracked (git check-ignore confirms; `git ls-files` shows none committed). Fork main == upstream 8ddaa79 (clean mirror); moat-port = 1 commit on top of base. No AMD-internal codenames/hosts/customers in the diff or message; author is the user's own public identity; fork is the public jeffdaily/gaussian_splatting.

## Validation 2026-06-01

Platform: linux-gfx90a. Device: AMD Instinct MI250X / MI250 (HIP_VISIBLE_DEVICES=3). Fork HEAD: a4d8b9aa308b23ba46a10d9734817237fd2474fd.

Build: fresh (`rm -rf build *.egg-info src/*.hip` then `PYTORCH_ROCM_ARCH=gfx90a MAX_JOBS=16 pip install -e . --no-build-isolation --no-deps`). Strategy B (torch auto-hipify). Completed successfully; `.so` at `src/splat_cuda.cpython-312-x86_64-linux-gnu.so`.

Commands:
```
# Build
bash utils/timeit.sh gaussian_splatting compile -- bash agent_space/gsplat_build.sh

# Decisive gate (run twice)
cd projects/gaussian_splatting/src
HIP_VISIBLE_DEVICES=3 python -m unittest test.test_rasterize_autograd test.test_cuda_autograd_functions -v

# Full suite
HIP_VISIBLE_DEVICES=3 python -m unittest discover test -v
```

Decisive gate (16 float64 gradchecks):
- Run 1: test_rasterize_autograd 8/8 PASS, test_cuda_autograd_functions 8/8 PASS. Ran 16 tests in 78.4s -- OK
- Run 2: test_rasterize_autograd 8/8 PASS, test_cuda_autograd_functions 8/8 PASS. Ran 16 tests in 80.3s -- OK
- Deterministic across both runs.

Full suite (33 tests): FAILED (failures=5, errors=3). Non-passing tests verified to be exactly the pre-documented stale-golden/missing-dataset set -- no new regressions:
- FAIL test_rasterize_full_sh_use_precompute, test_rasterize_full_sh_use_per_pixel_viewdir: stale SH goldens
- FAIL test_project_points, test_compute_projection_jacobian: places=4 tighter than float32 epsilon
- FAIL test_compute_rays_world_frame: places=7 tighter than float32 epsilon
- ERROR test_dataloader (3): hardcoded missing path /home/joe/Downloads/garden (CPU-only, dataset absent)
- All other GPU tests PASS: test_rasterize_no_sh, test_depth, test_tile_culling, test_structs, test_cuda_autograd_functions, test_rasterize_autograd, test_projection (other), test_utils (other)

Decision: PASS -> completed. validated_sha = a4d8b9aa308b23ba46a10d9734817237fd2474fd. Followers linux-gfx1100 and windows-gfx1151 auto-unblocked to port-ready.
