# gsplat ROCm/HIP port plan (lead: linux-gfx90a)

## Project
- Upstream: nerfstudio-project/gsplat @ 9ebed19 (v1.5.3).
- Differentiable Gaussian-splatting rasterization, a PyTorch CUDA extension.
- ext_type: **torch-extension** -> **Strategy B** (torch hipifies the .cu/.cuh at
  build time; do NOT hand-add a compat header or hand-rename symbols).

## Existing-AMD assessment
gsplat is NOT cleanly "already-supported", but it is also NOT a virgin CUDA
codebase. It already carries partial ROCm awareness in its build scripts:
`gsplat/cuda/build.py:175-178` (and the three sub-lib build.py under
`libs/{scene,geometry}` + `experimental/render`) branch on `torch.version.hip`
and add `-DUSE_ROCM -U__HIP_NO_HALF_CONVERSIONS__`, dropping the NVIDIA-only
`--expt-relaxed-constexpr`. There are NO `#if defined(USE_ROCM)` guards inside
the actual `.cu`/`.cuh` kernel source -- the device code is plain CUDA spelling
that relies on torch hipify. Per the GPUMD precedent in PORTING_GUIDE ("a project
may already ship an upstream HIP path; do NOT auto-skip as already-supported --
build+GPU-validate and fix the rot"), the disposition is to BUILD and
GPU-VALIDATE on gfx90a, fixing only what hipify cannot.

## ROCm torch env used
- conda env `py_3.12`, torch **2.13.0a0+gitb5e90ff**, `torch.version.hip
  7.2.53211`, `torch.cuda.is_available()=True`, device "AMD Instinct MI250X /
  MI250" (gfx90a). Host ROCm 7.2.1, hipcc at /opt/rocm/bin/hipcc, ninja present.
- `CUDA_HOME=None`, `ROCM_HOME=/opt/rocm` -> a `CUDAExtension` build hipifies and
  links amdhip64/c10_hip/torch_hip automatically (Strategy B).
- NOTE: must build/run with cwd OUTSIDE /var/lib/jenkins/pytorch (the source
  tree there shadows the installed `torch`). The gsplat src dir is fine.
- GPU pinned via `HIP_VISIBLE_DEVICES=2` (assigned ordinal) for ALL build+run.

## CUDA surface (warp/wave64 analysis)
The rasterizer/sorter do NOT use raw warp intrinsics (`__shfl`/`__ballot`/
`__any`/`__syncwarp`); grep finds ZERO. Instead they use CUDA **Cooperative
Groups** with explicit 32-thread tiles:
- `cg::tiled_partition<32>(block)` in the rasterize bwd + projection packed/fused
  kernels (RasterizeToPixels{3DGS,2DGS,FromWorld3DGS}Bwd.cu, ProjectionEWA3DGS*,
  Projection2DGS*).
- Reductions via `cg::reduce(warp, val, cg::plus/greater)` in include/Utils.cuh
  (`warpSum`/`warpMax`), and `warp.any(valid)`, leader election via
  `if (warp.thread_rank()==0) gpuAtomicAdd(...)`.
- Per-tile gaussian sort is HOST-side `cub::DeviceRadixSort::SortPairs` /
  `DeviceSegmentedRadixSort::SortPairs` (IntersectTile.cu:498,550), with
  **begin_bit=0** (so the cudaKDTree nonzero-begin_bit hipCUB bug does NOT apply).

Why this is wave64-SAFE by construction: `cg::tiled_partition<32>` requests a
32-thread sub-group regardless of hardware wave width. On gfx90a (wave64) a
256-thread block splits into 8 tiles of 32; `cg::reduce` and `thread_rank()==0`
operate per-tile, so each 32-tile behaves like a 32-lane NVIDIA warp and each
tile's rank-0 does its own atomicAdd. This avoids both popsift traps (no
positional 2-rows-per-block packing; no `lane=tid&31; if(lane==0)` election that
fires on every ballot bit). HIP supports cooperative_groups with explicit tile
sizes, so the reductions are warp-size-agnostic. **Expectation: no wave64 source
fix needed.** Verify empirically: the test compares device output AND gradients
vs a reference at 1e-4; a wave64 reduction bug would surface as wrong pixels/
grads or nondeterminism.

## Watch items (hipify may or may not handle)
- `csrc/TensorView.h:125` uses `#ifdef __CUDA_ARCH__` to pick `assert` (device)
  vs `TORCH_CHECK_INDEX` (host) in a host/device method. `__CUDA_ARCH__` is NOT
  defined during HIP device compile (HIP uses `__HIP_DEVICE_COMPILE__`), so the
  device pass would take the host branch. If `TORCH_CHECK_INDEX` is not
  device-callable this fails to compile (cudaKDTree fault class). Fix only if it
  actually breaks the build, guarded by USE_ROCM.
- `-use_fast_math` (FAST_MATH=1 default) is passed to hipcc; should be accepted.
- `c10/cuda/*`, `ATen/cuda/Atomic.cuh` (gpuAtomicAdd), `cub/cub.cuh`,
  `cooperative_groups.h`, glm third_party -- all standard, hipified by torch.

## Build plan
- `BUILD_NO_CUDA=0`, build the main `gsplat.csrc` extension (3DGS path is what the
  rasterization/basic tests exercise) via `pip install -e . --no-build-isolation`
  with `HIP_VISIBLE_DEVICES=2`. setup.py also builds the experimental render ext.
- The build is gated by env BUILD_* flags (default builds 3DGS etc). Keep
  defaults so `gsplat.has_3dgs()` is true (the rasterization tests require it).
- If an incremental edit is needed, torch re-hipifies on source change.

## Validation plan (real GPU, gfx90a, HIP_VISIBLE_DEVICES=2)
1. `tests/test_rasterization.py` -- compares CUDA-ext `rasterization()` render +
   alphas vs reference `_rasterization()` at rtol=atol=1e-4 (image correctness).
2. `tests/test_basic.py` -- forward AND `torch.autograd.grad` gradients vs the
   reference `_` path for projection, quat/scale, and rasterization (gradient
   correctness within the test's per-quantity tolerances).
3. `tests/test_rasterization.py` rerun / determinism spot-check (same-seed render
   equal run-to-run) since the bwd uses atomicAdd-combined reductions.
A render that runs but gives wrong pixels/grads is a FAIL.

## Disposition
Strategy B mechanical bring-up + GPU validation; fix only hipify-uncoverable
faults guarded by USE_ROCM (expected: little to none, given CG abstractions).
Record quirks in notes.md; append generalizable wave64/Strategy-B lessons to the
PORTING_GUIDE changelog.
