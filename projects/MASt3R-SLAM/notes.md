# MASt3R-SLAM notes

## Summary
CLEAN Strategy-B port (torch hipify). Two torch CUDA extensions; three `.cu`
translation units; no CUDA libraries, no warp intrinsics, no atomics, no
textures. Built and kernel-level-validated on gfx90a (MI250X, wave64) under
ROCm PyTorch 7.2.

- Fork: https://github.com/jeffdaily/MASt3R-SLAM (branch `moat-port`)
- Upstream base: rmurai0610/MASt3R-SLAM @ e6f4e3d474fad0e11f561482012be864ba8c3f17
- moat-port head: b2f86d46b91bc516b6813f1f5f189066cb5a243b
- Actions disabled on the fork.

## Submodule handling
The planner's recursive clone already vendored mast3r/dust3r/croco (curope is
in the tree at `thirdparty/mast3r/dust3r/croco/models/curope`). The src clone
was a shallow depth=1 mirror; unshallowed against the fork, repointed `origin`
to jeffdaily, added `upstream`, branched `moat-port` off upstream `main`.
The `thirdparty/eigen` submodule was registered but EMPTY (the GN host code
includes `<Eigen/Sparse>`); ran `git submodule update --init thirdparty/eigen`
to populate it (eigen is header-only, small, fetched fine over slow egress).
`.gitmodules` lists only eigen and pyimgui; the mast3r/dust3r/croco trees are
checked-in working-tree content, not submodules of this repo.

## Build environment
- conda env `py_3.12`, torch 2.13.0a0 with `torch.version.hip == 7.2.53211`.
- ROCm at /opt/rocm; device MI250X (gfx90a). 4 GCDs visible.

## Build recipe (gfx90a)
```
cd projects/MASt3R-SLAM/src
# main backends (mast3r_slam_backends)
PYTORCH_ROCM_ARCH=gfx90a MAX_JOBS=16 python setup.py build_ext --inplace
# vendored CroCo curope (built directly to avoid pulling mast3r's heavy deps)
cd thirdparty/mast3r/dust3r/croco/models/curope
PYTORCH_ROCM_ARCH=gfx90a MAX_JOBS=16 python setup.py build_ext --inplace
```
Incremental gotcha: torch hipify writes `*.hip` mirrors next to the `.cu`. After
editing a `.cu`, `rm -f <dir>/*.hip` and `rm -rf build` before rebuilding or the
stale mirror is recompiled (Strategy-B re-hipify gotcha in PORTING_GUIDE).

Multi-arch fat-binary check (warp-size policy): `PYTORCH_ROCM_ARCH="gfx90a;gfx1100"`
builds clean and `llvm-objdump --offloading mast3r_slam_backends*.so` emits BOTH
gfx90a and gfx1100 code objects -- the sources are wave-size-generic.

## What changed beyond building against ROCm torch
Five files, all minimal; no `cuda_to_hip.h` shim (hipify does it).
1. `setup.py` and `thirdparty/.../curope/setup.py`: gate nvcc `-gencode` /
   `--ptxas-options` / `--use_fast_math` / `cuda.get_gencode_flags()` behind
   `torch.version.cuda`; add a `torch.version.hip` branch that passes only
   `-O3` (hipcc gets the arch from `--offload-arch` via PYTORCH_ROCM_ARCH and
   defaults to fast math / `-ffp-contract=fast`).
2. PyTorch-version API drift (NOT ROCm-specific; also correct on CUDA; the
   ROCm torch we built against is new enough to require them):
   - `gn_kernels.cu` x3: `torch::linalg::linalg_norm` -> `torch::linalg_norm`.
   - `matching_kernels.cu`, `curope/kernels.cu`: `Tensor::type()` ->
     `scalar_type()` in `AT_DISPATCH_FLOATING_TYPES_AND_HALF`.
   - `matching_kernels.cu`: `<cuda/std/limits>` -> `<limits>` and
     `::cuda::std::numeric_limits` -> `std::numeric_limits` (hipify provides no
     `cuda/std` header on ROCm; semantics identical -- `numeric_limits<T>::min()`
     is the smallest positive normal for floating types, matching upstream).

## THE volatile reduction -- correct as-is on wave64, no __syncwarp needed
`gn_kernels.cu` `warpReduce(volatile float* sdata, tid)` + `blockReduce` is the
DROID-SLAM tree reduction with no `__syncwarp`. It is LEFT UNCHANGED. Validation
proved it is bit-exact deterministic on gfx90a (wave64): the final `tid<32`
warp step's first cross-lane read (`sdata[tid+32]`) spans lanes 0..63 of one
wavefront and is covered by the preceding `__syncthreads()` at the `tid<64`
step; the subsequent volatile steps are within one wavefront. A racing reduction
would have shown run-to-run variance -- none observed across 20 runs at every
point count straddling the lane boundaries. No lane width is hardcoded, so it is
also correct on wave32 (the follower validates gfx1100 on its own host; the
fat-binary check confirms it compiles for RDNA).

## Validation method and results (kernel-level gate)
No model weights / SLAM datasets (host egress ~40-160 KB/s; the ~2.6 GB MASt3R
checkpoints + multi-GB TUM/EuRoC are out of budget). Harness:
`agent_space/mast3r_validate.py`, run with `HIP_VISIBLE_DEVICES=0` to pin one
GCD. Seeded synthetic tensors; tolerance compares (never exact equality) to
absorb fast-math / `-ffp-contract` drift. 9/9 PASS:

- gauss_newton_points / gauss_newton_rays / gauss_newton_calib: bit-exact `dx`
  across 20 runs at n in {1,33,64,65,128,256,300} (calib uses image grids
  giving n in {64,72,256,320}); all finite; GN steps converge (e.g. calib
  |dx| 0.033 -> 1.5e-7 over 8 iters). This is the volatile-reduction gate.
- iter_proj: deterministic; output pixels clamped to [1,w-2]x[1,h-2]; the
  bilinear-interp ray at the returned pixel matches a torch gather.
- refine_matches: deterministic; EXACT pixel match vs an fp16 CPU
  descriptor-dot neighborhood-argmax reference (mismatch frac 0).
- curope rope_2d: deterministic; max abs diff 5.3e-7 (rel 1.3e-7) vs a CPU
  reimplementation of the kernel's exact `[u_Y,v_Y,u_X,v_X]` rotation layout.

Cross-arch consistency target for followers: the GN ops are deterministic, so a
gfx1100/gfx1151 follower should diff its `dx` for the same seeded inputs against
the gfx90a values (catches a wave32 reduction divergence a "sane output" gate
would miss). The harness is reusable as-is on the follower hosts.

## Fault classes encountered
- Strategy-B build flags (nvcc-only `-gencode`/`--use_fast_math` on the ROCm
  path) -- gated by `torch.version`.
- PyTorch-version API drift surfaced at compile (linalg_norm namespace,
  `.type()`, `cuda/std/limits`) -- arch-independent spelling fixes.
- fp drift (curope fast-math; GN `sqrtf`) -- handled by tolerance compares in
  validation, not source changes.
- Empty `thirdparty/eigen` submodule (build dependency) -- populated.
NONE of: warp intrinsics, hardcoded 32/64, atomics, textures/pitch,
rule-of-five, OOB neighbor reads (iter_proj/refine_matches clamp before the +1),
library swaps. The `iter_proj_kernel` reads `rays_img[...][v11+1][u11+1]` but
clamps u,v to [1,w-2]/[1,h-2] first, so +1 stays in bounds (safe on AMD).

## Aspirational end-to-end (not done; egress-bound)
A single short TUM sequence via `main.py` would be a stronger gate but needs the
2.6 GB checkpoints + a dataset; out of egress budget. The kernel-level gate is
the validation of record.
