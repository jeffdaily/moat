# pytorch_sparse -- porting plan

## Project
- Name: pytorch_sparse
- Upstream: https://github.com/rusty1s/pytorch_sparse
- Default branch: master
- Version inspected: 0.6.18 (depth-1 clone of master)

## Disposition: SKIP (already-supported)

rusty1s/pytorch_sparse has mature, AUTHORITATIVE, merged upstream ROCm/HIP
support plus maintained ROCm wheels. No MOAT port is warranted. Recorded via
`triage.py skip rusty1s/pytorch_sparse --reason already-supported`.

This is the third PyG-family sibling reaching the same conclusion
(pytorch_scatter, pytorch_cluster, mmcv were skipped earlier), but the evidence
below was verified directly against this repo, not assumed by family.

## Existing AMD support (authoritative, merged) + decision

Authoritative AMD-merged upstream history (`gh pr list --repo rusty1s/pytorch_sparse --search "ROCm OR HIP OR AMD"`):
- #282 "Enable ROCm builds" -- Dineshkumar Bhaskaran, MERGED 2022-10-17. The
  setup.py `torch.version.hip` path, `USE_ROCM`, the hipify abs-path work-around.
- #296 "Use macro for `__shfl_*` functions for ROCm" -- miaoneng, MERGED
  2022-11-28. The `SHFL_*_SYNC` macro shim (maskless `__shfl_up/__shfl_down`
  on ROCm; `_sync` forms on CUDA).
- #360 "[ROCm] fixes ambiguous calls to `shfl*` ... c10::Half to __half" --
  **ashwinma (Ashwin Aji, AMD)**, MERGED 2024-01-23. The `at::Half` shfl
  overloads in utils.cuh.
- #405 "Add ROCm 6.4.3+ support" -- Looong01 (pyg-rocm-build maintainer), OPEN.
  Newer-ROCm follow-up; the base ROCm path is already merged.

Maintained ROCm wheels: Looong01/pyg-rocm-build ships prebuilt torch-sparse
0.6.18 wheels; latest release 2026-04-21, ROCm 7.2.2, PyTorch 2.11,
Python 3.10-3.14.

Authoritativeness: AMD-engineer-authored (ashwinma) and community-merged ROCm
enablement, landed on master, plus maintained wheels. Per PORTING_GUIDE "assess
existing AMD support" this is mature/authoritative -> skip. Duplicating it would
re-port already-merged, AMD-owned work.

## Build classification (torch-extension) + evidence
- setup.py imports `CUDAExtension`/`CUDA_HOME` from `torch.utils.cpp_extension`
  (line 12); `WITH_CUDA = CUDA_HOME is not None or torch.version.hip` (line 23).
- setup.py line 98-102: under `if torch.version.hip:` adds `('USE_ROCM', None)`
  and undefs `__HIP_NO_HALF_CONVERSIONS__`; line 45-46 strips stale generated
  `hip` files on rebuild; line 145-147 the hipify abs-path work-around.
- CMakeLists.txt is a secondary libtorch consumer path (`find_package(Torch
  REQUIRED)`, `enable_language(CUDA)` under `WITH_CUDA`); the primary build is
  setup.py. Either way it is a pytorch extension.
- Strategy that WOULD apply if porting: Strategy B (torch hipify). It is already
  present in-tree; nothing to add.

## CUDA surface inventory (7 device files: csrc/cuda/*.cu, *.cuh)
- spmm_cuda.cu: merge-SpMM forward + value-backward. Uses 32-wide LOGICAL-warp
  grouping (`row = thread_idx >> 5`, `lane_idx = thread_idx & 31`), warp
  broadcast `SHFL_SYNC(FULL_MASK, mat_row, i)` for i in [0,32), and a
  `for(i=16;i>0;i/=2) SHFL_DOWN_SYNC` reduction. FULL_MASK = 0xffffffff.
- rw_cuda.cu (random walk), diag_cuda.cu, convert_cuda.cu: index/gather kernels,
  no warp intrinsics.
- atomics.cuh: `atomAdd` for float/double/half/bfloat16; `USE_ROCM`-guarded CAS
  path for double and half.
- utils.cuh: `at::Half` shfl overloads + `SHFL_*_SYNC` macros; under `USE_ROCM`
  they map to the MASKLESS `__shfl_up/__shfl_down/__shfl` plus a `__ldg(at::Half*)`
  shim.
- reducer.cuh: SUM/MEAN/MIN/MAX reducer policy, no warp intrinsics.
- No cub/thrust, no cuSPARSE/cuBLAS, no textures/surfaces, no managed/pinned
  memory, no streams/events beyond the current stream. spspmm is done in
  PyTorch (torch_sparse/spspmm.py -> matmul), not a cuSPARSE csrgemm.

## Risk note (NOT a porting task; recorded for the maintainers/wheels)
The spmm forward kernel's 32-wide logical-warp broadcast uses the maskless
`SHFL_SYNC` -> `__shfl(var, i)`, which on ROCm defaults to `width = warpSize`
(confirmed in /opt/rocm-7.2.1/include/hip/amd_detail/amd_warp_functions.h:142,
`int __shfl(..., int width = warpSize)`). On a 64-wide CDNA wavefront (gfx90a)
with THREADS=256, two 32-thread logical warps (different output rows) share one
wavefront, so `__shfl(mat_row, i)`, i in [0,32), reads absolute wavefront lanes
0-31 -- the LOWER logical warp -- for the upper logical warp too. That is the
classic wave64 cross-logical-warp hazard. The `SHFL_DOWN_SYNC` reduction
(i=16..1) stays within a logical warp on either width and is fine. This is a
potential latent correctness issue in the merged/wheel-shipped spmm forward on
wave64; it does NOT change the disposition. If a downstream consumer ever needs
spmm correctness re-confirmed on gfx90a, the right move is a targeted
upstream/wheel fix (give the broadcast an explicit width: `__shfl(var, i, 32)`),
NOT a from-scratch MOAT re-port of an AMD-owned, already-merged codebase. Not
filed here; left as an observation for whoever validates the wheels.

## File-by-file change list
None. No MOAT changes; the project already carries the ROCm build path and
fault-class handling.

## Build commands (only if a port were warranted -- not executed)
- `PYTORCH_ROCM_ARCH=gfx90a pip install . --no-build-isolation` against a ROCm
  torch (the CUDAExtension auto-hipifies csrc/cuda/*.cu).

## Test plan (only if a port were warranted -- not executed)
- GPU: `pytest test/test_spmm.py test/test_matmul.py test/test_spspmm.py` with
  the `devices` fixture including `cuda` (torch_sparse/testing.py). These
  exercise the spmm kernel directly.
- Non-GPU regression: the remaining `test/test_*.py` (CPU device parametrization).

## Open questions
None blocking. The wave64 spmm broadcast width is an observation for the
maintained wheels, not a MOAT deliverable.
