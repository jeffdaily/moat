# fused-ssim notes

## Summary

`rahul-goel/fused-ssim` is a PyTorch CUDA extension (fused differentiable SSIM
for Gaussian-splatting). Strategy B: torch hipifies the extension's `.cu` at
build time. Ported and GPU-validated on linux-gfx90a (MI250X, CDNA2, wave64,
ROCm 7.2.1) with ZERO source changes. The kernels have no warp-level primitives,
so there is no wave64 fault class to fix.

## Existing AMD support (assessment)

Already AMD-aware, but only at the build-config level:
- `setup.py:29-32` `if torch.version.hip:` branch (adds `-ffast-math`, skips
  `--maxrregcount`/`-gencode`). The only AMD commit, `88169c5` (#29, amd.com
  author), changed `setup.py` only.
- README advertises "2D (CUDA, Metal, ROCm) and 3D (CUDA only)".

NOT skipped as already-supported, because the 3D path post-dates that
enablement and had never been GPU-checked on AMD: `ext.cpp:5,15` gate the 3D
bindings on `FUSED_SSIM_CUDA`, which `setup.py` defines for HIP too, so
`fusedssim3d` IS compiled and exported on ROCm despite the "CUDA only" README
note. `fused_ssim/__init__.py:6` imports it unconditionally on any CUDA-or-HIP
device. The MOAT value here was to build + GPU-validate the current (post-3D-
rework) code on a modern ROCm, including that 3D path.

## ROCm torch env

- torch `2.13.0a0+gitb5e90ff`, `torch.version.hip 7.2.53211`, locally built.
- Must run from a NON-source dir; importing torch from `/var/lib/jenkins/pytorch`
  shadows `torch/_C` and fails. All runs below are from `/tmp`.
- conda env `py_3.12`; no project `.venv`.

## CUDA surface / fault classes

- Both `ssim.cu` (2D) and `ssim3d.cu` (3D) are separable 11x11 Gaussian
  convolutions with shared-memory tiling; block `(16,16)`=256 threads. The
  per-pixel reduction goes entirely through `__shared__` arrays + `block.sync()`;
  the 3D Z-axis pass uses a per-thread register ring buffer (no cross-thread
  exchange).
- NO warp primitives: grep `__shfl|__ballot|__activemask|__any|__all|__reduce|
  __popc|tiled_partition` over both `.cu` -> nothing. The two "same warp"
  strings (`ssim.cu:167`, `ssim3d.cu:252`) are misleading comments on plain
  `ly + BLOCK_Y` shared-memory indexing.
- The only literal 32 (`ssim.cu:328-335`, 2D backward load:
  `warp_id=tid/32; lane_id=tid%32; num_warps=(T+31)/32; col+=32`) is a benign
  loop-tiling stride distributing 256 threads over the 26x26 shared tile. It
  does not read `warpSize` and does no cross-lane op, so it is correct unchanged
  on wave64 (validated by the backward-gradient checks passing).
- No texture, atomics RMW, cuBLAS/cuFFT/cuRAND, rule-of-five handles, or layered
  arrays. => no Fault-class fixes needed.

## Build (Strategy B, zero source changes)

From the fork clone `projects/fused-ssim/src`:

```
HIP_VISIBLE_DEVICES=1 python -m pip install -e . --no-build-isolation -v
```

Torch hipify rewrote the sources to `ssim.hip` / `ssim3d.hip` (gitignored)
cleanly: `cooperative_groups` -> `<hip/hip_cooperative_groups.h>`,
`c10/cuda/CUDAGuard.h` -> `c10/hip/HIPGuard.h`, `__constant__` preserved,
`at::cuda::OptionalCUDAGuard` kept (works on HIP). Linked
`-lamdhip64 -lc10_hip -ltorch_hip`. setup.py HIP branch reported NVCC args
`['-O3','-DFUSED_SSIM_CUDA','-ffast-math']`. Module exports all four symbols
(`fusedssim`, `fusedssim_backward`, `fusedssim3d`, `fusedssim_backward3d`);
`is_3D_supported == True`.

## Validation (GPU 1, gfx90a) -- ALL PASS

Reference: `pytorch_msssim` 1.0.0 (pip) and a pure-PyTorch conv-based SSIM.

1. Repo `tests/test.py` (2D, 100 iters, B=5 CH=5 H=1080 W=1920): forward
   `torch.isclose` and full-tensor gradient `.all()` vs reference SSIM AND
   pytorch_msssim. Exit 0 (asserts inside the loop). fused fwd 3.67ms vs ref
   197ms.

   ```
   cd projects/fused-ssim/src/tests && HIP_VISIBLE_DEVICES=1 python test.py
   ```

2. Repo `tests/test_3D.py` (3D, 10 iters, B=2 CH=1 D=H=W=96, rtol=1e-6
   atol=1e-8): forward + gradient `torch.allclose` vs conv3d reference AND
   pytorch_msssim(spatial_dims=3). Exit 0.

   ```
   cd projects/fused-ssim/src/tests && HIP_VISIBLE_DEVICES=1 python test_3D.py
   ```

3. Independent harness `agent_space/fused_ssim_validate.py` over configs the
   repo tests do not cover (2D CH=1/3/16, B=1..4, several sizes; 3D three
   shapes) + bitwise determinism:

   ```
   HIP_VISIBLE_DEVICES=1 python agent_space/fused_ssim_validate.py
   ```

   - 2D fwd vs ref/pm: max|diff| ~1e-8; 2D grad: ~1e-10..1e-12.
   - 3D fwd vs pm: max|diff| ~5e-9; 3D grad: ~1e-11.
   - Determinism: 2D and 3D fwd+grad bitwise identical across two runs.
   - All measured diffs are 4-5 orders tighter than the asserted tolerances.

## Deliverable

Zero kernel/source edits. The port is "build against ROCm torch + validate";
no fork commit carries a code change for gfx90a. No GitHub Actions added; no
README/gen_readme changes.
