# DiffPhysDrone notes

## Summary
Strategy-B torch CUDAExtension port (`quadsim_cuda`: render raycast, nearest-obstacle collision, analytic forward + analytic backward rigid-body dynamics). Builds and validates on gfx90a (MI250X, ROCm 7.2, torch 2.13 / HIP 7.2.53211). Fork: jeffdaily/DiffPhysDrone @ moat-port, HEAD 2dc1a4b.

## The one source change
`Tensor.type()` -> `Tensor.scalar_type()` at the six `AT_DISPATCH_FLOATING_TYPES` sites (3 in `dynamics_kernel.cu`, 3 in `quadsim_kernel.cu`). `.type()` returns `DeprecatedTypeProperties`, which no longer converts to `ScalarType` in current torch:
```
error: no viable conversion from 'const DeprecatedTypeProperties' to 'torch::headeronly::ScalarType'
```
This is a torch forward-compat fix (not arch-related); the same edit is required to build against torch >= ~2.4 on CUDA or ROCm. No behavioral change.

The `#include <cuda.h>` risk (plan risk 2) did NOT materialize -- torch hipify maps it cleanly, no guard needed. No compat header (Strategy B); sources stay in CUDA spelling.

Added a `.gitignore` (build/, *.egg-info/, *.hip, *.so, __pycache__) so the hipify mirrors and compiled module stay out of the tree. The torch build emits `dynamics_kernel.hip` / `quadsim_kernel.hip` next to the .cu sources -- remove them before staging if they slip through.

## Build (gfx90a)
```
export HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx90a
cd projects/DiffPhysDrone/src/src
rm -rf build *.egg-info        # clean incremental hipify state between rebuilds
pip install -e . --no-build-isolation
```
Warnings only (deprecated `packed_accessor` -> use packed_accessor32/64; harmless).

## Validation (real GPU gate)
Primary gate = repo's own `src/src/test.py` (forward-sim allclose + analytic-backward vs `torch.autograd.backward`, all 4 outputs + 5 gradients on 64 random DOUBLE states). Passes deterministically:
```
cd src/src
python3 -c "import torch; torch.manual_seed(0)
import runpy; runpy.run_path('test.py', run_name='__main__')"   # PASS
```
Secondary GPU coverage: drove `render`, `find_nearest_pt` (nearest_pt/pos are 3-dim [traj][drone][3]), `update_state_vec`, `rerender_backward` (depth/dddp both 4-dim, output grid half the depth res) -- all produce finite outputs on gfx90a. AMD_LOG_LEVEL=3 confirms `run_forward_cuda_kernel<double>` dispatched via `hipLaunchKernel` with a native gfx90a:sramecc+:xnack- code object on MI250X.

## IMPORTANT gotcha: test.py is flaky under UNSEEDED randoms (upstream artifact, not a port bug)
test.py as-shipped uses unseeded randoms and the `v_next` allclose flaps ~7.5% of runs (17/20 pass; seeded sweep 44/50). Root cause is fully characterized and platform-independent:
- `ctl_dt` (and `grad_decay`) are declared `float` in the kernel signature (`run_forward_cuda(..., float ctl_dt, float airmode_av2a)`), while the PyTorch reference uses Python double `1/15`. The float-rounded `1/15` differs from double by ~5.2e-8 relative.
- That relative error is magnified on `v_next = v + 0.5*(a + a_next)*ctl_dt` because `a_next` includes the drag term, which has the widest dynamic range (occasionally O(50)); the absolute v_next error reaches ~2.5e-6, occasionally exceeding allclose's `rtol*|val|+atol` when a v_next element is near zero. The other three outputs (act_next, p_next, a_next) never flap.
- Proof it is purely the float ctl_dt: feeding the reference the same float-rounded ctl_dt the kernel receives makes kernel vs reference agree to 3.5e-15 (double eps). `-ffp-contract=off` did NOT change the flap rate (so it is not FMA contraction). The kernel even disagrees with `v + 0.5*(a + kernel's_own_a_next)*dt` by the same 2.5e-6 because internally it multiplies by the float ctl_dt.
- This flake occurs identically on stock CUDA (nvcc passes the same `float` ctl_dt). It is an upstream numeric/tolerance design choice, not a HIP regression. We did NOT widen ctl_dt to double: that would change the upstream float-path API/numerics and is out of scope for a mechanical port. Use a fixed seed when running test.py as a deterministic gate.

## Followers (gfx1100 / gfx1151, RDNA wave32)
No delta expected -- zero wave-width-sensitive constructs. Same moat-port branch should build and pass with only `PYTORCH_ROCM_ARCH=<arch>` changed. If a follower's test.py flaps it is the same unseeded float-ctl_dt artifact above (run seeded); a real failure would be a torch-on-RDNA build/runtime issue, not port logic. Do not re-plan; append findings here.
