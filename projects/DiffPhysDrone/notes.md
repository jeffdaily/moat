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

## Review 2026-06-02 (reviewer, linux-gfx90a)
Reviewed `git diff 2719361...HEAD` on moat-port @ 2dc1a4b. Verdict: review-passed. No problems found; all plan/notes fault-class analysis independently verified against source.

Verified (no findings):
- Diff scope is exactly .gitignore + 6 `.type()`->`.scalar_type()` edits (dynamics_kernel.cu:295,341,374; quadsim_kernel.cu:336,362,382). The only remaining `.type()` is a commented-out line (quadsim.cpp:78), untouched. Value-identical forward-compat fix; the dispatched `scalar_t` is unchanged, applies on CUDA too.
- Wave-agnostic: no __shfl/__ballot/__any/__all/__activemask/warpSize, no __shared__/__syncthreads/__syncwarp, no cub/hipcub/cooperative groups, no bare `32` literal. All 6 kernels are one-thread-per-output (blockIdx*blockDim+threadIdx with `if(idx>=N) return;`). wave64 cannot change results.
- Backward adjoint UB class (EnvGS lesson): NO value-returning `__device__` helper exists anywhere; all kernels are `__global__ void`. run_backward_cuda_kernel writes all 5 grad outputs by direct assignment on first touch then accumulates; local accumulators d_a_drag_2/d_a_drag_1 (dynamics_kernel.cu:214-215) are fully written at :223-224 before read at :240-245; d_v_fwd/left/up_s init to 0 (:235-237). No fall-off-end / uninitialized accumulator.
- OOB neighbor reads (colmap class): rerender_backward (quadsim_kernel.cu:293-295) reads depth[b][0][u*2..u*2+1][v*2..v*2+1]; output H,W = half depth res, so max index 2H-1 = depth dim-1, in-bounds by construction, no clamp needed. nearest_pt "others" loop guarded `i>=B` (:191); obstacle loops iterate `.size(1)`.
- Strategy correct (B): sources stay CUDA-spelled, no compat header, hipify at build. `<cuda.h>`/`<cuda_runtime.h>` retained (hipify maps cleanly per notes).
- Commit hygiene: `[ROCm] Build quadsim_cuda extension on ROCm/HIP` (47 chars), mentions Claude, no noreply trailer, no ghstack, no em-dash, has Test Plan. Fork master = clean upstream mirror @ 2719361. Actions disabled on fork (enabled:false). No AMD-internal account references.

Note for validator: GPU gate not re-run at review (expected). Validator runs src/src/test.py with a FIXED seed (forward 4-output + analytic-backward-vs-autograd 5-grad allclose on 64 double states) plus the env_cuda.py secondary drive. The documented ~7.5% unseeded v_next flake is the pre-existing float ctl_dt vs double-reference gap (platform-independent), not a port defect; run seeded.

## Validation 2026-06-02 (validator, linux-gfx90a)

Platform: AMD Instinct MI250X (gfx90a:sramecc+:xnack-), ROCm 7.2, torch 2.13, HIP_VISIBLE_DEVICES=0.
Fork: jeffdaily/DiffPhysDrone @ moat-port, validated_sha=2dc1a4b4d2a1cfb7341925e0bd5591f891dc098d.

### Build
```
export HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx90a
cd projects/DiffPhysDrone/src/src && rm -rf build *.egg-info
pip install -e . --no-build-isolation
# result: Successfully installed quadsim_cuda-0.0.0
```
Compile: PASS. Extension .so loads cleanly.

### Primary gate: test.py (fixed seed, run x2)
```
# Run 1
python3 -c "
import os, sys; os.chdir('/var/lib/jenkins/moat/projects/DiffPhysDrone/src/src'); sys.path.insert(0, '.')
import torch; torch.manual_seed(42)
import runpy; runpy.run_path('test.py', run_name='__main__')
print('PASS run 1 (seed=42)')
"
# Run 2 (identical)
```
Both runs: PASS. All 4 forward outputs (act_next, p_next, v_next, a_next) and all 5 gradients (d_act_pred, d_act, d_p, d_v, d_a) pass torch.allclose vs reference. Deterministic across runs.

### Device dispatch confirmation
AMD_LOG_LEVEL=3 output (excerpt):
```
hipLaunchKernel ( ... )
Using native code object for device: amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack- co: amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-
hipLaunchKernel: Returned hipSuccess
```
Native gfx90a code object confirmed. No fallback.

### Secondary: env_cuda.py drive (5 steps + render + find_nearest_pt + rerender_backward)
```
cd projects/DiffPhysDrone/src
python3 -c "
from env_cuda import Env; import torch
torch.manual_seed(42)
env = Env(batch_size=8, width=32, height=24, grad_decay=0.4, device='cuda')
for step in range(5): env.run(torch.randn((8,3),device='cuda'), ctl_dt=1/15, v_pred=torch.randn((8,3),device='cuda'))
canvas, _ = env.render(ctl_dt=1/15)
vec = env.find_vec_to_nearest_pt()
import quadsim_cuda
dddp = torch.empty((4,3,24,32), device='cuda')
quadsim_cuda.rerender_backward(dddp, torch.rand((4,1,48,64),device='cuda').add(0.1), 0.53)
assert torch.isfinite(canvas).all() and torch.isfinite(vec).all() and torch.isfinite(dddp).all()
print('SECONDARY PASS')
"
```
PASS: render (8,24,32), find_nearest_pt (10,8,3), rerender_backward (4,3,24,32) -- all finite on gfx90a.

Verdict: **completed** (linux-gfx90a). Pass count: 4 forward + 5 gradient allclose assertions x2 seeded runs = 18/18. No GPU fault. No non-GPU regression (no CPU-only path exists).
