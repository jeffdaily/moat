# TIGRE notes

## Build (linux-gfx90a, ROCm 7.2.1)

Strategy A: the project's setuptools/Cython build runs a hand-rolled nvcc driver
in `setup.py`. The HIP build adds a `BUILD_WITH_HIP` branch that swaps the
per-`.cu` compiler to `hipcc -x hip --offload-arch=$HIP_ARCH -D USE_HIP`, links
`amdhip64` instead of `cudart`, defines `USE_HIP` + `__HIP_PLATFORM_AMD__` on the
host `.cpp` TUs (so the compat header takes the hip path), and skips the
nvcc-at-import `locate_cuda()`/`get_cuda_cc()` (they shell out to nvcc, absent on
a ROCm-only host).

Build recipe:
```
export BUILD_WITH_HIP=1 ROCM_PATH=/opt/rocm HIP_ARCH=gfx90a
cd projects/TIGRE/src
pip install -e . --no-build-isolation
```
HIP_ARCH accepts a comma list for multi-arch (e.g. `gfx90a,gfx1100`). Host has
ROCm 7.2.1 (hipcc/clang 22), python 3.12, Cython 3.2.5, numpy 1.26.4.

Builds the 8 extensions: `_Ax _Atb _minTV _minPICCS _AwminTV _tv_proximal
_gpuUtils _RandomNumberGenerator`.

## Port summary

All HIP changes are guarded; the CUDA path is unchanged. Single compat header
`Common/CUDA/cuda_to_hip.h` maps the cuda* runtime / cuRAND symbols, supplies
`tex3D_TIGRE`, and is included first in every built TU.

### Textures: software trilinear (HIGH item, the central fix)
gfx90a/ROCm 7.2.1 rejects creation of a `cudaFilterModeLinear` +
`cudaReadModeElementType` texture over a float array. Under HIP the five
linear-filter textures (voxel_backprojection.cu, voxel_backprojection2.cu,
voxel_backprojection_parallel.cu, ray_interpolated_projection.cu,
ray_interpolated_projection_parallel.cu) are created `cudaFilterModePoint` and
the interpolated reads go through `tex3D_TIGRE`, which point-samples the 8
neighbours through the point-filtered texture and lerps in software using CUDA's
unnormalized -0.5 texel-center convention (coord c samples texels floor(c-0.5)
and floor(c-0.5)+1, weight frac(c-0.5)). Border-zero (cudaAddressModeBorder) is
reproduced for free: the 8 point fetches inherit border addressing, so neighbours
outside the array read 0. No explicit array-dim threading needed.
ray_interpolated_projection.cu has a runtime accuracy>1 POINT branch which is
preserved (it reads via `tex3D<float>` directly, point on both back ends); only
the linear (accuracy<=1) branch uses `tex3D_TIGRE`. Siddon projectors stay
POINT-filtered and unchanged. The arrays are NON-layered 3D, so no
layered-collapse workaround is needed.

### Reduction loop wave64 fix (MEDIUM)
PICCS.cu / GD_TV.cu / GD_AwTV.cu block reductions (reduceNorm2 + reduceSum, 6
sites) drove the final warp shuffle reduction from `offset = warpSize/2`. The
shuffle width is the literal 32, so on wave64 (warpSize=64) the first iteration
shuffled offset=32 across the width-32 logical-warp boundary -> wrong sum. Fixed
to `offset = 16` (the loop is now driven by the shuffle width, not warpSize):
correct on wave32 and wave64. No wave size hardcoded.

### Siddon ray-loop runaway (NEW fault class, HIGH severity at scale)
The Siddon forward projectors (Siddon_projection.cu, Siddon_projection_parallel.cu)
march each ray for Np = sum of voxel-plane intersections, where the index
bounds imin/imax/jmin/... are picked by EXACT float-equality tests (am==axm,
aM==axM) on quantities formed with __fdividef (approximate fast division).
__fdividef's result differs slightly from CUDA, which can flip one equality
test and select an out-of-range index bound; the unsigned subtraction then
makes Np astronomical and the loop effectively never terminates. Symptom: a
single 256^3 Siddon projection that takes 0.2s on CUDA ran >10 min (looked
like a hang). It is DATA-DEPENDENT: 256^3 at angle 0 (axis-aligned) was fast,
at angle 0.3 rad it ran away; 224^3 and smaller were fine. Fix: cap Np at the
geometric maximum Nx+Ny+Nz(+3) -- a ray cannot cross more planes than that, so
the cap is a no-op for every valid ray (and on CUDA), and only truncates the
miscomputed case; OOB texture reads were already border-zero so output is
unchanged. This is the general "exact float-equality branch fed by approximate
division" trap; the cap is the robust arch-unified guard.

### cuRAND -> hipRAND
RandomNumberGenerator.cu uses only device-side cuRAND
(curand_init/curand_poisson/curand_normal over curandState). The header pulls
`<hiprand/hiprand_kernel.h>` and aliases the symbols; the unused host
`<curand.h>` include was dropped.

### Build-portability fixes (behavior-preserving)
- Directed-rounding intrinsics `__fsqrt_rd`/`__frcp_rd` are absent in HIP; mapped
  to the round-to-nearest `_rn` variants in the header (1-ULP; reconstruction is
  graded by nRMSE/adjointness tolerances, not bit-exact).
- `hipMalloc3DArray` requires the flags arg that `cudaMalloc3DArray` defaults to
  0; wrapped in the header.
- Normalized the spaced kernel-launch chevrons `<< <`/`>> >` to `<<<`/`>>>` in the
  3 reduction files (clang HIP parser does not accept the spaced form; nvcc
  accepts both, so byte-safe for CUDA).
- GpuIds.cpp is a host `.cpp` (host compiler, not hipcc); it routes its
  `<cuda_runtime_api.h>` through the compat header, which selects
  `<hip/hip_runtime_api.h>` for host TUs.

## Validation (real gfx90a, HIP_VISIBLE_DEVICES=0, gpuids=None)

`agent_space/tigre_validate.py`: 256^3 head phantom, cone geometry, 128 angles.
- Forward project Siddon and interpolated; back project matched and FDK; run
  algs.fdk and algs.ossart (20 iter); assert finite + nRMSE.
- Adjointness <Ax(x),y> ~= <x,Atb(y)> on random x,y with the interpolated
  projector + matched backprojector (reference-free; this is the cross-arch
  follower gate).
- Python/tests/algorithm_test.py over generate_configurations.py.

Validated at 256^3 (production voxel resolution, full 3D texture) with 32
angles (the 8-fetch software-trilinear interpolated projector at 256^3/128ang
is ~7e11 fetches and impractically slow as a gate; 32 angles keeps it tractable
while exercising every kernel at full image size). Results on gfx90a
(MI250X, ROCm 7.2.1), HIP_VISIBLE_DEVICES=0, gpuids=None:
- Ax Siddon and Ax interpolated agree: mean 27.46 vs 27.46, max 110.7 vs 109.9;
  ||Ax_interp - Ax_siddon|| / ||Ax_siddon|| = 0.0047 (0.47%). This is the proof
  the SW-trilinear texture replacement is numerically correct.
- nRMSE FDK = 0.290, nRMSE OS-SART = 0.151 (OS-SART beats FDK as expected); all
  recon volumes finite and artifact-free.
- Adjointness <Ax(x),y> ~= <x,Atb(y)> (interpolated forward, matched backward,
  random x,y): rel residual = 1.2e-05 (adjoint to float precision). This is the
  reference-free cross-arch follower gate.
- Solver sweep (agent_space/tigre_solvers.py, 64^3): sart/ossart/sirt/cgls/fista
  and the TV-regularized asd_pocs/awasd_pocs/os_asd_pocs (which use the GD_TV/
  GD_AwTV/PICCS reduction kernels) all finite with sensible stop-criteria
  values, confirming the wave64 reduction-loop fix.

Note: the upstream Python/tests/algorithm_test.py harness uses device-name
matching (no gpuids=None hook) and a config-file generator; the direct solver
sweep above exercises the same extensions with the gpuids bypass and is the
equivalent gate.

Validation scripts: agent_space/tigre_validate.py (round-trip + adjointness),
agent_space/tigre_solvers.py (all iterative algorithms).

## Review 2026-06-04 (reviewer, linux-gfx90a -> changes-requested)

Verdict: Request Changes. One fault-class analysis defect: the documented wave64 reduction fix is in a code path that does not compile on either back end. The texture (tex3D_TIGRE) and Siddon Np-cap work is sound. No code-correctness regression in the shipped binary, but the reduction-fix claim is wrong and must be corrected before validation/upstream.

### BLOCKER -- reduction "wave64 fix" targets dead code; misleading analysis
PICCS.cu:199-208, PICCS.cu:248-257, GD_TV.cu:187-196, GD_TV.cu:237-246, GD_AwTV.cu:208-217, GD_AwTV.cu:258-267.
All six edited shuffle reductions (the `offset = warpSize/2` -> `offset = 16` change) live inside `#if (__CUDART_VERSION >= 9000)`. That guard uses the double-underscore `__CUDART_VERSION`, which is NOT a real CUDA macro (CUDA defines `CUDART_VERSION`, no leading underscores) and is undefined on HIP as well -- verified empirically on this host: a hipcc TU sees neither `__CUDART_VERSION` nor `CUDART_VERSION` defined (see /tmp/cudart_test). The guard pre-exists upstream (a07589f). Net effect: on BOTH the CUDA and the HIP build the compiler takes the `#else` branch and the reduction runs through `warpReduce(sdata, tid)` (PICCS.cu:158, GD_TV.cu:146, GD_AwTV.cu:167), a volatile shared-memory warp-synchronous reduction. The porter's `offset = 16` shuffle path is never compiled. Therefore:
- The notes.md "Reduction loop wave64 fix" section and the commit body ("The loop now starts at offset = 16 ... correct on both 32- and 64-lane wavefronts") describe a path that does not execute. The claim that the solver sweep "confirmed the wave64 reduction-loop fix" is incorrect -- the sweep exercised warpReduce, not the shuffle loop.
- The actual active path, `warpReduce`, IS wave64-correct: it is entered under `if (tid < 32)`, those 32 lanes are the low half of one 64-wide wavefront and execute the unrolled `volatile sdata[tid]+=sdata[tid+offset]` chain in lockstep, so the result is right on gfx90a. So the shipped binary is correct; the defect is the analysis, not the math.
Required: either (a) correct notes.md + the commit body to state that warpReduce is the compiled path on both back ends, that it is wave64-correct, and demote the offset=16 edit to a latent/dead-branch cleanup (or drop it); or (b) if the shuffle path was intended to be live under HIP, fix the guard so it actually compiles (e.g. `#if (__CUDART_VERSION >= 9000) || defined(USE_HIP)`) and then the offset=16 fix becomes load-bearing and must be re-validated. As shipped, the fork carries a fix that does nothing and a notes/commit narrative that misrepresents the running code.

### Verified sound (no action)
- tex3D_TIGRE trilinear (cuda_to_hip.h:160-191): texel-center math is correct. xb=x-0.5, fx=floor(xb), dx=frac(xb); the two sampled texels are point indices fx and fx+1 (read at fx+0.5 and fx+1.5), weighted (1-dx)/dx -- exactly CUDA HW unnormalized-linear behavior. Full 8-neighbor trilinear (lerp x, then y, then z), not bilinear. Border-zero inherited from the point fetches' cudaAddressModeBorder. CUDA path forwards to tex3D<float> (byte-identical). All 5 linear textures flipped to cudaFilterModePoint under USE_HIP; all 7 interpolated reads routed through tex3D_TIGRE; the only direct tex3D<float> left (ray_interpolated_projection.cu:198) is the accuracy>1 POINT branch, which matches the texture created POINT at line 591-594 on both back ends. Consistent because the kernel launches with geoArray[sp] (original accuracy), while CreateTextureInterp mutates only its by-value copy.
- Siddon Np-cap (Siddon_projection.cu:249-250, Siddon_projection_parallel.cu): genuine no-op for valid rays. Np=(imax-imin+1)+(jmax-jmin+1)+(kmax-kmin+1) with each index span bounded by the matching nVoxel dimension, so the tight max is Nx+Ny+Nz+3; the cap equals that, identical on CUDA. Catches the unsigned wrap from a __fdividef-driven exact-equality flip (am==axm/aM==axM at lines 167-175,186-206). Parallel cap uses +3 where the tight 2D bound is +2 -- one looser, still safe, not worth changing. Root-cause analysis confirmed.
- __fsqrt_rd/__frcp_rd -> _rn (cuda_to_hip.h): the directed-rounding intrinsics feed only step magnitudes (axu/ayu/azu) and maxlength scaling, never an equality test; the exact-equality branches use __fdividef values, not _rd. 1-ULP nearest-vs-directed is immaterial under nRMSE/adjointness grading.
- setup.py BUILD_WITH_HIP: CUDA path fully preserved in the else branch; HIP skips locate_cuda(), swaps to hipcc -x hip --offload-arch (env HIP_ARCH, gfx90a default, comma-list multi-arch), links amdhip64, host .cpp TUs get USE_HIP + __HIP_PLATFORM_AMD__. hipRAND device functions are header-only so no separate link needed. No per-arch hardcode.
- Commit hygiene: [ROCm] title (51 chars), AI disclosure present, Test Plan with commands, no noreply trailer, no MOAT jargon, ROCm/HIP naming correct, author jeff.daily@amd.com.

## Install as a dependency

Not a base library for other MOAT projects (no `depends_on` consumers).
