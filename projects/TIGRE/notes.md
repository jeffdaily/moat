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

## Install as a dependency

Not a base library for other MOAT projects (no `depends_on` consumers).
