# gsplat notes (ROCm/HIP port, lead linux-gfx90a)

## Summary
gsplat (nerfstudio-project/gsplat) v1.5.3, a PyTorch CUDA extension for
differentiable Gaussian-splatting rasterization. Strategy B (torch hipifies the
.cu/.cuh at build time). Upstream already carries a partial ROCm flag path in its
build scripts (`torch.version.hip` -> `-DUSE_ROCM`), but it was never completed:
the kernels using cooperative-groups reduce, cub, GLM, labeled_partition, and
cuda::std did not build under ROCm. Per the GPUMD precedent (do not auto-skip a
project that merely declares an AMD path), this port builds + GPU-validates and
fixes the rot. The core 3DGS/2DGS path is fully ported; the 3DGUT path (needs
cuda::std::optional) is also now enabled on ROCm by vendoring ROCm/libhipcxx --
see "Enabling 3DGUT on ROCm" below.

## Environment
- torch 2.13.0a0+gitb5e90ff, torch.version.hip 7.2.53211, device MI250X (gfx90a).
- Host ROCm 7.2.1, hipcc /opt/rocm/bin/hipcc, ninja. CUDA_HOME=None, ROCM_HOME=/opt/rocm.
- conda env py_3.12. Deps installed: jaxtyping, rich.
- IMPORTANT: build/run with cwd OUTSIDE /var/lib/jenkins/pytorch (its source tree
  shadows the installed torch). Use the gsplat src dir.
- GPU pinned: HIP_VISIBLE_DEVICES=2 for ALL build + validation.

## Build command (lead, gfx90a)
    cd projects/gsplat/src
    git submodule update --init --recursive gsplat/cuda/csrc/third_party/glm
    HIP_VISIBLE_DEVICES=2 PYTORCH_ROCM_ARCH=gfx90a BUILD_3DGUT=0 MAX_JOBS=16 \
        pip install -e . --no-build-isolation -v
- PYTORCH_ROCM_ARCH=gfx90a scopes to the lead arch (torch otherwise builds
  gfx90a/942/950/1100 from the wheel default; gfx1100 is a follower platform).
- BUILD_3DGUT=0 excludes the 3DGUT/eval3d kernels (RasterizeToPixelsFromWorld*,
  ProjectionUT3DGS*), which use cuda::std::optional/nullopt. libcu++ (<cuda/std/*>)
  is NOT shipped with ROCm 7.2.1. The core 3DGS + 2DGS rasterization/sort/
  compositing (the port's target) is unaffected. NOTE: 3DGUT IS now buildable on
  ROCm by supplying cuda::std from ROCm/libhipcxx -- see "Enabling 3DGUT" below.
- The bundled GLM submodule MUST be initialized; an empty submodule silently falls
  back to system GLM 0.9.9 (no HIP support, wrong errors).

## Enabling 3DGUT on ROCm with ROCm/libhipcxx (gfx90a, ROCm 7.2.1)
The only blocker for 3DGUT on ROCm was `cuda::std::optional` (3 files include
`<cuda/std/optional>`): RasterizeToPixelsFromWorld3DGS{Fwd,Bwd}.cu and
ProjectionUT3DGSFused.cu. ROCm/libhipcxx (header-only, provides the `cuda::std`
namespace; see findings/libhipcxx/NOTES.md) fills it. Steps:

    # 1. Vendor libhipcxx (header-only; amd-develop validated on ROCm 7.2.1)
    git clone --depth 1 --branch amd-develop \
        https://github.com/ROCm/libhipcxx.git <somewhere>/libhipcxx

    # 2. Build gsplat with 3DGUT + the libhipcxx include. Config.h treats
    #    "enable ANY module" as "disable all UNSPECIFIED modules", so to keep the
    #    core 3DGS/2DGS/adam/reloc/losses you MUST list them all explicitly:
    cd projects/gsplat/src
    git submodule update --init --recursive gsplat/cuda/csrc/third_party/glm
    HIP_VISIBLE_DEVICES=1 PYTORCH_ROCM_ARCH=gfx90a \
      LIBHIPCXX_INCLUDE=<somewhere>/libhipcxx/include \
      BUILD_3DGUT=1 BUILD_3DGS=1 BUILD_2DGS=1 BUILD_ADAM=1 BUILD_RELOC=1 \
      BUILD_LOSSES=1 BUILD_CAMERA_WRAPPERS=1 MAX_JOBS=16 \
        pip install -e . --no-build-isolation -v

- `gsplat/cuda/build.py` reads `LIBHIPCXX_INCLUDE` and, on ROCm, appends
  `-I$LIBHIPCXX_INCLUDE` to `extra_cuda_cflags` (the device-compile flags; reaches
  hipcc via setup.py's `nvcc` key). It also flips the ROCm `BUILD_3DGUT` default to
  1 when `LIBHIPCXX_INCLUDE` is set (still 0 when unset, so a plain `pip install`
  without libhipcxx does not hard-error on the missing header). CUDA path unchanged.
- `<cuda/std/optional>` and `cuda::std::optional`/`nullopt` pass through torch's
  hipify UNCHANGED (no mapping), so the only source change beyond the include was
  one latent fault: the two FromWorld kernels' `cudaFuncSetAttribute(kernel,...)`
  needed the `(const void*)kernel<...>` cast (fault #9) -- it was already applied to
  the non-3DGUT rasterizers but those 2 files were never compiled at BUILD_3DGUT=0.
- libhipcxx does NOT provide `<cuda/ptx>`; gsplat does not use cuda::ptx, so fine.

## CONFIG GOTCHA (gsplat Config.h module selection)
Config.h (csrc/Config.h): if you explicitly set GSPLAT_BUILD_<X>=1 for ANY module,
every module you did NOT name defaults to 0. So `BUILD_3DGUT=1` ALONE builds ONLY
3DGUT and silently drops 3DGS/2DGS/adam/reloc/losses (has_3dgs()==False, only 2 ops
register). To add 3DGUT to the full extension, enable all desired modules together
(see the build command above). This is upstream behavior, not ROCm-specific.

## CUDA surface (warp / wave64)
gsplat does NOT use raw warp intrinsics (`__shfl`/`__ballot`/`__any`/`__syncwarp`)
-- grep finds zero. It uses CUDA Cooperative Groups:
- `cg::tiled_partition<32>(block)` warp tiles in the rasterize bwd + projection
  kernels; reductions via `cg::reduce(warp, val, cg::plus/greater)` (Utils.cuh
  warpSum/warpMax), `warp.any(valid)`, leader atomics `if (warp.thread_rank()==0)`.
- per-tile gaussian sort = host-side `cub::DeviceRadixSort::SortPairs` /
  `DeviceSegmentedRadixSort::SortPairs`, **begin_bit=0** (IntersectTile.cu:498,550)
  -> the cudaKDTree nonzero-begin_bit hipCUB bug does NOT apply here.
- `cg::labeled_partition(warp, gid/cid)` in the packed/fused projection backward
  to coalesce per-gaussian/per-camera gradient atomics.

Wave64 verdict: the 32-thread tiles are wave-size-agnostic by construction. HIP's
`thread_block_tile<32>` shuffles (__shfl_*, width=numThreads=32) and the
member-mask coalesced shuffles stay within the tile's lanes, so a tiled_partition
<32> on a 64-lane gfx90a wavefront reduces correctly per 32-lane group, exactly
like a 32-lane NVIDIA warp. No positional 2-rows-per-block packing and no
`lane=tid&31; if(lane==0)` election (the popsift traps) are present, so no explicit
wave64 lane-math rework was required -- the fixes below are HIP-CG / hipify gaps.

## Fault classes hit + fixes (guarded by USE_ROCM unless noted)

1. nvcc-only compiler flags leak to hipcc/clang++ (build.py x2).
   `gsplat/cuda/build.py` and `experimental/render/kernels/cuda/build.py` added
   `--forward-unknown-opts`, `-use_fast_math`, `-diag-suppress 20012,186`
   unconditionally; hipcc/clang++ rejects all three ("unknown argument", "no such
   file 20012,186"). Fix: guard on `not torch.version.hip`; use `-ffast-math`
   instead of `-use_fast_math` on ROCm. (build.py ~107/131/135 + experimental.)

2. hipify rewrites `std::min`/`std::max` -> `::min`/`::max`. On the HIP HOST pass
   (plain g++) global `::min`/`::max(float,float)` do not exist -> host TUs
   (Lidar.cpp, Projection.cpp) including Cameras.cuh/ExternalDistortion.cuh fail
   ("::min has not been declared"). Fix: Common.h provides host-only
   (`!__HIP_DEVICE_COMPILE__`) global min/max forwarding to std::min/std::max; the
   device pass keeps HIP's builtin ::min/::max, CUDA is untouched.

3. Bundled GLM mangled by hipify (the biggest one). torch's hipify
   (CUDAExtension AOT path) walks every .hpp under cwd + the extension include dirs
   into its file set, then content-rewrites any GLM header a source pulls in: it
   drops GLM's .inl files (hipify copies only .hpp/.h) and corrupts GLM's
   __CUDACC__/CUDA_VERSION compiler detection -> "GLM requires CUDA 7.0",
   "scalar_constants.inl: No such file", "no matching function for 'length'/
   'quat_cast'/... (host fn from device fn)". GLM 1.0.2 already detects __HIP__ and
   compiles verbatim under -x hip. Fix: setup.py monkeypatches
   `hipify_python.hipify` to add the glm dir to `ignores` and strip it from
   `header_include_dirs`, leaving GLM pristine and resolved from source via -I.
   (torch's AOT hipify call forwards no `ignores`, so it must be injected at the
   setup.py level; the JIT load() path does pass ignores but only for ROCM/TORCH.)

4. HIP cooperative_groups has no cg::reduce and no <cooperative_groups/reduce.h>
   (ROCm 7.2.1; grep of hip headers finds no reduce). gsplat's warpSum/warpMax
   (Utils.cuh) and 3 rasterizer bwd 'warp_bin_final' sites rely on it. Fix:
   Utils.cuh warpReduceSum/Max = a butterfly `shfl_xor` all-reduce over the tile
   (USE_ROCM), cg::reduce on CUDA; warpSum/warpMax + the 3
   `cg::reduce(warp,bin_final,greater<int>())` sites route through it. width-32 tile
   shuffle is wave64-correct.

5. No libcu++ (<cuda/std/*>) in the ROCm 7.2.x RELEASE. Utils.cuh used
   cuda::std::is_floating_point_v / numeric_limits. Fix (core path): `namespace
   gstd = std` on ROCm (std numeric_limits/type_traits are constexpr +
   device-usable under HIP), cuda::std on CUDA -- avoids an extra dependency for
   the core build. The 3DGUT files instead use cuda::std::optional, which has no
   std drop-in; that is now supplied by vendoring the header-only ROCm/libhipcxx
   (provides the `cuda::std` namespace) and adding its include/ to the device
   compile -- see "Enabling 3DGUT on ROCm" above. (Utils.cuh keeps the gstd=std
   alias on ROCm even in the 3DGUT build; only the 3DGUT .cu files' direct
   cuda::std::optional needs libhipcxx.)

6. hipify maps `<cub/cub.cuh>` -> `<hipcub/hipcub.hpp>` but leaves the `cub::`
   namespace unrenamed -> cub::DoubleBuffer/DeviceRadixSort undeclared
   (IntersectTile.cu, IntersectTileLidar.cu). Fix: `namespace cub = hipcub;` under
   USE_ROCM after the include (the cudaKDTree alias lesson).

7. HIP cooperative_groups has no cg::labeled_partition (packed/fused projection
   backward, 4 files: Projection2DGS{Fused,Packed}, ProjectionEWA3DGS{Fused,Packed}).
   Fix: Utils.cuh `LabeledGroup` rebuilt from `warp.match_any(label)` (HIP CG DOES
   provide match_any): holds the 64-bit same-label mask; size()=__popcll(mask),
   thread_rank()=__popcll(mask & below-lane), and an all_reduce over only the
   same-label lanes (iterate set bits, __shfl each). Only the lowest same-label lane
   (rank 0) does the atomicAdd. Dispatched via `LABELED_PARTITION(warp,label)`
   (labeled_partition_compat on HIP, cg::labeled_partition on CUDA), warpReduceSum/
   Max overloaded for LabeledGroup. IMPORTANT: a first cut used a trivial 1-lane
   "self group" (every lane does its own atomic). That is numerically correct (float
   atomicAdd is associative) BUT the per-lane atomic fan-out adds enough float
   accumulation-ORDER noise that v_viewmats -- summed over thousands of gaussians --
   flapped the tests' tight 2e-3 gradient tolerance at ~1 element run-to-run. The
   match_any version restores the CUDA atomic count (one per distinct label per warp)
   and the flake goes away. Lesson: match the CUDA atomic granularity, do not fan out.

8. static-constexpr member read through an instance is not a constant expression
   under clang (Lidars.cuh:122 `constexpr float kToAngle = 1.f/lidar.ANGLE_..`);
   nvcc is lenient. Fix: `const` instead of `constexpr` (identical runtime; not
   USE_ROCM-guarded since const is portable).

9. cudaFuncSetAttribute(kernel, ...) -> hipFuncSetAttribute requires the kernel as
   `const void*` on HIP (CUDA has a templated T* overload; HIP only `const void*`).
   6 rasterizer launch sites (RasterizeToPixels{3DGS,2DGS}{Fwd,Bwd},
   RasterizeToIndices{2DGS,3DGS}). Fix: cast `(const void *)kernel<...>` -- portable,
   CUDA's cudaFuncSetAttribute also has a const void* overload (not USE_ROCM-guarded).

10. __CUDA_ARCH__ undefined during HIP device compilation (TensorView.h:125 picked
    host TORCH_CHECK_INDEX/throw on the HIP device pass inside a __host__ __device__
    method). Fix: `#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)`
    (the cudaKDTree lesson, reconfirmed). Only surfaces with BUILD_CAMERA_WRAPPERS=1.

11. std::array CTAD deduction guide and std::array::at() are __host__-only under
    clang/HIP -> illegal in __host__ __device__ code (TensorView.h:111 `std::array{}`;
    Cameras.cuh:1202/1205 FThetaCameraModel `.at()` pulls __throw_out_of_range_fmt).
    Fix: spell the type `std::array<int64_t, ndims>{}` (no CTAD) and use operator[]
    instead of .at() (no bounds-check throw). Only with BUILD_CAMERA_WRAPPERS=1.

12. The experimental inference renderer (experimental/render, a separate
    NVIDIA-tuned extension: inline PTX asm, half2 intrinsics __hmax2/__hlt2_mask/
    __hgeu2_mask, cub/block internals, 32-bit warp masks, hipFuncSetAttribute) is
    not part of the core differentiable rasterizer and is not imported by the gsplat
    API. setup.py skips it on ROCm (build only gsplat.csrc). Porting it is a
    separate, larger effort.

## Files changed (port diff)
- setup.py (hipify GLM-ignore monkeypatch; skip experimental render ext on ROCm)
- gsplat/cuda/build.py, experimental/render/kernels/cuda/build.py (nvcc flag guards;
  build.py also: LIBHIPCXX_INCLUDE -> -I on ROCm device flags, flips ROCm 3DGUT
  default on when LIBHIPCXX_INCLUDE set)
- gsplat/cuda/include/Common.h (host ::min/::max shim)
- gsplat/cuda/include/Utils.cuh (cg::reduce -> shfl_xor, cuda::std -> gstd,
  LabeledGroup via match_any + LABELED_PARTITION)
- gsplat/cuda/include/Lidars.cuh (constexpr -> const)
- gsplat/cuda/include/Cameras.cuh (FThetaCameraModel .at() -> operator[])
- gsplat/cuda/csrc/TensorView.h (__CUDA_ARCH__||__HIP_DEVICE_COMPILE__; std::array CTAD)
- gsplat/cuda/csrc/IntersectTile.cu, IntersectTileLidar.cu (cub=hipcub alias)
- gsplat/cuda/csrc/Projection2DGS{Fused,Packed}.cu,
  ProjectionEWA3DGS{Fused,Packed}.cu (LABELED_PARTITION)
- gsplat/cuda/csrc/RasterizeToPixels{3DGS,2DGS}{Fwd,Bwd}.cu,
  RasterizeToIndices{2DGS,3DGS}.cu (const void* kernel cast)
- gsplat/cuda/csrc/RasterizeToPixelsFromWorld3DGS{Fwd,Bwd}.cu (const void* kernel
  cast -- the 3DGUT-only FromWorld rasterizers, surfaced once BUILD_3DGUT=1 compiled
  them; same fault #9 as the non-3DGUT rasterizers)
- bundled GLM submodule initialized to 1.0.2 (delivered by fork, not a source edit)
- (3DGUT requires the external ROCm/libhipcxx clone for cuda::std; not a source edit,
  wired via LIBHIPCXX_INCLUDE -- see "Enabling 3DGUT on ROCm" above)

## Validation (gfx90a / MI250X, HIP_VISIBLE_DEVICES=2, ROCm 7.2.1)
Build flags: PYTORCH_ROCM_ARCH=gfx90a BUILD_3DGUT=0 BUILD_CAMERA_WRAPPERS=1
(fast-math defaults OFF on ROCm -- see fault 13 below).

Fault 13: ROCm's -ffast-math is more aggressive than CUDA's -use_fast_math and
perturbs a few ill-conditioned projection-covariance gradients past gsplat's own
tolerances: test_proj/test_projection[batch_dims=(2,),(1,2),pinhole] each fail on
exactly 1/6,036,390 elements of v_covars (|d|=0.47 vs 0.1 allowed, at one
near-degenerate gaussian). Confirmed root cause: rebuilding with FAST_MATH=0 makes
all 15 pinhole proj/projection tests pass. Fix: build.py defaults FAST_MATH off on
ROCm (FAST_MATH=1 still opt-in); CUDA default unchanged. NOT a wave64/port defect.

1. Upstream pytest tests/test_basic.py (HIP extension vs pure-torch reference,
   forward + autograd.grad), non-lidar core math:
     -k "(test_quat_scale_to_covar_preci or test_proj or test_projection or
          test_fully_fused_projection_packed or test_isect or test_sh) and not lidar"
   Result (final LabeledGroup build): 102 passed, 6 skipped, 0 non-lidar failures,
   STABLE across repeated runs (projection fwd+bwd grads incl. fused+packed,
   quat/scale covariance, spherical harmonics, tile intersection). These use the
   garden test scene (assets/test_garden.npz) and the tolerances baked into the
   upstream tests. (The earlier per-lane-atomic labeled_partition cut flapped 1-2
   v_viewmats elements run-to-run at the 2e-3 tolerance; the match_any LabeledGroup
   -- fault 7 -- fixed it. The fast-math-off default fixed the v_covars edges.)

2. Standalone rasterization/compositing validation (agent_space/gsplat_validate2.py),
   since test_rasterize_to_pixels needs nerfacc (CUDA-only wheel; from-source build
   not run):
   [B] gsplat HIP render vs an INDEPENDENT pure-torch front-to-back over-compositor
       fed gsplat's own projected gaussians, matching Common.h constants exactly
       (MAX_ALPHA=0.99, ALPHA_THRESHOLD=1/255, TRANSMITTANCE_THRESHOLD=1e-4):
       max|dRGB|=2.45e-4 (mean 9.8e-7), max|dAlpha|=2.8e-4. PASS (residual is
       -ffast-math + float accumulation order).
   [C] analytic d(loss)/d(colors) vs central finite difference of the HIP forward:
       max rel err 5.3e-5. PASS (rasterization backward correct).
   [D] determinism, 2 same-seed runs: render + alpha BIT-EXACT; gradients match
       within float-atomicAdd noise (max rel diff ~1.6e-7 means, ~1.2e-7 colors).
       PASS. (Gradient atomicAdd order is non-deterministic on any GPU incl. CUDA;
       the forward has no shared-output atomics so it is bit-exact.)

## Validation 2026-05-30 (gfx1100, ROCm 7.2.1)

Platform: AMD Radeon Pro W7800 48GB, gfx1100 (RDNA3, wave32). ROCm 7.2.1.
conda env py_3.12, torch 2.13.0a0+gitb5e90ff, torch.version.hip 7.2.53211.
Fork tip validated: 621ebd6e74b4dfabb3e64fdaaa2e7ed3941a5f08.

### Build command
    cd projects/gsplat/src
    git submodule update --init --recursive gsplat/cuda/csrc/third_party/glm
    HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx1100 BUILD_3DGUT=0 BUILD_CAMERA_WRAPPERS=1 MAX_JOBS=16 \
        pip install -e . --no-build-isolation -v
No fork modification required (follower reuses gfx90a moat-port commit).

### Code-object evidence
    roc-obj-ls gsplat/csrc.so | grep gfx1100 | wc -l  -> 20
All 20 code objects are hipv4-amdgcn-amd-amdhsa--gfx1100; no other arch present.

### pytest test suite (tests/test_basic.py)
Command:
    cd projects/gsplat/src && HIP_VISIBLE_DEVICES=0 python3 -m pytest tests/test_basic.py \
        -k "(test_quat_scale_to_covar_preci or test_proj or test_projection or \
             test_fully_fused_projection_packed or test_isect or test_sh) and not lidar" -v
Result: 102 passed, 6 skipped, 0 failures -- identical to gfx90a result.
Run twice for determinism: both runs 102 passed, 6 skipped.
The 6 skips: 3 test_isect + 3 test_projection_ut_* (3DGUT/lidar, excluded by BUILD_3DGUT=0).

Tests covered (forward + autograd.grad backward):
- test_quat_scale_to_covar_preci: covariance/precision matrix computation
- test_proj, test_projection: 3DGS projection fwd+bwd for pinhole/ortho/fisheye
- test_fully_fused_projection_packed: packed/fused LabeledGroup projection bwd
- test_sh: spherical harmonics
All compare HIP ext output vs pure-torch reference at the upstream test tolerances.

### Standalone rasterization fwd+bwd validation (agent_space/gsplat_validate_gfx1100.py)
(nerfacc CUDA wheel not available on ROCm; test_rasterization.py test_rasterization[...]
failures are all ModuleNotFoundError: nerfacc, same as on gfx90a, not a numeric defect.)

[B] Forward: render [0, 0.84], alpha [0, 1.0], 95.8% pixels lit (alpha>0.1). No NaN/Inf. PASS.
[C] Backward: analytic d(loss)/d(colors) vs finite-diff over top contributing gaussians:
    max rel err = 1.19e-3 (<2%). PASS.
[D] Determinism: render + alpha bit-exact across same-seed runs. Gradient rel diff 8.73e-06. PASS.

### wave32 warp-tile reductions
gfx1100 is RDNA3 (wave32). tiled_partition<32> creates tiles that are exactly ONE full
wavefront on gfx1100, so each tile's butterfly shfl_xor reduction (warpReduceSum/Max) and
LabeledGroup match_any logic operate on a single wavefront with no inter-wavefront
coordination needed. The 102/102 projection fwd+bwd and the rasterization fwd+bwd results
confirm the custom warpSum/warpMax (fault 4) and LabeledGroup via match_any (fault 7) are
correct on wave32. No wave-size source fix required for gfx1100.

## 3DGUT now ENABLED on ROCm via ROCm/libhipcxx (2026-05-30, gfx90a, ROCm 7.2.1)
The earlier BUILD_3DGUT=0 limitation existed ONLY because libcu++ (`<cuda/std/*>`,
for cuda::std::optional) is absent from ROCm 7.2.x. Supplying it from the
header-only ROCm/libhipcxx (amd-develop, commit fa4ccc6) closes the gap; see
"Enabling 3DGUT on ROCm" above for the exact flags.

Build: full extension with `BUILD_3DGUT=1 BUILD_3DGS=1 BUILD_2DGS=1 BUILD_ADAM=1
BUILD_RELOC=1 BUILD_LOSSES=1 BUILD_CAMERA_WRAPPERS=1 LIBHIPCXX_INCLUDE=<clone>/include`
on gfx90a -> exit 0. `has_3dgut()`, `has_3dgs()`, `has_2dgs()`, `has_adam()` all
True; all 5 op namespaces register (3DGUT: projection_ut_3dgs_fused,
intersect_tile_lidar, rasterize_to_pixels_from_world_3dgs_fwd).

Tests previously blocked, now passing on gfx90a (HIP_VISIBLE_DEVICES=1):
- test_isect_lidar_corner_cases: 102 passed (was 100% "op not registered"
  AttributeError -- the test has no has_3dgut() skip). [collection is 102, not the
  ~107 the earlier note estimated.]
- test_isect_lidar + test_projection_ut_* + test_ut_params_*: 137 passed (were
  has_3dgut()-skipped).
- test_external_distortion.py + test_ftheta.py: 57 passed (were has_3dgut()-skipped).
Total: 296 3DGUT/lidar/UT tests that were unavailable at BUILD_3DGUT=0 now pass.

Core regression unaffected: the notes' core 3DGS/2DGS filter still passes (104-105
of 105; one test_projection[pinhole] element flaps run-to-run at the tight gradient
tolerance -- the pre-existing float-atomic accumulation-order noise of fault #7,
passes in isolation and on rerun, NOT introduced by 3DGUT/libhipcxx). FAST_MATH
remains off by default on ROCm (fault #13).

## Scope / known limitations on this ROCm stack
- 3DGUT eval3d tests (test_rasterize_to_pixels_eval3d, 28 cases) still FAIL, but
  ONLY because their pure-torch REFERENCE (gsplat/cuda/_torch_impl_eval3d.py) does
  `from nerfacc import ...` and nerfacc is absent on ROCm (its CUDA wheel's _C is
  None anyway). This is a missing-reference-package limitation, NOT a HIP kernel or
  libhipcxx defect -- the eval3d HIP kernel itself built and its op registered. Same
  nerfacc gap already noted for test_rasterization below.
- experimental/render inference renderer skipped on ROCm (NVIDIA PTX/half2/cub),
  separate from the core API.
- nerfacc CUDA wheel's _C is None on ROCm; the nerfacc-backed reference path
  (_rasterization / _rasterize_to_pixels accumulate) was replaced by the
  independent compositor above for the compositing-kernel check.
- libhipcxx does NOT provide `<cuda/ptx>` (cuda::ptx); gsplat does not use it.
