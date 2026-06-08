# nerfacc ROCm/HIP port plan (lead: linux-gfx90a)

## Project
- Upstream: nerfstudio-project/nerfacc @ 57ccfa1 ("Use int64_t instead of long (#301)"), default branch `master`. 1457 stars.
- A General NeRF Acceleration Toolbox: efficient volumetric ray-marching / sampling
  for NeRFs, shipped as a PyTorch CUDA extension (`nerfacc.csrc`, the `_C` module).
- ext_type: **torch-extension** -> **Strategy B** (torch hipifies the `.cu`/`.cuh`
  at build time via `torch.utils.cpp_extension`; do NOT add a compat header or
  hand-rename symbols). Same family as the completed gsplat and FastGeodis ports.
- Why in MOAT (see notes.md): nerfacc is gsplat's optional pure-torch reference
  package. nerfacc ships a CUDA-only wheel whose `_C` is unavailable on ROCm, so
  `import nerfacc` fails and 38 gsplat reference tests (the eval3d accumulator +
  several rasterize_to_pixels comparisons) are a documented known-fail on every
  platform. A ROCm build of nerfacc closes that gap and stands on its own.

## Existing AMD support
- **Finding: NONE. No existing ROCm/HIP port, authoritative or community.**
- Evidence:
  - `grep -rniE 'amd|rocm|hip|gfx[0-9]' README* docs/` -> only false positives
    (the scene names "Hotdog"/"Ship" match `hip`; PSNR benchmark tables). No AMD
    section, no "notable forks" link. Upstream does NOT use a link-the-fork merge
    policy (no platform-fork section), so an upstream PR is the correct vehicle
    if/when we land it.
  - Repo-wide grep: the ONLY rocm/hip hits are in `setup.py` -- nerfacc already
    anticipates a hipify build: `setup.py:35-36` strips generated `*hip*` files on
    rebuild; `setup.py:71-75` branches on `torch.version.hip` to define `USE_ROCM`
    and undef `__HIP_NO_HALF_CONVERSIONS__` (and drops nvcc-only
    `--expt-relaxed-constexpr`). There are NO `#if defined(USE_ROCM)` guards inside
    the `.cu`/`.cuh` kernel source -- device code is plain CUDA spelling relying on
    torch hipify. This is build-scaffold awareness only, never validated on AMD.
  - `gh api repos/nerfstudio-project/nerfacc/forks` (120 forks): none under
    ROCm/AMD/GPUOpen orgs, none with rocm/hip/amd in the name.
  - `gh api repos/ROCm/nerfacc` -> 404; `repos/AMD/nerfacc` -> 404.
  - WebSearch ("nerfacc ROCm AMD GPU HIP port", "nerfacc AMD Instinct MI300 gfx9"):
    no nerfacc-specific AMD result; only generic ROCm/HIP docs.
- **Decision: proceed with a from-scratch Strategy B port (correctness-first).**
  The pre-existing `torch.version.hip` build branch is a hint that the author
  expected hipify to work, not a completed port; per the GPUMD/gsplat precedent we
  BUILD + GPU-validate and fix only what hipify cannot. There is no community `.cu`
  edit to inherit (and none to audit for wave64 hardcodes).

## ROCm torch env (grounds the build commands; identical to gsplat/FastGeodis)
- conda env `py_3.12`, torch **2.13.0a0+gitb5e90ff**, `torch.version.hip
  7.2.53211`, `torch.cuda.is_available()=True`, device "AMD Instinct MI250X /
  MI250" (gfx90a, GFX Version gfx90a per rocm-smi). Host ROCm 7.2.1, hipcc at
  /opt/rocm/bin/hipcc.
- `CUDA_HOME` empty, `ROCM_HOME=/opt/rocm` -> a `CUDAExtension` build hipifies the
  `.cu`/`.cuh` and links amdhip64/c10_hip/torch_hip automatically (Strategy B).
- GPU pinned via `HIP_VISIBLE_DEVICES=<assigned ordinal>` for ALL build + run
  (this host shares 4 GCDs with a sibling CLI; pick a free ordinal at run time).
- NOTE (gsplat caveat, applies here): run pytest with cwd OUTSIDE
  /var/lib/jenkins/pytorch (its source tree shadows the installed `torch`). The
  nerfacc src dir, or /tmp with the tests copied in, is fine.

## Build classification + evidence
- **torch-extension.** `setup.py` imports `from torch.utils.cpp_extension import
  BuildExtension` (line 19) and `CUDAExtension` (line 29); the ext is built as a
  single `CUDAExtension("nerfacc.csrc", sources=glob('csrc/*.cu')+glob('*.cpp'))`
  (lines 80-90), `cmdclass={"build_ext": BuildExtension...}` (line 129). No CMake.
  Per PORTING_GUIDE "Build classification" this is unambiguously Strategy B.
- ext_type already set to `torch-extension` in upstream.json + status.json.

## Port strategy: B (torch hipify), rationale
- Building a `CUDAExtension` against a ROCm torch runs `torch.utils.hipify` on the
  extension `.cu`/`.cuh` and links the HIP runtime. Keep sources in CUDA spelling;
  fix only fault classes hipify cannot (none of warp-size, textures, layered arrays
  apply here -- see inventory). The CUDA surface is small and standard (cub
  DeviceScan, thrust reverse_iterator, PyTorch philox RNG), all with 1:1 ROCm
  spellings via hipify.
- No AMD-native rewrite is warranted: nerfacc has no CUTLASS/CuTe/wgmma/Hopper
  kernels. The hot paths are ray-marching, a Blelloch shared-memory scan, and
  host-side cub scans -- a mechanical HIP translation is the right call.

## CUDA surface inventory
Five `.cu` translation units + headers under `nerfacc/cuda/csrc/`:

| File | Kernels | Notable surface |
|------|---------|-----------------|
| `camera.cu` (183) | `opencv_lens_undistortion{,_fisheye}` | pure per-thread math, `<<<>>>`; no cub/thrust/rng/atomics |
| `grid.cu` (519) | `traverse_grids_kernel`, `ray_aabb_intersect_kernel` | per-thread ray AABB / grid DDA; includes curand headers but kernels use NO rng/atomics/warp/texture |
| `pdf.cu` (456) | `importance_sampling_kernel`, `compute_intervels_kernel`, `searchsorted_kernel`, `compute_ray_ids_kernel` | PyTorch philox RNG (only here): `at::cuda::philox::unpack` + `curandStatePhilox4_32_10_t` + `curand_init` + `curand_uniform` (pdf.cu:140-143); `gen->philox_cuda_state(4)` host-side |
| `scan.cu` (303) | `device::{inclusive,exclusive}_scan_kernel<float,16,32>` | hand-rolled Blelloch scan over `__shared__` (utils_scan.cuh); `thrust::make_reverse_iterator` (host iterator adaptor); `std::plus`/custom `Multiply` binary op |
| `scan_cub.cu` (287) | (host wrappers only) | `cub::DeviceScan::{Exclusive,Inclusive}{Sum,Scan}ByKey` via `CUB_WRAPPER`; gated by `CUB_SUPPORTS_SCAN_BY_KEY()` |

Headers: `utils_scan.cuh` (the scan kernels), `utils.cub.cuh` (CUB_VERSION /
CUB_WRAPPER macros), `utils_math.cuh` (1476 lines of pure inline vec/quat math --
NO warp/atomic/texture/special-fn surface), `utils_camera.cuh`,
`utils_contraction.cuh`, `utils_grid.cuh`, `data_spec*.{hpp,cuh}`, `utils_cuda.cuh`
(CHECK_* macros, `CUDA_GET_THREAD_ID`, `DEVICE_GUARD`).

Surface NOT present (grepped, all zero): `__shfl*`, `__ballot`, `__any`/`__all`,
`__activemask`, `__syncwarp`, `warpSize`, any hardcoded-32 warp logic, cooperative
groups, textures/surfaces (`tex*`/`surf*`/`cudaArray` -- the lone `surf` hit is the
word "surface" in a comment), `__constant__`, device atomics
(atomicAdd/Max/Min/CAS), `cuda::std`/`<cuda/std/*>`, `__half`/`half2` intrinsics,
inline PTX `asm()`, cuBLAS/cuFFT/cuSPARSE, raw `cudaMalloc`/`cudaMemcpy`/streams/
events (PyTorch owns the stream via `at::cuda::getCurrentCUDAStream()`).

Library mapping (all handled by torch hipify):
- `<cub/cub.cuh>`, `<cub/version.cuh>`, `cub::DeviceScan`, `cub::Equality` ->
  hipcub. **Caveat below (CUB_VERSION).**
- `<thrust/iterator/reverse_iterator.h>`, `thrust::make_reverse_iterator` ->
  rocThrust 1:1 (host-side; trivial).
- `<curand*.h>`, `curandStatePhilox4_32_10_t`, `curand_init`, `curand_uniform`,
  `at::cuda::philox::unpack`, `at::PhiloxCudaState` -> hiprand/rocrand; this is the
  exact RNG pattern PyTorch's own ROCm build uses, so the headers resolve.
- `c10/cuda/CUDAGuard.h`, `ATen/cuda/{CUDAContext,CUDAGeneratorImpl,Exceptions,
  CUDAGraphsUtils}.{h,cuh}` -> c10/hip + ATen HIP, auto-mapped.

## Risk list
1. **cub `is_cub_available()` silently becomes false on ROCm (primary functional
   note, LOW risk to correctness).** `utils.cub.cuh` defines
   `CUB_SUPPORTS_SCAN_BY_KEY()` as `CUB_VERSION >= 101500`, and `CUB_VERSION`
   defaults to `0` unless `<cub/version.cuh>` defines it. hipify rewrites
   `<cub/version.cuh>` -> `<hipcub/hipcub_version.hpp>`, which defines
   `HIPCUB_VERSION` (measured 400200 on this stack) but **NOT** `CUB_VERSION`. So
   on ROCm `CUB_VERSION` stays 0 -> `CUB_SUPPORTS_SCAN_BY_KEY()` == 0 ->
   `is_cub_available()` returns **false** and the entire `*_cub` body is
   `#if`-compiled out. Consequence: the build SUCCEEDS, and the Python layer
   (`nerfacc/scan.py:56,120,190,256` and `nerfacc/volrend.py:206,266`) falls back
   to the hand-rolled packed/`pack_info` scan whenever `is_cub_available()` is
   false. The library is fully FUNCTIONAL and correct without the cub path -- but
   the `*_cub` ops register as unavailable and test_scan's cub-specific assertions
   would exercise the fallback. This is the same "tested fallback path" situation
   gsplat accepted. **Plan:** first validate the default build (cub off). Then, as
   a small, contained improvement, make `CUB_SUPPORTS_SCAN_BY_KEY()` also true on
   ROCm by checking `HIPCUB_VERSION` (or `defined(USE_ROCM)`), so the hipcub
   DeviceScan-by-key path is actually compiled and validated. Guard with
   `USE_ROCM`/`HIPCUB_VERSION` so CUDA is byte-identical. Decide keep-vs-improve
   based on whether the hipcub `*ByKey` + custom `Product` functor build cleanly;
   if hipcub `ExclusiveScanByKey` with a user functor misbehaves, leave the
   graceful fallback (correctness-first) and note it.
2. **Warp size (wave64 vs wave32): LOW.** No warp intrinsics anywhere. The only
   shared-memory collective is the Blelloch scan in utils_scan.cuh, launched as
   `<<<blocks, dim3(16,32)>>>` (`num_threads_x=16`, `num_threads_y=32`): the scan
   tree runs over the 16-wide x-dimension with an explicit `__syncthreads()` after
   EVERY up-sweep and down-sweep step (utils_scan.cuh:71,79,88,97,198,206,215,224),
   so it does NOT rely on implicit wave-lockstep (the icicle cross-logical-warp
   hazard) and is correct on both wave64 (gfx90a) and wave32 (gfx1100/gfx1151).
   `__shared__ T sbuf[num_threads_y][2*num_threads_x]` is sized by compile-time
   template constants, not by runtime warp count, so no static-array warp-sizing
   trap. Expectation: no wave-size source fix. Followers must still re-validate on
   RDNA per policy (the scan is deterministic, so a cross-arch output diff is the
   right follower gate if anything looks off).
3. **Philox RNG numeric reproducibility: LOW-MEDIUM.** pdf.cu's stratified
   importance sampling seeds `curandStatePhilox4_32_10_t` from
   `at::cuda::philox::unpack(philox_args)` and draws `curand_uniform`. hiprand's
   Philox4_32_10 matches curand's algorithm, and the test (`test_pdf.py`) compares
   the `_C` importance_sampling against a pure-torch reference. The stratified path
   uses randomness, so the reference comparison may use `stratified=False` or a
   tolerance; verify the test's exact contract. If a stratified test compares
   bit-exact RNG draws it could differ -- but the non-stratified deterministic path
   is the main correctness gate. Not a wave-size issue.
4. **hipify abs-path / generated-file hygiene: LOW.** `setup.py:35-36` already
   strips `*hip*` paths from the source glob on rebuild (the known torch-hipify
   incremental gotcha). The commented-out `include_package_data=False` block
   (setup.py:94-96) hints the author hit a hipify abs-path packaging issue; if a
   rebuild picks up a stale hipified mirror, clean the build dir / re-hipify (the
   Strategy B incremental-build caveat). Build with `MAX_JOBS` set;
   `BuildExtension` here uses `use_ninja=False`.
5. **`-Wno-sign-compare` / `-O3` cxx flags + `-s` link strip: LOW.** These are
   plain g++/clang flags, accepted on ROCm. `extra_compile_args["nvcc"]` gets
   `-O3` only on ROCm (the `--expt-relaxed-constexpr` is correctly NVIDIA-gated at
   setup.py:76-77). No nvcc-only flag leaks to hipcc (unlike gsplat fault #1), so
   no build.py-style guard is needed.
6. **No texture / rule-of-five / OOB-neighbor / 256B-pitch / layered-array / smid /
   stream-callback / `__fsqrt_rn` exactness risks** -- none of those surfaces exist
   in this codebase (all grepped zero). The float math is ordinary `fmaxf`/`/`/
   vector ops in utils_math.cuh; `-ffast-math` is NOT forced (no FAST_MATH flag), so
   the gsplat fast-math gradient-edge class does not apply.

## File-by-file change list (expected)
Strategy B: most files compile unchanged via hipify. Anticipated edits, all minimal
and `USE_ROCM`-guarded so the CUDA build is byte-identical:
- `nerfacc/cuda/csrc/include/utils.cub.cuh` -- (improvement, optional) make
  `CUB_SUPPORTS_SCAN_BY_KEY()` true on ROCm via `HIPCUB_VERSION`/`USE_ROCM` so the
  hipcub DeviceScan-by-key path is compiled + tested instead of silently disabled.
  If hipcub `*ScanByKey` with the custom `Product` functor does not build cleanly,
  DROP this edit and keep the upstream graceful fallback (cub off) -- still correct.
- `setup.py` -- only if a real build issue appears (e.g. re-enable the hipify
  abs-path `include_package_data=False` workaround the author left commented at
  94-96, or add a missing include dir). No change expected for the default build.
- Everything else (`camera.cu`, `grid.cu`, `pdf.cu`, `scan.cu`, `scan_cub.cu`,
  `utils_scan.cuh`, `utils_math.cuh`, `utils_camera.cuh`, `utils_contraction.cuh`,
  `utils_grid.cuh`, `data_spec*.{hpp,cuh}`, `utils_cuda.cuh`, `nerfacc.cpp`):
  expected NO source change -- hipify + the existing `torch.version.hip` setup.py
  branch handle them. Add a `USE_ROCM` guard only for a concrete fault that
  surfaces at build/validate time.

## Build commands (lead, gfx90a)
    cd projects/nerfacc/src
    HIP_VISIBLE_DEVICES=<free ordinal> PYTORCH_ROCM_ARCH=gfx90a MAX_JOBS=16 \
        pip install -e . --no-build-isolation -v
- `PYTORCH_ROCM_ARCH=gfx90a` scopes the lead arch (a follower passes its own arch:
  gfx1100 / gfx1101 / gfx1201; no source change needed -- the arch is a build var).
- `--no-build-isolation` so it builds against the installed ROCm torch (not a fresh
  CPU torch from PyPI). `setup.py` auto-defines `USE_ROCM` under `torch.version.hip`.
- Verify the ext loaded as an AMD build:
    python -c "import torch, nerfacc; from nerfacc.cuda import _backend; print(_backend._C); print('hip', torch.version.hip)"
    python -c "from nerfacc.cuda import _backend; print('cub_available', _backend._C.is_cub_available())"
- Code-object evidence (per arch): `roc-obj-ls $(python -c 'import nerfacc.csrc as c;
  print(c.__file__)') | grep gfx90a`.

## Test plan (real GPU, gfx90a; this feeds the validator)
nerfacc has a genuine GPU pytest suite under `tests/` that compares the `_C`
extension against pure-torch references (`_xxx` functions) on `device="cuda:0"`
(== HIP). Run with the GPU pinned and cwd outside /var/lib/jenkins/pytorch:

    cd projects/nerfacc/src
    HIP_VISIBLE_DEVICES=<ordinal> python -m pytest tests/ -v --tb=short

GPU tests (the validation gate):
- `tests/test_scan.py` -- inclusive/exclusive sum + prod, fwd AND `.backward()`,
  comparing flattened/packed_info/indices variants. Exercises scan.cu, scan_cub.cu
  (cub-or-fallback), utils_scan.cuh. PRIMARY scan correctness + autograd gate.
- `tests/test_grid.py` -- `ray_aabb_intersect` and `traverse_grids` vs `_ray_aabb_
  intersect`/`_query` references (grid.cu). Ray-marching correctness.
- `tests/test_pdf.py` -- `importance_sampling` + `searchsorted` vs references
  (pdf.cu, the philox RNG path). Watch the stratified vs deterministic contract.
- `tests/test_rendering.py` -- `render_visibility/weight_from_alpha` etc. (volrend,
  which calls is_cub_available + the scan ops). End-to-end volumetric rendering.
- `tests/test_camera.py` -- `opencv_lens_undistortion{,_fisheye}` vs `_` references
  (camera.cu), `torch.allclose` atol=1e-5.
- `tests/test_pack.py` -- `pack_info` (uses the scan path).
- `tests/test_vdb.py` -- needs optional `fvdb`; warns + returns if absent (treat as
  skip, NOT a failure; do not install fvdb just for this).

Pass criteria: every non-vdb test passes on gfx90a, with `_C` output matching the
pure-torch reference within the tests' own tolerances. A build that loads but
returns wrong values, or a scan whose `.backward()` mismatches, is a FAIL. Record
the cub-available state (true after the improvement, false if kept on fallback) and
which test_scan assertions cover which path. Run test_scan + test_pdf twice for
determinism (scan backward and stratified sampling).

Non-GPU regression set: nerfacc is GPU-centric (the CUDA ext IS the library); there
is no separate CPU test path that must not regress beyond the pure-torch reference
functions the tests already invoke. Do NOT add a GitHub Actions smoketest (policy);
the upstream `.github/workflows/{building,code_checks,doc,publish}.yml` stay as-is,
and Actions on the fork get disabled after `gh repo fork`.

## Open questions
1. Does test_pdf compare the stratified (RNG-driven) importance_sampling path
   bit-exactly, or only the deterministic path / with a tolerance? If bit-exact on
   RNG draws, confirm hiprand Philox4_32_10 reproduces curand's stream for the same
   seed (it should -- same algorithm); otherwise scope the stratified assertion.
2. cub keep-vs-improve: will hipcub `DeviceScan::ExclusiveScanByKey` /
   `InclusiveScanByKey` with the user-defined `Product` functor (and the `1.0f`
   init) build and run correctly on ROCm 7.2.1? If yes, enable the cub path
   (risk #1 improvement) and validate it; if not, keep the upstream graceful
   fallback and document. Resolved at porter/validator time, not blocking.
3. Confirm the gsplat-side payoff after this lands: rebuild gsplat with nerfacc
   importable and check that the 38 nerfacc-gated gsplat tests
   (test_rasterize_to_pixels_eval3d + rasterize_to_pixels reference) move from
   known-fail toward passing -- that is the cross-project reason nerfacc is in MOAT
   (tracked separately from this port's own validation).
