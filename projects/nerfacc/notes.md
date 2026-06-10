# nerfacc notes

## Why this project is in MOAT

nerfstudio-project/nerfacc (1457 stars) is a PyTorch CUDA extension for efficient
volumetric ray-marching / sampling in NeRFs. It was a discovery blind spot (a
.cu-based CUDA library in a Python-dominant repo, no "cuda" in name/topics).

Added because gsplat's test suite uses nerfacc as an optional pure-torch
reference (the eval3d accumulator and several rasterize_to_pixels comparisons).
nerfacc ships a CUDA-only wheel whose `_C` extension is unavailable on ROCm, so
`import nerfacc` fails and 38 gsplat reference tests are a documented known-fail
on every platform. Porting nerfacc to ROCm closes that gap and stands on its own
as a widely used library.

## Classification (preliminary, for the planner)

- ext_type: torch-extension (Strategy B -- torch hipifies the .cu/.cuh at build
  time), same family as gsplat and FastGeodis.
- Sibling of gsplat in the nerfstudio ecosystem; no MOAT-internal deps.
- Upstream default branch: master.

## Planner findings (2026-06-08, lead linux-gfx90a)

- Existing AMD support: NONE (no authoritative or community ROCm/HIP port).
  120 forks, none under ROCm/AMD/GPUOpen or with rocm/hip/amd in the name;
  `ROCm/nerfacc` and `AMD/nerfacc` both 404; web search negative; README/docs grep
  only false positives (scene names Hotdog/Ship match `hip`). Decision: from-scratch
  Strategy B port, correctness-first. Full rationale in plan.md.
- setup.py ALREADY branches on `torch.version.hip` (defines USE_ROCM, undefs
  __HIP_NO_HALF_CONVERSIONS__, drops --expt-relaxed-constexpr; strips *hip* files on
  rebuild). This is build-scaffold awareness only; never validated on AMD, no
  in-source USE_ROCM guards. Treat as a hint, not a completed port.
- CUDA surface is small + clean: 5 .cu (camera/grid/pdf/scan/scan_cub). NO warp
  intrinsics, NO cooperative groups, NO textures/surfaces/layered arrays, NO
  __constant__, NO device atomics, NO cuda::std, NO half2/PTX. Libs: cub DeviceScan
  ByKey (host), thrust::make_reverse_iterator (host), PyTorch philox RNG (pdf.cu
  only). All hipify-mapped.
- KEY caveat -- cub disables itself on ROCm: utils.cub.cuh gates
  CUB_SUPPORTS_SCAN_BY_KEY() on CUB_VERSION>=101500, but hipify maps
  <cub/version.cuh> -> <hipcub/hipcub_version.hpp> which defines HIPCUB_VERSION
  (400200 here) NOT CUB_VERSION, so CUB_VERSION stays 0 -> is_cub_available()==false
  and the *_cub body is #if-compiled out. The library STILL works (Python falls back
  to the hand-rolled packed scan), but the cub path is untested unless we make
  CUB_SUPPORTS_SCAN_BY_KEY() true on ROCm via HIPCUB_VERSION (a small optional
  improvement, USE_ROCM-guarded). Keep-vs-improve decided at porter/validate time.
- Wave64 safe: the only shared collective (Blelloch scan, utils_scan.cuh,
  <<<,dim3(16,32)>>>) has an explicit __syncthreads() after every sweep step; no
  implicit wave-lockstep dependency, shared buffer sized by template constants.
- Real GPU pytest suite present: tests/test_{scan,grid,pdf,rendering,camera,pack}.py
  compare _C vs pure-torch references on device="cuda:0" (incl. autograd .backward()
  in test_scan). test_vdb needs optional fvdb (skips). This is the validation gate.
- Env (same as gsplat/FastGeodis): py_3.12, torch 2.13.0a0+gitb5e90ff, hip
  7.2.53211, MI250X gfx90a, ROCM_HOME=/opt/rocm, CUDA_HOME empty. Build:
  HIP_VISIBLE_DEVICES=<ord> PYTORCH_ROCM_ARCH=gfx90a pip install -e . --no-build-isolation.
  Run pytest with cwd OUTSIDE /var/lib/jenkins/pytorch (it shadows installed torch).

## Porter findings (2026-06-09, lead linux-gfx90a) -- state: ported

Fork: https://github.com/jeffdaily/nerfacc, branch moat-port, HEAD
2298cb55073791bdcfaff496c273c0ba5758a08d. Actions disabled on the fork.
Built + GPU-validated on MI250X gfx90a. 3 source files edited (36 insertions),
all USE_ROCM-guarded so the CUDA build is byte-identical. setup.py needed NO
edit -- its existing torch.version.hip branch handled USE_ROCM + the half-conv
undef. No GLM, no warp intrinsics, no textures, no fast-math; the gsplat fault
classes did not apply.

Three hipify gaps fixed (NONE were anticipated as build-blockers in plan.md;
plan expected a near-clean Strategy B build):

1. utils_math.cuh = NVIDIA helper_math.h. On HIP, float2/3/4 are HIP_vector_type,
   which ALREADY provides component-wise operator* and operator/ (a non-template
   `friend` operator* at amd_hip_vector_types.h:397 + free templates), so
   helper_math's same-type `operator*(float2,float2)` / `operator/` are AMBIGUOUS
   (not redundant-but-ignored -- a hard "use of overloaded operator is ambiguous"
   error). Guard out the SIX same-type float2/3/4 operator*/operator/ on ROCm
   (HIP's are component-wise identical). Note: only the same-type-vector `*` and
   `/` conflict -- `+`/`-` don't (HIP provides them only as free templates, which
   lose to helper_math's non-template exact match) and the vector-scalar forms
   don't. ALSO: scalar `lerp(float,float,float)` conflicts with C++20 std::lerp
   ("declaration conflicts with target of using declaration already in scope") --
   guard it out too; std::lerp(a,b,t)=a+t*(b-a) is identical and the vector lerps
   don't call the scalar one. GENERAL: any port that vendors NVIDIA helper_math.h
   hits exactly this same-type-vector */÷ + scalar-lerp ambiguity on HIP.

2. cub scan-by-key was silently disabled (plan risk #1, but the CUB_VERSION guard
   detail differed). hipify rewrites the `#if defined(CUDA_VERSION) && CUDA_VERSION
   >= 11000` guard around `<cub/version.cuh>` to TORCH_HIP_VERSION (a small value,
   NOT >= 11000), so <cub/version.cuh> is never included and CUB_VERSION stays 0 ->
   CUB_SUPPORTS_SCAN_BY_KEY()==0. So HIPCUB_VERSION is ALSO never defined (the plan
   assumed it would be) -- gating on HIPCUB_VERSION does NOT work. Fix: on USE_ROCM
   just `#define CUB_SUPPORTS_SCAN_BY_KEY() 1` (hipcub DeviceScan ByKey is present
   across the supported ROCm range). This enables + validates the hipcub by-key
   path instead of the slower hand-rolled fallback.

3. With (2) enabled, scan_cub.cu fails to compile: hipify maps cub::DeviceScan ->
   hipcub::DeviceScan but leaves `cub::Equality` UNTOUCHED (no mapping-table entry)
   -> "use of undeclared identifier 'cub'". Fix: `namespace cub = hipcub;` under
   USE_ROCM inside the cub block (after the hipified `#include <hipcub/hipcub.hpp>`).
   hipcub::Equality exists (it is the DeviceScan default EqualityOpT).

Copyright/attribution added (parallel AMD line + Jeff Daily author) to the 3
substantially-edited files: utils_math.cuh, utils.cub.cuh, scan_cub.cu.

GENERATED-FILE HYGIENE: building creates an untracked nerfacc/hip/ hipify mirror
(+ build/, nerfacc/csrc.so). NONE belong in the commit -- setup.py:36 strips *hip*
from the source glob and regenerates the mirror each build. Commit ONLY the 3
nerfacc/cuda/csrc edits. After editing a .cu/.cuh you MUST `rm -rf nerfacc/hip`
before rebuilding, else hipify skips it ("[skipped, already hipified]") and the
stale mirror is recompiled. `python setup.py build_ext --inplace` writes the .so
to build/lib.../ ; the editable install loads nerfacc/csrc.so -- re-run
`pip install -e . --no-build-isolation` (or cp the built .so) to refresh it.

### Build + test commands (lead gfx90a, GPU 0 free; GPU 1 had a sibling, GPU 2 busy)
    source /opt/conda/etc/profile.d/conda.sh && conda activate py_3.12
    cd projects/nerfacc/src && rm -rf nerfacc/hip build
    HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx90a MAX_JOBS=16 \
        pip install -e . --no-build-isolation
    # verify AMD build + cub path active:
    cd /tmp && HIP_VISIBLE_DEVICES=0 python -c \
      "import torch; from nerfacc.cuda import is_cub_available; \
       print(torch.version.hip, is_cub_available())"   # 7.2.53211 True
    # GPU test suite (cwd /tmp, copy tests in):
    cp -r projects/nerfacc/src/tests /tmp/nerfacc_tests
    HIP_VISIBLE_DEVICES=0 python -m pytest /tmp/nerfacc_tests/ -v -p no:cacheprovider

### Result: 23/23 pass on gfx90a (test_vdb no-ops without optional fVDB).
cub_available True -- the hipcub DeviceScan ByKey path (with the custom Product
functor) is compiled and exercised by test_scan inclusive/exclusive sum+prod incl.
.backward(); the prior "indices without CUB available is slow" warnings are gone.
test_scan + test_pdf run twice, deterministic. gfx90a code objects confirmed in
csrc.so via roc-obj-ls. test_pdf importance_sampling (philox RNG) passes within
tolerance -- no bit-exact-RNG issue (plan open question 1 resolved: not bit-exact).

### Follower notes
RDNA followers (gfx1100/gfx1101/gfx1201): the fixes are wave-agnostic (operator
guards + cub enablement + namespace alias touch no lane math), so re-validate the
build + same pytest suite per policy; no expected source change. cub-by-key uses
hipcub's own portable wavefront handling. The Blelloch scan (utils_scan.cuh) is
the only shared-mem collective and has an explicit __syncthreads() after every
sweep -- wave64/wave32 safe (plan risk #2).

## Review 2026-06-09 (reviewer, lead linux-gfx90a) -- state: review-passed

Reviewed moat-port @ 2298cb5 (diff vs upstream master 57ccfa1) with /pr-review.
3 files, 36 insertions, all USE_ROCM-guarded. No problems found; CUDA build is
byte-identical (every edit sits behind `#if defined(USE_ROCM)` or
`#if !defined(USE_ROCM)`, so the CUDA preprocessor sees zero change).

Verified, not assumed:
- helper_math ambiguity fix: the 6 same-type binary float2/3/4 operator* /
  operator/ guarded out are exactly the 6 present (utils_math.cuh:753,824,901,
  988,1013,1040); the `*=`/scalar/`+`/`-` forms are correctly left unguarded.
  Scalar lerp guarded (1159); the vector lerps don't call it (use +/-/scalar*),
  so removing it on ROCm is safe; std::lerp(float,float,float) covers the scalar.
- CUB_SUPPORTS_SCAN_BY_KEY()==1 on ROCm (utils.cub.cuh:18-24): the CUDA branch is
  untouched, only an additive USE_ROCM arm.
- namespace cub = hipcub (scan_cub.cu:14-19) placed AFTER the hipified
  <hipcub/hipcub.hpp> include and inside CUB_SUPPORTS_SCAN_BY_KEY(); the only
  cub:: spellings are cub::DeviceScan and cub::Equality, both covered.
- Real ROCm build evidence: csrc.so carries gfx90a code objects (roc-obj-ls).
- Fault classes N/A and correctly untouched: no warpSize/32, no warp intrinsics,
  no textures/rule-of-five, no OOB neighbor reads, no pitch, no library mis-swap.
- Commit hygiene clean: [ROCm] title 47 chars, no noreply/co-authored/ghstack,
  no MOAT jargon, ASCII-only, Claude named, Test Plan present, only public
  jeff.daily@amd.com reference. Copyright: parallel AMD line + Jeff Daily author
  on all 3 edited files in house style.
- Working tree: only untracked nerfacc/hip/ + build/ artifacts (correctly NOT
  committed; setup.py:36 strips *hip* and regenerates).

Real-GPU validation is the validator's next gate (porter ran 23/23 locally).
Verdict: Approve -> review-passed.

## Validation 2026-06-09 (validator, lead linux-gfx90a) -- state: completed

Platform: linux-gfx90a, GPU: AMD Instinct MI250X (gfx90a), ROCm 7.2.1, torch 2.13.0a0+gitb5e90ff, hip 7.2.53211.
Fork: jeffdaily/nerfacc @ moat-port, HEAD 2298cb55073791bdcfaff496c273c0ba5758a08d.

Build: started from a clean tree (`rm -rf nerfacc/hip build`).

    source /opt/conda/etc/profile.d/conda.sh && conda activate py_3.12
    cd projects/nerfacc/src && rm -rf nerfacc/hip build
    HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx90a MAX_JOBS=16 \
        pip install -e . --no-build-isolation
    # ~264s wall, exit 0

Verified post-build:
- `is_cub_available()` returns True -- hipcub DeviceScan ByKey path is compiled and active.
- roc-obj-ls confirms 5 gfx90a code objects in csrc.so (one per .cu TU: camera, grid, pdf, scan, scan_cub).

GPU test suite (cwd /tmp, tests copied from projects/nerfacc/src/tests/):

    HIP_VISIBLE_DEVICES=0 python -m pytest /tmp/nerfacc_tests/ -v -p no:cacheprovider

Result: 23/23 PASS (~5.7s), 5 warnings (3x torch.jit.script deprecation in test_camera; 2x "fVDB not installed" in test_vdb -- tests return early, still reported PASSED).

All GPU tests pass:
- test_scan (4/4): inclusive/exclusive sum+prod, all exercising hipcub ByKey path + .backward(). Run twice, deterministic.
- test_pdf (3/3): searchsorted, importance_sampling (philox RNG), pdf_loss. Run twice, deterministic.
- test_rendering (6/6): render_visibility, weight_from_alpha, weight_from_density, accumulate_along_rays, grads, rendering.
- test_grid (5/5): ray_aabb_intersect, traverse_grids, traverse_grids_test_mode, traverse_grids_with_near_far_planes, sampling_with_min_max_distances, mark_invisible_cells. (6 tests total)
- test_pack (1/1): pack_info.
- test_camera (1/1): opencv_lens_undistortion.
- test_vdb (2/2): skip-early on missing fVDB, PASSED.

Non-GPU regressions: none -- nerfacc is GPU-centric; the pure-torch reference functions the tests invoke are unchanged.

Fork working tree clean: only untracked nerfacc/hip/ + build/ (never committed; setup.py:36 strips *hip* and regenerates).

validated_sha: 2298cb55073791bdcfaff496c273c0ba5758a08d

## Validation 2026-06-09 (validator, linux-gfx1100) -- state: completed

Platform: linux-gfx1100, GPU: AMD Radeon Pro W7800 48GB (gfx1100), ROCm 7.2.1, torch 2.13.0a0+gitb5e90ff, hip 7.2.53211.
Fork: jeffdaily/nerfacc @ moat-port, HEAD 2298cb55073791bdcfaff496c273c0ba5758a08d.

Build: clean tree (rm -rf nerfacc/hip build), HIP_VISIBLE_DEVICES=0 pinned.

    source /opt/conda/etc/profile.d/conda.sh && conda activate py_3.12
    cd projects/nerfacc/src && rm -rf nerfacc/hip build
    HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx1100 MAX_JOBS=16 \
        pip install -e . --no-build-isolation
    # exit 0; editable install built csrc.so for gfx1100

Verified post-build:
- `is_cub_available()` returns True -- hipcub DeviceScan ByKey path active on gfx1100.
- roc-obj-ls confirms 5 gfx1100 code objects in csrc.so (camera, grid, pdf, scan, scan_cub).

GPU test suite (cwd /tmp, tests copied from projects/nerfacc/src/tests/):

    HIP_VISIBLE_DEVICES=0 python -m pytest /tmp/nerfacc_tests_gfx1100/ -v -p no:cacheprovider

Result: 23/23 PASS (~12.3s), 6 warnings (1x NumPy 1.x/2.x warning; 3x torch.jit.script deprecation; 2x "fVDB not installed" in test_vdb -- tests return early, still PASSED).

All GPU tests pass:
- test_scan (4/4): inclusive/exclusive sum+prod, all exercising hipcub ByKey path + .backward().
- test_pdf (3/3): searchsorted, importance_sampling (philox RNG), pdf_loss.
- test_rendering (6/6): render_visibility, weight_from_alpha, weight_from_density, accumulate_along_rays, grads, rendering.
- test_grid (6/6): ray_aabb_intersect, traverse_grids, traverse_grids_test_mode, traverse_grids_with_near_far_planes, sampling_with_min_max_distances, mark_invisible_cells.
- test_pack (1/1): pack_info.
- test_camera (1/1): opencv_lens_undistortion.
- test_vdb (2/2): skip-early on missing fVDB, PASSED.

Follower notes (gfx1101/gfx1201): same build + test recipe with the arch env var changed. No source changes needed; fixes are wave-agnostic.

Non-GPU regressions: none.

Fork working tree clean: only untracked nerfacc/hip/ (never committed).

validated_sha: 2298cb55073791bdcfaff496c273c0ba5758a08d

## Validation 2026-06-09 (validator, windows-gfx1201) -- state: completed

Platform: AMD Radeon RX 9070 XT (gfx1201, RDNA4, wave32), Windows 11 Pro for Workstations.
ROCm via TheRock pip wheels: rocm-sdk 7.14.0a20260604 (hip 7.14.60850-d34cbb64),
torch 2.9.1+rocm7.14.0a20260604. Python 3.12.
Fork tip validated: 3e88a726661743d5b00d420726081f69964c5e38 (new commit on top of 2298cb5).

### Windows+HIP build fixes (new commit 3e88a72 on top of 2298cb5)

Two Windows-specific build issues encountered and fixed in setup.py (both gated
`sys.platform == "win32" and torch.version.hip`, so Linux/CUDA builds are byte-identical):

1. ninja required on Windows+ROCm: the non-ninja MSVC compile path does not escape
   spaces in -I paths before handing them to hipcc; include dirs under "Program Files
   (x86)" are split on the space when hipcc forwards them to clang as separate
   arguments. The ninja path escapes spaces (replaces with `\`). Fix: `use_ninja=True`
   on Windows+ROCm only (was hardcoded `False`).

2. Binding .cpp must compile via hipcc: torch routes .cpp -> MSVC cl.exe. MSVC cannot
   parse the GCC `__attribute__` syntax in HIP runtime headers (amd_hip_vector_types.h)
   pulled in through torch/extension.h. Fix: copy the binding nerfacc.cpp to a
   nerfacc_winhip.cu shim at build time; the `.cu` extension makes torch's
   `_is_cuda_file()` return True, routing it to hipcc (amdclang). The shim is
   transient (gitignored, regenerated each build).

Both fixes are the standard pattern for Windows+ROCm torch extensions (also used in
gsplat). Also added nerfacc/hip/ and *_winhip.cu to .gitignore.

### Build command (gfx1201, Windows)

    VENV="B:/develop/TheRock/external-builds/pytorch/.venv"
    ROCM_HOME="$VENV/Lib/site-packages/_rocm_sdk_devel"
    MSVC_BIN="/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64"
    export PATH="$MSVC_BIN:$PATH"
    export ROCM_HOME HIP_DEVICE_LIB_PATH="$ROCM_HOME/lib/llvm/amdgcn/bitcode"
    export DISTUTILS_USE_SDK=1 HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx1201 MAX_JOBS=64

    cd projects/nerfacc/src && rm -rf nerfacc/hip build
    pip install -e . --no-build-isolation
    # exit 0, ~41s wall

Verified post-build:
- `is_cub_available()` returns True -- hipcub DeviceScan ByKey path active on gfx1201.
- import nerfacc OK; torch.cuda.get_device_name(0) = "AMD Radeon RX 9070 XT".

### GPU test suite

    cp -r projects/nerfacc/src/tests /tmp/nerfacc_tests_gfx1201
    HIP_VISIBLE_DEVICES=0 python -m pytest /tmp/nerfacc_tests_gfx1201/ -v -p no:cacheprovider

Result: 23/23 PASS (~6.85s), 2 warnings (fVDB not installed in test_vdb -- tests return early, PASSED).

All GPU tests pass:
- test_scan (4/4): inclusive/exclusive sum+prod, hipcub ByKey path + .backward().
- test_pdf (3/3): searchsorted, importance_sampling (philox RNG), pdf_loss.
- test_rendering (6/6): render_visibility, weight_from_alpha, weight_from_density, accumulate_along_rays, grads, rendering.
- test_grid (6/6): ray_aabb_intersect, traverse_grids, traverse_grids_test_mode, traverse_grids_with_near_far_planes, sampling_with_min_max_distances, mark_invisible_cells.
- test_pack (1/1): pack_info.
- test_camera (1/1): opencv_lens_undistortion.
- test_vdb (2/2): skip-early on missing fVDB, PASSED.

Non-GPU regressions: none -- nerfacc is GPU-centric; pure-torch reference functions unchanged.

Fork working tree clean: .gitignore updated, only untracked nerfacc/hip/ and
nerfacc/cuda/csrc/nerfacc_winhip.cu (both gitignored build artifacts, never committed).

Note: linux-gfx90a and linux-gfx1100 flipped to revalidate because advance-head
classified setup.py as mixed. The changes are gated `sys.platform=="win32" and
torch.version.hip`, so the Linux builds are byte-identical. Linux validators can
use codeobj_diff.py to carry forward without re-running GPU tests.

validated_sha: 3e88a726661743d5b00d420726081f69964c5e38

## Revalidation 2026-06-10 (validator, linux-gfx90a) -- state: completed (carry-forward)

Platform: linux-gfx90a, GPU: AMD Instinct MI250X (gfx90a), ROCm 7.2.1.
Delta: 2298cb55073791bdcfaff496c273c0ba5758a08d -> 3e88a726661743d5b00d420726081f69964c5e38 (1 commit).

Delta classification: setup.py + .gitignore only. All setup.py changes are gated
`sys.platform == "win32" and torch.version.hip` (or `bool(torch.version.hip)`);
on Linux, sys.platform == "linux", so these branches are dead code. .gitignore
is metadata only. The Linux build is byte-identical to the previously validated build.

Binary-equivalence check: built both SHAs for gfx90a into separate dirs, then ran
utils/codeobj_diff.py:

    # Build old SHA (2298cb5)
    cd projects/nerfacc/src && git checkout 2298cb55073791bdcfaff496c273c0ba5758a08d
    rm -rf nerfacc/hip build
    HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx90a MAX_JOBS=16 pip install -e . --no-build-isolation
    cp nerfacc/csrc.so agent_space/nerfacc_build_old/csrc.so

    # Build new SHA (3e88a72)
    git checkout moat-port
    rm -rf nerfacc/hip build
    HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx90a MAX_JOBS=16 pip install -e . --no-build-isolation
    cp nerfacc/csrc.so agent_space/nerfacc_build_new/csrc.so

    python3 utils/codeobj_diff.py agent_space/nerfacc_build_old agent_space/nerfacc_build_new
    # verdict=identical
    # csrc.so: identical (exported symbols + device ISA identical (370 exports))

Conclusion: binary-equivalent on gfx90a. Carried forward via moatlib carry-forward
(method: binary-equiv). No GPU re-run needed.

validated_sha: 3e88a726661743d5b00d420726081f69964c5e38
