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
