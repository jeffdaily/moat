# pyslam Porting Plan

## Project

- Name: pyslam
- Upstream: https://github.com/luigifreda/pyslam
- Default branch: main

## Existing AMD support

**Decision: SKIP -- not portworthy, dependency composition problem**

pyslam is a Visual SLAM **framework** that orchestrates many third-party components. It has **no native CUDA code** -- all CUDA lives in bundled thirdparty submodules:

1. **lietorch** (princeton-vl/lietorch) -- Lie groups for PyTorch, CUDA kernels for SE3/SO3 operations
2. **MonoGS** (muskie82/MonoGS) -- Gaussian Splatting SLAM, depends on:
   - **diff-gaussian-rasterization** (graphdeco-inria) -- differentiable Gaussian rasterizer
   - **simple-knn** (graphdeco-inria) -- CUDA k-NN for spatial queries

The pyslam core code is pure Python/C++ (no `.cu` files outside thirdparty/). GPU features (Gaussian splatting, lietorch-based operations) are optional and come from these bundled submodules.

### Existing AMD efforts assessed

- **ROCm/gsplat**: AMD's official HIP port of gsplat (a Gaussian rasterization library). This is a DIFFERENT library than diff-gaussian-rasterization -- it is NOT a drop-in replacement. gsplat has a different API, different feature set (batch rasterization, N-D features, sparse gradients), and cannot simply substitute for diff-gaussian-rasterization without significant integration work.

- **EmmanuelMess/lietorch-ROCm**: A community fork with 1 star, uses `HSA_OVERRIDE_GFX_VERSION=10.3.0` (a crutch the PORTING_GUIDE forbids), no releases, no validation. NON-authoritative; would require from-scratch redo.

- **graphdeco-inria/diff-gaussian-rasterization**: No AMD/ROCm support upstream. Known HIPIFY issues (ROCm/HIPIFY#1630: vector_types.h conflicts). AMD recommends using gsplat instead (but that is not API-compatible).

- **pytorch/lietorch**: No AMD support upstream.

### Why not portworthy

1. **Dependency composition problem**: pyslam would require porting THREE independent upstream projects (lietorch, diff-gaussian-rasterization, simple-knn), none of which are within pyslam's control. Each is a separate porting effort with its own risks.

2. **AMD already has the underlying capability via gsplat**: The core Gaussian splatting functionality exists in AMD's official gsplat library. The gap is MonoGS-to-gsplat integration, not CUDA-to-HIP translation.

3. **Relationship to plvs**: The user asked about overlap with luigifreda/plvs (same author's C++ SLAM). plvs is a standalone C++ SLAM system with its own CUDA integration points. pyslam and plvs are SEPARATE projects by the same author -- pyslam is Python-first with PyTorch/neural features, plvs is C++-first with different volumetric integration. There is no shared CUDA code between them.

4. **Scope mismatch**: MOAT targets leaf CUDA libraries with direct GPU code. pyslam is a framework composition layer. The right approach is:
   - Port lietorch independently (Strategy B, pytorch extension)
   - Either port diff-gaussian-rasterization independently OR adapt MonoGS to use ROCm/gsplat
   - Port simple-knn independently
   - THEN pyslam becomes usable on AMD without any changes to pyslam itself

5. **Optional GPU features**: pyslam's core SLAM functionality (feature tracking, bundle adjustment, loop closure) runs on CPU with optional PyTorch acceleration. Only Gaussian splatting integration (a single VolumetricIntegratorType) requires the CUDA submodules. The framework is already usable on AMD for non-Gaussian-splatting volumetric integrators.

### Recommendation

Do NOT port pyslam as a MOAT target. Instead:

1. **If lietorch AMD support is needed**: Port princeton-vl/lietorch directly as a separate MOAT target (Strategy B pytorch extension). This would benefit RAFT-3D, DPVO, DROID-SLAM, and other lietorch consumers.

2. **If Gaussian splatting SLAM on AMD is needed**: Either:
   - Port MonoGS to use ROCm/gsplat instead of diff-gaussian-rasterization (significant API adaptation)
   - Or wait for community/AMD to provide a diff-gaussian-rasterization drop-in

3. **For pyslam specifically**: No pyslam fork is needed. Once upstream deps have AMD support, pyslam works unchanged.

## Disposition

Skip with reason: not-a-target (framework composition, no native CUDA code; underlying deps are separate porting targets)
