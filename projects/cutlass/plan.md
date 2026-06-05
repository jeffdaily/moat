# CUTLASS ROCm Port Plan

## Project

- **Name:** cutlass
- **Upstream:** https://github.com/nvidia/cutlass
- **Default branch:** main
- **Version assessed:** 4.5.2 (May 2026)

## Existing AMD support

**Status: ALREADY SUPPORTED via AMD-native alternatives -- recommend SKIP**

### AMD's official alternatives

1. **Composable Kernel (CK)** -- AMD's native CUTLASS equivalent
   - URL: https://rocm.docs.amd.com/projects/composable_kernel/
   - Repository (now in rocm-libraries): https://github.com/ROCm/composable_kernel (deprecated mirror)
   - Provides GEMM, convolution, attention, reduction, and fused operators
   - Uses MFMA (Matrix Fused Multiply Add) instructions on CDNA
   - Production quality, maintained by AMD

2. **rocWMMA** -- WMMA abstraction for AMD matrix cores
   - Header-only library for MFMA/WMMA operations
   - Analogous to nvcuda::wmma namespace

3. **FlyDSL** -- AMD's CuTe DSL equivalent (Python front-end)
   - URL: https://github.com/ROCm/FlyDSL
   - Inspired by CuTe layout algebra
   - Lowers through Fly MLIR dialect to ROCDL
   - Supports gfx942 (MI300), gfx950 (MI350), gfx1201 (Radeon AI PRO)

### Existing HIP port attempts

1. **ROCm/cutlass-internal** (private)
   - Description: "HIP port of Nvidia CUTLASS. Optimized for Instinct MI100. For internal evaluation."
   - Last updated: 2022-01-27 -- stale and inaccessible
   - NOT authoritative for current use

2. **cctry/cutlass_HIP** (community)
   - "Modified CUTLASS 3 (CUTE) for HIP" -- April 2024
   - 0 stars, appears abandoned
   - NON-authoritative community experiment

3. **gahan9/hip-cutlass** (fork)
   - Just a fork mirroring upstream (v4.1 release); no HIP work visible

### Decision: SKIP (already-supported via Composable Kernel)

CUTLASS is not a general CUDA library that can be mechanically hipified. It is an architecture-specific template library where the core value is:
1. NVIDIA PTX inline assembly (~3100 asm blocks)
2. NVIDIA tensor core ISA (mma.sync, ldmatrix, stmatrix, cp.async, tensormap -- ~2074 occurrences)
3. NVIDIA SM-versioned code paths (sm50/60/61/70/75/80/89/90/100/120)
4. CuTe layout algebra tied to NVIDIA warp semantics

A mechanical HIP translation is infeasible because:
- NVIDIA PTX has no HIP/AMDGPU equivalent -- these are ISA intrinsics, not runtime calls
- NVIDIA tensor core shapes (m16n8k8, m16n8k16, etc.) differ from AMD MFMA shapes
- Warp = 32 vs wavefront = 64 fundamentally changes register mapping
- NVIDIA TMA (Tensor Memory Accelerator) has no AMD analog

AMD's Composable Kernel IS the correct AMD answer to CUTLASS. It provides the same algorithmic value (tiled GEMM, attention, convolution) using AMD-native MFMA instructions and wavefront semantics. A CUTLASS HIP port would be an inferior, unvalidatable reimplementation of what CK already delivers.

For projects that depend on CUTLASS (xformers, fastertransformer, etc.), the path is:
1. Use Composable Kernel or hipBLASLt for GEMM/attention kernels
2. Use FlyDSL for CuTe-style kernel authoring
3. Use Triton for portable high-level kernel DSL

## Build classification

N/A -- not proceeding with port.

## Port strategy

N/A -- not proceeding with port.

## CUDA surface inventory (for reference)

| Category | Count/Details |
|----------|---------------|
| Inline assembly blocks | ~3,104 |
| PTX tensor instructions (mma.sync, ldmatrix, etc.) | ~2,074 |
| Header files | 783 |
| SM-versioned implementations | sm50/60/61/70/75/80/89/90/100/120 |
| NVIDIA-only dependencies | libcu++ (cuda/std/*), cuBLAS reference |

## Risk list

Port is not recommended. Risks of attempting anyway:
1. **Infeasible scope** -- rewriting 3000+ asm blocks to AMD ISA
2. **No 1:1 mapping** -- tensor core shapes differ
3. **Duplication** -- Composable Kernel already exists and is AMD-maintained
4. **Validation impossible** -- cannot verify parity with NVIDIA behavior on AMD hardware
5. **Maintenance burden** -- CUTLASS evolves rapidly (Hopper->Blackwell); tracking is unrealistic

## Recommendation

Mark nvidia/cutlass as `skip` with reason `already-supported` (previously marked `cant-port`). The disposition note should reference Composable Kernel and FlyDSL as the AMD equivalents.

Projects in MOAT that depend on CUTLASS should:
1. Identify which CUTLASS functionality they use (GEMM, attention, etc.)
2. Substitute with CK/rocWMMA/hipBLASLt for those specific operations
3. Use FlyDSL if CuTe-style kernel authoring is needed

## References

- Composable Kernel: https://rocm.docs.amd.com/projects/composable_kernel/
- FlyDSL: https://github.com/ROCm/FlyDSL
- rocWMMA: https://rocm.docs.amd.com/projects/rocWMMA/
- AMD matrix cores blog: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores/README.html
- CK vs CUTLASS interface request: https://github.com/ROCm/composable_kernel/issues/900
