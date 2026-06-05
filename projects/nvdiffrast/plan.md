# nvdiffrast Porting Plan

## Project

- **Name**: nvdiffrast
- **Upstream**: https://github.com/NVlabs/nvdiffrast
- **Default branch**: main
- **Upstream HEAD**: 253ac4fcea7de5f396371124af597e6cc957bfae
- **Description**: Modular primitives for high-performance differentiable rendering (PyTorch library)

## Existing AMD Support

**Assessment**: Two community forks exist with extensive HIP ports, but neither is authoritative:

1. **ATLAS-0321/nvdiffrast-rocm** (https://github.com/ATLAS-0321/nvdiffrast-rocm)
   - Updated: 2026-05-02
   - Single commit: "Add ROCm/HIP port: hipraster + AMD bug fixes (L-036/L-038/L-043/L-044)"
   - Targets: gfx1100 (RDNA3), ROCm 7.1.1, PyTorch 2.11.0+rocm7.1
   - Quality: Good -- detailed bug-fix documentation in commit message:
     - L-036: Fixed `getLaneMaskLe` UB for lane==31 (1u<<32)
     - L-038: CoarseRaster.inl tileSegCount bounds check
     - L-043: CoarseRaster.inl triHeader bounds checks
     - L-044: FineRaster.inl bounds checks
   - Contains extensive wave-sync learning documentation
   - Verified via TRELLIS.2 4B image-to-3D pipeline

2. **Lamothe/nvdiffrast_rocm** (https://github.com/Lamothe/nvdiffrast_rocm)
   - Updated: 2026-04-26
   - Single commit: "Updates for ROCm 7.2.1"
   - Similar hipraster structure but with fewer documented fixes
   - Appears to be an earlier iteration

**Decision**: These are non-authoritative community forks (personal repos, 0 stars/forks, unvalidated on CDNA/gfx90a). They target RDNA3 (wave32) only; gfx90a (wave64/CDNA) is untested. Per PORTING_GUIDE, do NOT adopt as base -- port from scratch our way, use as non-authoritative HINTS only.

The ATLAS fork's documented bug fixes (getLaneMaskLe UB, bounds checks) and wave-sync learnings are valuable reference material for the porter.

**Upstream merge policy**: README states "We do not currently accept outside code contributions in the form of pull requests." License (NVIDIA Source Code License, 1-Way Commercial) allows derivative works. A MOAT port delivers value via the jeffdaily fork without upstream PR.

## Build Classification

**Type**: PyTorch extension (Strategy B)

**Evidence** (`setup.py:29-30`):
```python
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
```

Uses `CUDAExtension` with sources including `.cu` kernels. No standalone CMake.

## Port Strategy

**Strategy B**: PyTorch extension, torch hipify at build time.

**Rationale**: Standard torch CUDAExtension build. However, the cudaraster/ module contains extensive PTX inline assembly that cannot be hipified mechanically -- requires a parallel hipraster/ implementation.

**Approach**:
1. Build against ROCm PyTorch; torch.utils.hipify handles most `.cu` files
2. Port the cudaraster/ rasterizer to a parallel hipraster/ module with:
   - PTX assembly replaced with portable C++ or HIP builtins
   - Wave intrinsics mapped: `__ballot_sync` -> `__ballot`, etc.
   - bounds checks for uninitialized memory (AMD does not zero-init like CUDA)
3. setup.py conditionally selects hipraster/ or cudaraster/ based on `torch.version.hip`
4. Keep NVIDIA path byte-identical (dual backend)

## CUDA Surface Inventory

| Category | Files | Risk |
|----------|-------|------|
| `.cu` kernels | 5 | Low -- hipify handles |
| `__global__/__device__` | ~252 occurrences | Low |
| `__shfl/__ballot/__all/__any` | ~45 occurrences (BinRaster.inl, CoarseRaster.inl, FineRaster.inl, antialias.cu, interpolate.cu) | Medium -- wave64 |
| PTX inline assembly | ~40 asm statements in Util.inl | High -- requires rewrite |
| Textures/surfaces | None | -- |
| cuBLAS/cuFFT/cuRAND/cuSPARSE | None | -- |
| Thrust/CUB | None | -- |
| Streams/events | Standard CUDA streams | Low |

### PTX Assembly in cudaraster/impl/Util.inl

Must be replaced with portable equivalents:

| PTX | HIP Equivalent |
|-----|----------------|
| `mov.u32 %lanemask_lt` | `__lanemask_lt()` |
| `mov.u32 %lanemask_le` | `__lanemask_lt() \| (1u << __lane_id())` (avoids UB) |
| `mov.u32 %lanemask_gt/ge` | Derived from above |
| `bfind.u32` | `31 - __clz(v)` |
| `cvt.rni.sat.s32.f32` | `(S32)rintf(fminf(fmaxf(a,-2147483648.f),2147483647.f))` |
| `vadd/vsub` 16-bit half-word ops | Portable shift/mask arithmetic |
| `vmad` byte ops | Portable shift/mask arithmetic |
| `prmt` byte shuffle | Portable shift/mask |

### Warp Intrinsics

All uses are wave32-based (warp of 32 lanes). On gfx90a (wave64):
- `__ballot_sync(~0u, pred)` -> `__ballot(pred)` returns 64-bit mask
- Code uses 32-bit masks (`U32`) extensively -- needs audit for wave64 correctness
- Lane mask operations assume 32 lanes

**Critical**: The rasterizer block sizes (32 threads) align with CUDA warp size. Wave64 on gfx90a may require block size adjustments or logical-warp handling.

## Risk List

1. **PTX inline assembly** (HIGH): ~40 PTX statements in Util.inl must be rewritten. Community forks provide working implementations.

2. **Wave64 correctness** (HIGH): Code assumes 32-lane warps. On gfx90a:
   - Lane masks are 32-bit but wave is 64 lanes
   - `getLaneMaskLe` fix documented by ATLAS applies
   - Block sizes (32 threads) may need wave64 handling

3. **Uninitialized memory** (MEDIUM): CUDA runtime zeros memory; HIP does not always. Bounds checks needed in CoarseRaster/FineRaster (per ATLAS learnings).

4. **Wave sync semantics** (MEDIUM): `__syncwarp` -> `__builtin_amdgcn_wave_barrier()` needed (not just `__threadfence_block`). ATLAS fork documents 3x MODE1 resets with wrong mapping.

5. **gfx90a vs gfx1100 divergence** (MEDIUM): Community ports tested on gfx1100 (wave32) only. gfx90a (wave64) requires additional validation.

6. **No formal test suite** (LOW): Only sample scripts. Validation via visual rendering output comparison.

## File-by-File Change List

### setup.py
- Detect HIP build via `torch.version.hip`
- Conditional source list: hipraster/ on HIP, cudaraster/ on CUDA
- HIP compiler flags: `-DNVDR_TORCH`, `-D__HIP_PLATFORM_AMD__`

### csrc/common/hipraster/ (NEW)
Mirror of cudaraster/ with HIP rewrites:
- `impl/Util.inl` -- PTX -> portable C++/HIP builtins
- `impl/BinRaster.inl` -- wave intrinsics
- `impl/CoarseRaster.inl` -- wave intrinsics, bounds checks
- `impl/FineRaster.inl` -- wave intrinsics, bounds checks
- `impl/RasterImpl_kernel.hip` -- wave sync defines
- Other files: minimal changes from hipify

### csrc/common/common_hip.h (NEW)
HIP compatibility header:
- `__ballot_sync` -> `__ballot`
- `__all_sync` / `__any_sync` -> `__all` / `__any`
- `__syncwarp` -> `__builtin_amdgcn_wave_barrier()`
- `__frcp_rz` -> `1.0f / x`

### csrc/common/antialias.cu, interpolate.cu, rasterize.cu, texture_kernel.cu
- torch hipify handles
- May need common_hip.h include for wave intrinsics

### csrc/torch/*.cpp
- No changes expected (hipify handles)

## Build Commands

### Configure + Build (gfx90a)
```bash
# Activate ROCm PyTorch environment
source /opt/rocm/venv/bin/activate  # or conda activate rocm-pytorch

# Build with HIP
pip install --no-build-isolation -v -e /var/lib/jenkins/moat/projects/nvdiffrast/src
```

### Verify Build
```bash
python -c "import nvdiffrast.torch as dr; print('nvdiffrast loaded')"
```

## Test Plan

### GPU Validation (no formal test suite)

1. **Basic import test**:
```bash
python -c "import nvdiffrast.torch as dr; ctx = dr.RasterizeCudaContext(); print('RasterizeCudaContext OK')"
```

2. **Sample scripts** (samples/torch/):
```bash
cd /var/lib/jenkins/moat/projects/nvdiffrast/src/samples/torch
python triangle.py --outdir /tmp/nvdiffrast_test
python cube.py --outdir /tmp/nvdiffrast_test
python earth.py --outdir /tmp/nvdiffrast_test
```

3. **Correctness validation**:
- Compare rendered output images against CUDA reference (if available)
- Visual inspection for obvious rendering errors
- Run each sample 3x for determinism check

4. **Operators to exercise**:
- `dr.rasterize()` -- the hipraster path
- `dr.interpolate()` -- simple, likely works with torch hipify
- `dr.texture()` -- texture sampling
- `dr.antialias()` -- edge antialiasing, uses `__ballot_sync`

### Non-GPU Tests
None present in upstream.

## Open Questions

1. **Wave64 block sizing**: The rasterizer uses 32-thread blocks. On gfx90a (wave64), does this cause inefficiency or correctness issues? May need runtime warpSize query.

2. **Multi-arch fat binary**: Can a single build support both gfx90a and gfx1100? Wave intrinsics differ.

3. **Performance**: Community ports report successful rendering but no perf comparison. May need AMD-native tuning for production use.

4. **ATLAS vs Lamothe fork**: ATLAS has more detailed fixes. Use ATLAS as primary reference for the port.
