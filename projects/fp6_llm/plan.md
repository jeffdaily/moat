# fp6_llm Port Plan

## Project

- **Name**: fp6_llm (Quant-LLM)
- **Upstream**: https://github.com/usyd-fsalab/fp6_llm
- **Default branch**: main
- **Description**: FP6-LLM provides efficient GPU support for LLM inference with 6-bit quantization (FP6) and other x-bit formats (FP5). It implements TC-FPx, a Tensor Core-based scheme for mixed-precision GEMM with quantized weights.

## Existing AMD support

**Finding**: No existing AMD/ROCm/HIP support found.

Searches performed:
- `grep -rniE 'amd|rocm|hip|gfx[0-9]' README* docs/` in upstream: no matches
- WebSearch for "fp6_llm ROCm", "fp6_llm AMD GPU", "fp6_llm HIP": no results
- GitHub forks API scan: 22 forks examined, none from ROCm/AMD/GPUOpen orgs or with rocm/hip/amd in name
- gh search repos for "fp6 llm rocm", "TC-FPx amd": no results

The upstream README explicitly states: "Currently, FP6-LLM is only tested and verified on A100 GPUs" and targets "Tensor Core GPUs like NVIDIA H100 and GH200." No AMD/ROCm branch or PR exists.

**Decision**: Proceed with a from-scratch port. However, this project is a REWRITE-REQUIRED case, not a mechanical HIP translation. See Risk list below.

## Build classification

**Type**: PyTorch extension (torch-extension)

**Evidence**: `setup.py` lines 2, 32-41:
```python
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
...
ext_modules=[
    CUDAExtension(
        name="fp6_llm_cuda",
        sources=["fp6_llm/csrc/pybind.cpp", "fp6_llm/csrc/fp6_linear.cu"],
        extra_compile_args=extra_compile_args,
    ),
],
```

The project also has standalone Makefiles for building a C++ shared library (`fp6_llm/Makefile`) and C++ tests (`tests/cpp/Makefile`), but the primary interface is the PyTorch extension.

## Port strategy

**CRITICAL: This project CANNOT be mechanically ported to ROCm/HIP.**

FP6-LLM is fundamentally built around NVIDIA Tensor Cores and uses:
1. **PTX assembly for Tensor Core MMA operations** (`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`)
2. **PTX assembly for async copy** (`cp.async.cg.shared.global`, `cp.async.commit_group`, `cp.async.wait_group`)
3. **PTX assembly for ldmatrix** (`ldmatrix.sync.aligned.x2.m8n8.shared.b16`, `ldmatrix.sync.aligned.x4.m8n8.shared.b16`)
4. **PTX `__cvta_generic_to_shared`** for address space conversion

These are NVIDIA-specific instructions with NO HIP/AMD equivalents. There is no `hipify` or compat-header solution for PTX assembly.

**Recommended approach**: AMD-native rewrite using one of:
1. **rocWMMA** (Wavefront Matrix Multiply Accumulate) - AMD's equivalent to Tensor Core MMA operations
2. **Composable Kernel (CK)** - AMD's high-performance kernel library for GEMM and similar ops
3. **MFMA intrinsics** - Direct Matrix Fused Multiply-Add intrinsics for CDNA GPUs

The algorithmic innovations (FP6 bit-packing, ahead-of-time weight prepacking, SIMT-efficient dequantization) are portable, but the compute kernel (`QUANT_GEMM_Kernel`) requires a ground-up rewrite to use AMD matrix instructions.

**Disposition recommendation**: Mark as `blocked` with reason `rewrite-required` unless there is explicit appetite for an AMD-native kernel rewrite (a multi-week effort, not a mechanical port).

## CUDA surface inventory

### PTX Assembly (NOT portable)

| File | Line | PTX Instruction | Purpose |
|------|------|-----------------|---------|
| `ptx_mma.cuh:43-45` | `ldmatrix.sync.aligned.x2.m8n8.shared.b16` | Load matrix from shared memory for MMA |
| `ptx_mma.cuh:51-53` | `ldmatrix.sync.aligned.x4.m8n8.shared.b16` | Load matrix from shared memory for MMA |
| `ptx_mma.cuh:62-70` | `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` | Tensor Core matrix multiply-accumulate |
| `ptx_cp.async.cuh:27-34` | `cp.async.cg.shared.global` | Async copy from global to shared memory |
| `ptx_cp.async.cuh:40` | `cp.async.commit_group` | Commit async copy group |
| `ptx_cp.async.cuh:46` | `cp.async.wait_group N` | Wait for async copy group |
| `ptx_cp.async.cuh:55` | `cp.async.wait_all` | Wait for all async copies |
| `ptx_mma.cuh:41` | `__cvta_generic_to_shared` | Address space conversion |
| `ptx_cp.async.cuh:26` | `__cvta_generic_to_shared` | Address space conversion |

### Warp Intrinsics

| File | Line | Intrinsic | Notes |
|------|------|-----------|-------|
| `utils_parallel_dequant.cuh:111` | `__shfl_sync(0xffffffff, tmpReg, i, 4)` | Width=4 subwarp shuffle for scale broadcast |

The `__shfl_sync` with width=4 operates within a 4-lane subgroup, which is portable to HIP with a 64-bit mask fix. However, this is a minor part of the codebase compared to the PTX assembly.

### Hardcoded Warp Size

| File | Line | Value | Usage |
|------|------|-------|-------|
| `configs.h:9` | `#define WARP_SIZE 32` | Used throughout for lane_id, warpId, loop bounds, block dimensions |

This is used in:
- Block dimension: `dim3 BlockDim(WARP_SIZE * TilingConfig::BLOCK_WARPS, 1, 1)`
- Lane calculations: `threadIdx.x % WARP_SIZE`, `threadIdx.x / WARP_SIZE`
- Loop bounds: `SMEM_SIZE_IN_BYTES_PER_WARP/WARP_SIZE`
- Static config: `BLOCK_THREADS = BLOCK_WARPS * WARP_SIZE`

On AMD CDNA (gfx90a), warp size is 64. The tiling strategy, shared memory layout, and register allocation are all tuned for 32-wide warps. An AMD port would need to re-derive these parameters for 64-wide wavefronts.

### CUDA Runtime API

Standard `cuda*` calls that could be hipified:
- `cudaFuncSetAttribute` (fp6_linear.cu:31)
- `cudaGetLastError` (fp6_linear.cu:103)
- `cudaStream_t` (fp6_linear.cu:14, etc.)
- `cudaError_t` (fp6_linear.cu:47, etc.)

These are trivially portable via hipify or compat-header.

### Library Dependencies

- **cuBLAS**: Used only in tests as a baseline for correctness comparison (`-lcublas` in tests/cpp/Makefile)
- **PyTorch**: ATen/CUDA for tensor operations and stream management (`<ATen/cuda/CUDAContext.h>`)

### Host-side Code (Portable)

The following are CPU-only and fully portable:
- `utils/weight_prepacking.h` - FP6 bit packing (pure C++)
- `utils/weight_dequant.h` - FP6 to FP16 dequantization (pure C++)
- `utils/weight_quant.h` - FP16 to FP6 quantization (pure C++)

## Risk list

1. **PTX Assembly (BLOCKER)**: The core GEMM kernel uses NVIDIA PTX assembly for Tensor Core operations. There is NO mechanical translation path. Requires AMD-native rewrite with rocWMMA/CK/MFMA.

2. **Tensor Core Tiling Strategy**: The M16N8K16 MMA shape, shared memory layout, and register allocation are optimized for NVIDIA Tensor Cores. AMD's matrix units have different shapes (e.g., MFMA 16x16x16, 32x32x8) requiring algorithmic rework.

3. **Async Copy**: `cp.async` is NVIDIA-specific. HIP/ROCm does not have an exact equivalent. Would need to restructure memory prefetching using standard `hipMemcpyAsync` or remove the multi-stage pipeline.

4. **Warp Size 32 -> 64**: The entire kernel is designed around 32-lane warps. Block sizes, loop unrolling factors, and register allocation would need adjustment for 64-lane wavefronts on CDNA.

5. **Performance Parity Uncertain**: Even with a correct AMD rewrite, FP6-LLM's performance claims (2.6x vs cuBLAS, 7.2x vs bitsandbytes) may not transfer to AMD hardware without extensive tuning.

6. **FP6 Native Support on MI350+**: AMD's MI355X natively supports FP4/FP6 datatypes via MFMA scale instructions (per ROCm 7.0.0 release notes). A port targeting MI300 (gfx90a) would use emulated FP6, but MI350+ could potentially use native instructions. This affects the optimal implementation strategy.

## File-by-file change list

Given the rewrite-required nature, a detailed file-by-file list is not applicable. However, if an AMD-native rewrite were undertaken:

**Files to rewrite from scratch (keep algorithm, replace instructions):**
- `csrc/include/ptx_mma.cuh` -> AMD MFMA/rocWMMA version
- `csrc/include/ptx_cp.async.cuh` -> Standard HIP memory operations
- `csrc/include/kernel_matmul.cuh` -> Re-derived for AMD matrix units
- `csrc/include/utils_gmem.cuh` -> Standard HIP async copy

**Files to adapt (minor changes):**
- `csrc/include/configs.h` -> WARP_SIZE=64 for CDNA, adjust tiling
- `csrc/include/utils_core.cuh` -> Adjust for wavefront size
- `csrc/include/utils_parallel_dequant.cuh` -> Fix `__shfl_sync` mask to 64-bit

**Files portable as-is:**
- `csrc/utils/weight_prepacking.h`
- `csrc/utils/weight_dequant.h`
- `csrc/utils/weight_quant.h`
- `csrc/utils/common.h`

**Build files to modify:**
- `setup.py` -> Add ROCm/HIP extension path
- `fp6_llm/Makefile` -> hipcc instead of nvcc
- `tests/cpp/Makefile` -> hipcc, hipBLAS

## Build commands

N/A - Project cannot be built for ROCm without kernel rewrite.

If it could be built (hypothetically after rewrite):
```bash
# PyTorch extension (after HIP enablement)
HIP_VISIBLE_DEVICES=0 pip install .

# C++ shared library
cd fp6_llm && make HIPCC=hipcc SMS=gfx90a
```

## Test plan

N/A until kernel rewrite is complete.

The upstream test suite:
1. **Python tests** (`tests/python/run.sh`): Correctness and performance comparison vs cuBLAS baseline
2. **C++ tests** (`tests/cpp/make && ./run.sh`): Standalone kernel tests

For an AMD port, the test plan would:
1. Compare FP6-LLM output against rocBLAS FP16 baseline
2. Verify relative error matches upstream tolerance
3. Measure performance vs rocBLAS/hipBLAS

## Open questions

1. **Proceed with rewrite?** This is a significant engineering effort (estimated 2-4 weeks for an experienced GPU programmer). Is there explicit demand for FP6 quantization on AMD?

2. **Target architecture?** gfx90a (MI200 series) uses emulated FP6. gfx94x (MI300 series) may have different matrix unit characteristics. gfx950 (MI350/MI355) has native FP4/FP6 support.

3. **rocWMMA vs Composable Kernel?** CK has more examples and may be faster to implement, but rocWMMA is closer to the upstream WMMA style.

4. **Upstream appetite?** Given the project is "NVIDIA Tensor Core focused," would upstream accept an AMD-native alternative implementation?

## Recommendation

**Set `blocked` with reason `rewrite-required`**.

This project is not a porting candidate in the usual MOAT sense. It requires an AMD-native kernel implementation from scratch rather than mechanical CUDA-to-HIP translation. The effort level is comparable to writing a new library feature rather than porting an existing one.

If AMD FP6 quantization for LLMs is a priority, consider:
1. Contributing to an existing AMD quantization effort (e.g., check if Composable Kernel or rocm-ds has FP6 GEMM)
2. Building on AMD's native FP6 support in MI350+ (gfx950) when that hardware is available
3. Commissioning a dedicated AMD-native implementation rather than treating this as a "port"
