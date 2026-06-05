# Plan: baspacho

## Project

- **Name**: baspacho
- **Upstream**: https://github.com/facebookresearch/baspacho
- **Default branch**: main
- **Description**: BaSpaCho (Batched Sparse Cholesky) is a direct solver for symmetric positive-definite sparse matrices. Implements supernodal Cholesky decomposition with GPU (CUDA) support for batched operations.

## Existing AMD support

**None found.** Searched:
- `grep -rniE 'amd|rocm|hip|gfx[0-9]'` on README/docs -- only mentions "AMD" as the "Approximate Minimum Degree" reordering algorithm from SuiteSparse, not AMD GPUs
- WebSearch for "baspacho ROCm", "baspacho AMD GPU", "baspacho HIP", "baspacho MI300/gfx9" -- no results
- GitHub forks via `gh api repos/facebookresearch/baspacho/forks` -- no forks under ROCm/AMD/GPUOpen orgs or with rocm/hip/amd in the name
- No rocm/hip branches in upstream
- No ROCm/HIP-related PRs/issues

**Decision**: Proceed with a from-scratch port. The project is a pure CMake build with CUDA kernels and cuBLAS/cuSolver usage, suitable for Strategy A (minimal-footprint compat header).

## Build classification

**Pure CMake** (Strategy A).

Evidence:
- `CMakeLists.txt` line 36-37: `set(BASPACHO_USE_CUBLAS ON CACHE BOOL ...)` and `enable_language(CUDA)`
- `CMakeLists.txt` line 44: `find_package(CUDAToolkit REQUIRED)`
- No `find_package(Torch)`, no `torch.utils.cpp_extension`, no `CUDAExtension`, no setup.py
- Build uses `enable_language(CUDA)`, links `CUDA::cudart`, `CUDA::cublas`, `CUDA::cusolver`

## Port strategy

**Strategy A: pure CMake, minimal footprint**

Rationale:
- Single CUDA source file: `baspacho/baspacho/MatOpsCuda.cu` (1473 lines)
- Single CUDA header: `baspacho/baspacho/CudaAtomic.cuh` (76 lines)
- Clean separation between CUDA and CPU code
- Uses `enable_language(CUDA)` pattern, easily extended to `enable_language(HIP)`
- No warp intrinsics (no `__shfl*`, `__ballot`, `__activemask`, `__syncwarp`)
- No texture/surface usage
- Standard cuBLAS/cuSolver calls with direct hipBLAS/hipSOLVER equivalents

Approach:
1. Add `cuda_to_hip.h` compat header with CUDA->HIP symbol aliases
2. Add `-DUSE_HIP=ON` CMake option that gates HIP builds
3. Use `set_source_files_properties(... PROPERTIES LANGUAGE HIP)` for `.cu` files
4. Link hipBLAS and hipSOLVER instead of cuBLAS/cuSolver

## CUDA surface inventory

### CUDA source files
| File | Lines | Description |
|------|-------|-------------|
| `baspacho/baspacho/MatOpsCuda.cu` | 1473 | Main CUDA implementation: kernels, cuBLAS/cuSolver calls |
| `baspacho/baspacho/CudaAtomic.cuh` | 76 | atomicAdd wrappers for older architectures |

### CUDA API symbols (by category)

**Runtime API** (CudaDefs.h, MatOpsCuda.cu):
- `cudaMalloc`, `cudaFree`, `cudaMemcpy`, `cudaMemcpyHostToDevice`, `cudaMemcpyDeviceToHost`
- `cudaStreamSynchronize`, `cudaDeviceSynchronize`
- `cudaDeviceReset`, `cudaGetErrorString`, `cudaError_t`, `cudaSuccess`

**cuBLAS** (MatOpsCuda.cu):
- `cublasHandle_t`, `cublasCreate`, `cublasDestroy`, `cublasSetStream`
- `cublasStatus_t`, `CUBLAS_STATUS_SUCCESS`, `CUBLAS_STATUS_*` enums
- `cublasDgemm`, `cublasSgemm`, `cublasDgemmBatched`, `cublasSgemmBatched`
- `cublasDtrsm`, `cublasStrsm`, `cublasDtrsmBatched`, `cublasStrsmBatched`
- `cublasDsymm`, `cublasSsymm`
- `CUBLAS_SIDE_LEFT`, `CUBLAS_FILL_MODE_UPPER`, `CUBLAS_OP_C`, `CUBLAS_OP_N`

**cuSolver** (MatOpsCuda.cu):
- `cusolverDnHandle_t`, `cusolverDnCreate`, `cusolverDnDestroy`, `cusolverDnSetStream`
- `cusolverStatus_t`, `CUSOLVER_STATUS_SUCCESS`, `CUSOLVER_STATUS_*` enums
- `cusolverDnDpotrf`, `cusolverDnSpotrf` (single-matrix Cholesky)
- `cusolverDnDpotrfBatched`, `cusolverDnSpotrfBatched` (batched Cholesky)
- `cusolverDnDpotrf_bufferSize`, `cusolverDnSpotrf_bufferSize`

**cuSPARSE** (included but error string only):
- `cusparseStatus_t` -- only for error enum in CublasError.cpp, not actually used

**Device code** (MatOpsCuda.cu, CudaAtomic.cuh, MathUtils.h):
- `__global__` kernels: `factor_lumps_kernel`, `factor_spans_kernel`, `sparse_elim_2loops_kernel`, `sparse_elim_straight_kernel`, `assemble_kernel`, `assembleVec_kernel`, `assembleVecT_kernel`, `sparseElim_diagSolveL`, `sparseElim_subDiagMult`, `sparseElim_diagSolveLt`, `sparseElim_subDiagMultT`
- `__device__` functions: `do_sparse_elim`, `stridedMatSubDev`, `stridedTransAdd`, `stridedTransSet`, `locked_sub_product`, `locked_sub_AxB`, `locked_sub_ATxB`
- `__host__ __device__`: `cholesky`, `solveUpperT`, `solveUpper`, `bisect`, `toOrderedPair` (via `__BASPACHO_HOST_DEVICE__` macro)
- Kernel launches: 25+ uses of `<<<gridDim, blockDim>>>`

**Atomic operations** (CudaAtomic.cuh):
- `atomicAdd` for double/float accumulation in elimination
- `atomicCAS` for legacy (<arch 60) double atomicAdd workaround

### Library mapping

| CUDA Library | HIP Library | Notes |
|--------------|-------------|-------|
| cuBLAS (`CUDA::cublas`) | hipBLAS | 1:1 API for GEMM, TRSM, SYMM |
| cuSolver (`CUDA::cusolver`) | hipSOLVER | 1:1 API for potrf (Cholesky) |
| cuSPARSE (header only) | hipSPARSE | Only error enum used, trivial |

## Risk list

1. **No warp-size risk**: No `__shfl*`, `__ballot`, `warpSize`, or hardcoded 32/64. The kernels are element-wise or per-lump/span indexed, not warp-collective. Safe for both wave64 (gfx90a) and wave32 (gfx1100/gfx1151).

2. **hipSOLVER batched potrf**: `cusolverDnDpotrfBatched`/`cusolverDnSpotrfBatched` must map to `hipsolverDnDpotrfBatched`/`hipsolverDnSpotrfBatched`. Verify API compatibility; hipSOLVER should have these.

3. **Atomic operations**: Uses `atomicAdd` for double/float. HIP supports `atomicAdd` for double natively on gfx90a (CDNA2). The project's `CUDA_DOUBLE_ATOMIC_ADD_WORKAROUND` for <arch60 is unnecessary on HIP but harmless.

4. **Architecture detection**: The project detects CUDA arch via a compiled probe (`cmake/detect_cuda.cu`). For HIP builds, this needs to be bypassed or replaced. The manual arch specification path (`-DBASPACHO_CUDA_ARCHS="..."`) can be adapted for HIP.

5. **Legacy cuda arch handling**: CMake code separates architectures <60 vs >=60 for the atomicAdd workaround. This CUDA-specific logic can be removed/simplified for HIP builds.

6. **CUBLAS_FILL_MODE_UPPER enum**: Maps to `HIPBLAS_FILL_MODE_UPPER` in hipBLAS. Need to alias or the compat header handles it.

7. **No textures/surfaces**: Confirmed absence avoids pitch alignment and layered-array issues.

8. **No pinned/managed memory**: Only `cudaMalloc`/`cudaMemcpy`, no special memory types.

9. **`__CUDACC__` macro**: Used in `Utils.h` (line 143) to define `__BASPACHO_HOST_DEVICE__`. HIP defines `__HIPCC__` instead. The compat header must define `__CUDACC__` when `__HIPCC__` is defined, or patch the macro.

## File-by-file change list

### New files

| File | Purpose |
|------|---------|
| `baspacho/baspacho/cuda_to_hip.h` | CUDA-to-HIP compat header with symbol aliases |

### Modified files

| File | Changes |
|------|---------|
| `CMakeLists.txt` | Add `USE_HIP` option; conditional `enable_language(HIP)`; arch default logic; link hipBLAS/hipSOLVER |
| `baspacho/baspacho/CMakeLists.txt` | Gate CUDA vs HIP language on `USE_HIP`; link hip libs; simplify arch handling for HIP |
| `baspacho/baspacho/CudaDefs.h` | Include `cuda_to_hip.h`; alias error check macros |
| `baspacho/baspacho/MatOpsCuda.cu` | Include compat header (no other source changes needed if compat header is comprehensive) |
| `baspacho/baspacho/Utils.h` | Make `__BASPACHO_HOST_DEVICE__` work with `__HIPCC__` |

### Files unchanged
- All test files (they use the library API, not CUDA directly)
- All example files
- All CPU-only source files

## Build commands

### Configure (gfx90a)
```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DBASPACHO_USE_BLAS=ON \
  -DBLA_VENDOR=OpenBLAS
```

### Build
```bash
cmake --build build -- -j$(nproc)
```

### Configure (multi-arch for validation)
```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100"
```

## Test plan

### GPU tests (must pass on real GPU)
All tests in `baspacho/tests/CMakeLists.txt` when `BASPACHO_USE_CUBLAS` (becomes `USE_HIP`):
- `CudaFactorTest` -- single-matrix CUDA factorization
- `CudaSolveTest` -- single-matrix CUDA solve
- `BatchedCudaFactorTest` -- batched CUDA factorization
- `BatchedCudaSolveTest` -- batched CUDA solve
- `CudaPartialTest` -- partial CUDA operations

### Non-GPU tests (must not regress)
- `AccessorTest`
- `CoalescedBlockMatrixTest`
- `CreateSolverTest`
- `SparseStructureTest`
- `EliminationTreeTest`
- `FactorTest`
- `SolveTest`
- `MathUtilsTest`
- `PartialFactorSolveTest`

### Run tests
```bash
ctest --test-dir build --output-on-failure
```

### Benchmarking (optional validation)
```bash
build/baspacho/benchmarking/bench -B 1_CHOLMOD  # if CHOLMOD available
```

## Open questions

1. **hipSOLVER batched API parity**: Confirm `hipsolverDnDpotrfBatched`/`hipsolverDnSpotrfBatched` exist with identical signatures. ROCm docs suggest they do.

2. **BLAS vendor on ROCm**: The project supports OpenBLAS/MKL/ATLAS. On ROCm systems, rocBLAS is available but the host BLAS (for non-GPU operations) should still be OpenBLAS or another CPU BLAS. Verify this interaction.

3. **Dispenso threading library**: Fetched automatically via FetchContent. Should work unchanged on ROCm. No known compatibility issues but not verified.

4. **Arch autodetection**: The `cmake/detect_cuda.cu` probe won't work for HIP. Either always require `-DCMAKE_HIP_ARCHITECTURES` or add a HIP detection mechanism (rocminfo parsing or similar).
