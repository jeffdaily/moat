# cuSZ Porting Plan (linux-gfx90a)

## Project
- Name: cuSZ (pSZ/cuSZ)
- Upstream: https://github.com/szcompressor/cuSZ
- Default branch: main
- Description: GPU-based error-bounded lossy compression for scientific data

## Existing AMD Support

**Assessment:** Incomplete, bitrotted, authoritative (upstream-maintained but broken)

The cuSZ upstream already includes HIP/ROCm backend support (`-DPSZ_BACKEND=HIP`) with:
- `cmake/hip.cmake` build configuration
- Dedicated `.hip` source files in `psz/src/`, `test/src/`, `example/src/`
- CUDA-to-HIP translation headers in `portable/include/macro/c_cu2hip_*.h`
- Guards via `PSZ_USE_HIP` and `_PORTABLE_USE_HIP` defines

However, the HIP backend is **non-functional** and out-of-sync with the CUDA backend:
1. Wrong file paths in `cmake/hip.cmake`:
   - `src/cusz_version.h.in` should be `psz/src/cusz_version.h.in`
   - `CUSZConfig.cmake.in` should be `cmake/CUSZConfig.cmake.in`
2. References non-existent source files (e.g., `src/hf/hfclass.hip`, `src/hf/hf_codec.hip`)
3. Several `.hip` files are 0-byte stubs (`codec/hf/src/hf_kernels.hip`, `psz/src/utils/viewer.hip`)
4. Missing codec subproject HIP builds (codec/hf and codec/fzg only have CUDA CMake)
5. Open upstream issue #86 confirms the HIP build is broken (Mar 2025)

**Decision:** MOAT value is to fix the existing HIP support, not port from scratch. The upstream authors intend HIP support (they created the infrastructure) but it bitrotted. The porter should update `cmake/hip.cmake` to mirror the CUDA build structure, fix missing files, and validate.

**Separate project:** hipSZ (hpdps-group/hipSZ) exists as a HIP implementation of cuSZp (a different algorithm variant), developed independently for Chinese DCU hardware. It is NOT a fork of cuSZ and implements cuSZp rather than cuSZ, so it does not make cuSZ's HIP port redundant.

## Build Classification

**Classification:** Pure CMake (Strategy A, with modifications)

**Evidence:**
- Root `CMakeLists.txt` selects backend via `PSZ_BACKEND` cache variable
- No PyTorch/torch dependency (`find_package(Torch)` not present)
- Standard `enable_language(HIP)` pattern in `cmake/hip.cmake`
- Pure C++/CUDA/HIP build with external deps: rocthrust, rocprim, hiprand, rocrand

The project already follows the Strategy A model with `enable_language(HIP)` and a CUDA-to-HIP compat header layer. The fix is to update the bitrotted HIP CMake, not introduce a new porting strategy.

**ext_type:** cmake

## Port Strategy

**Strategy:** Fix existing HIP infrastructure

Since the project already has HIP support infrastructure but it's broken, the approach is:

1. Fix `cmake/hip.cmake` path references
2. Sync the HIP build structure with `cmake/cuda.cmake` (portable/, codec/hf/, codec/fzg/)
3. Fill in empty `.hip` stub files with proper includes of `.cuhip.inl` counterparts
4. Add missing HIP builds for codec/hf and codec/fzg subprojects
5. Validate the build compiles and tests pass on gfx90a

This is NOT a from-scratch CUDA-to-HIP port -- the translation layer (`c_cu2hip_*.h`) and `.cuhip.inl` shared sources already exist.

## CUDA Surface Inventory

### Kernels and Device Code
- Lorenzo predictors (1D/2D/3D): `psz/src/kernel/lrz_c.cu`, `lrz_x.cu`, `proto_lrz_*.cu`
- Histogram: `hist_generic.cu`, `histsp.cu`
- Sparse vector: `spvn.cu`
- Spline interpolation: `spline3.cu`
- Huffman encoding: `codec/hf/src/hf_kernels.cu`
- FZG codec: `codec/fzg/src/fzg_kernel.cu`
- Statistics: `psz/src/stat/*.cu`

### Warp Intrinsics
- `__shfl_up_sync`, `__shfl_down_sync`, `__shfl_xor_sync`, `__shfl_sync` (mapped via `c_cu2hip_1_fix_primitives.h`)
- `__ballot_sync` in `lrz_c.cuhip.inl`, `fzg_c.cuhip.inl`, `fzg_x.cuhip.inl`
- Hardcoded mask `0xffffffff` for 32-lane operations

### Warp Size
- File `wave32.cuhip.inl` explicitly uses 32-lane warp operations
- `const static unsigned int WARP_SIZE = 32;` in `hist.cuhip.inl`
- Operations use width-32 shuffles (`__shfl_up_sync(..., d, 32)`)

### Libraries
- **cuRAND/hipRAND**: Used in `test/src/utils/rand.cu_hip.cc` (test utilities only)
- **Thrust/rocThrust**: Used in `psz/src/archive/glue.cuh` and `*.thrust.cu` stat files
- **CUB (legacy)**: Used in `_legacy_lrz_cx_21.cuhip.inl` (marked deprecated)

### Runtime API
- Standard cudaMalloc/cudaFree/cudaMemcpy (all mapped via compat header)
- cudaStream_t for async operations
- cudaDeviceGetAttribute for device properties

### Textures/Surfaces
- None found (pure compute, no texture sampling)

## Risk List

1. **Warp size 32 hardcoding (LOW RISK)**: The code explicitly uses width-32 logical warp operations (`__shfl_up_sync(..., 32)`). Per PORTING_GUIDE, width-32 logical-warp ops are arch-agnostic and work on wave64 (they operate within a 32-lane subgroup). The `wave32.cuhip.inl` naming reflects this design choice. No changes needed.

2. **Empty .hip stubs (HIGH RISK)**: `codec/hf/src/hf_kernels.hip` and `psz/src/utils/viewer.hip` are 0-byte empty files. These must be populated with proper includes of corresponding `.cu` or `.cuhip.inl` files.

3. **Missing codec HIP builds (HIGH RISK)**: `codec/hf/CMakeLists.txt` and `codec/fzg/CMakeLists.txt` only define CUDA builds. HIP versions must be added or the HIP build must inline these sources.

4. **CMake path mismatches (HIGH RISK)**: `cmake/hip.cmake` has at least 5 path errors relative to actual file locations. Must be corrected.

5. **Thrust usage with `thrust::cuda::par` (MEDIUM RISK)**: The `glue.cuh` archive code uses `thrust::cuda::par` explicitly. On HIP this should be `thrust::hip::par` or just `thrust::device`. The compat header may need to map this.

6. **Test utilities cuRAND (LOW RISK)**: Test random number generation uses cuRAND API directly. The `hip.cmake` already links `${hiprand_LIBRARIES}` and this is test-only code.

## File-by-File Change List

### CMake fixes
- `cmake/hip.cmake`: Fix all path references, sync structure with `cuda.cmake`
- Add HIP build support for `codec/hf/` and `codec/fzg/` (either inline in hip.cmake or add HIP CMake files)

### Empty stub fills
- `codec/hf/src/hf_kernels.hip`: Add `#include "hf_kernels.cuhip.inl"` (create the .inl if needed)
- `psz/src/utils/viewer.hip`: Add proper content or merge with viewer.cu

### Potential compat header additions
- `portable/include/macro/c_cu2hip_0_translation.h`: May need `thrust::cuda::par` -> `thrust::hip::par` mapping

### Test cmake
- `test/cmake/hip-test.cmake`: Review for path and target consistency with the fixed hip.cmake

## Build Commands

### Configure (gfx90a)
```bash
cmake -B build-hip \
  -DPSZ_BACKEND=HIP \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DPSZ_BUILD_EXAMPLES=ON \
  -S .
```

### Build
```bash
cmake --build build-hip -j$(nproc)
```

### Install (optional)
```bash
cmake --install build-hip --prefix install-hip
```

## Test Plan

### GPU Tests (via CTest)
```bash
cd build-hip
ctest --output-on-failure
```

Expected tests from `test/cmake/hip-test.cmake`:
- `test_spv_hip`: Sparse vector operations
- `test_zigzag`: Zigzag encoding (CPU, should not regress)
- `test_l1_scan`: Level-1 scan operations
- `test_l1_compact`: Level-1 compact operations
- `test_l2_cudaproto`: Level-2 kernel tests
- `test_histsp`: Histogram operations
- `test_l3_cuda_pred`: Level-3 predictor tests
- `test_lrz_seq`: Sequential Lorenzo (CPU, should not regress)
- `test_lrzsp_hip`: Lorenzo sparse
- `test_statfn`: Statistics functions

### Integration Test (compression/decompression)
```bash
# Download sample data
./script/sh.download-sdrb-data  # or use provided small samples

# Run CLI tool
./hipsz -i sample.f32 -z -x 512 -y 512 -e 1e-3 -o compressed.cusz
./hipsz -i compressed.cusz -d -o decompressed.f32
```

### Non-GPU Tests
- `test_zigzag`: CPU-only zigzag codec
- `test_lrz_seq`: CPU-only Lorenzo predictor

## Open Questions

1. **Upstream coordination**: The maintainer (jtian0) noted in issue #86 they plan to fix HIP support after merging CUDA changes. Should we contribute fixes upstream or wait? Decision: Proceed with fixes; the port can be contributed as a PR to accelerate their roadmap.

2. **codec/hf and codec/fzg integration**: Should these be built as separate HIP libraries (mirroring CUDA structure) or inlined into the main hip.cmake? The current hip.cmake inlines Huffman code; follow that pattern.

3. **Version parity**: The HIP build version is 0.6.0 while CUDA is 0.10.0. This should be synced in the final PR.
