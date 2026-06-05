# Velvet - HIP Port Notes

## Build

```bash
export HIP_VISIBLE_DEVICES=0
cd projects/Velvet/src
cmake -B build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/bin/Velvet
```

## Port Gotchas

### GLM 1.0.1 Required
System GLM 0.9.9.8 lacks `GLM_COMPILER_HIP` detection, so `__device__ __host__` qualifiers are missing and device code fails. GLM 1.0.1 adds HIP support. Bundled in `glm_local/`.

### rocThrust THRUST_DEVICE_SYSTEM
rocThrust checks `__CUDACC__` before `__HIP__`. To avoid the CUDA backend, define `THRUST_DEVICE_SYSTEM=5` (HIP) before including Thrust. This is done in `cuda_to_hip.h`.

### .cpp to .cu Rename
CMake HIP targets apply `HIP_ARCHITECTURES` to all sources. Mixed CXX/HIP targets cause the C++ files to receive HIP architecture flags which errors. All .cpp files are renamed to .cu so the HIP compiler handles everything.

### Windows Path Separators
The original code uses `#include <glm\ext\...>` which fails on Linux (case-sensitive, wrong separator). Fixed to forward slashes.

### Case-Sensitive Includes
`SpatialhashGPU.cuh` vs `SpatialHashGPU.cuh` -- Windows ignores case, Linux does not.

### fmt 10+ const format()
Modern fmt requires `format()` method to be const in custom formatters.

### GLAD2 API Change
GLAD2 uses `gladLoadGL((GLADloadfunc)...)` not the older `gladLoadGLLoader((GLADloadproc)...)`.

### hipGraphicsResource Typedef
HIP uses `typedef struct ihipGraphicsResource* hipGraphicsResource_t;` whereas CUDA uses `struct cudaGraphicsResource`. Cannot use `struct` prefix with HIP.

### HOST_INIT Macro
The macro suppresses dynamic initialization for `__device__ __constant__` variables. Must check `__HIPCC__` in addition to `__CUDACC__` or `__CUDA_ARCH__`.

## Dependencies (bundled)

- GLM 1.0.1 (header-only, in `glm_local/`)
- GLAD (generated, in `glad/`)
- ImGui (GUI toolkit, in `imgui/`)

## Known Warnings

- hipDeviceSynchronize nodiscard warnings in VtBuffer.hpp -- return value not checked (original CUDA code behavior)

## Review 2026-06-05

### Port Correctness

**CRITICAL: CUDA_CALL macros are no-ops on HIP** -- `Velvet/Common.cuh:34` gates the CUDA_CALL/CUDA_CALL_S/CUDA_CALL_V macros on `#ifdef __CUDACC__`. hipcc defines `__HIPCC__` but NOT `__CUDACC__`, so on HIP the macros resolve to empty bodies (lines 47-50) and all kernel launches silently do nothing. The binary compiles but simulation kernels never execute. Fix: change line 34 to `#if defined(__CUDACC__) || defined(__HIPCC__)`.

### Minimal Footprint

**.gitignore rewrite breaks upstream CUDA** -- The port replaced the entire .gitignore content (VS user files, x64/, x86/, .vs/) with just `build/`. The upstream Windows build would no longer ignore its build artifacts. Add `build/` without deleting the existing Windows ignores.

### Backward Compatibility

**VtCallback template parameter renamed** -- `Velvet/Common.hpp:88-91` changes `TArgs` to `Args`. This is a breaking API change for any downstream code that explicitly named the template parameter. Keep the original name `TArgs` or ensure this is intentional.

### Recommendation

**Request Changes**

The __CUDACC__ guard on CUDA_CALL means kernel launches are silent no-ops. This must be fixed before validation.

## Review Fix 2026-06-05

Fixed both issues:

1. **CUDA_CALL macros**: Changed `#ifdef __CUDACC__` to `#if defined(__CUDACC__) || defined(__HIPCC__)` so kernel launches work on HIP.

2. **.gitignore**: Restored upstream Windows ignores (*.user, x64/, x86/, .vs/) and added `build/` at end.

## Re-review 2026-06-05

Previous findings verified as fixed:

1. **CUDA_CALL macros** -- `Velvet/Common.cuh:34` now checks `defined(__CUDACC__) || defined(__HIPCC__)`. Kernel launches will work on HIP.

2. **.gitignore** -- Upstream Windows ignores (*.user, x64/, x86/, .vs/) restored; `build/` appended. No minimal footprint violation.

3. **VtCallback template rename** -- The `TArgs -> Args` change at method level (line 91) is harmless: it removes a name shadow with the class-level `TArgs` (line 82). Method template parameters are never explicitly named in calls, so this is non-breaking.

### Fault Class Verification

- **warpSize/32**: No warp intrinsics (__shfl*, __ballot, __activemask). No hardcoded 32 for warp operations. BLOCK_SIZE=256 is a launch config, not a warp assumption.
- **Rule-of-five**: VtBuffer/VtRegisteredBuffer/VtMergedBuffer delete copy constructors/assignment. No texture/surface handles requiring special cleanup.
- **OOB neighbor reads**: Neighbor caching in SpatialHashGPU.cu clamps via `cellEnd[h]` and uses sentinel 0xffffffff.
- **Texture pitch**: No texture usage in this project.
- **Library swaps**: CUB -> hipCUB via namespace alias (SpatialHashGPU.cu:3-6). rocThrust via THRUST_DEVICE_SYSTEM=5 in cuda_to_hip.h.
- **Arch-unified**: No per-arch fixes; single unified port.

### Build System

- CMakeLists.txt correctly uses `enable_language(HIP)` when USE_HIP=ON, `enable_language(CUDA)` otherwise.
- CMAKE_HIP_ARCHITECTURES defaulted to gfx90a but can be overridden.
- CUDA build path preserved (USE_HIP=OFF).

### Commit Hygiene

- `[ROCm]` prefix present on both commits.
- No `Co-Authored-By: noreply` trailer.
- Mentions Claude by name.
- No AMD-internal account references.

### Recommendation

**Approve** -- ready for validation.
