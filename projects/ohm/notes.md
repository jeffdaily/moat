# ohm notes

## Port summary

HIP backend added to ohm for AMD GPU support. The project has an OpenCL-style kernel abstraction: kernels are written in .cl files using OpenCL semantics, compiled via a compat header that maps OpenCL primitives to the backend (CUDA or HIP).

### Changes
- New `cmake/OhmHip.cmake` for HIP feature detection
- New `gputil/hip/` directory with HIP implementation (mirrors `gputil/cuda/`)
- New `.hip` kernel wrappers in `ohmgpu/gpu/` that include `.cl` files via HIP compat header
- Updated CMake files to support `OHM_FEATURE_HIP` option
- Fixed two upstream bugs: missing `#include <utility>` in `PlyPointStream.cpp` and `VoxelBuffer.cpp`

### Build (gfx90a)

```bash
cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DOHM_FEATURE_HIP=ON \
  -DOHM_FEATURE_CUDA=OFF \
  -DOHM_FEATURE_OPENCL=OFF \
  -DOHM_FEATURE_TEST=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a

cmake --build build -j$(nproc)
```

### Test

```bash
HIP_VISIBLE_DEVICES=3 ./build/bin/gputiltesthip  # 8/8 tests PASS
HIP_VISIBLE_DEVICES=3 ./build/bin/ohmtesthip     # 58/58 tests PASS
```

## Key implementation notes

1. The project already has excellent GPU abstraction -- CUDA and OpenCL share the same .cl kernel files. HIP follows the same pattern.

2. HIP provides vector operators for its vector types (float3, int3, etc.), so hutil_math.h disables its operator definitions to avoid ambiguity. Only the math functions (clamp, dot, length, normalize) are kept.

3. The `__local` macro in HIP conflicts with our definition; this causes harmless warnings.

4. HIP sources are identified by `.hip` extension and compiled with `LANGUAGE HIP`.

## Review 2026-06-05

**Reviewer**: MOAT reviewer agent

**Verdict**: APPROVE

The port correctly adds a HIP backend that mirrors the existing CUDA backend architecture. Key observations:

1. **Strategy**: Appropriate adaptation of Strategy A -- instead of a single cuda_to_hip.h compat header, this project uses its existing OpenCL-style abstraction layer with a parallel `gputil/hip/` directory containing `hutil_importcl.h` (analogous to `cutil_importcl.h` for CUDA).

2. **Fault classes**: No warp intrinsics, no hardcoded warp size, no textures/surfaces. Atomics correctly mapped. The float CAS (`gputilAtomicCasF32`) uses `__float_as_int`/`__int_as_float` which are available in HIP.

3. **hutil_math.h**: Correctly disables vector operators (`#if 0` block) that would conflict with HIP's built-in operators, while keeping math functions (dot, length, normalize, clamp).

4. **CMake**: OhmHip.cmake properly uses `check_language(HIP)`, defaults `CMAKE_HIP_ARCHITECTURES` only when unset (allowing follower platforms to override), and gates all HIP dependencies.

5. **Upstream fixes**: The two `#include <utility>` additions are legitimate portability fixes.

6. **Commit hygiene**: Title 43 chars, properly prefixed `[ROCm]`, mentions Claude, no noreply trailer, no MOAT jargon.

No issues found. Ready for validation.

## Validation 2026-06-05 (linux-gfx90a)

**GPU arch**: gfx90a (MI250X)  
**ROCm version**: 7.2.1  
**Build command**:
```bash
cmake -B build_gfx90a -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DOHM_FEATURE_HIP=ON \
  -DOHM_FEATURE_CUDA=OFF \
  -DOHM_FEATURE_OPENCL=OFF \
  -DOHM_FEATURE_TEST=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a

cmake --build build_gfx90a -j$(nproc)
```

**Test results**:
- `gputiltesthip`: 8/8 tests PASSED
- `ohmtesthip`: 58/58 tests PASSED

**Total**: 66/66 tests PASSED

All GPU tests executed successfully on real hardware with no failures.

## Validation 2026-06-05 (linux-gfx1100)

**GPU arch**: gfx1100 (Navi31)  
**ROCm version**: 7.2.1  
**Build command**:
```bash
/usr/bin/cmake -B build_gfx1100 -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DOHM_FEATURE_HIP=ON \
  -DOHM_FEATURE_CUDA=OFF \
  -DOHM_FEATURE_OPENCL=OFF \
  -DOHM_FEATURE_TEST=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100

/usr/bin/cmake --build build_gfx1100 -j$(nproc)
```

**Test results**:
- `gputiltesthip`: 8/8 tests PASSED
- `ohmtesthip`: 58/58 tests PASSED

**Total**: 66/66 tests PASSED

All GPU tests executed successfully on real hardware with no failures. Note: Used system cmake (/usr/bin/cmake 3.28.3) instead of conda cmake to avoid googletest compatibility issues.
