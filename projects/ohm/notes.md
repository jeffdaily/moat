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
