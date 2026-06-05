# CUDA-ScanMatcher-ICP notes

## Build (HIP/ROCm)

```bash
cd src
mkdir build && cd build
cmake .. -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
```

For other architectures, override CMAKE_HIP_ARCHITECTURES:
```bash
cmake .. -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100  # RDNA3
```

## Test (non-visual mode)

The project is a visualization demo with no automated tests. For headless validation,
set `VISUALIZE 0` in main.cpp (already set for this port):

```bash
HIP_VISIBLE_DEVICES=2 ./bin/cis565_ScanMatching
# Expected output: 10 ICP iterations with ~480us NN timing per step
```

## Port details

- Strategy A: single cuda_to_hip.h compat header
- All sources compiled as HIP (required because hip::host target pollutes CXX with HIP flags)
- GLM device support via glm_device.h (defines __CUDACC__ + CUDA_VERSION for GLM's qualifier detection)
- Include order matters: Thrust before GLM (rocThrust backend selection checks __CUDACC__ before __HIP__)
- GL-HIP interop updated to modern graphics resource API (hipGraphicsGLRegisterBuffer)
- Headless testing works with VISUALIZE=0 (GL interop fails without a real GPU-backed display)

## Gotchas

1. rocThrust/rocPRIM require C++17; upstream used C++11
2. svd3.h uses rsqrt() which is device-only on HIP; replaced with 1/sqrtf()
3. Kernel launch syntax `<< <` (with space) rejected by hipcc; fixed to `<<<`
4. Old cudaGL* interop functions (cudaGLSetGLDevice, cudaGLRegisterBufferObject) do not exist
   on HIP; ported to modern hipGraphicsGLRegisterBuffer/MapResources API
5. hip::host CMake target adds USE_PROF_API=1 which breaks hip_prof_str.h under plain gcc;
   compile all sources as HIP to avoid
