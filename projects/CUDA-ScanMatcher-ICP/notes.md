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

## Review 2026-06-05

### Port Correctness

1. **main.cpp:80** -- Direct use of `hipSetDevice(gpuDevice)` instead of going through the compat header. This breaks the CUDA build because `hipSetDevice` is not defined without HIP headers. Should use `cudaSetDevice` and add the mapping to cuda_to_hip.h.

2. **main.cpp:133-139** -- Mixed HIP/CUDA symbols in the same code block: uses `cudaGraphicsGLRegisterBuffer` (compat-mapped) but directly uses `hipError_t`, `hipSuccess`, and `hipGetErrorString` without mappings. These HIP-specific symbols will cause CUDA build failures. The error handling code should use the CUDA spellings (`cudaError_t`, `cudaSuccess`, `cudaGetErrorString`) which the compat header already maps.

### Minimal Footprint

3. **CMakeLists.txt:112** -- All sources including host `.cpp` files are marked as `LANGUAGE HIP`. Per PORTING_GUIDE Strategy A, only `.cu` files should be marked HIP; host C++ should remain untouched by the HIP toolchain. The current approach compiles main.cpp, glslUtility.cpp, and utilityCore.cpp with hipcc, which is unnecessary and deviates from the minimal-footprint principle.

### Backward Compatibility (upstream)

4. **The CUDA build is broken** -- Due to issues #1 and #2 above, building with `USE_HIP=OFF` will fail. The compat header should map:
   - `cudaSetDevice` -> `hipSetDevice`
   - And code should use CUDA spellings consistently, letting the compat header do the translation.

### Recommendation

**Request Changes**

The port breaks the CUDA build path due to direct use of HIP-specific symbols in main.cpp. All four issues must be fixed:
- Add `cudaSetDevice` mapping to cuda_to_hip.h
- Replace `hipSetDevice` with `cudaSetDevice` in main.cpp:80
- Replace `hipError_t`/`hipSuccess`/`hipGetErrorString` with `cudaError_t`/`cudaSuccess`/`cudaGetErrorString` in main.cpp:133-139
- Consider marking only `.cu` files as LANGUAGE HIP (minor, but per Strategy A guidelines)

## Fix 2026-06-05

Resolved review findings:
1. Added `cudaSetDevice` mapping to cuda_to_hip.h
2. Changed `hipSetDevice(gpuDevice)` to `cudaSetDevice(gpuDevice)` in main.cpp:80
3. Changed `hipError_t`/`hipSuccess`/`hipGetErrorString` to `cudaError_t`/`cudaSuccess`/`cudaGetErrorString` in main.cpp:133-139

The port now uses CUDA spellings consistently; cuda_to_hip.h performs translation on HIP builds. CUDA build path (USE_HIP=OFF) should work unchanged.

Kept all sources as LANGUAGE HIP because gotcha #5: hip::host CMake target adds USE_PROF_API=1 which breaks hip_prof_str.h under plain gcc. Compiling .cpp files with hipcc avoids this.

## Review 2026-06-05 (re-review after fixes)

Previous review findings resolved:
1. cudaSetDevice mapping added to cuda_to_hip.h
2. main.cpp:80 now uses cudaSetDevice (mapped via compat header)
3. main.cpp:133-139 now uses cudaError_t/cudaSuccess/cudaGetErrorString (mapped via compat header)

Fault classes verified:
- No warp intrinsics or hardcoded warpSize=32
- No textures, no pitch alignment concerns
- No resource handle rule-of-five issues
- No OOB neighbor reads requiring clamps

LANGUAGE HIP on all sources (including host .cpp) justified by gotcha #5.

Ready for GPU validation.

## Validation 2026-06-05

### Platform: linux-gfx90a

**Build**: gfx90a
```bash
cd src
rm -rf build && mkdir build && cd build
cmake .. -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
```

Build succeeded with warnings (nodiscard return values ignored, not critical).
llvm-objdump confirms gfx90a code objects embedded.

**Test**: HIP_VISIBLE_DEVICES=2 (MI250X gfx90a)
```bash
HIP_VISIBLE_DEVICES=2 ./bin/cis565_ScanMatching
```

**Results**: PASS
- 10 ICP iterations completed successfully
- NN timing: 14320us (first), ~447-546us (warmup), then stable ~481-484us per iteration
- No crashes, no errors, no NaN values
- GPU computation works correctly on gfx90a
