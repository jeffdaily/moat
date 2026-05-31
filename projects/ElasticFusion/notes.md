# ElasticFusion notes

Heavy deps (OpenGL, Pangolin) and older codebase; verify build prerequisites first.

## Lead platform: linux-gfx90a (MI250X, ROCm 7.2.1) -- PORTED

Strategy A (compat header + `enable_language(HIP)` + LANGUAGE HIP), correctness-first.
Fork: https://github.com/jeffdaily/ElasticFusion , port on `moat-port`, master = clean
upstream mirror. Actions disabled on the fork.

### THE KEY FINDING: GL/HIP interop is infeasible on a headless MI250X

ElasticFusion's tracking inputs reach the CUDA/HIP kernels ONLY through OpenGL --
`RGBDOdometry` maps GL textures (the GLSL-splatted surfel prediction, filtered depth,
RGB) to a `cudaArray_t` via `cudaGraphicsGLRegisterImage`/`MapResources`/
`SubResourceGetMappedArray`, then feeds that array to the kernels. There is no
CUDA-only ingest path. The feasibility gate (the porter's first experiment,
`rocm_validation/gl_interop_probe.cpp`) is whether `hipGraphicsGLRegisterImage` can
round-trip a GL texture to a HIP-readable array on this node. Result: it cannot, for
two independent and fundamental reasons.

1. The gfx90a (CDNA2 MI250X) is a COMPUTE-ONLY chip with no graphics/display blocks.
   Mesa's hardware GL driver (radeonsi) refuses to create a context on it:
   `radeonsi: error: can't create a graphics context on a compute chip`. EGL on each
   amdgpu DRM node (card1-4 / renderD128-131) enumerates fine and advertises 112
   OpenGL configs, but `eglCreateContext` then fails (BAD_MATCH / BAD_ALLOC) with that
   radeonsi error. So there is no GPU-resident GL context to share with HIP.
2. No GPU-backed GL fallback exists here: there is NO Vulkan ICD installed
   (`/usr/share/vulkan/icd.d` empty; only `libvulkan.so.1` loader), so zink
   (GL-over-Vulkan) fails `vkCreateInstance -> VK_ERROR_INCOMPATIBLE_DRIVER`. Software
   GL (the EGL software device, llvmpipe) DOES create a GL 4.5 context, but it lives in
   host memory: `hipGLGetDevices -> hipErrorNoDevice` (count 0), and
   `hipGraphicsGLRegisterImage -> hipErrorUnknown`. HIP cannot import a CPU-rendered
   texture onto the gfx90a device.

The HIP GL-interop API itself is fully present in ROCm 7.2.1 (`hip/hip_gl_interop.h`,
`hipGraphicsGLRegisterImage/Buffer`, `hipGLGetDevices`, map/unmap/unregister, the flags
enum) and all symbols are in hipify's map. The wall is purely the absence of a
GPU-backed GL stack on a headless compute accelerator -- environmental, not a
HIP-correctness gap, and not fixable in the port.

The plan's documented fallback (b), a GL-readback bounce, does NOT help here either:
the readback still needs a GL context to RENDER the surfel-prediction textures (the
GLSL splatting), and llvmpipe (the only GL available) is a host-memory CPU renderer.
You would be running the entire SLAM front end on llvmpipe and bouncing every texture
host<->device each frame -- not a meaningful GPU validation of the tracking core, and
enormous bringup (full Pangolin + a working dataset + a CUDA reference) for a result
that proves nothing the kernel harness does not already prove. So the principled
validation is the device-array kernel harness below. On a follower with a real display
GPU (gfx1100 has graphics blocks + a hardware radeonsi GL), interop may actually work;
that is the right place to validate the live GL path.

### What was ported (the diff to review, in order)

1. `Core/Cuda/cuda_to_hip.h` (NEW) -- the compat header, force-included on HIP TUs
   (`CMAKE_HIP_FLAGS -include`). On HIP: includes `<hip/hip_runtime.h>` +
   `<hip/hip_gl_interop.h>`, aliases only the cuda* symbols used to hip*, defines
   `CUDA_WARP_FULL_MASK`. CRUCIAL: `__CUDACC__`/`EIGEN_NO_CUDA` are defined ONLY under
   `#if defined(__HIPCC__)` (the HIP compiler, i.e. the .cu TUs), NOT merely under
   USE_HIP -- the host .cpp are built by g++ with -DUSE_HIP and must get the symbol
   aliases but must NOT see __CUDACC__ (see gotcha below). On CUDA the whole header is
   a no-op `#include <cuda_runtime.h>` + 32-bit mask.
2. `hip_compat/{cuda_gl_interop,cuda_runtime,cuda_runtime_api,driver_types,vector_types,
   vector_functions}.h` (NEW) -- forwarding shims (each `#include
   "../Core/Cuda/cuda_to_hip.h"`), on the include path only under USE_HIP, so host
   .cpp keep their toolkit-named includes unchanged. HIP ships no toolkit-named
   `<vector_types.h>`/`<vector_functions.h>` (float3/make_float3 come from hip_runtime),
   so those need shims too.
3. `CMakeLists.txt` (top) -- `option(USE_HIP OFF)`; under USE_HIP skip
   `find_package(CUDA)`, add the hip_compat dir + ROCm include + `-DUSE_HIP
   -D__HIP_PLATFORM_AMD__` for the host CXX files.
4. `Core/CMakeLists.txt` -- under USE_HIP: `enable_language(HIP)`, default
   `CMAKE_HIP_ARCHITECTURES=gfx90a` only-when-unset (followers pass -D, no edit), mark
   ONLY the `.cu` (filtered out of the glob, NOT the `.cuh` -- see gotcha) LANGUAGE HIP,
   add them straight to `efusion`, set `HIP_ARCHITECTURES`, force-include the compat
   header. The legacy FindCUDA / CUDA_COMPILE / CudaDetect / CudaComputeTargetFlags flow
   is preserved verbatim in the `else()`.
5. `Core/Cuda/operators.cuh` -- USE_HIP-guard the project's `operator+`/`operator-`
   on float3 (HIP's HIP_vector_type<float,3> ships them; ambiguous otherwise). Keep
   `cross`/`dot`/`norm`/`normalized` (HIP has no such named helpers) and
   `operator*(mat33,float3)` (custom type, no collision).
6. `Core/Cuda/reduce.cu` + `cudafuncs.cu` -- `#include "cuda_to_hip.h"` first (so
   CUDA_WARP_FULL_MASK is defined on both backends), and the 42 `__shfl_down_sync(
   0xFFFFFFFF,...)` -> `__shfl_down_sync(CUDA_WARP_FULL_MASK,...)` (the wave64 fix).
7. `Core/ElasticFusion.h` -- guard the UNUSED `#include <pangolin/gl/glcuda.h>` under
   `#if !defined(USE_HIP)` (Pangolin's glcuda.h uses CUDA interop symbols with no 1:1
   HIP equivalent, e.g. cudaGraphicsMapFlagsNone; EF references nothing from it).

### wave64 reduction (the flagged correctness wall) -- FIXED + PROVEN

`warpReduceSum`/`blockReduceSum` (templated over JtJJtrSE3 / JtJJtrSO3 / int2) loop
`offset = warpSize/2 .. 1`, so they are correctly warpSize-parameterized (span the full
64-lane wavefront on CDNA, 32 on RDNA). `shared[32]` is oversized-but-safe (16
wavefronts at 1024 threads / wave64). The block collective uses real `__shfl_down_sync`
+ `__syncthreads()`, so there is NO unsynchronized warp-tail race (unlike the
MPPI-Generic class). The ONE real issue was the lane mask: the CUDA 32-bit `0xFFFFFFFF`
fails HIP's `sizeof(mask)==8` static_assert (compile error) and is not the wave64 full
mask. Fixed arch-unified with `CUDA_WARP_FULL_MASK` = `0xffffffffffffffffULL` on HIP,
`0xFFFFFFFFu` on CUDA. No divergent early-return precedes the reduction, so the full
mask (not `__activemask()`) is correct (all lanes of each warp participate; inactive
shared slots are zero-padded). Re-validated on RDNA wave32 by warpSize=32 making the
same code reduce a 32-lane wave -- followers need no change.

### GPU validation on real gfx90a (HIP_VISIBLE_DEVICES=2) -- PASSED

`rocm_validation/kernel_harness.cpp` (no GL): feeds fixed DeviceArray inputs into the
ported entry points. All checks pass, deterministic across repeated process runs:

- createVMap: vmap == CPU back-projection (exact, all pixels).
- createNMap: 2961 unit-length normals.
- tranformMaps: identity pose bit-stable.
- resizeVMap: correct half-res dims + finite.
- icpStep (SE3 wave64): JtJ symmetric, 6 positive diagonal entries, residual 0 at
  identity, inliers=2961, BIT-IDENTICAL across two runs.
- so3Step (SO3 wave64): inliers=2852, BIT-IDENTICAL across two runs.
- computeRgbResidual (int2 wave64): device count=2773 and sigmaSum=69325 EXACTLY equal a
  CPU recount of the device-written correspondences. Integer adds are order-independent,
  so this is a hard arithmetic proof the wave64 reduction sums all 64 lanes with none
  lost/doubled. Also exercises Sobel `__constant__` + cudaMemcpyToSymbol.
- rgbStep (SE3 photometric wave64): symmetric, BIT-IDENTICAL across two runs.

Plus the full HIP library links: `libefusion.so` built (USE_HIP=ON, gfx90a) including
the 2 .cu (LANGUAGE HIP, `.hip_fatbin` present), the host GL-interop .cpp via the shims,
and all the surfel/fusion C++; `ldd` shows libamdhip64, no CUDA.

Non-GPU regression: USE_HIP=OFF configure proceeds into the unchanged legacy FindCUDA
branch and stops only at `find_package(CUDA)` (no CUDA toolkit on this ROCm host), proving
the port's CMake/source changes are inert on the CUDA path.

### Build (gfx90a)

Deps via apt: `libeigen3-dev libsuitesparse-dev libglew-dev freeglut3-dev libegl1-mesa-dev
libjpeg-dev zlib1g-dev libopenni2-dev`. Pangolin + Sophus are the pinned submodules
(`git submodule update --init third-party/Sophus third-party/Pangolin`). Pangolin v0.5
needs `-DCMAKE_CXX_FLAGS="-include cstdint"` to build on modern GCC (its image_io_jpg.cpp
uses uint8_t without <cstdint>). OpenNI2 is only the live-sensor `Tools/` path, not the
`efusion` library.

```
cmake -S . -B build-hip -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
      -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_BUILD_TYPE=Release
make -C build-hip efusion -j16
```

Followers (gfx1100/gfx1151) reuse the same commit with only
`-DCMAKE_HIP_ARCHITECTURES=<arch>`; no source/CMake edit.

## Gotchas (HIP, gfx90a)

- Define `__CUDACC__` / `EIGEN_NO_CUDA` ONLY under `#if defined(__HIPCC__)`, never merely
  under USE_HIP. The host .cpp (GPUTexture, RGBDOdometry, container layer) are built by
  g++ with -DUSE_HIP and need the cuda*->hip* symbol aliases, but if __CUDACC__ leaks into
  that host compile, (a) Pangolin's headers hide their Eigen declarations via their own
  `#ifndef __CUDACC__` guards -> "Eigen does not name a type", and (b) types.cuh excludes
  its Eigen ctor of mat33 -> "conversion from Eigen::Matrix to mat33 requested" at every
  `mat33 m = eigenMat;`. __HIPCC__ is set only by the HIP compiler on the .cu, which is
  exactly where __CUDACC__ is wanted (kernel_containers.hpp GPU_HOST_DEVICE__,
  types.cuh's Eigen-ctor exclusion). Defining __CUDACC__ is safe here only because EF
  pulls in NO rocThrust (cf. the Open3D/cudf rocThrust-backend caveat).
- A legacy `file(GLOB ... *.cu *.cuh)` + `set_source_files_properties(${glob} LANGUAGE
  HIP)` makes CMake try to COMPILE the `.cuh` headers as standalone HIP TUs (operators.cuh
  references mat33 from types.cuh -> "unknown type name 'mat33'" in isolation). Filter the
  glob to just `*.cu` before tagging LANGUAGE HIP.
- HIP's `__device__ memset`/`memcpy` (amd_device_functions.h) enter unqualified lookup
  once __CUDACC__ is defined, so a host-only helper doing `memset(&desc,0,...)` fails
  ("call to __device__ function from __host__ function") unless `<cstring>` is included
  to keep the global ::memset a viable candidate. Compat header includes <cstring> on HIP.
- HIP ships no toolkit-named `<vector_types.h>`/`<vector_functions.h>`; float3/short2/
  make_float3 come from `<hip/hip_runtime.h>`. The hip_compat shims cover these.
- `__HIP_PLATFORM_AMD__` is NOT defined at the preprocessor stage before a HIP header is
  included, so a compat header that branches on `defined(__HIP_PLATFORM_AMD__)` must also
  accept `USE_HIP` (passed as -D) to activate in TUs that include it before hip_runtime.h.

## Install as a dependency

Not applicable -- ElasticFusion is a leaf application, not a base library consumed by
other MOAT targets.
