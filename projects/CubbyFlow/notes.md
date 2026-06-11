# CubbyFlow notes

## Port complete 2026-06-11 (linux-gfx90a)

Strategy A (compat header + modern CMake HIP language). ROCm 7.2.1, gfx90a.
The earlier "blocked: __CUDA_ARCH__ return-type selection" determination was
wrong; the real fix is below.

### The __CUDA_ARCH__ host/device return-type blocker (resolved)

CUDAArrayBase / CUDAStdVector select between a device accessor (returns T&) and
a host accessor (returns a copy-back wrapper / value) via `#ifdef __CUDA_ARCH__`.
This is NOT impossible on HIP; the prior attempt mis-diagnosed it.

Root cause of the failure: nvcc does NOT parse `__host__` function BODIES during
its device pass, so a host function calling `arr[i]` is never type-checked in the
device pass. clang (HIP) parses ALL bodies in BOTH passes and defers the
cross-space-call diagnostic. With the original `#ifdef __CUDA_ARCH__` the host
overloads are entirely ABSENT in the device pass, so when clang parses a `__host__`
function body during the device pass, `operator[]` resolves to the only visible
(device) overload and errors "call to __device__ function from __host__ function".
Confirmed: the error fires in the DEVICE-ONLY compile, not the host pass.

Fix (CUDAArrayBase.hpp/-Impl, CUDAStdVector.hpp/-Impl): under `__HIP__`, declare
and define BOTH the device and host accessor overloads at once, distinguished by
`__host__`/`__device__` attributes, and let clang resolve by call context in each
pass. CUDA path kept byte-identical with `#elif defined(__CUDA_ARCH__) / #else`.
Pattern used everywhere: `#if defined(__HIP__) || defined(__CUDA_ARCH__)` (device)
paired with `#if defined(__HIP__) || !defined(__CUDA_ARCH__)` (host). The shim
also defines `__CUDA_ARCH__` only in the HIP device pass (guarded by
`__HIP_DEVICE_COMPILE__`) so the remaining intra-body `#ifdef __CUDA_ARCH__` device
selections resolve correctly per pass.

### Other faults fixed

- __CUDACC__ vs __HIPCC__: project gates kernels/attribute macros on `__CUDACC__`
  (hipcc does not define it). Do NOT `#define __CUDACC__` -- rocThrust keys its
  backend on it and would pick the CUDA backend (missing CUB header). Instead
  extend each guard to `defined(__CUDACC__) || defined(__HIPCC__)` (Macros.hpp,
  CUDAArray-Impl.hpp, CUDAAlgorithms.hpp). Also force `-DTHRUST_DEVICE_SYSTEM=5`
  (HIP) belt-and-suspenders.
- -Impl.hpp definitions omitted the `__host__ __device__` their declarations
  carry; nvcc merges, clang requires the match. Added CUBBYFLOW_CUDA_HOST_DEVICE
  to every shared definition in CUDAStdArray-Impl, CUDAArrayBase-Impl,
  CUDAArrayView-Impl, CUDASPHKernels2/3-Impl.
- HIP nodiscard: hipError_t and hipDeviceReset are nodiscard (cudaError_t /
  cudaDeviceReset are not). Bound the result to a typed local in
  _CUBBYFLOW_CUDA_CHECK and `static_cast<void>` the reset; one test discarded
  cudaDeviceSynchronize -> cast to void.
- HIP vector types provide arithmetic/compound/equality operators for floatN, so
  CUDAUtils.hpp's CUDA-side floatN operators are ambiguous -> guard them out with
  `#if !defined(__HIP__)`; kept the named helpers (Dot/Length/To*).
- clang -Werror flags nvcc tolerates: -Wno-class-memaccess is GNU-only (scope to
  C/CXX language); -Wno-reorder-ctor / -Wno-unused-private-field / -Wno-unused-
  variable must come AFTER -Werror (clang honours a later -Wno-* over -Werror),
  so appended as `$<$<COMPILE_LANGUAGE:HIP>:...>` at the end of
  DEFAULT_COMPILE_OPTIONS, not in CMAKE_HIP_FLAGS.
- cuda_runtime.h includes -> `#if defined(__HIP__) #include <Core/CUDA/cuda_to_hip.h>
  #else #include <cuda_runtime.h> #endif` (9 sites).
- CMake: USE_HIP option, enable_language(HIP), CMAKE_HIP_ARCHITECTURES default
  gfx90a, force-include the shim via CMAKE_HIP_FLAGS `-include`, mark the CUDA
  directory's .cu AND its .cpp (they touch device types/thrust) as LANGUAGE HIP.
  Tests/CUDATests and Examples/CUDASPHSim: same .cu->HIP marking; the example's
  source glob was CUDA-only -> now `USE_CUDA OR USE_HIP`. Python bindings stay
  excluded on GPU builds (USE_GPU).

### Build

```
cd projects/CubbyFlow/src
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON
cmake --build build -j$(nproc)
```

`git submodule update --init --recursive` first. CMake wants clang directly for
HIP, NOT the hipcc wrapper (CMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++).

### Validation (gfx90a, real GPU)

- CUDATests: 35/35 cases, 3168/3168 assertions PASS (CUDA array/vector/stdarray,
  particle system data, particle system solver, point hash grid searcher; the
  hash-grid test cross-checks Keys/Start/End/SortedIndices and nearby-point
  callbacks against the CPU searcher).
- UnitTests (CPU regression): 722/722 PASS.
- CUDASPHSim example: runs the full GPU pipeline end-to-end (512 particles), all
  kernels execute, densities verified sane (~350 vs target 1000).
- WCSPH solver with a container set: 5 frames on GPU, all positions finite,
  particles settle to the floor (y-range [0, 0.205]) -- confirms neighbor search,
  density, EOS pressure, viscosity, integration, and boundary collision are all
  numerically correct on ROCm/gfx90a.

### Known non-issue: CUDASPHSim x,y = FLT_MAX in output

The shipped CUDASPHSim example never calls SetContainer(), so the solver's
m_container is the default-empty BoundingBox3F (lowerCorner=+FLT_MAX,
upperCorner=-FLT_MAX, from BoundingBox::Reset). The integration kernels clamp
x.x/x.y to [lower, upper]; with the empty box every finite coordinate is
`< lowerCorner` so it gets set to +FLT_MAX. This is pure CPU-side float math in
unmodified upstream code, identical on CUDA and HIP (the solver .cu files and
Geometry/BoundingBox are untouched by the port) -- NOT a port defect, an upstream
example-config detail. PCISPH's integration clamps only x,y (no z), so z shows
correct free-fall while x,y saturate; setting a container makes all axes finite
and physical (verified above). Not deferred/blocking.

### Files changed

Compat header: Includes/Core/CUDA/cuda_to_hip.h (new).
Host/device overload restructure: CUDAArrayBase.hpp, CUDAArrayBase-Impl.hpp,
CUDAStdVector.hpp, CUDAStdVector-Impl.hpp.
Attribute matching: CUDAStdArray-Impl.hpp, CUDAArrayView-Impl.hpp,
CUDASPHKernels2-Impl.hpp, CUDASPHKernels3-Impl.hpp.
Macros/guards: Includes/Core/Utils/Macros.hpp, CUDAAlgorithms.hpp,
CUDAArray-Impl.hpp, CUDAUtils.hpp, and the 9 cuda_runtime.h include sites.
CMake: CMakeLists.txt, Sources/Core/CMakeLists.txt, Tests/CUDATests/CMakeLists.txt,
Examples/CUDASPHSim/CMakeLists.txt, Builds/CMake/CompileOptions.cmake.
Test cast: Tests/CUDATests/CUDAArray2Tests.cu.
