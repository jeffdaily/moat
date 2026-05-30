# cupoch notes (ROCm/HIP port)

Upstream: neka-nat/cupoch @ bd6cec71d788060d06e4ce8fae903c8bc217d434
Lead platform: linux-gfx90a (MI250X, CDNA2, wave64), ROCm 7.2.1, CMake 3.31.6.

## Existing AMD support

None. `grep -riE 'hip|rocm|gfx|amdgpu'` over src/cmake/CMakeLists = 0 hits. Fresh
CUDA -> HIP port. Strategy A (pure CMake, not a torch extension).

## CUDA surface

- Thrust-dominated (94 files: transform / sort_by_key / reduce_by_key / scan /
  gather / async copy). rocThrust is a header drop-in -- no Thrust source swap.
- NO cuBLAS / cuSOLVER / cuSPARSE / cuRAND / CUB anywhere -> no math-library compat
  header needed. Dense solves go through Eigen (host + device headers).
- Small raw CUDA runtime surface (utility/platform.cu + a cudaSafeCall macro):
  cudaStream*, cudaGetDevice/SetDevice/GetDeviceCount/GetDeviceProperties,
  cudaDeviceSynchronize, cudaMemcpy(+kinds), cudaMallocHost/FreeHost,
  cudaGetLastError/GetErrorString. All 1:1 hip* spellings.
- Custom __global__ kernels in only a few files (knn/bruteforce_nn, knn/lbvh_knn,
  geometry/distancetransform, integration/scalable_tsdfvolume, registration/
  permutohedral) + flann's nearestKernel. NO __shfl/__ballot/__activemask/
  warpSize/atomicMin/atomicMax on the core path -> wave64 risk did not
  materialize (no warp-collective or warp-width-dependent code was hit).
- CUDA-OpenGL interop (cudaGraphicsGLRegisterBuffer ...) only in visualization
  (deferred) and one vestigial include in utility/platform.cu (guarded out).

## Scope achieved: VALIDATED CORE

Built + GPU-validated under HIP (gfx90a):
- cupoch_utility, cupoch_camera, cupoch_knn, cupoch_geometry
- third-party GPU deps: stdgpu (HIP backend), flann CUDA kdtree, jsoncpp
- geometry built minus 4 deferred files (below)

Deferred (documented, NOT on the validated path; clang/HIP stricter than nvcc or
OpenGL/codec system deps absent on the headless host):
- Full third_party graph + examples + the pybind11 Python module: need OpenGL dev
  libs (GL/glfw/glew all MISSING on this host) for visualization, plus libSGM
  (stereo), imgui, urdfdom. The Python module links every module incl.
  visualization, so it cannot build here.
- Modules not built: visualization (CUDA-GL interop), imageproc/libSGM, io (beyond
  what geometry needs), integration, odometry, collision, planning, kinematics,
  kinfu.
- geometry/distancetransform.cu: __shared__ array of DistanceVoxel (non-trivially
  constructible -> clang "initialization is not supported for __shared__
  variables").
- geometry/{trianglemesh,occupancygrid,voxelgrid_factory}.cu: pull in
  intersection_test.inl / distance_test.inl whose function DEFINITIONS drop the
  __host__ __device__ that their header DECLARATIONS carry; clang errors
  ("__host__ function cannot overload __host__ __device__ function"). Fix is the
  cudaKDTree attribute-matching lesson (add __host__ __device__ to the ~12 .inl
  defs) but it is off the validated path; left for a follow-up.
- RMM allocator path (USE_RMM): RMM is CUDA-only; build with USE_RMM=OFF, which
  falls back to thrust::device_vector (zero source edit on that switch).

## Build (lead, gfx90a)

    cmake <src> -DUSE_HIP=ON -DCUPOCH_CORE_ONLY=ON -DUSE_RMM=OFF \
        -DBUILD_UNIT_TESTS=OFF -DBUILD_PYTHON_MODULE=OFF -DBUILD_PYBIND11=OFF \
        -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_BUILD_TYPE=Release
    cmake --build . --target cupoch_geometry cupoch_knn cupoch_camera cupoch_utility -j$(nproc)

Clean from-scratch build verified: cfg exit 0, build exit 0, 0 errors, all 7 libs
(cupoch_{utility,camera,knn,geometry} + stdgpu + flann_cuda_s + jsoncpp).

Followers (gfx1100/gfx1151) reuse the same source with only
`-DCMAKE_HIP_ARCHITECTURES=<arch>` (the arch is never hardcoded; see CMakeLists
USE_HIP block + cupoch_core_3rdparty.cmake). A revalidate needs no new commit.

USE_HIP=ON without CUPOCH_CORE_ONLY would wire the same compat header +
cuda_add_library shim into the full module graph; it is gated OFF here because the
GUI/codec system deps are absent on this host, not because of a HIP problem.

## Port design (minimal footprint, CUDA path unchanged, USE_HIP-guarded)

The whole HIP path is behind `option(USE_HIP OFF)` / `#if defined(USE_HIP)`; the
NVIDIA build is byte-identical.

1. Compat header `src/cupoch/utility/cuda_to_hip.h`: the only file that knows HIP;
   aliases the ~20 cuda* runtime symbols cupoch uses to hip* and includes the HIP
   runtime; on NVIDIA a plain `#include <cuda_runtime.h>`. Pulled into platform.h,
   device_vector.h, intersection_test.h, distance_test.h in place of bare CUDA
   includes. Includes <cstring>/<cstdlib> before the HIP runtime (cudaKDTree
   lesson: HIP's __device__ memcpy/memset overloads otherwise shadow libc).

2. Build system. cupoch uses the LEGACY FindCUDA module: top-level
   `find_package(CUDA REQUIRED)` + `enable_language(CUDA)` and every GPU lib via
   `cuda_add_library(...)` (CUDA_NVCC_FLAGS). Strategy A's normal `LANGUAGE HIP`
   recipe assumes modern target-based CUDA. Bridge: under USE_HIP,
   `enable_language(HIP)` and define a `cuda_add_library()` SHIM macro that does
   `add_library()` + marks sources `LANGUAGE HIP`, so every module CMakeLists
   (`cuda_add_library(cupoch_geometry ...)`) is untouched. Arch defaulted only
   when unset (never a literal gfx90a).

3. CUPOCH_CORE_ONLY option + cmake/cupoch_core_3rdparty.cmake: builds only the
   minimal third-party set (stdgpu HIP, flann, jsoncpp) and the 4 core module
   dirs, skipping the full third_party (which needs absent OpenGL/codec libs).

## ROCm gotchas hit + fixes (generalizable ones also in PORTING_GUIDE changelog)

- thrust::cuda::par does NOT exist in rocThrust; it is `thrust::hip::par`. cupoch's
  utility/device_vector.h exec_policy() and flann's device_vector.h both used
  thrust::cuda::par.on(stream). Fixed cupoch's with a USE_HIP guard; for flann
  aliased `namespace thrust { namespace cuda = hip; }` in a prelude. Same for
  thrust::system::cuda::unique_eager_event (async copy) -> thrust::system::hip::
  (down_sample.cu UniformDownSample).

- pinned_allocator: cupoch ships a vendored copy of Thrust's CUDA pinned_allocator
  (utility/pinned_allocator.h) that includes <thrust/system/cuda/error.h> (CUDA-only
  error enums) and defines thrust::cuda::experimental::pinned_allocator. Under HIP
  this drags in undefined cudaSuccess/cudaError* and breaks ANY TU that includes
  device_vector.h (i.e. all of geometry). Fix: under USE_HIP, replace the body with
  `using pinned_allocator = thrust::hip::universal_host_pinned_allocator<T>` (from
  <thrust/system/hip/memory.h>) under the same thrust::cuda::experimental name so
  pinned_host_vector callers are unchanged.

- HIP's float4 is HIP_vector_type<float,4> with an EXPLICIT ctor: `float4_t res =
  {0}` (CUDA plain-struct aggregate init) fails copy-init. Use value-init
  `float4_t res = {}` (kdtree_flann.inl). HIP_vector_type also ships componentwise
  + - * / for BOTH vector-vector AND vector-scalar plus make_floatN, so a CUDA
  helper header full of those (flann's cutil_math.h, ~270 defs) is entirely
  redundant under HIP and only collides ("ambiguous operator", "redefinition of
  max"). Replaced cutil_math.h with a tiny HIP shim providing only the named
  helpers flann actually uses (dot/length/fabs on vectors) on top of HIP's
  operators, selected via a flann-private BEFORE include dir.

- stdgpu 1.3.0 bitrots against ROCm 7 in three ways, all fixable from cupoch's side
  (no submodule edit): (a) its bundled Findthrust.cmake, on seeing THRUST_VERSION
  >= 2.0.0, requires CUDA-only CUB (cub/version.cuh) + libcudacxx (cuda/std/version)
  that rocThrust 2.x does not ship -> override with a HIP-aware Findthrust on
  CMAKE_MODULE_PATH; (b) `find_package(hip 5.1 REQUIRED)` is rejected by ROCm 7's
  SameMajorVersion hip-config (7 != 5) -> a hip version-compat config shim on
  CMAKE_PREFIX_PATH (the real hip::host/device targets already exist from the
  top-level find); (c) stdgpu compiles its generic impl/*.cpp with the CXX compiler,
  but on the HIP backend those include rocThrust -> rocPRIM headers that only the
  HIP (clang) compiler can parse (__builtin_amdgcn_wavefrontsize, clang template
  syntax) -- the stock HIP backend assumes hipcc IS the CXX compiler. Force every
  stdgpu TU to LANGUAGE HIP via set_source_files_properties(... TARGET_DIRECTORY
  stdgpu) with ABSOLUTE source paths (a relative path silently no-ops across
  directories).

- cuda_add_library mixes .cu and .cpp; several cupoch .cpp (utility/helper.cpp via
  helper.h) and stdgpu .cpp include rocThrust/stdgpu headers, so the SHIM marks
  BOTH .cu and .cpp as LANGUAGE HIP (nvcc drove the whole cuda_add_library, mirror
  that). clang compiles plain host C++ unchanged.

- hip::device interface flag leak: linking roc::rocthrust / hip::device propagates
  their INTERFACE_COMPILE_OPTIONS `-x hip --offload-arch=...` to ALL languages of a
  consuming target, so g++ chokes on `--offload-arch` for any host .cpp. Link only
  hip::host (runtime, no -x hip); rocThrust/hipCUB are header-only under
  /opt/rocm/include (default HIP search path), so they need no link target.

- flann/lbvh include CUDA driver/vector headers (<cuda.h>, <cuda_runtime.h>,
  <vector_functions.h>, <vector_types.h>) that ROCm lacks. Per-target PRIVATE+BEFORE
  compat include dirs map each onto the HIP runtime/vector-type header (kept
  target-private so they never shadow <cuda_runtime.h> for rocThrust's own
  internals, which DO need the real one on the CUDA system path). lbvh_index's
  vec_math.h redefines max/min under `#if !defined(__CUDACC__)`; HIP defines
  __HIPCC__ not __CUDACC__, so extend both guards with `|| defined(__HIPCC__)` to
  take the device path and skip the host max/min (HIP provides them).

- clang two-phase / standards strictness: explicit template instantiation
  `template class LineSet<2>;` at file scope under a `using namespace
  cupoch::geometry;` errors ("must occur in namespace geometry"); wrap in the real
  `namespace cupoch { namespace geometry { ... } }`.

## Validation (REAL GPU, gfx90a, HIP_VISIBLE_DEVICES=1)

Harness agent_space/cupoch_validate/validate.cpp (out of git): 20000-point wavy-plane
cloud, fixed seed. AMD_LOG_LEVEL=3 confirms 1836 GPU kernel dispatches + 342 H2D/D2H
memcpy (real device execution, not a CPU fallback).

- VoxelDownSample (pure rocThrust: transform -> sort_by_key -> reduce_by_key):
  709 occupied-voxel centroids match a CPU grid-bin reference (same
  floor((p-min)/voxel) convention) to max centroid error 2.17e-07 (float eps);
  bitwise-identical output across 3 runs (deterministic).
- EstimateNormals (flann CUDA kdtree KNN + covariance kernels): per-point normals
  match a CPU brute-force KNN + 3x3 covariance smallest-eigenvector reference,
  sign-free worst |dot(n_gpu,n_cpu)| = 0.999998 over all 20000 points (0 points
  below 0.99); deterministic across 3 runs (worst |dot| 1.000000).

Both checks PASS -> the core point-cloud GPU path is correct and deterministic on
gfx90a. (Harness CPU reference had a latent bug -- Eigen Vector3f default ctor does
NOT zero, so a std::map accumulator must be explicitly zero-initialized before +=;
fixed in the harness, not a cupoch issue.)

## Validation 2026-05-30 (gfx1100, ROCm 7.2.1)

Platform: AMD Radeon Pro W7800 48GB (gfx1100, RDNA3, wave32), ROCm 7.2.1, HIP_VISIBLE_DEVICES=0.
Fork sha: 8fd480654b900040a1fa03140435916bcbd93d47 (moat-port, unchanged -- no follower code change needed).

### Build

Submodules initialized (stdgpu, eigen, spdlog, dlpack; lbvh/lbvh_index in-tree):

    cmake <src> -DUSE_HIP=ON -DCUPOCH_CORE_ONLY=ON -DUSE_RMM=OFF \
        -DBUILD_UNIT_TESTS=OFF -DBUILD_PYTHON_MODULE=OFF -DBUILD_PYBIND11=OFF \
        -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
        -DCMAKE_BUILD_TYPE=Release
    cmake --build . --target cupoch_geometry cupoch_knn cupoch_camera cupoch_utility -j$(nproc)

Result: 0 errors, 7 libs built: libcupoch_{utility,camera,knn,geometry}.a + libflann_cuda_s.a + libstdgpu.a + libjsoncpp.a. Build time ~132s.

### gfx1100 code-object evidence

    llvm-objdump --offloading libcupoch_geometry.a | grep "Extracting"
    -> all .cu.o entries: "hipv4-amdgcn-amd-amdhsa--gfx1100" (25 objects)
    -> no gfx90a entries

Same confirmed for libcupoch_knn.a (kdtree_flann.cu.o, lbvh_knn.cu.o both gfx1100).

### GPU dispatch confirmation

AMD_LOG_LEVEL=3 on harness (3 GPU runs): 1710 hipLaunchKernel calls + 324 hipMemcpy ops (real device execution, not CPU fallback). All hipLaunchKernel: Returned hipSuccess. GPU agent: amdgcn-amd-amdhsa--gfx1100.

### Validation harness

agent_space/cupoch_validate_gfx1100/validate.cpp (gitignored): 20000-point wavy-plane cloud, fixed seed 42, voxel_size=0.05, KNN=30. Same approach as gfx90a harness.

### VoxelDownSample (rocThrust transform -> sort_by_key -> reduce_by_key)

2804 occupied-voxel centroids (consistent count: gfx90a used 709 because the original harness used a larger voxel; here voxel_size=0.05 on 20000 pts gives 2804).
- GPU vs CPU (floor((p-min)/voxel) convention): max centroid error = 1.33e-07 (well within float eps). PASS.
- Bitwise-identical output across 3 runs (deterministic). PASS.

### EstimateNormals (flann CUDA kdtree KNN + covariance smallest-eigenvector)

- No NaN in 20000 normals. PASS.
- Per-point normals vs CPU brute-force KNN + 3x3 covariance reference (500-point sample): worst sign-free |dot(n_gpu, n_cpu)| = 1.000000, 0 points below 0.99. PASS.
- Deterministic across 3 runs; worst |dot(run1, run2)| over all 20000 points = 0.999999. PASS.

### Result: 6 / 6 PASS. linux-gfx1100 -> completed.

## Deliverable / git

Port lives in projects/cupoch/src (gitignored; parent delivers the fork). Build dirs
in agent_space (out of git). Committed to MOAT: status.json, plan.md, notes.md, and
the PORTING_GUIDE changelog lines.
