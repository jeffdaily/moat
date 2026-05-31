# gtsam_points -- ROCm/HIP port notes

Fork: https://github.com/jeffdaily/gtsam_points (branch `moat-port`, master is a clean upstream mirror).
Lead: linux-gfx90a (MI250X, ROCm 7.2.1). Strategy A (colmap/cupoch model): compat header + `LANGUAGE HIP`, CUDA path byte-identical behind `BUILD_WITH_HIP`.

## Dependency: GTSAM (resolved via source build of 4.3a0)

The build hard-requires `find_package(GTSAM 4.2)` AND `find_package(GTSAM_UNSTABLE 4.2)` and links both `gtsam` and `gtsam_unstable`.

- Ubuntu 24.04 `apt install libgtsam-dev` (4.2.0+dfsg) ships ONLY `libgtsam.so` + a `GTSAMConfig.cmake`. It has NO `gtsam_unstable` lib, NO `gtsam_unstable` headers, and NO `GTSAM_UNSTABLEConfig.cmake`. There is no separate `libgtsam-unstable-dev` apt package. So the apt package CANNOT satisfy `find_package(GTSAM_UNSTABLE REQUIRED)` -- it fails at configure. (apt does not work for this project.)
- Resolution: build GTSAM 4.3a0 from source (fast on this host). `GTSAM_BUILD_UNSTABLE=ON` is the default, so `libgtsam_unstable.so` + `GTSAM_UNSTABLEConfig.cmake` are produced. Use the system Eigen 3.4 (`GTSAM_USE_SYSTEM_EIGEN=ON`) so GTSAM and gtsam_points share one Eigen ABI.

GTSAM source build (one-time, into a local prefix):
```
git clone --depth 1 --branch 4.3a0 https://github.com/borglab/gtsam.git <gtsam-src>
cmake -S <gtsam-src> -B <gtsam-build> -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=<gtsam-install> \
  -DGTSAM_BUILD_UNSTABLE=ON \
  -DGTSAM_BUILD_TESTS=OFF -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
  -DGTSAM_BUILD_DOCS=OFF -DGTSAM_BUILD_PYTHON=OFF \
  -DGTSAM_INSTALL_MATLAB_TOOLBOX=OFF \
  -DGTSAM_WITH_TBB=OFF -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF \
  -DGTSAM_USE_SYSTEM_EIGEN=ON
cmake --build <gtsam-build> -j 16 && cmake --install <gtsam-build>
```
GTSAM_VERSION ends up 4.3, so gtsam_points selects the DEFAULT ISAM2 sources (`GTSAM_POINTS_ISAM2_SRC_DIR` empty), not the `gtsam4.2/` ones.

Eigen 3.4 and Boost 1.83 are already present on the host (system packages).

## Build (gfx90a), with the source-built GTSAM

In projects/gtsam_points/src:
```
cmake -S . -B build_hip -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DBUILD_WITH_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=<gtsam-install> \
  -DBUILD_WITH_OPENMP=ON -DBUILD_WITH_TBB=OFF -DBUILD_TESTS=ON \
  -DBUILD_DEMO=OFF -DBUILD_EXAMPLE=OFF -DBUILD_TOOLS=OFF
cmake --build build_hip -j 16
```
Followers (gfx1100/gfx1151) reuse the SAME commit with only `-DCMAKE_HIP_ARCHITECTURES=<arch>` (the target reads `${CMAKE_HIP_ARCHITECTURES}`; no source change).

### Runtime / test gotcha: source-built GTSAM .so are not on the loader path

The source-built GTSAM (and its bundled `libmetis-gtsam.so`, `libcephes-gtsam.so`) live in `<gtsam-install>/lib`, which is not in the default loader path. The test binaries' RUNPATH covers their DIRECT deps but NOT the transitive `libmetis-gtsam.so` (RUNPATH, unlike RPATH, is not used for transitive deps). So `gtest_discover_tests` (which runs each binary at build time to enumerate tests) and ctest both need:
```
export LD_LIBRARY_PATH=<gtsam-install>/lib:/opt/rocm/lib
```
This is purely an environment concern of the source-built GTSAM; it is NOT baked into the fork's CMake. With a system GTSAM it would not arise.

## What the port does (Strategy A)

The only files that know HIP:
- `include/gtsam_points/cuda/cuda_to_hip.h` -- force-included into the GPU library (`gtsam_points_cuda`); aliases the cuda* runtime symbols to hip*, `#define cub hipcub`, pins `CUDA_VERSION` into [11000,13000), aliases `namespace thrust::cuda = thrust::hip` (guarded on `__HIPCC__`), maps `CURAND_STATUS_*` -> `HIPRAND_STATUS_*`.
- `include/gtsam_points/cuda/cuda_to_hip_types.h` -- the opaque-type bridge ONLY (forward-decls + `#define CUstream_st ihipStream_t` etc.); force-included into BOTH the main `gtsam_points` lib AND every test target (they call the GPU API across `CUstream_st*`).
- `hip_compat/` -- forwarding shims for `<cuda.h>`, `<cuda_runtime.h>`, `<cuda_runtime_api.h>`, `<device_atomic_functions.h>`, `<curand.h>`, `<cusparse.h>`, and `<cub/device/device_{reduce,select}.cuh>` -> hipCUB. On the HIP include path PRIVATE+BEFORE on `gtsam_points_cuda` only; absent on the CUDA build.

CMake: a `BUILD_WITH_HIP` option mirrors `BUILD_WITH_CUDA`; `enable_language(HIP)`, `find_package(hip)`, the same `gtsam_points_cuda` source list with `.cu` marked `LANGUAGE HIP`, link `hip::host` (NOT `hip::device`), arch from `CMAKE_HIP_ARCHITECTURES`. `GTSAM_POINTS_USE_CUDA` stays defined for HIP so all existing test/host `#ifdef GTSAM_POINTS_USE_CUDA` guards compile unchanged.

### Key correctness/build points (full detail in the commit body)

- Opaque-type bridge MUST be consistent across all 3 consumers (GPU lib, main lib, tests) or the link fails (`CUstream_st*` vs `ihipStream_t*` name-mangling mismatch on `overlap_gpu` / `merge_frames_gpu` / `OffloadableGPU::touch`).
- `thrust::cuda::par[_nosync]` (~12 sites) does not exist on the HIP backend; namespace alias `thrust::cuda = thrust::hip`. Must be guarded on `__HIPCC__`: only the `-x hip` .cu can parse the rocThrust execution-policy header; the GPU lib's host .cpp (compiled by g++) cannot, and do not use `thrust::cuda::par`.
- `cuda_malloc_async.hpp` CUDA_VERSION-undefined trap: pin CUDA_VERSION so `hipMallocAsync` (stream-ordered, fine-grained device memory) is used, not a downgrade to sync hipMalloc.
- The unsigned `atomicMax` on voxel intensity (GaussianVoxelMapGPU::insert) is the cudaKDTree "atomicMax dropped on coarse-grained memory" fault class, but the buffer is fine-grained `hipMallocAsync` device memory where it is correct -- VERIFIED on GPU (test_voxelmap VoxelMapGPU_Intensity passes). No atomicCAS-loop emulation needed.
- The GPU lib's 3 `.cpp` (nonlinear_factor_set_gpu*, integrated_vgicp_factor_gpu) are host-only orchestration (streams + async alloc, no kernels, no thrust) -- kept CXX, not LANGUAGE HIP.

## Validation (gfx90a, ROCm 7.2.1)

Full suite, serial on one GCD: `HIP_VISIBLE_DEVICES=3 ctest --test-dir build_hip --output-on-failure -j1` -> 100% passed, 0 failed out of 87.
GPU gates:
- test_matching_cost_factors VGICP_CUDA_NONE/_OMP: IntegratedVGICPFactorGPU converges across FORWARD/BACKWARD/UNARY/MULTI_FRAME. Measured rot error 0.0004-0.0025 rad (<< 0.015), trans error 0.008-0.050 m (<< 0.15).
- test_voxelmap VoxelMapGPU / VoxelMapGPU_Intensity / VoxelMapGPU_IO.
- test_types TestPointCloudGPU.
- Device dispatch confirmed (AMD_LOG_LEVEL=3, ~2856 dispatch records / VGICP_CUDA case); two runs agree to tolerance.
Non-GPU suite unchanged (alignment, kdtree, loam, bundle adjustment, colored/compact GICP, continuous time/trajectory, global registration, voxel raycaster, headers, CPU ICP/GICP/VGICP).

The CUDA Graph path (cuda_graph.cu / cuda_graph_exec.cu) COMPILES (4-arg hipGraphAddDependencies, 5-arg hipGraphInstantiate for ROCm 7.x) but is not exercised by the GPU gate -- the VGICP factor drives streams + hipCUB directly, not graph capture. Flagged for the validator: compile-only at validation time.

## Install as a dependency

gtsam_points is a base-ish library (GTSAM factors + GPU VGICP/voxelmap). A dependent MOAT project consumes the ROCm build like this.

1. Clone + build the ported lib (needs the source-built GTSAM 4.3a0 above on CMAKE_PREFIX_PATH):
```
git clone -b moat-port https://github.com/jeffdaily/gtsam_points _deps/gtsam_points/src
cmake -S _deps/gtsam_points/src -B _deps/gtsam_points/build -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DBUILD_WITH_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=<arch> \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=<gtsam-install> \
  -DCMAKE_INSTALL_PREFIX=/var/lib/jenkins/moat/_deps/gtsam_points/install \
  -DBUILD_WITH_OPENMP=ON -DBUILD_WITH_TBB=OFF \
  -DBUILD_TESTS=OFF -DBUILD_DEMO=OFF -DBUILD_EXAMPLE=OFF -DBUILD_TOOLS=OFF
cmake --build _deps/gtsam_points/build -j 16 --target install
```
2. Consume it: `find_package(gtsam_points REQUIRED)` + `target_link_libraries(<tgt> gtsam_points::gtsam_points)`, with `-DCMAKE_PREFIX_PATH=/var/lib/jenkins/moat/_deps/gtsam_points/install;<gtsam-install>`. The installed package config gains a HIP branch (`find_dependency(hip)` when `GTSAM_POINTS_USE_HIP`), so the exported `gtsam_points_cuda` target (which links `hip::host`) resolves on a ROCm-only host. The GPU library target is `gtsam_points::gtsam_points_cuda`; the public stream type in installed headers is `CUstream_st*`, which a HIP consumer must treat as `hipStream_t` (force-include `gtsam_points/cuda/cuda_to_hip_types.h` with `-DUSE_HIP`, exactly as this project's main lib/tests do).
