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

## Review 2026-05-31 (reviewer, linux-gfx90a)

Reviewed `git diff 85d0f4c...HEAD` (HEAD 09346fd) with the /pr-review skill, ROCm-fault-class aware. Verdict: review-passed (no changes requested). 333 insertions / 4 deletions, additive, CUDA path behind BUILD_WITH_HIP. No defects found; the five load-bearing claims were independently fact-checked against the ROCm 7.2.1 headers on this host (`/opt/rocm/include/hip/hip_runtime_api.h`) and the actual sources.

Verified (no problems):
- Opaque-type bridge identities EXACT vs the authoritative HIP typedefs: `CUstream_st->ihipStream_t`, `CUevent_st->ihipEvent_t`, `CUgraph_st->ihipGraph`, `CUgraphNode_st->hipGraphNode`, `CUgraphExec_st->hipGraphExec` (hip_runtime_api.h:685/716/1449/1453/1457 -> `typedef struct ihipStream_t* hipStream_t` etc.). After the `#define`, a `CUstream_st*` is mangling-identical to `hipStream_t`; no ABI mismatch. Bridge force-included consistently into all three consumers: GPU lib `gtsam_points_cuda` (via cuda_to_hip.h which includes the types bridge at cuda_to_hip.h:41), main lib `gtsam_points` (CMakeLists.txt:266-269), and every test target (CMakeLists.txt:442-448). Headers carry `struct CUstream_st;` forward decls (e.g. gaussian_voxelmap_gpu.hpp:13, cuda_graph_exec.hpp:6-8) which the macro rewrites to the HIP struct uniformly. Tests do not name `CUstream_st` directly but include the public headers and call across it; the identical `-DUSE_HIP` + `-include cuda_to_hip_types.h` on every test makes the mangled symbols agree.
- thrust::cuda=thrust::hip alias guarded on `__HIPCC__` (cuda_to_hip.h:110-115). The rocThrust execution-policy header is pulled ONLY in that guard, so the 3 host `.cpp` in the GPU lib (nonlinear_factor_set_gpu.cpp, nonlinear_factor_set_gpu_create.cpp, integrated_vgicp_factor_gpu.cpp -- CMakeLists.txt:288/289/302, NOT in the LANGUAGE HIP list, compiled CXX) never parse it. No host TU pulls the rocThrust policy header. 12 `thrust::cuda::par[_nosync]` call sites confirmed all in `.cu`.
- cuda_malloc_async.hpp: CUDA_VERSION pinned to 12000 (cuda_to_hip.h:34-36). 12000 >= 11000 so the `#if (CUDA_VERSION < 11000)` downgrade body is inert -> `cudaMallocAsync` keeps mapping to `hipMallocAsync` (stream-ordered), NOT a sync hipMalloc downgrade. 12000 < 13000 so cuda_graph.cu:18-22 takes the 4-arg `hipGraphAddDependencies(graph,&from,&to,1)` (hip_runtime_api.h:8061-8062: 4 params) and cuda_graph_exec.cu:11 the 5-arg `hipGraphInstantiate(&instance,graph,nullptr,nullptr,0)` (hip_runtime_api.h:8217-8218: 5 params). Both arities match ROCm 7.x exactly.
- voxelmap unsigned atomicMax (gaussian_voxelmap_gpu.cu:139) operates on `voxel_intensities`, allocated by `cudaMallocAsync` (gaussian_voxelmap_gpu.cu:221) = hipMallocAsync fine-grained device memory. No `cudaMallocManaged`/`__managed__`/`hipMallocManaged`/`cudaMemAdvise` anywhere in the tree (grep empty), so the cudaKDTree coarse-grained-drop fault class does not apply and no atomicCAS-loop emulation is needed. Claim is correct.
- hipCUB: `#define cub hipcub` lives only in the full cuda_to_hip.h (GPU lib only); absent from the lightweight cuda_to_hip_types.h (so it does not leak into main lib or tests). hip_compat/cub/device/device_{reduce,select}.cuh map the `<cub/...>` include path to `<hipcub/...>` (headers present at /opt/rocm/include/hipcub/device/). No bare `cub` identifier in non-cub:: code to clobber. All toolkit angle-bracket includes are shim-covered: `<cuda.h>` (cuda_graph.cu), `<cusparse.h>` (check_error_cusolver.cu), `<curand.h>`, `<device_atomic_functions.h>`, `<cuda_runtime[_api].h>`, `<cub/device/*>`. `<cuda_gl_interop.h>` has no shim but is included only by gl_buffer_map.cu, which is commented out in CMakeLists.txt:287 (not compiled in either build) -- non-issue.
- BUILD_WITH_HIP CMake arm: CUDA path byte-identical under NOT BUILD_WITH_HIP (all additions gated). nvcc-only flags (`-Xcudafe`, `--expt-relaxed-constexpr`, `-Wno-c99-extensions`, CMAKE_CUDA_FLAGS) confined to the `if(BUILD_WITH_CUDA)` block (CMakeLists.txt:88-106); the HIP arm does not inherit them. Arch read from `${CMAKE_HIP_ARCHITECTURES}` defaulted only when unset (CMakeLists.txt:127-129) -- no hardcoded literal, followers reuse the commit. Links `hip::host` not `hip::device` (CMakeLists.txt). config.cmake.in checks `GTSAM_POINTS_USE_HIP` FIRST then `elseif USE_CUDA` (config:31-35) -- correct, since a HIP build sets both USE_CUDA=1 and USE_HIP=1, so HIP-first is required to call find_dependency(hip). "Install as a dependency" export valid.
- wave64: zero `__global__` kernels, zero warp intrinsics (`__shfl/__ballot/__activemask/__any/__all/warpSize/__syncwarp/__popc/cooperative_groups/tiled_partition` grep empty). GPU surface is entirely rocThrust/hipCUB over `__host__ __device__` Eigen functors. No hand-rolled warp code. No wave64 exposure on the lead; gfx1100/gfx1151 followers inherit zero warp-size risk from the source (their risk is library availability + RDNA, validated on their own HW).
- Commit hygiene: title `[ROCm] Port gtsam_points GPU library to HIP (gfx90a)` = 52 chars (<=72), `[ROCm]` prefix, mentions Claude, no Co-Authored-By noreply trailer, ASCII (uses `--`), Test Plan with fenced commands. `master` == base 85d0f4c (clean upstream mirror, no port commits). All work under jeffdaily; no AMD-internal account references.

Note for the validator (carried from the porter, not a review blocker): the CUDA Graph path (cuda_graph.cu / cuda_graph_exec.cu) compiles but is not exercised by the GPU gate (VGICP drives streams + hipCUB directly, not graph capture) -- compile-only at validation time. The GTSAM 4.3a0 source-build dependency is a build-env matter (documented above), not a code issue.

## Validation 2026-05-31 (validator, linux-gfx90a)

Device: AMD Instinct MI250X gfx90a, ROCm 7.2.53211 (HIP 7.2.53211.e1a6bc5663, Direct Dispatch: 1), HIP_VISIBLE_DEVICES=3.

GTSAM dep: reused source-built borglab/gtsam @ 4.3a0 at /var/lib/jenkins/moat/_deps/gtsam/install (libgtsam.so.4.3a0, libgtsam_unstable.so.4.3a0, libmetis-gtsam.so, libcephes-gtsam.so all present).

Build: reused intact build_hip/ at HEAD 09346fdeaa9e179e45ba23c8264356ab59884e50 (ninja: no work to do).

Commands:
```
export LD_LIBRARY_PATH=/var/lib/jenkins/moat/_deps/gtsam/install/lib:/opt/rocm/lib
export HIP_VISIBLE_DEVICES=3
# compile check (no-op, build was current)
utils/timeit.sh gtsam_points compile -- ninja -C /var/lib/jenkins/moat/projects/gtsam_points/src/build_hip -j 16
# run 1
utils/timeit.sh gtsam_points test -- ctest --test-dir /var/lib/jenkins/moat/projects/gtsam_points/src/build_hip --output-on-failure -j1
# run 2 (determinism)
ctest --test-dir /var/lib/jenkins/moat/projects/gtsam_points/src/build_hip --output-on-failure -j1
```

Results (both runs): 87/87 passed, 0 failed, ~39 s total.

GPU gate results:
- test_matching_cost_factors VGICP_CUDA_NONE, VGICP_CUDA_OMP: PASSED both runs. Convergence EXPECT_LT(rot, 0.015 rad) and EXPECT_LT(trans, 0.15 m) held across FORWARD/BACKWARD/UNARY/MULTI_FRAME (no EXPECT failures in output). Porter-measured: rot 0.0004-0.0025 rad, trans 0.008-0.050 m -- reproduced within tolerance on both runs.
- test_voxelmap VoxelMapGPU, VoxelMapGPU_Intensity (atomicMax on hipMallocAsync fine-grained memory -- intensity path accumulates correctly, no coarse-grained drop), VoxelMapGPU_IO: all PASSED.
- test_types TestPointCloudGPU: PASSED.
- Device dispatch confirmed: AMD_LOG_LEVEL=3 shows ShaderName records for gtsam_points kernels (voxel_coord_kernel, voxel_bucket_assignment_kernel, accumulate_points_kernel, finalize_voxels_kernel via rocThrust) on gfx90a GCD 3. All hipPeekAtLastError/hipGetLastError calls returned hipSuccess.

Non-GPU suite: test_alignment, test_bundle_adjustment, test_colored_gicp, test_compact_mahalanobis, test_continuous_time, test_continuous_trajectory, test_global_registration, test_headers, test_kdtree, test_loam_factors, test_voxel_raycaster -- all PASSED, no regression.

CUDA Graph path (cuda_graph.cu / cuda_graph_exec.cu): compile-only at this validation, as noted by porter. Not exercised by the GPU gate.

State: linux-gfx90a -> completed (validated_sha: 09346fdeaa9e179e45ba23c8264356ab59884e50). Followers linux-gfx1100 and windows-gfx1151 unblocked to port-ready.

## Validation 2026-06-01 (gfx1100, ROCm 7.2.1)

Device: AMD Radeon Pro W7800 48GB gfx1100 (RDNA3, wave32), ROCm 7.2.1 (HIP 7.2.53211-e1a6bc5663), HIP_VISIBLE_DEVICES=0.

GTSAM dep: built borglab/gtsam @ 4.3a0 from source into /var/lib/jenkins/moat/_deps/gtsam/install. Libraries present: libgtsam.so.4.3a0, libgtsam_unstable.so.4.3a0, libmetis-gtsam.so, libcephes-gtsam.so. CMake configure of gtsam 4.3a0:
```
cmake -S /var/lib/jenkins/moat/_deps/gtsam/src -B /var/lib/jenkins/moat/_deps/gtsam/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/var/lib/jenkins/moat/_deps/gtsam/install \
  -DGTSAM_BUILD_UNSTABLE=ON \
  -DGTSAM_BUILD_TESTS=OFF -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
  -DGTSAM_BUILD_DOCS=OFF -DGTSAM_BUILD_PYTHON=OFF \
  -DGTSAM_INSTALL_MATLAB_TOOLBOX=OFF \
  -DGTSAM_WITH_TBB=OFF -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF \
  -DGTSAM_USE_SYSTEM_EIGEN=ON
cmake --build /var/lib/jenkins/moat/_deps/gtsam/build -j 16 && cmake --install /var/lib/jenkins/moat/_deps/gtsam/build
```

Build (gfx1100): fresh clone of jeffdaily/gtsam_points@09346fd (moat-port) into projects/gtsam_points/src, configured with gfx1100 only:
```
cmake -S /var/lib/jenkins/moat/projects/gtsam_points/src \
  -B /var/lib/jenkins/moat/projects/gtsam_points/src/build_hip \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DBUILD_WITH_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=/var/lib/jenkins/moat/_deps/gtsam/install \
  -DBUILD_WITH_OPENMP=ON -DBUILD_WITH_TBB=OFF -DBUILD_TESTS=ON \
  -DBUILD_DEMO=OFF -DBUILD_EXAMPLE=OFF -DBUILD_TOOLS=OFF
export LD_LIBRARY_PATH=/var/lib/jenkins/moat/_deps/gtsam/install/lib:/opt/rocm/lib
utils/timeit.sh gtsam_points compile -- ninja -C /var/lib/jenkins/moat/projects/gtsam_points/src/build_hip -j 16
```
Build: 114 targets, all succeeded (timeit: 89 s).

Code-object verification (roc-obj-ls libgtsam_points_cuda.so): 6 ELF bundles, all hipv4-amdgcn-amd-amdhsa--gfx1100. Zero gfx90a objects. No source change required; -DCMAKE_HIP_ARCHITECTURES=gfx1100 was sufficient (CMakeLists.txt reads ${CMAKE_HIP_ARCHITECTURES}).

Commands:
```
export LD_LIBRARY_PATH=/var/lib/jenkins/moat/_deps/gtsam/install/lib:/opt/rocm/lib
export HIP_VISIBLE_DEVICES=0
# run 1
utils/timeit.sh gtsam_points test -- ctest --test-dir /var/lib/jenkins/moat/projects/gtsam_points/src/build_hip --output-on-failure -j1
# run 2 (determinism)
ctest --test-dir /var/lib/jenkins/moat/projects/gtsam_points/src/build_hip --output-on-failure -j1
```

Results (both runs): 87/87 passed, 0 failed, ~30 s total. Matches gfx90a 100%/0-failed out of 87.

GPU gate results:
- test_matching_cost_factors VGICP_CUDA_NONE, VGICP_CUDA_OMP: PASSED both runs. VGICP_CUDA_TBB self-skips (BUILD_WITH_TBB=OFF, same as gfx90a). IntegratedVGICPFactorGPU converged across FORWARD/BACKWARD/UNARY/MULTI_FRAME with no EXPECT_LT failures (rot < 0.015 rad, trans < 0.15 m all held). Porter-measured range on gfx90a: rot 0.0004-0.0025 rad, trans 0.008-0.050 m; gfx1100 results within same coarse tolerance range (no ULP-level divergence visible at this tolerance).
- test_voxelmap VoxelMapGPU, VoxelMapGPU_Intensity (atomicMax on hipMallocAsync fine-grained device memory -- correct on gfx1100, no coarse-grained drop), VoxelMapGPU_IO: all PASSED.
- test_types TestPointCloudGPU: PASSED.
- No NaN/HIP fault, clean exit on both runs.

Non-GPU suite: test_alignment, test_bundle_adjustment, test_colored_gicp, test_compact_mahalanobis, test_continuous_time, test_continuous_trajectory, test_global_registration, test_headers, test_kdtree, test_loam_factors, test_voxel_raycaster -- all PASSED, no regression vs gfx90a.

Determinism: both runs 87/87, timing within 1%. Two VGICP_CUDA runs agree to tolerance.

CUDA Graph path (cuda_graph.cu / cuda_graph_exec.cu): compile-only at this validation, consistent with gfx90a (VGICP drives streams + hipCUB directly, not graph capture).

No fork interaction. No CI change. No source change needed; same commit 09346fd validated unchanged on gfx1100 with only -DCMAKE_HIP_ARCHITECTURES=gfx1100 at configure time.

State: linux-gfx1100 -> completed (validated_sha: 09346fdeaa9e179e45ba23c8264356ab59884e50).
