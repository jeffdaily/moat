# LichtFeld-Studio notes (ROCm/HIP port, lead linux-gfx90a)

## What this port covers (SCOPE -- read first)
LichtFeld-Studio is a full 3DGS workstation (GUI + Vulkan viewer + Python/MCP +
USD + nvjpeg). MOAT ports only the **libtorch-free GPU COMPUTE tranche** and
GPU-validates it; the GUI/Vulkan-interop/Python/MCP/USD/nvImageCodec layers are
explicitly DEFERRED (gated out under USE_HIP). Compute tranche:
- lfs_core / its tensor library (src/core/tensor + src/core/cuda)
- lfs_training + lfs_training_kernels (Adam, schedulers, losses, SSIM, MCMC,
  densification, pruning, bilateral grid, ppisp)
- fastlfs_backend (fastgs fwd/bwd rasterizer + fused Adam)
- gsplat_backend_lfs (vendored gsplat: projection, intersect/tile, rasterize
  fwd/bwd, SH, relocation)
- edge_compute_backend (igs+ edge rasterizer)
- support libs: lfs_geometry, lfs_diagnostics, lfs_logger, lfs_event_bridge

## Environment (gfx90a host)
- 4x MI250X GCD (gfx90a, wave64), ids 0-3. Use a FREE GCD via HIP_VISIBLE_DEVICES
  (check `rocm-smi --showuse`/`--showmemuse`; GPU 3 was free at bringup).
- ROCm 7.2.1, hipcc /opt/rocm/bin/hipcc, clang++ /opt/rocm/llvm/bin/clang++,
  cmake+ninja from conda env py_3.12.
- Deps NOT via vcpkg (VCPKG_ROOT empty). Supplied directly:
  - glm: VENDORED 1.0.1 at /var/lib/jenkins/moat/_deps/glm-1.0.1 (passed via
    -DLFS_GLM_INCLUDE_DIR). The SYSTEM glm is 0.9.9.8, which needs __CUDACC__ to
    emit device qualifiers -- and defining __CUDACC__ breaks rocThrust (see GLM
    fix below). 1.0.1 has a native GLM_COMPILER_HIP path (keys on __HIPCC__) so
    glm:: math compiles in __device__ code under hipcc with NO macro hack.
  - GTest: conda /opt/conda/envs/py_3.12/lib/cmake/GTest.
  - libtorch (test parity oracle): the ROCm python torch (torch 2.13.0a0, hip
    7.2.53211) at /opt/conda/envs/py_3.12/.../torch/share/cmake (Torch_DIR).
  - args.hxx (Taywee, used only by argument_parser.cpp): VENDORED at
    /var/lib/jenkins/moat/_deps/lfs_args (-DLFS_ARGS_INCLUDE_DIR). No apt pkg.
  - apt-installed: libstdc++-14-dev + gcc-14 (C++23 <print> is absent from the
    system libstdc++ 13; clang picks gcc-14 automatically), nlohmann-json3-dev,
    libspdlog-dev, libopenimageio-dev (for image_io.cpp), libopenmesh-dev (for
    mesh_data.cpp). TBB from /usr.
  - OpenImageIO / OpenMesh CMake configs are built BY HAND in HipCompute.cmake
    (find_library + INTERFACE target): the apt -dev OpenImageIOConfig references
    CLI tool binaries (iconvert) the package omits, which trips find_package;
    OpenMesh ships only OpenMeshCore (the leaf CMake asks for OpenMeshCoreStatic).
- rocThrust is a drop-in: <thrust/...> resolves from /opt/rocm/include unchanged
  (no shim). Only cub/cuda*/curand/cooperative_groups/nvtx need shims.

## Build command (lead gfx90a) -- script agent_space/lfs_build.sh
```
cmake -S projects/LichtFeld-Studio/src -B build-hip -G Ninja \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
  -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release \
  -DTorch_DIR=/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/share/cmake/Torch \
  -DLFS_GLM_INCLUDE_DIR=/var/lib/jenkins/moat/_deps/glm-1.0.1 \
  -DLFS_ARGS_INCLUDE_DIR=/var/lib/jenkins/moat/_deps/lfs_args
cmake --build build-hip -j16
```
CMAKE_PREFIX_PATH must include the torch cmake dir + conda GTest + /usr (the
script sets it). Do NOT pass -DCMAKE_HIP_COMPILER (it triggers a reconfigure that
drops -DUSE_HIP); HIP is enabled at project() time (LANGUAGES HIP) and CMake finds
clang at /opt/rocm. For followers: pass -DCMAKE_HIP_ARCHITECTURES=gfx1100/gfx1151
unchanged (no source edit). gfx90a is wave64; followers are wave32.

## How -DUSE_HIP gates the build (CMakeLists.txt + cmake/HipCompute.cmake)
USE_HIP is a plain cache BOOL (NOT option(): option() resets to its default on
the compiler-detection reconfigure and silently falls back to the NVIDIA path).
When USE_HIP, the top-level CMakeLists does `project(... LANGUAGES HIP CXX C)`,
`include(cmake/HipCompute.cmake)`, then `return()` -- the entire NVIDIA
find_package storm + GUI subdirs + main exe + monolithic tests are skipped.
HipCompute.cmake: enable HIP arch (default gfx90a only when unset), force-include
the compat header on every HIP TU, add src/hip_compat + src/core/include +
/opt/rocm/include to the include path, retag every .cu LANGUAGE HIP via a
top-scope add_library/add_executable override, stub CUDA::cudart/curand/
cuda_driver/cupti/nvToolsExt as INTERFACE targets (HIP runtime is implicit;
hiprand linked for RNG), then add_subdirectory the compute libs + cmake/hip_tests.

## Why a dedicated compute-test exe (open question #2 resolved)
The monolithic `lichtfeld_tests` target (tests/CMakeLists.txt) irreducibly links
lfs_mcp, lfs_visualizer, lfs_rendering, lfs_sequencer + nanobind/imgui/implot/
assimp/Vulkan -- the whole GUI/USD/Python graph -- and mixes ~93 compute tests
with ~85 GUI/MCP/Python/USD tests in one exe. It cannot be decoupled without
source surgery, so under USE_HIP we build a NEW `lfs_compute_tests` exe (build-file
addition only) linking just the compute libs + GTest + ROCm libtorch + the compute
test sources. The upstream `lichtfeld_tests`/`lichtfeld_benchmarks` + main app +
GUI find_package storm are gated behind `if(NOT USE_HIP)`.

## Library-coupling caveat (not in original plan)
lfs_core PRIVATE-links OpenImageIO + lfs_geometry; lfs_training links lfs_io
(FFMPEG/WebP/archive/nvimgcodec) + lfs_python_utils. So the compute libs are NOT
cleanly isolated at link time. Under USE_HIP these GUI/io deps are severed (see
the per-CMakeLists USE_HIP gating) and the few host .cpp that use OpenImageIO /
io are excluded or stubbed for the compute build.

## Fault classes hit + fixes (all guarded #if USE_HIP || __HIP_PLATFORM_AMD__ unless noted)

The compat header is src/core/include/core/cuda/cuda_to_hip.h (the ONLY file that
knows HIP); shim headers in src/hip_compat/ resolve the toolkit angle-includes.

1. WAVE SIZE (the only correctness-bearing source edits; arch-unified, correct on
   wave64 AND wave32). A kWarpSize abstraction (64 on CDNA __GFX9__, else 32) +
   64-bit shuffle mask:
   - src/core/tensor/internal/warp_reduce.cuh AND src/training/include/lfs/core/
     warp_reduce.cuh (a vendored duplicate): warp_reduce_{sum,max,min,prod} offset
     16->kWarpSize/2, mask 0xffffffff->64-bit; block_reduce_* lane/warp_id %/ 32
     -> kWarpSize. These recombine per-warp partials via shared memory, so the
     native-64 fold is correct (PORTING_GUIDE AutoDock atomicAdd-combine class).
   - tensor_warp_reduce.cu warp_medium_segment_reduce_* (one segment per warp +
     stride-32 gather): warp_id/lane/warps_per_block AND the gather strides
     (+= 32, base += 32) all -> kWarpSize, so a 64-lane warp owns one segment and
     gathers stride-64. Host WARPS_PER_BLOCK left at /32 (a grid-sizing floor; the
     kernel is a grid-stride loop so any value is correct).
   - tensor_broadcast_ops.cuh broadcast_channel3d_kernel (one pixel per warp +
     stride-32 channel gather): warp_id/lane_id/num_warps + ch_base += 32 ->
     kWarpSize.
   - tensor_debug.cu validate_tensor_kernel (test_nan_inf_gpu_check backend) and
     mcmc_kernels.cu histogram (one index per warp): WARP_SIZE 32 -> kWarpSize,
     __shfl_down_sync mask -> 64-bit.
   - ssim.cu warp_id/lane_id/32 left AS-IS: these are a tid-based 2D loop
     decomposition for shared-tile loading (no shfl/ballot/syncwarp), recombined
     by block.sync(); provably correct on wave64. tensor_strided_ops.cu:570
     WARP_SIZE=32 is a host coalescing HEURISTIC (iteration-order choice), not lane
     math; left as-is.
2. cg::reduce: ROCm CG has none. gsplat Utils.cuh warpSum/warpMax (GSPLAT_WARP_SUM/
   MAX macro -> butterfly warp.shfl_xor all-reduce on HIP, cg::reduce on CUDA) and
   RasterizeToPixelsFromWorld3DGSBwd warp_bin_final (int-max butterfly). The
   thread_block_tile<32> is wave-agnostic (gsplat MOAT-port lesson). ProjectionUT
   uses "optional" only in comments -- NO cuda::std, so NO libhipcxx needed.
3. GLM in device code: vendored GLM 1.0.1 (native HIP path), so the compat header
   deliberately does NOT define __CUDACC__/CUDA_VERSION/GLM_FORCE_CUDA (defining
   __CUDACC__ makes rocThrust take its broken CUDA-system path:
   <cub/detail/detect_cuda_runtime.cuh> not found).
4. float3 operator ambiguity: fastgs + edge_compute helper_math.h -- HIP's
   HIP_vector_type provides all float2/3/4 arithmetic operators, so the hand-rolled
   operator overloads (negate/+/-/*//, ~lines 218-854) AND the host scalar int
   min/max are skipped on HIP (using std::min/std::max brought in for the vector
   helpers' unqualified calls). The named helpers (dot/cross/clamp/lerp/...) and
   make_*(1-arg) stay -- HIP lacks those. Host fminf/fmaxf/rsqrtf kept (guarded on
   the device-compile macro, not USE_HIP, so the HIP HOST pass still gets them).
5. __ballot_sync 64-bit mask: fastgs + edge kernels_forward.cuh
   `__ballot_sync(0xffffffffu, active) == 0` -> LFS_BALLOT_MASK (64-bit on HIP).
   The "any lane active" early-exit is wave-agnostic.
6. CUB: cub/* shim -> hipcub + `namespace cub = hipcub`. IntersectTile
   DeviceRadixSort::SortPairs(begin_bit=0) + DeviceScan::InclusiveSum work on
   hipCUB (the nonzero-begin_bit hipCUB bug does NOT apply -- begin_bit=0).
   hipCUB's DeviceSegmentedReduce::Reduce unifies the begin/end offset iterators
   into ONE OffsetIteratorT (tensor_ops.cu: use a single shared lambda).
7. thrust::cuda -> thrust::hip alias (rocThrust execution policy namespace) in the
   5 thrust-using .cu (tensor_ops/masking/warp_reduce, selection_ops, mcmc_kernels,
   + tensor_generic_ops.cuh). par/par_nosync resolve unchanged.
8. __CUDA_ARCH__ defined only in the HIP DEVICE pass (compat header, guarded by
   __HIP_DEVICE_COMPILE__) so the device-intrinsic guards (powf/fmodf/__ldcs) take
   the device branch; host pass + rocThrust unaffected. The `#ifdef __CUDACC__`
   guards gating device template code (tensor_ops.hpp, tensor_broadcast_ops.cuh,
   tensor_functors.hpp, logger.hpp) changed to `|| defined(__HIPCC__)`.
9. CUdeviceptr is integer on CUDA, void* on HIP (memory_arena.cu VMM): pointer
   arithmetic on the virtual base goes through LFS_DPTR_OFF (byte pointer on HIP).
   CUDA driver VMM symbols (cuMemCreate/Map/SetAccess/..., CUmem*) aliased to hip*.
10. packed128.cuh __ldcs/__stcs/__stcg on int4 -> plain load/store on HIP (HIP has
    these only for a few scalar types; the cache hint is perf-only). __nv_bfloat16
    -> __hip_bfloat16 alias.
11. __half conversion ambiguity (tensor.cpp): HIP __half has operator=(float) AND
    (double), so int/uint8 -> __half is ambiguous. Extracted the per-element CPU
    convert into a template helper (convert_one<F,T>) so `if constexpr` discards
    dead branches; __half routes through float. (Portable, not USE_HIP-guarded.)
12. cudaFuncSetAttribute(kernel,...) -> (const void*)kernel cast (gsplat 2 sites;
    portable). NVTX nvtxRangePush/Pop -> no-ops on HIP.
13. libtorch (ROCm) test oracle: c10/cuda headers reference stream-capture/graph/
    IPC cuda* symbols literally (torch hipifies them only at CUDAExtension build
    time); aliased in the compat header's "libtorch interop" block + the test exe
    defines C10_CUDA_NO_CMAKE_CONFIGURE_FILE and USE_ROCM (torch's documented AMD
    escape hatch for the unported cuda_cmake_macros.h).

## DEFERRED (gated out under USE_HIP, NOT failures)
- GUI / Vulkan viewer (src/visualizer), CUDA<->Vulkan external-memory interop
  (src/rendering/cuda_vulkan_interop.*), Python/nanobind (src/python,
  lfs_python_utils), MCP server (src/mcp), USD/assimp import, nvImageCodec/nvjpeg
  GPU image decode (external/nvImageCodec, lfs_io's nvcodec path), FFmpeg video
  (lfs_video). The top-level CMake never configures these under USE_HIP.
- lfs_io (PLY/SOG/SPZ/USD/colmap loaders + FFmpeg/WebP/archive/nvjpeg): NOT built;
  lfs_training's link to it is severed under USE_HIP and the io-coupled training
  orchestration (trainer.cpp, training_setup.cpp, checkpoint.cpp, metrics.cpp,
  strategies/mrnf.cpp, improved_gs_plus.cpp, strategy_factory.cpp,
  control/command_api.cpp -> all pull dataset.hpp/io) is dropped from the compute
  lfs_training. The compute gtests drive kernels/optimizer/rasterizers directly.
- src/core/cuda/exportable_storage.cpp (CUDA-IPC cross-process memory export):
  excluded (HIP has no handle-type-supported device attribute); not used by the
  compute kernels.
- Headless training smoke (--headless on a COLMAP scene): NOT run -- the trainer/
  dataloader/io tranche is deferred, so the headless entrypoint is not built in
  the compute configuration. The compute gtest set (incl. rasterizer fwd/bwd +
  Adam + losses against the libtorch oracle) is the GPU correctness gate here.

## VALIDATION (gfx90a / MI250X, ROCm 7.2.1, HIP_VISIBLE_DEVICES=0, serial)
Fork tip validated: 580e0012 (moat-port).
Built: all 9 compute libs (lfs_core, lfs_core_cuda, lfs_tensor_kernels,
lfs_geometry, lfs_diagnostics, lfs_training, gsplat_backend_lfs, fastlfs_backend,
edge_compute_backend) + lfs_compute_tests, clean.

lfs_compute_tests (876 GPU-compute gtests, single-process => serial; ROCm
libtorch parity oracle): **874 passed, 2 failed**, BIT-IDENTICAL across two runs
(determinism confirmed; 12.6 s). The wave64-critical subset -- 307 tests covering
tensor reductions (the offset-16/kWarpSize fix), warp/block reductions,
tensor-vs-torch + torch-comparison parity, gsplat + fastgs rasterizers (cg::reduce
shim + ballot), SSIM, fused L1+SSIM, losses, MCMC tensor/memory ops, sort, matrix,
random/curand -- ALL pass (only the MCMC quantization-design test below appears in
that filter).

The 2 failures are DETERMINISTIC and NOT wave64 / port-correctness defects (they
fail identically on CUDA by construction):
- MCMCTest.RemoveGaussiansSoftDeletesRows: AdamParamState.exp_avg is a uint8
  QUANTIZED moment buffer with QUANTIZED_MOMENT_ZERO_POINT=128 (adam_optimizer:32).
  Soft-delete zeroes the per-primitive SCALE (dequant -> 0), not the raw quant
  buffer. The test reads the RAW uint8 buffer (.to_vector() -> 128 = zero-point =
  dequantized 0.0) and asserts == 0.0f, which is incompatible with the quantized
  representation (128, not 0, is "zero"). Pure integer quantization, identical on
  any GPU. A test/impl design mismatch, not HIP.
- ImageKernelsTest.FusedCannyUInt8MatchesNormalizedFloatInput: compares the Canny
  edge output of a float input (CPU-normalized byte*(1/255)) vs a uint8 input
  (GPU-normalized byte*(1.0f/255.0f)) at a tight 1e-5 tolerance. The two
  normalizations differ by ~1 ULP; the Canny non-max-suppression hysteresis
  (roundf(grad/mag) -> integer neighbor direction) is discontinuous, so one ULP
  flips one pixel's edge direction -> a 0.409 single-pixel diff. The kernel is
  bit-deterministic and wave-agnostic (shared-memory stencil, no shfl/ballot);
  this is a cross-input FP-decision-boundary sensitivity in the TEST (gsplat
  fault-13 / gfx1151-radius class), not a kernel bug.
