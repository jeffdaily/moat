# DDN-SLAM notes

DDN-SLAM (Real-time Dense Dynamic Neural Implicit SLAM, RA-L 2025) = ORB-SLAM3 front-end
fused with an instant-ngp NeRF for dense neural-implicit mapping (the Orbeez-SLAM design),
plus a YOLOv5/TensorRT dynamic-object detector. Lead platform: linux-gfx90a (MI250X), ROCm 7.2.1.

Staged port (correctness-first):
- Stage 0 (de-risk, make-or-break R1): tiny-cuda-nn has no ROCm; prove the HIP fork
  tiny-rocm-nn builds + converges on gfx90a. -- DONE, PASSES.
- Stage 1 (deliverable): HIP-port instant-ngp-kf (the NeRF core) + the ORB-SLAM3 host code,
  headless, no TensorRT/OptiX/GUI; validate rgbd on a Replica RGB-D sequence (UseDynamic:0).
- Stage 2 (DEFERRED): YOLOv5 dynamic path on MIGraphX/ORT-ROCm. See "Stage 2 deferral".

## Repo / fork
- Upstream: DrLi-Ming/DDN-SLAM (default `main`, HEAD 76478ea). Fork: jeffdaily/DDN-SLAM.
  Port on branch `moat-port`; fork `main` stays a clean upstream mirror. Actions disabled on the fork.
- The DrLi-Ming repo has NO .gitmodules and ships only Thirdparty/DBoW2. Every other dep
  (instant-ngp-kf, g2o, Sophus, OpenCV, the rgbd_replica/tum/scannet example entry points) is
  ABSENT and must be fetched. Submodule URLs/SHAs come from the parent MarvinChung/Orbeez-SLAM:
  - Thirdparty/instant-ngp-kf  = MarvinChung/instant-ngp-kf @ 1adb67d17b9b8117f6276fb406530cad34692f5b
  - Thirdparty/g2o             = RainerKuemmerle/g2o        @ 26f775d144f3b09bc072b90b903631036a1e4107
  - Thirdparty/opencv-4.5.5    = opencv/opencv              @ dad26339a975b49cfb6c7dbe4bd5276c9dcb36e2 (system OpenCV 4.6 used instead)
  - Examples/RGB-D/rgbd_replica.cu, rgbd_tum.cu, rgbd_scannet.cu -- present in Orbeez-SLAM, ABSENT in DrLi-Ming.

## Stage 0 -- tiny-rocm-nn de-risk (R1, the make-or-break): PASSED on gfx90a
tiny-cuda-nn (hash-grid encoding + fully-fused FP16 MLP) is the one hard dependency and has no
upstream ROCm. Mitigation: PhysicalAI-AIM/tiny-rocm-nn (HIP/hipBLAS/rocWMMA port; keeps the
tiny-cuda-nn/ header namespace + full public API: gpu_memory.h, gpu_matrix.h, cpp_api.h,
config.h, encodings/grid.h=HashGrid, networks/fully_fused_mlp.h, losses, optimizers incl Adam).
tiny-rocm-nn was authored/tested on ROCm 6.4.3; two ROCm-7.2.1 version-drift fixes were needed:

1. hipBLAS v2 datatype removal (include/tiny-cuda-nn/cublas_matmul.h).
   `hipblasDatatype_t` and `HIPBLAS_R_32F`/`HIPBLAS_R_16F` were removed in the hipBLAS v2 API.
   ROCm 7.2.1 `hipblasGemmEx` takes `hipDataType` (HIP_R_32F/HIP_R_16F) for in/out and
   `hipblasComputeType_t` (HIPBLAS_COMPUTE_32F) for compute. Fixed all three gemm wrappers.
   The supported combo HIP_R_16F x HIP_R_16F -> HIP_R_16F with COMPUTE_32F (per the hipblas.h
   table) is exactly the FP16-data/FP32-accumulate path the fused MLP wants.

2. rocWMMA 7.2.x ambiguous mfma specialization (src/fully_fused_mlp.cpp). ROOT CAUSE:
   in ROCm 7.2.1 rocwmma, `hfloat16_t == __half` (types.hpp) and there are TWO viable partial
   specializations of `amdgcn_mfma<__half,__half,__half,16,16,16>` -- the dedicated hfloat16_t
   one (mfma_impl.hpp:421, gated `!ROCWMMA_NO_HALF`) AND the generic `sizeof(ComputeT)<4` one
   (:131). Partial ordering cannot disambiguate -> "ambiguous partial specializations" (the build
   wall). rocWMMA's canonical 16-bit fragment element is `float16_t` (== _Float16), bit-identical
   to __half on gfx9, which selects the explicit float16_t specializations (:199-366) cleanly.
   FIX (arch-correct, the rocWMMA-idiomatic spelling): a `wmma_elem<T>` trait mapping
   __half -> rocwmma::float16_t used for FRAGMENT ELEMENT TYPES only; all host/shared __half
   buffers stay __half. rocWMMA static_asserts that the load/store pointer type exactly matches
   the fragment element type, so each load_matrix_sync/store_matrix_sync pointer is reinterpreted
   via wmma_ptr(); fill_fragment scalars use (rocwmma::float16_t)0.0f; warp_activation<OUT_T> ->
   warp_activation<wmma_elem_t<OUT_T>> so the activation arithmetic type matches the fragment.
   Confirmed standalone (agent_space/wmma_probe): __half fragments = 16 errors; float16_t = 0.

MICRO-GATE result (HIP_VISIBLE_DEVICES=2, gfx90a):
  ./build/mlp_learning_an_image data/images/albert.jpg data/config_hash.json 2000
  (HashGrid + FullyFusedMLP, the NeRF-representative config)
    Step#0:    loss=9.39851
    Step#10:   loss=0.272781
    Step#100:  loss=0.0196025
    Step#1000: loss=0.00890704      -- 3 orders of magnitude drop, no NaN, no GPU fault
  Default config.json: loss 9.46 -> 0.16 @100; output image reconstructed.
  => the fused FP16 MFMA MLP + hash-grid encoding converge correctly on gfx90a. R1 RESOLVED.

WAVE_SIZE note (follower platforms): tiny-rocm-nn hardcodes `WAVE_SIZE = 64` in fully_fused_mlp.cpp
and is gfx9/CDNA-only by design (rocWMMA MFMA). Correct for the gfx90a lead. The RDNA followers
(gfx1100/gfx1151, wave32) will need tiny-rocm-nn's wave size made arch-aware before they can use
the fused MLP -- a follower delta, not a lead concern.

## Build recipe (Stage 0)
```
cmake -S _deps/tiny-rocm-nn -B _deps/tiny-rocm-nn/build -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a
cmake --build _deps/tiny-rocm-nn/build -j 16
```
(tiny-rocm-nn's CMakeLists sets CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc before project(); LANGUAGES CXX HIP.)

## Stage 1 -- instant-ngp-kf + ORB-SLAM3 HIP port (in progress)
Surface inventory (verified by scanning all instant-ngp-kf src/*.cu at the pinned SHA):
- NO warp intrinsics anywhere (no __shfl/__ballot/__activemask/warpSize/cooperative_groups). The
  warp-cooperative work is entirely inside tcnn/tiny-rocm-nn (handled in Stage 0). => no warp_size
  abstraction needed in instant-ngp-kf itself.
- NO cudaMallocManaged anywhere => R2 (atomicMax-on-managed silently dropped) does NOT apply. The
  density/sharpness-grid atomicMax((uint32_t*)&grid, __float_as_uint(...)) in testbed_nerf.cu (2
  sites) is on plain GPUMemory device memory; uint atomicMax is correct on gfx90a (author's own
  comment confirms it is acceptable).
- Textures/surfaces: testbed.cu (GUI-only: GLTexture/dlss/ImGui) and render_buffer.cu. In
  render_buffer.cu the GLTexture/cudaGraphicsGLRegisterImage path is #ifdef NGP_GUI, but the
  CudaSurface2D + tonemap/overlay surf2Dread/surf2Dwrite + cudaMallocArray(...SurfaceLoadStore)
  are UNCONDITIONAL. They map 1:1 to HIP (hipMallocArray/surf2D*); point-sampled, so the 256B
  surface-pitch class is benign here. testbed_nerf.cu has zero imgui/GUI refs.
- cuda* runtime surface in testbed_nerf.cu is trivial (cudaMemsetAsync, cudaMemcpyAsync,
  cudaStream*, cudaMemcpy*, cudaStreamSynchronize, cudaDeviceSynchronize) -- all 1:1 HIP.
- instant-ngp uses its OWN vendored Eigen (Tom94/eigen) and Eigen IN device code (common_device.cuh
  includes <Eigen/Dense>). R9: define EIGEN_NO_CUDA so Eigen takes its __HIP_DEVICE_COMPILE__ path
  (avoids the absent math_constants.h CUDA path).
- OptiX is correctly gated (find_package(OptiX) -> if not found, NGP_OPTIX OFF; all optixTrace/GAS
  in src/optix/* + triangle_bvh.cu under #ifdef NGP_OPTIX). triangle_bvh.cu ships a __host__
  __device__ software ray_intersect fallback. testbed_nerf.cu has ZERO OptiX. => NOT B7-gated.
- instant-ngp-kf deps: in-tree (filesystem, nanovdb, tinyexr, tinyobjloader, stb_image); submodules
  needed headless = args, eigen, tinylogger (+ tiny-cuda-nn -> tiny-rocm-nn). GUI/Python submodules
  (glfw, imgui, dlss, pybind11) excluded by NGP_BUILD_WITH_GUI=OFF + Python off.

Build flags (headless): -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -DORBEEZ_BUILD_WITH_GUI=OFF
-DNGP_BUILD_WITH_OPTIX=OFF -DNGP_BUILD_WITH_GUI=OFF -DUSE_TENSORRT=OFF.

### instant-ngp-kf HIP-port fixes applied (the R1 tcnn<->ngp integration, the real work)
CMake: USE_HIP path (LANGUAGES C CXX HIP; GUI/Vulkan/OptiX default OFF; skip the glfw FATAL_ERROR
and the Python/pybind11 subdir on HIP; C++14->17; mark .cu LANGUAGE HIP; force-include the compat
header via CMAKE_HIP_FLAGS -include; HIP_ARCHITECTURES + --offload-compress on the ngp target;
build the standalone `testbed` exe only on CUDA). Vendored tiny-rocm-nn as dependencies/tiny-cuda-nn
with `add_library(tiny-cuda-nn ALIAS tiny-rocm-nn)` so ngp's `target_link_libraries(... tiny-cuda-nn)`
resolves; skip its samples/test subdirs when vendored.

Compat header (include/neural-graphics-primitives/cuda_to_hip.h, force-included on HIP TUs): aliases
the cuda* runtime surface ngp uses; defines EIGEN_NO_CUDA (Eigen-in-device R9); provides a CAS-loop
atomicAdd(__half*) emulation (see below).

tcnn-API compat shims added to the vendored tiny-rocm-nn (it followed a NEWER tcnn API than the
pinned NVlabs tcnn instant-ngp-kf @ 1adb67d was written against):
- common.h: `vector_t<T,N> = tvec<T,N>`, `vector_fullp_t<N> = tvec<float,N>`, and
  `TCNN_NAMESPACE_BEGIN/END` macros (the fork writes `namespace tcnn {` directly).
- object.h: `using EGradientMode = GradientMode` (the E-prefix was dropped).
- encodings/grid_interface.h: a storing `set_quantize_threshold(float)` on GridEncoding (the fork
  dropped lookup-time quantization; storing it keeps Testbed::set_min_level compiling -- the
  feature only affects early-training anti-aliasing of coarse hash levels, not final reconstruction).

ngp source fixes (all USE_HIP / __HIP__-guarded so the CUDA build is untouched):
- common.h NGP_HOST_DEVICE: `#ifdef __NVCC__` -> also `|| __HIPCC__ || __HIP__` (else sign()/etc.
  became host-only and device refs failed). Same for the cudaMemcpyAsync helper guards in nerf.h
  (RaysNerfSoa::copy_from_other_async) and sdf.h (RaysSdfSoa::enlarge/copy_from_other_async).
- testbed_nerf.cu + nerf.h: the `inline constexpr __device__` NERF_GRIDSIZE/NERF_CASCADES/NERF_STEPS/
  STEPSIZE/... helpers -> `__host__ __device__` (they are called from host marching/setup code; nvcc
  allows constexpr-__device__ from host, clang/HIP does not).
- testbed.cu: the __CUDACC_VER_MAJOR__/MINOR__ CUDA-version floor check guarded out on HIP;
  dump_parameters_as_images reads Trainer::params() in network_precision_t (the fork stores params in
  __half, not a float master copy) and widens to float for the EXR dump (debug-only).
- nerf_network.h + trainable_buffer.cuh + takikawa_encoding.cuh: adapted the tcnn subclass interface
  from the OLD tcnn (4-arg set_params with backward_params, 7-arg initialize_params, set_alignment/
  min_alignment override) to the fork's NEW interface (3-arg set_params_impl, 3-arg
  initialize_params(rnd, params_full_precision, scale), set_padded_output_width/
  required_output_alignment). NerfNetwork already used the _impl compute virtuals (inference_mixed_
  precision_impl/forward_impl/backward_impl) so only the param-plumbing virtuals needed adapting.
  testbed_nerf.cu's trainable_verts->initialize_params call updated to init-then-set_params.
- common_device.cuh deposit_image_gradient: ROCm 7.2.1 has NO atomicAdd(__half*) or
  atomicAdd(__half2*) (only int/uint/ulong/float/double). Guard the CUDA __half2 fast path to
  non-HIP; on HIP use the compat header's CAS-emulated atomicAdd(__half*) for the FP16 gradient
  buffer and native atomicAdd(float*) for float buffers. Guard on __HIP__ (set in both host+device
  passes) so neither HIP pass parses the __half2 path. (FP16-atomic fault class -- the
  cuCollections sub-word-CAS lesson generalized to __half gradient accumulation.)
- testbed_sdf.cu: CUDA `umin` builtin (absent on HIP) -> `min`.
- dependencies/tinyexr/tinyexr.h: anonymous `typedef struct {...} HeaderInfo;` (with std::vector
  members) -> named `struct HeaderInfo_tag` (clang rejects the anonymous-non-C-typedef-for-linkage
  where nvcc tolerates it; vendored-header 1-line fix).
- camera_path.h / render_buffer.cu: moved imgui/cuda_gl_interop GUI includes inside #ifdef NGP_GUI
  (the .cu guarded them but the header did not).
- pinned the absent submodule deps to their exact instant-ngp-kf SHAs after a master-branch fetch
  pulled a C++20 tinylogger (`requires` clauses) that broke the C++17 build: tinylogger
  9b7a92ff44ffca3e80dca3549fa4eb599d039eeb, args a48e1f8808..., eigen 3a8eda01....

### Top-level DDN-SLAM (ORB-SLAM3) HIP-port fixes
- CMakeLists: USE_HIP (LANGUAGES CXX HIP) + USE_TENSORRT options; mark the 27 host-only .cu
  LANGUAGE HIP + force-include the compat header; exclude src/YoloDetector.cu and drop
  -lnvinfer/libyolov5.so/-lcudart when USE_TENSORRT OFF; an add_orbeez_example() helper that skips
  absent entry points (rgbd_tum/scannet were absent) and marks them HIP; C++17; nvcc-only
  CUDA_NVCC_FLAGS gated to the CUDA path.
- Config.h: REMOVE_DYNAMIC_OBJECT (the YOLO-feature flag, was unconditional) now gated on
  USE_TENSORRT, so the YOLO construction (System.cu), GetImg (System.cu), and dynamic-mask code
  (Frame.cu) compile out cleanly on ROCm; mpYoloDetector default-initialized to nullptr.
- YoloDetector.h: the TensorRT-specific pieces (NvInferRuntime.h, `using nvinfer1`, IExecutionContext*
  member, doInference(IExecutionContext...)) guarded by USE_TENSORRT; YoloBoundingBox (the data type
  used by Frame/Tracking) stays always-available.
- Pangolin v0.6 built locally (system GL/GLEW/EGL present) into _deps/pangolin-install; system
  OpenCV 4.6 satisfies find_package(OpenCV 4.5). Sophus 1.22.10 (header-only) + g2o @ pinned SHA.

### YOLO / TensorRT coupling (must compile-guard for Stage 1)
- nvinfer symbols are confined to YoloDetector.{h,cu}. But YoloDetector.h includes
  <NvInferRuntime.h> + `using namespace nvinfer1` and is pulled by Frame.h, KeyFrame.h,
  FrameDrawer.h, System.h, Tracking.h -- so the TensorRT header leaks into nearly every TU.
- YoloBoundingBox (a pure data type used by Frame::isInBox + Tracking) lives in YoloDetector.h and
  must stay always-available. Plan: guard ONLY the TensorRT pieces (#include NvInferRuntime.h,
  using nvinfer1, IExecutionContext* members, doInference(IExecutionContext...)) behind
  #ifdef USE_TENSORRT; keep YoloBoundingBox + the class shell. Guard `new YoloDetector()` + thread
  launch in System.cu (~224) behind USE_TENSORRT. Top-level CMake: when USE_TENSORRT=OFF drop
  -lnvinfer, libyolov5.so, and src/YoloDetector.cu from the build. Replica configs set
  UseDynamic:0, so the static-scene path is functionally unaffected.

## Stage 2 deferral -- YOLOv5 dynamic-object path (TensorRT -> MIGraphX)
src/YoloDetector.cu runs YOLOv5 via NVIDIA TensorRT (createInferRuntime, deserializeCudaEngine
on model/yolov5x.engine, IExecutionContext::enqueue) and links -lnvinfer +
Thirdparty/yolov5_tensorrtx/lib/libyolov5.so. TensorRT has no ROCm equivalent; a faithful dynamic
path is a MIGraphX (or ORT-ROCm) reimplementation behind the same YoloDetector interface, plus
YOLOv5 weight conversion. MIGraphX is NOT installed on this host. The dynamic detection is
runtime-gated by the YAML UseDynamic flag (Replica=0, OpenLORIS=1), so Stage 1 compile-guards it
out (USE_TENSORRT=OFF) and validates the static-scene neural SLAM. This is the same Stage-1-core /
Stage-2-vendor-backend split as EnvGS (OptiX) -- see UPSTREAM_FINDINGS B-class TensorRT cluster.

## Concurrency / host
- 4 GCDs; sibling agents use GCD 0/1. This port: GPU runs on HIP_VISIBLE_DEVICES=2, builds capped -j 16.
- HIP compiler: /opt/rocm/llvm/bin/clang++ (ROCm 7.2.1, AMD clang 22.0.0). System OpenCV 4.6.0,
  Eigen3, boost_serialization present; Pangolin + Sophus + MIGraphX absent.
</content>
