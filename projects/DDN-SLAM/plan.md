# DDN-SLAM -- ROCm/HIP Port Plan (linux-gfx90a lead)

## Project
- Name: DDN-SLAM ("Real-time Dense Dynamic Neural Implicit SLAM", RA-L 2025, DOI 10.1109/LRA.2025.3546130)
- Upstream: https://github.com/DrLi-Ming/DDN-SLAM (default branch `main`, HEAD 76478ea "Update README.md")
- What it is: ORB-SLAM3 (RGB-D / monocular tracking + mapping) fused with an instant-ngp NeRF for dense neural-implicit mapping (the Orbeez-SLAM design), plus a YOLOv5 dynamic-object detector for masking moving objects. Three upstream lineages, per its README: Orbeez-SLAM (ORB-SLAM2 + instant-ngp), ORB-SLAM2/3, and NVlabs/instant-ngp.

## Decisive question answered: neural-render backend
**Custom-CUDA ray-march + tiny-cuda-nn. NOT OptiX. -> PORTABLE (not B7-gated).**

- The repo's own 28 `.cu` files (System/Tracking/LocalMapping/Optimizer/...) are ORB-SLAM3 C++ renamed `.cc`->`.cu`; they contain **zero `__global__` and zero `__device__`** (verified). They are compiled by nvcc only so they can `#include <neural-graphics-primitives/testbed.h>` and drive instant-ngp's `Testbed` object. All GPU compute lives in two missing thirdparty deps.
- The neural core is `Thirdparty/instant-ngp-kf` = a fork of NVlabs/instant-ngp (MarvinChung/instant-ngp-kf @ submodule sha `4e09053`-era tiny-cuda-nn). DDN-SLAM drives it through the canonical instant-ngp `Testbed` API: `add_training_image`, `add_sparse_point_cloud`, `frame()` (train+render), `m_nerf.training.optimize_extrinsics`, `compute_and_save_marching_cubes_mesh`, `cull_empty_region` (src/Map.cu, src/System.cu).
- **The NeRF training+rendering path is custom CUDA volume-marching** (`src/testbed_nerf.cu`, 4237 lines, 36 `__global__` kernels: sample generation along rays, density-grid update/bitfield, ray compaction, CDF construction, `composite_kernel_nerf` integration, `shade_kernel_nerf`, plus the SLAM tracking/mapping gradient kernels `compute_cam_gradient_train_nerf*`, `compute_loss_kernel_train_nerf`) plus tiny-cuda-nn for the hash-grid encoding + fully-fused MLP.
- **OptiX is optional and unused by the SLAM path.** instant-ngp-kf's CMake: `option(NGP_BUILD_WITH_OPTIX ... ON)` then `if ((OptiX_FOUND OR OptiX_INCLUDE) AND NGP_BUILD_WITH_OPTIX)` -> `-DNGP_OPTIX`, else `set(NGP_OPTIX OFF)` with a warning that only SDF "raystab"/"pathescape" training gets slower. All OptiX code (`optixTrace`/GAS in `src/triangle_bvh.cu`) is `#ifdef NGP_OPTIX`; `testbed_nerf.cu` has **zero** OptiX or cutlass references. The triangle BVH also ships a full `__host__ __device__` software `ray_intersect`/`closest_triangle` fallback. SDF mode is not used by RGB-D/mono NeRF SLAM. So B7 (OptiX->HIPRT) does NOT gate this project; HIPRT (absent on this host) is not needed.
- DLSS (`src/dlss.cu`, NVIDIA-only) is `#ifdef NGP_VULKAN` and auto-disables (`m_dlss_supported=false`) when Vulkan/NGX is absent -- GUI-only, not a hard dep.

### The real hard dependency: tiny-cuda-nn (hash-grid + fully-fused MLP)
tiny-cuda-nn has **no upstream ROCm/HIP support** (verified: no hip/rocm/amd paths in NVlabs/tiny-cuda-nn tree; the fully-fused MLP needs TensorCores/`wmma`/`__half`, and modern tcnn hard-requires CUTLASS -- which per MOAT policy does not port to ROCm). This is the make-or-break piece.

**Mitigation found: `PhysicalAI-AIM/tiny-rocm-nn`** -- a HIP/hipBLAS/rocWMMA port of tiny-cuda-nn that explicitly lists **gfx90a (MI250X)** as supported (also gfx942), benchmarked on MI300X, fully-fused FP16 MLP on MFMA, image-fit converges. Critically it **keeps the `tiny-cuda-nn/` header namespace and the full public API** (`config.h`, `cpp_api.h`, `network_with_input_encoding.h`, `encodings/grid.h` = HashGrid, `networks/fully_fused_mlp.h`, all losses + optimizers incl. Adam) and ships `cuda_compat/` shim headers. Its `src/` module set matches the pinned tcnn `4e09053` era exactly, minus `cutlass_mlp` (replaced by rocWMMA/hipBLAS). instant-ngp's NeRF default network IS `FullyFusedMLP` + `HashGrid`, both present, so the missing `CutlassMLP` (wide-network fallback) is not on the critical path.
- Maturity risk: tiny-rocm-nn is young (1 star, pushed 2026-04) and is a from-scratch reimplementation, so API edge-cases (exact gradient semantics, `network_precision_t`, the GPUMemory/GPUMatrix helpers instant-ngp's kernels touch directly, `cpp_api` ABI) may not be 1:1 with the pinned tcnn. This is the primary technical uncertainty of the whole port and must be de-risked first (Stage 0 below).

## The dynamic component
**YOLOv5 object detection on NVIDIA TensorRT. No ROCm 1:1; this is a backend rewrite, deferrable.**
- `src/YoloDetector.cu` (+ `include/YoloDetector.h`) runs YOLOv5 via TensorRT: `NvInferRuntime.h`, `createInferRuntime`, `deserializeCudaEngine("model/yolov5x.engine")`, `IBuilder`/`buildEngineWithConfig`, `IExecutionContext::enqueue`. It links `-lnvinfer` and `Thirdparty/yolov5_tensorrtx/lib/libyolov5.so` (the tensorrtx YOLOv5 builder + custom TRT plugins). It is "dynamic" by detecting moving-object bounding boxes (person/etc.), then Tracking culls ORB features inside dynamic boxes and the NeRF masks those regions (`YoloBoundingBox.isDynamic`, `mvKeysDynam`, `proStatic`).
- Coupling: `YoloDetector` is **unconditionally constructed** in `System.cu:224` (`new YoloDetector()`) and its thread `Run()` launched, but the detection is **runtime-gated** by the `UseDynamic` YAML flag (`mbYOLO = fsSettings["UseDynamic"]!=0`). Replica configs set `UseDynamic: 0`; OpenLORIS sets `1`. The constructor still deserializes the TRT engine regardless.
- ROCm reality: **MIGraphX is not installed** on this host and **TensorRT/nvinfer is (correctly) absent**. A faithful dynamic path means re-implementing YOLOv5 inference on MIGraphX (or torch-ROCm/ONNXRuntime-ROCm) -- a separate backend rewrite, exactly the EnvGS Stage-1/Stage-2 split pattern.

## Existing AMD support
**None.** No ROCm/HIP fork of DDN-SLAM or Orbeez-SLAM exists (searched); zero hip/rocm/gfx/USE_HIP references in the tree. Upstream has a single branch `main`, no tags. Decision: **fresh CUDA-to-HIP port** (not a finish/improve of an existing port). Do NOT skip as already-supported.

## Build classification: pure CMake (Strategy A), with a torch-extension-shaped sub-dependency
- Evidence: `CMakeLists.txt:2` `project(ORB_SLAM3 LANGUAGES CXX CUDA)`; sources listed as a `SOURCES` var compiled into `add_library(ORB_SLAM3 STATIC ...)` + `add_executable(rgbd_replica ...)` etc.; **no `find_package(Torch)`, no `CUDAExtension`, no setup.py/pyproject.toml** (verified). This is a standalone CMake/CUDA app, NOT a pytorch extension.
- So the top-level project is **Strategy A** (compat-header + `enable_language(HIP)` + mark `.cu` `LANGUAGE HIP`). BUT the bulk of the GPU work is in two vendored deps:
  1. `instant-ngp-kf` (its own CMake; pulls tiny-cuda-nn via `add_subdirectory(dependencies/tiny-cuda-nn)`) -> replace the tcnn submodule with **tiny-rocm-nn** (already a HIP CMake project, `LANGUAGES CXX HIP`, `hipcc`, links `roc::rocblas roc::hipblas`, rocWMMA headers) and HIP-port instant-ngp-kf's own `src/*.cu`.
  2. `yolov5_tensorrtx` (TensorRT) -> deferred to Stage 2.

## Port strategy
Staged, correctness-first, following the EnvGS precedent (port the non-OptiX/non-TensorRT core first, defer the vendor-locked backend):

- **Stage 0 (de-risk, do FIRST): prove tiny-rocm-nn builds on gfx90a and runs instant-ngp's NeRF kernels.** Build tiny-rocm-nn standalone (`-DCMAKE_HIP_ARCHITECTURES=gfx90a`), run its `mlp_learning_an_image` sample (converges = fused MLP + grid encoding work). Then build instant-ngp-kf's `ngp` library against tiny-rocm-nn instead of the tcnn submodule and resolve the API/ABI delta (the GPUMemory/GPUMatrix/`network_precision_t`/`cpp_api` surface that `testbed_nerf.cu` consumes directly). If tiny-rocm-nn cannot satisfy instant-ngp's tcnn API without major surgery, that is the gating finding -- escalate (port-vs-bigger-rewrite decision) rather than grinding.
- **Stage 1 (the deliverable): NeRF neural-implicit SLAM on a STATIC scene, headless, no TensorRT.** HIP-port instant-ngp-kf's `src/*.cu` (Strategy A: `enable_language(HIP)`, compat header, `.cu` -> `LANGUAGE HIP`, `NGP_BUILD_WITH_OPTIX=OFF`, `NGP_BUILD_WITH_GUI=OFF`). Compile-guard the YOLO/TensorRT path out of the top-level build (a `-DUSE_TENSORRT=OFF`/`NO_TENSORRT` guard around `YoloDetector` construction + link) so the app links without `-lnvinfer`/`libyolov5.so`. Validate the full ORB-SLAM3-front-end + NeRF-mapping pipeline on a Replica RGB-D sequence with `UseDynamic: 0`.
- **Stage 2 (deferred): dynamic-object path on MIGraphX.** Re-implement YOLOv5 inference on MIGraphX (or ORT-ROCm) behind the same `YoloDetector` interface. Requires installing MIGraphX (absent here) and converting the YOLOv5 weights. Track as a deferred stage in notes.md, like EnvGS Stage-2 OptiX reflections.

Rationale: the neural core is genuinely portable (custom CUDA + a tcnn HIP fork that targets gfx90a), so Stage 1 produces a real, validatable ROCm neural-SLAM. The two NVIDIA-locked pieces (OptiX-SDF, which is unused; TensorRT-YOLO, which is runtime-optional) are cleanly separable and need backend rewrites, so they are deferred, not blocking.

## CUDA surface inventory (the part we actually port)
- Top-level DDN-SLAM `.cu` (28 files): pure host C++ (ORB-SLAM3); no kernels. The only CUDA contact is `cudaStream_t`/`cudaMalloc`/`cudaMemcpyAsync` in `YoloDetector.cu` (deferred) and including ngp headers elsewhere. Hipify/Strategy-A handles these trivially once YOLO is guarded out.
- instant-ngp-kf `src/testbed_nerf.cu` (NeRF core, 36 kernels): per-sample/per-ray volume marching, density-grid (`grid_to_bitfield`, `bitfield_max_pool`), `composite_kernel_nerf`, `shade_kernel_nerf`, CDF (`construct_cdf_*`), training-sample generation, and the SLAM gradient kernels (camera-pose `compute_cam_gradient_train_nerf*`, extrinsics, loss). Atomics: `atomicAdd` (counters + gradient accumulation, dominant) and `atomicMax((uint32_t*)..., __float_as_uint(...))` on the density/sharpness grid. **No `__shfl`/`__ballot`/cooperative-groups/`warpSize`/hardcoded-32** (verified) -- the warp-cooperative work is all inside tcnn/tiny-rocm-nn.
- instant-ngp-kf `src/testbed.cu` (108 KB), `common_device.cu`, `marching_cubes.cu` (mesh export), `nerf_loader.cu`, `camera_path.cu`: support kernels; no warp intrinsics, no textures in the compute path.
- instant-ngp-kf `src/render_buffer.cu`: `cudaMallocArray`/`cudaSurfaceObject_t`/`surf2Dread`/`surf2Dwrite`/`tonemap_kernel`/DLSS -- **GUI/display surface only** (GLTexture interop). Excluded by `NGP_BUILD_WITH_GUI=OFF` for headless validation, so the texture/surface fault classes are sidestepped on the lead bringup.
- instant-ngp-kf `src/triangle_bvh.cu`: OptiX (`#ifdef NGP_OPTIX`) + software BVH fallback. Build with `NGP_OPTIX` undefined; the software `ray_intersect` covers any non-SDF use.
- tiny-cuda-nn -> replaced wholesale by tiny-rocm-nn (rocWMMA fused MLP + hipBLAS, hash-grid encoding). hipBLAS/rocBLAS present; rocWMMA headers present; hipRTC present.
- Libraries: cuBLAS (via tcnn) -> hipBLAS (handled by tiny-rocm-nn). cuRAND: check tcnn `random.h` (tiny-rocm-nn uses pcg32, likely no cuRAND). No cuFFT/cuSPARSE/cuSOLVER. Thrust/CUB: check instant-ngp-kf usage -> rocThrust/hipCUB (drop-in per cudaKDTree lesson).

## Risk list
- **R1 (gating): tiny-rocm-nn API/ABI fidelity vs the pinned tcnn `4e09053`.** instant-ngp's `testbed_nerf.cu` consumes tcnn's `GPUMemory`/`GPUMatrix`/`network_precision_t`/`cpp_api`/`json` config surface directly, not just the high-level trainer. A from-scratch HIP reimpl may diverge in signatures, half-precision type, or gradient semantics. De-risk in Stage 0; this is the make-or-break.
- **R2: `atomicMax` on the density/sharpness grid silently dropped on coarse-grained/managed memory (cudaKDTree fault class).** instant-ngp allocates the grid via tcnn `GPUMemory` (plain `hipMalloc` device memory, not managed), so likely fine -- but if any grid buffer is managed, the uint `atomicMax` no-ops on gfx90a and the density grid stays empty (degenerate NeRF). Verify the allocation kind; if managed, use an atomicCAS-loop max.
- **R3: wave64 warp-synchronous reductions inside tiny-rocm-nn / tcnn-era reduce_sum.** The classic unrolled `volatile` warp reduction races on wave64 (MPPI lesson). tiny-rocm-nn ships its own `reduce_sum.cpp`; trust its rocWMMA/HIP path but add a fixed-seed determinism check. instant-ngp's own kernels have no warp reductions (verified).
- **R4: half-precision (`__half`/`network_precision_t`) numerics.** NeRF trains in FP16 with FP16 MFMA on gfx90a vs TensorCore on NVIDIA. Expect ULP-level (and FP16-level) divergence; validate by SLAM trajectory/PSNR tolerance, NOT bit-exactness.
- **R5: GUI/Vulkan/GL/DLSS/Pangolin coupling.** Build headless (`NGP_BUILD_WITH_GUI=OFF`, `ORBEEZ_BUILD_WITH_GUI=OFF`). Pangolin is also pulled by ORB-SLAM3's `Viewer`/`MapDrawer`; confirm it can be disabled or built CPU-only for headless runs.
- **R6: TensorRT hard-link even when UseDynamic=0.** The top-level CMake unconditionally links `-lnvinfer` + `libyolov5.so` and the `YoloDetector` ctor deserializes a TRT engine. Must compile-guard the whole YOLO path out for Stage 1 (not just runtime-gate), or the app won't link on ROCm.
- **R7: heavy thirdparty bring-up (build scale).** instant-ngp + tcnn + g2o + DBoW2 + Sophus + OpenCV 4.5.5 + Pangolin + Eigen. The vendored opencv-4.5.5/g2o/Sophus submodules are absent from the DrLi-Ming repo (only DBoW2 ships); they must be fetched from Orbeez-SLAM's submodule URLs or system packages. Large templated HIP TUs (tcnn/instant-ngp) may hit the `--offload-compress` link-size wall (cudf lesson) -- apply if R_X86_64_PC32 overflow appears.
- **R8: `-march=native` + `-Xcompiler=-mf16c`** in CUDA_NVCC_FLAGS are nvcc-passthrough; map to hipcc host-compiler flags (`-Xcompiler` works under hipcc, but `-mf16c` is x86 host SIMD -- keep on host pass only).
- **R9: instant-ngp-kf vendored Eigen/`__CUDA_ARCH__` device path.** Eigen used in device code takes its CUDA path under `__CUDA_ARCH__` and `#include <math_constants.h>` (absent on ROCm); define `EIGEN_NO_CUDA` so Eigen takes its native `__HIP_DEVICE_COMPILE__` path (MPPI lesson). instant-ngp uses its own `vec.h`/Eigen mix -- check which.

## File-by-file change list (Stage 0 + Stage 1)
- `Thirdparty/instant-ngp-kf/dependencies/tiny-cuda-nn` -> swap for `tiny-rocm-nn` (vendor it; reconcile its CMake target name `tiny-rocm-nn` vs the `tiny-cuda-nn`/`tcnn` names instant-ngp-kf links). Resolve API deltas (R1).
- `Thirdparty/instant-ngp-kf/CMakeLists.txt`: add `USE_HIP` path -- `enable_language(HIP)`, set `CMAKE_HIP_ARCHITECTURES` from cache (default gfx90a only when unset, never literal), mark `src/*.cu` `LANGUAGE HIP`, force `NGP_OPTIX OFF`, `NGP_BUILD_WITH_GUI OFF`. Exclude `dlss.cu`, GUI sources, and OptiX program objects.
- `Thirdparty/instant-ngp-kf/`: add one `cuda_to_hip.h` compat header (force-included on HIP TUs via `CMAKE_HIP_FLAGS -include`); alias the cudaXxx symbols instant-ngp uses; define `EIGEN_NO_CUDA` if Eigen-in-device (R9); `#define cub hipcub` if cub:: is used.
- `Thirdparty/instant-ngp-kf/src/triangle_bvh.cu`: ensure the `#ifndef NGP_OPTIX` software path compiles under HIP (the `__host__ __device__` BVH traversal); guard the OptiX block out.
- `Thirdparty/instant-ngp-kf/src/render_buffer.cu`: excluded by GUI-off; if any non-GUI symbol leaks, guard the surface code with `NGP_GUI`.
- Top-level `CMakeLists.txt`: add `USE_HIP` option; `enable_language(HIP)`; mark the 28 `.cu` `LANGUAGE HIP`; add a `USE_TENSORRT` (default OFF on ROCm) guard that drops `-lnvinfer`, `libyolov5.so`, and the `YoloDetector` sources from the build. Bake configurable arch.
- Top-level compat header for the 28 host-only `.cu` (mostly a no-op; they touch almost no CUDA directly).
- `src/System.cu` (~line 224), `src/Tracking.cu`, `include/System.h`, `include/Tracking.h`: `#ifdef USE_TENSORRT` around `YoloDetector` construction/thread/`SetYoloDetector` and the `mYoloDetector` members so Stage-1 links without TensorRT; static-scene path (`UseDynamic 0`) is unaffected functionally.
- `src/YoloDetector.cu`: excluded from Stage-1 build (Stage-2 MIGraphX rewrite target).

## Build commands (gfx90a)
Prereqs to stage (absent from the DrLi-Ming repo; fetch per Orbeez-SLAM submodule URLs or system pkgs): `Thirdparty/instant-ngp-kf` (+ its submodules incl. tiny-cuda-nn -> replaced by tiny-rocm-nn), `Thirdparty/g2o`, `Thirdparty/Sophus`, `Thirdparty/opencv-4.5.5` (or system OpenCV 4.5+), `Thirdparty/yolov5_tensorrtx` (Stage-2 only), Pangolin, Eigen3, the ORBvoc.txt vocabulary.

Stage 0 (tiny-rocm-nn standalone):
```
git clone --recursive https://github.com/PhysicalAI-AIM/tiny-rocm-nn _deps/tiny-rocm-nn
cmake -S _deps/tiny-rocm-nn -B _deps/tiny-rocm-nn/build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_HIP_ARCHITECTURES=gfx90a
cmake --build _deps/tiny-rocm-nn/build -j
./_deps/tiny-rocm-nn/build/mlp_learning_an_image _deps/tiny-rocm-nn/data/images/albert.jpg   # must converge
```
Stage 1 (DDN-SLAM headless, no TensorRT, no GUI, no OptiX):
```
cmake -S projects/DDN-SLAM/src -B projects/DDN-SLAM/src/build \
  -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DUSE_TENSORRT=OFF -DORBEEZ_BUILD_WITH_GUI=OFF -DNGP_BUILD_WITH_OPTIX=OFF -DNGP_BUILD_WITH_GUI=OFF \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
cmake --build projects/DDN-SLAM/src/build -j --target rgbd_replica
```
(If link hits R_X86_64_PC32 overflow on the tcnn/instant-ngp TUs, add `--offload-compress` to the HIP compile options -- cudf lesson.)

## Test plan
There is **no unit-test suite** (ORB-SLAM3 lineage ships none; instant-ngp ships none usable here). Validation is end-to-end on a sample RGB-D sequence.
- **Primary GPU gate (Stage 1):** run `rgbd_replica` on a small Replica RGB-D sequence (e.g. office0/room0, `UseDynamic: 0`) headless. Assert: (a) the run completes without GPU fault/NaN; (b) the NeRF actually trains -- `loss` (`m_loss_scalar`) decreases over iterations and stays finite (guards R2 density-grid-empty and R4 FP16 NaN); (c) a sane camera trajectory is produced (`KeyFrameTrajectory.txt`) and ATE vs the Replica GT pose file is within a tolerance comparable to the published Orbeez/DDN numbers (not bit-exact -- FP16 + atomic-order nondeterminism); (d) `compute_and_save_marching_cubes_mesh` exports a non-degenerate mesh; optionally PSNR of a rendered training view above a floor.
- **Determinism check:** fixed RNG seed, two runs; assert trajectory/loss agree to a sensible tolerance (atomicAdd gradient accumulation is order-nondeterministic, so expect small drift, not bitwise -- GPUMD/MPPI lesson). If wildly nondeterministic, suspect R3 (warp-sync reduction) or R2.
- **Stage 0 micro-gate:** tiny-rocm-nn `mlp_learning_an_image` converges (loss drops, output image matches input) -- proves fused MLP + hash-grid on gfx90a before the SLAM integration.
- **Non-GPU regression set:** none to regress (no upstream tests). The CUDA build is the reference; do not break it (keep all HIP behind `USE_HIP`).

## Open questions
1. **R1 / Stage 0:** Does tiny-rocm-nn satisfy the exact tcnn `4e09053` API + ABI that instant-ngp-kf's `testbed_nerf.cu` links against (GPUMemory/GPUMatrix/network_precision_t/cpp_api/json config), or does the integration need non-trivial glue? This decides effort and is gating -- resolve before Stage 1.
2. **Dataset:** which Replica/TUM sequence + GT to stage for the validation gate (none on host yet); the validator fetches a small one.
3. **Pangolin headless:** can ORB-SLAM3's Viewer/MapDrawer be fully disabled (or built CPU-only) so the headless gfx90a run needs no GL/X? (Likely `bUseViewer=false` in the System ctor.)
4. **Thrust/CUB in instant-ngp-kf:** confirm rocThrust/hipCUB drop-in (cudaKDTree says yes) and whether any `DeviceRadixSort`/`atomicMin/Max`-on-managed patterns hit the gfx90a fault classes.
5. **Stage 2 scope:** MIGraphX is not installed; the YOLOv5->MIGraphX (or ORT-ROCm) rewrite + weight conversion is a separate effort. Confirm Stage 1 (static-scene neural SLAM) is an acceptable lead deliverable with Stage 2 deferred (EnvGS precedent says yes).

## Recommendation
**PORT (staged), recommended.** The neural-render backend is custom-CUDA ray-march + tiny-cuda-nn, NOT OptiX -- so this is NOT B7-gated and is genuinely portable on ROCm 7.2.1/gfx90a. The single hard dependency (tiny-cuda-nn) has a HIP/rocWMMA fork (`tiny-rocm-nn`) that already targets gfx90a and keeps the tcnn API, which makes a correctness-first Stage-1 neural-implicit SLAM tractable. The two NVIDIA-locked pieces are cleanly separable and deferrable: OptiX (SDF mode) is unused by the SLAM path, and the TensorRT YOLO "dynamic" component is runtime-optional and gets a Stage-2 MIGraphX rewrite. Gate the go/no-go on Stage 0 (tiny-rocm-nn <-> instant-ngp-kf API fit, risk R1); if tiny-rocm-nn cannot back instant-ngp's tcnn API without a major rewrite, escalate the port-vs-larger-rewrite decision rather than grinding.
