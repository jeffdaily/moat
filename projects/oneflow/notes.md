# oneflow notes

ROCm/HIP port of oneflow's CUDA GPU backend. Lead platform linux-gfx90a (MI250X,
CDNA2, wave64), ROCm 7.2.1. Strategy A (pure CMake: USE_HIP option, one compat header,
.cu retagged LANGUAGE HIP). Fork: https://github.com/jeffdaily/oneflow (branch moat-port).

## Key architectural decision: WITH_CUDA defined FOR the HIP build
oneflow's entire GPU backend is gated on `#ifdef WITH_CUDA` (204 files) and `kCUDA` is the
GPU DeviceType. Rather than introduce a `kROCM` enum (proto + every registry + thousands
of switch arms), the port keeps `BUILD_CUDA=ON` so `-DWITH_CUDA` is defined and `kCUDA`
keeps meaning "the GPU backend", and adds a parallel `USE_HIP` that routes the toolkit,
libraries, and the `.cu` source language to ROCm. `device="cuda"` in Python maps to the
HIP backend (mirrors how torch-ROCm keeps `torch.cuda`). AutoDock-GPU lesson, framework
scale.

## Monolith constraint
All ~220 `.cu` + all `.cpp` link into ONE `liboneflow.so` (`oneflow_add_library(oneflow
SHARED ${of_all_obj_cc})`); there is no per-op sub-target. The whole lib must compile +
link before any op runs, so the validation slice (ep/cuda/primitive tensor ops) is a
VALIDATION scope, not a build scope.

## Build script
`projects/oneflow/src/build_rocm.sh [config|tp|oneflow|all]`. gfx90a, -j16, ccache.
Configure flags: `-DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -DBUILD_CUDA=ON
-DWITH_CUTLASS=OFF -DWITH_MLIR=OFF -DBUILD_RDMA=OFF -DBUILD_CPP_API=OFF -DBUILD_HWLOC=OFF
-DBUILD_TESTING=OFF -DTREAT_WARNINGS_AS_ERRORS=OFF -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF`.
HIP compiler auto-detected at /opt/rocm/lib/llvm/bin/clang++ (no -DCMAKE_HIP_COMPILER
needed). Third-party (LLVM-16 prebuilt download, protobuf, grpc, opencv, oneDNN, half,
eigen, flatbuffers, etc.) builds from source via oneflow's own cmake/third_party; the CUDA
deps (NCCL/CUTLASS/flash-attn/cub) are skipped under USE_HIP (RCCL/hipCUB/rocThrust are
header drop-ins; CUTLASS/flash/MIOpen deferred).

## Followers (gfx1100/gfx1151)
Same commit; only `-DCMAKE_HIP_ARCHITECTURES=<arch>` changes (the wavefront width
OF_GPU_WARP_SIZE is derived from the arch in cmake/cuda.cmake, no source edit). RDNA is
wave32 -- the warp-size fixes below are arch-unified and correct for both 32 and 64.

## Where things are
- `oneflow/core/device/cuda_to_hip.h` -- the single compat header. Force-included on every
  HIP TU (`CMAKE_HIP_FLAGS -include ...`). Aliases the cuda*/cublas*/curand*/cufft*/cusolver*
  symbols oneflow uses to hip*; complex types/conv; CUDART_INF/NAN -> HIP_INF/NAN;
  `__trap`->abort; defines `__CUDACC__` and device-pass `__CUDA_ARCH__=1`; 64-bit warp mask
  `OF_SHFL_FULL_MASK`; libc-before-hip include order; handle-type aliases for
  cudnn/cusolver/cublasLt (operations deferred).
- `oneflow/core/device/hip_compat/` -- forwarding shim headers (cuda_runtime.h, cublas_v2.h,
  curand.h, cufft.h, cuComplex.h, nccl.h -> rccl/rccl.h, cub/cub.cuh -> `namespace cub =
  hipcub`, ...) on the HIP include path ONLY. The CUDA build never sees them.

## Fault classes -- status

### CLEARED (validated by compiling the ep/cuda/primitive slice as HIP, 30/30 TUs)
1. wave64 warp size. `kWarpSize`/`kCudaWarpSize` were hardcoded 32 in
   core/cuda/{softmax,layer_norm,rms_norm}.cuh and core/device/cuda_util.h. Fixed to
   `OF_GPU_WARP_SIZE` (64 on gfx9xx, 32 on RDNA/CUDA), injected from the target arch by
   cmake/cuda.cmake into BOTH the host and device passes of every .cu via CMAKE_HIP_FLAGS.
   This is mandatory because kWarpSize flows from host launch config into device kernel
   template args, so it MUST be one compile-time constant -- `__GFX9__` is device-pass-only
   (proven: a static_assert keyed on __GFX9__ fails in the host pass) and cannot be used.
   RmsNormParamGrad's full-warp WarpReduceSum now starts its butterfly at kWarpSize/2 so all
   64 lanes fold in (was hardcoded 16).
2. 32-bit shfl mask. `__shfl_*_sync(0xffffffff, ...)` does not compile on ROCm 7.x
   (static_assert sizeof(mask)==8). Compat header defines `OF_SHFL_FULL_MASK`
   (0xffffffffffffffffULL on HIP, 0xffffffff on CUDA); the reduction headers route through
   it. Warp-tiled shuffles also pass explicit `thread_group_width` (confines the butterfly
   to the group on a 64-lane wavefront; matches the codebase's existing convention).
3. `__CUDA_ARCH__` undefined on HIP. Compat header defines `__CUDA_ARCH__=1` in the device
   pass only (guarded by `__HIP_DEVICE_COMPILE__`), so device-vs-host `#ifdef`s take the
   device branch and numeric `#if __CUDA_ARCH__ >= NNN` intrinsic gates (TanhApprox PTX,
   atomic.cuh native paths) naturally fall to the portable `#else` -- no per-site edits
   needed for the slice. Paired with `__CUDACC__` (defined when `__HIPCC__`) so header-only
   device code is selected. EIGEN_NO_CUDA set so Eigen takes its native HIP path.
4. Missing symbol aliases surfaced by compiling: cuComplex/cuDoubleComplex + conversions,
   cuComplexFloatToDouble/DoubleToFloat, CUDART_INF/INF_F/NAN/NAN_F, __trap, the cufft/
   cusolver enums and types.
5. cuda_pseudo_half.h redefinition. HIP's __half (hip_fp16.h) already ships the full
   operator/intrinsic set the file emulates for pre-sm53 CUDA; with __CUDA_ARCH__=1 it would
   redefine them and clash. Gated the body with `&& !defined(USE_HIP)`.

### Third-party bringup fixes (not HIP-specific, needed for the monolith)
- flatbuffers 1.12.0 hardcodes -Werror; gcc-13 -Wstringop-overflow on libstdc++
  stl_algobase.h breaks its build. PATCH_COMMAND strips the -Werror flags.
- kineto auto-enables a roctracer backend on ROCm and demands ROCTRACER_LIBRARY from a wrong
  path; LIBKINETO_NOROCTRACER=ON under USE_HIP (profiler is off anyway).

### DEFERRED (excluded from the first slice; documented for a continuation)
- cuDNN -> MIOpen (NOT a drop-in: different descriptor/algo-find API). The conv/pool/
  batchnorm/normalization primitives that call cuDNN. cuda_util.h gates `<cudnn.h>` and the
  cudnn error-string helper behind `!USE_HIP`; the handle type is aliased so cuda_stream.h
  parses, but the operations are not. These TUs need real MIOpen adaptation.
- cuSOLVER -> hipSOLVER (handle/stream aliased; the getrf/getri dense ops in the few
  cusolver kernels deferred).
- cuBLASLt fused-MLP / grouped-GEMM kernels (cublas_fused_mlp*, fused_matmul_bias*) --
  hipBLASLt API differs; deferred.
- nvjpeg/npp image decode (NVIDIA-only) -- gated behind `!USE_HIP`.
- CUTLASS/flash-attention fused kernels -- `-DWITH_CUTLASS=OFF` (already #ifdef-guarded);
  CUTLASS does not port to ROCm (would be a CK/ck_tile reimplementation).

## GPU validation (test oracle = PyTorch ROCm)
torch 2.13 ROCm (torch.cuda.is_available()==True on HIP) is importable as the autotest
oracle. 4 MI250X GCDs; run on one free GCD via HIP_VISIBLE_DEVICES (rocm-smi to pick).
Slice autotests (after monolith links): test_add, test_cast, test_where, test_masked_fill,
test_softmax, test_layer_norm, test_matmul, test_broadcast_ops, test_permute, run SERIALLY
(pytest, not -n) on one GCD.
