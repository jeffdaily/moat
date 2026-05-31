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

### Monolith compile progress (whole-tree .cu under HIP)
A keep-going `ninja -k 0` of all 220 .cu objects drove the fault-finding. Starting from
138/220 compiling with just the compat header, the fixes below cleared the bulk; the
genuinely-deferred cuDNN/cuBLASLt/wmma TUs (~12) are list(FILTER EXCLUDE)-ed from the HIP
monolith glob in cmake/oneflow.cmake. The 30-TU ep/cuda/primitive validation slice all
compiles as HIP.

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
5. cuda_pseudo_half.h / cuda_pseudo_bfloat16.h redefinition. HIP's __half (hip_fp16.h) and
   __hip_bfloat16 (hip_bf16.h) already ship the full operator/intrinsic set these files
   emulate for pre-sm53/pre-sm80 CUDA; with __CUDA_ARCH__=1 they would redefine them and
   clash ("__device__ function cannot overload __host__ __device__ function"). Gated both
   bodies with `&& !defined(USE_HIP)`.
6. bf16 type identity. HIP has two bf16 structs: __hip_bfloat16 (modern, what
   __float2bfloat16 etc. return) and hip_bfloat16 (legacy). oneflow's nv_bfloat16/
   __nv_bfloat16 must both alias __hip_bfloat16 so functor return/param types match the
   conversion results (else "no viable conversion from __hip_bfloat16 to hip_bfloat16").
7. More 32-bit shfl/ballot masks tree-wide: layer_norm_gpu_kernel.cu, data_shuffle_kernel.cu
   (per-arch kWarpSize + OF_SHFL_FULL_MASK), batch_norm_kernel_utils.h (WARP_SIZE per-arch +
   WARP_SHFL_XOR default mask 64-bit), lru_cache.cu (kept 32 ways = 32 lanes; 64-bit ballot
   folded with __ffsll/__popcll, lanes 32-63 idle). Also CUDART_INF/NAN, __trap->abort,
   cuComplex* + conversions, CUDA_VERSION defined to 11080, bare curand->hiprand,
   CUDNN_BN_MIN_EPSILON (numeric bound), cufft/cusolver enums.
8. clang strictness: `dst.template Mut(i)` where Mut is non-template -> drop the spurious
   `.template` (the -Wmissing-template-arg-list-after-template-kw family). The cub shim must
   make `cub` a real namespace (`namespace cub { using namespace hipcub; }`) not an alias,
   because ndarray_reduce_impl.cu REOPENS namespace cub to add its own functors.
   layer_norm.cuh MaximizeDynamicSharedMemorySize casts the kernel ptr to const void* for
   hipFuncGetAttributes/hipFuncSetAttribute (gsplat lesson).
9. Host GPU .cpp TUs (ep/cuda/*.cpp, device/cuda_util.cpp, ...) compile as CXX not hipcc, so
   they get the compat header force-included + shim dir + /opt/rocm/include + warp-size macro
   PER oneflow target (set_compile_options_to_oneflow_target), NOT globally -- a global
   force-include leaks into third-party subdir builds (TBB) which lack the ROCm include path.

### Third-party bringup fixes (not HIP-specific, needed for the monolith)
- flatbuffers 1.12.0 hardcodes -Werror; gcc-13 -Wstringop-overflow on libstdc++
  stl_algobase.h breaks its build. PATCH_COMMAND strips the -Werror flags.
- kineto auto-enables a roctracer backend on ROCm and demands ROCTRACER_LIBRARY from a wrong
  path; LIBKINETO_NOROCTRACER=ON under USE_HIP (profiler is off anyway).

### Compile tally (whole oneflow/ .cu tree, gfx90a)
203 of 220 .cu compile as HIP. The 17 deferred TUs (list(FILTER EXCLUDE) in
cmake/oneflow.cmake, each with a genuine library/API gap -- not a mechanical port):
- cuDNN, no MIOpen drop-in: grid_sample_kernel_util.cu (spatial-transformer),
  normalization_kernel.cu (cuDNN batchnorm). cuda_util.h/cudnn_util.h gate `<cudnn.h>` and
  the cudnn error-string helper behind `!USE_HIP`; the handle type is aliased so
  cuda_stream.h parses, but the operations are not. cudnn_conv_util.cpp also deferred.
- cuBLASLt fused-MLP / grouped-GEMM / matmul-bias family: cublas_fused_mlp_kernel.cu,
  cublas_fused_mlp_grad_kernel.cu, cublas_bias_add_relu_matmul_grad_kernel.cu,
  cublas_fused_matmul_bias_add_grad.cu, fused_matmul_bias_kernel.cu,
  fused_matmul_bias_add_relu_dropout.cu, grouped_matmul_bias.cu (hipBLASLt descriptor
  surface + hipblasGemmStridedBatchedEx signature differ from cuBLASLt).
- cuSOLVER: lu_decomposition_kernel.cu (getrf/getrs; deferred to hipSOLVER).
- CUTLASS/CuTe + wmma + TensorRT-flash: fused_dot_feature_interaction_kernel.cu (wmma),
  fused_self_attention_query_mul_key_and_value_kernel.cu (cublasGemmStridedBatchedEx),
  fused_attention_kernels/fused_glu_kernel/fused_multi_head_attention_inference (CUTLASS,
  already -DWITH_CUTLASS=OFF). CUTLASS does not port to ROCm (CK/ck_tile reimplementation).
- libcu++: lru_cache.cu (`cuda::std::semaphore`, no ROCm equivalent).
- P2P/IPC embedding shuffle: one_embedding_embedding_shuffle_p2p_kernel.cu,
  one_embedding_embedding_gradient_shuffle_p2p_kernel.cu.
- nvjpeg/npp image decode (NVIDIA-only) -- gated behind `!USE_HIP`.
The fixable-but-not-fixed bf16x2 broadcast (__float2bfloat162_rn -> a HIP helper in the
compat header) was added, clearing dropout_kernel.cu.

### REMAINING WORK to a linked liboneflow.so (continuation)
The 203 .cu compile; the host .cpp GPU TUs (ep/cuda/*.cpp etc.) get the per-target compat
flags but have not yet been driven to a clean compile + link of the full liboneflow.so.
Next steps for a continuation: (1) build the `oneflow` target to surface host-.cpp and
link faults (the host TUs reference the excluded ops via registries -- may need the
excluded kernels stubbed or their registrations guarded); (2) once liboneflow.so + the
oneflow_internal pybind module link, run the slice autotests on GPU.

## GPU validation (test oracle = PyTorch ROCm)
torch 2.13 ROCm (torch.cuda.is_available()==True on HIP) is importable as the autotest
oracle. 4 MI250X GCDs; run on one free GCD via HIP_VISIBLE_DEVICES (rocm-smi to pick).
Slice autotests (after monolith links): test_add, test_cast, test_where, test_masked_fill,
test_softmax, test_layer_norm, test_matmul, test_broadcast_ops, test_permute, run SERIALLY
(pytest, not -n) on one GCD.
