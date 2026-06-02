# ZhiLight notes

Fork: https://github.com/jeffdaily/ZhiLight  (branch `moat-port`)
Lead platform: linux-gfx90a (CDNA2, wave64). ROCm 7.2.1, ROCm PyTorch 2.13.

## Build (gfx90a)

```
HIPCXX=/opt/rocm/lib/llvm/bin/clang++ ZHILIGHT_USE_HIP=1 ENABLE_NCCL_TP=on TESTING=1 \
cmake -S projects/ZhiLight/src -B agent_space/zhilight-build -GNinja \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ -DCMAKE_HIP_PLATFORM=amd \
  -DCMAKE_BUILD_TYPE=Release -DWITH_TESTING=ON -DPYTHON_EXECUTABLE=$(which python3)
cmake --build agent_space/zhilight-build --target C internals_ -j16
```

Followers (gfx1100/gfx1151) reuse the same commit with only
`-DCMAKE_HIP_ARCHITECTURES=<arch>` (no source edit; the targets read the cache var).

## Validate (real GPU, GCD 1)

The validation gate is the in-tree torch-reference kernel tests (no model weights),
which exercise the wave64-sensitive reduce/softmax/attention/ff paths and the
hipBLASLt GEMM. Symlink the two built modules into the package, then:

```
ln -sf agent_space/zhilight-build/C.cpython-312-x86_64-linux-gnu.so \
       projects/ZhiLight/src/zhilight/
ln -sf agent_space/zhilight-build/tests/py_export_internal/internals_.cpython-312-x86_64-linux-gnu.so \
       projects/ZhiLight/src/zhilight/
cd projects/ZhiLight/src
HIP_VISIBLE_DEVICES=1 PYTHONPATH=. python3 -m pytest \
  tests/test_softmax.py tests/test_attn_softmax.py tests/test_rotary_embedding.py \
  tests/test_feedforward.py tests/test_linear.py tests/test_embedding.py \
  tests/test_arthmetic.py tests/test_concat_tensor.py tests/test_index_along_dim.py \
  tests/test_log_prob.py tests/test_attention.py tests/test_lazy_loader.py
# 80 passed on gfx90a, deterministic across runs.
```
(Remove the two `*.so` symlinks before committing; they are build artifacts.)

## Port shape (Strategy A)

- `hip_compat/`: one central `zhilight_cuda_to_hip.h` (force-included on HIP TUs)
  plus toolkit-named forwarding shims (cuda_runtime.h, cublasLt.h, cublas_v2.h,
  nccl.h, curand.h, cuda_fp16/bf16.{h,hpp}, mma.h, vector_types.h, library_types.h,
  cub/...) on the HIP-only include path so host .cpp resolve CUDA spellings to HIP
  without editing include lines. CUDA build never sees this dir.
- USE_HIP gating in the three CMakeLists (top, 3rd/bmengine, 3rd/bmengine/bmengine)
  and the internals test CMake. cuBLAS/cuBLASLt/cuRAND/NCCL -> hipBLAS/hipBLASLt/
  hipRAND/RCCL. `.cu` marked LANGUAGE HIP; `--offload-compress` on the HIP TUs.

## Gotchas / lessons (gfx90a)

- WAVE64 ROOT CAUSE (reduce.cuh): warpReduce* used 32-lane offsets + 0xFFFFFFFF
  masks and blockReduce* hardcoded %32 / /32 / shared[33]. On wave64 only 32 of 64
  lanes were reduced -> silently wrong amax/amin/softmax (no crash). Fixed
  wave-agnostically (BM_WARP_SIZE per __GFX9__, full-width no-mask shuffles on HIP,
  num_warps-driven block reduce). amax/amin tests fail before, pass after.
- round_up_thread() (core/utils.h) rounded block size to 32. On wave64 that
  launches a HALF-filled wavefront, so a block reduce's full-wavefront shuffle
  reads inactive lanes -> wrong amax/amin. Round to 64 on HIP (correct on wave32
  too, just over-provisioned). This is the second half of the wave64 fix.
- attention_kernel.cu scalar paths tile a 128-dim head as 32 lanes * 4 (lane*4
  short4 loads). Kept WARP_SIZE=32 logical warps and used WIDTH-32 reductions
  (warpReduceSumWidth<float,32> + attnShfl32 width-32 shuffles), correct on wave32
  AND wave64. The wmma tensor-core variants are gated off (use_mma=false on HIP;
  KERNEL_mqa_self forced to the scalar high-precision path).
- hipBLASLt rejects 16F compute/scale for fp16 GEMM (cublasLtMatmulDescCreate ->
  INVALID_VALUE on gfx90a). Use HIPBLAS_COMPUTE_32F + 32F scale with fp16 I/O.
  The cublasLt algo-introspection API (AlgoInit / AlgoConfigGetAttribute / tile+
  stage enums / search-by-algo-id) has no hipBLASLt analogue; find_algo() is
  rewritten under USE_HIP to take the top hipblasLtMatmulAlgoGetHeuristic result.
- pybind11 enables IPO/LTO; the HIP link does not finalize LTO, leaving a slim
  PyInit-less .so (ImportError). Use pybind11_add_module(... NO_EXTRAS ...) +
  INTERPROCEDURAL_OPTIMIZATION OFF on the HIP build (gpuRIR lesson).
- Torch (c10) was pulled transitively via the flash-attn .so on CUDA; with flash
  dropped on HIP, link torch/torch_cpu/c10 explicitly into `backend` or the C /
  internals_ modules fail with undefined c10::detail::torchCheckFail.
- ROCm PyTorch 2.13 c10 headers (TensorImpl SetDimsTemplate) use C++20 `requires`;
  bump CMAKE_CXX_STANDARD/CMAKE_HIP_STANDARD to 20 on the HIP build (CUDA stays 17).
- hip_bf16.h is clang-only (uses __builtin_elementwise_*), so host g++ TUs cannot
  include it; the compat header includes it only under __clang__ (no host .cpp uses
  the bf16 type directly). hip_fp16.h compiles fine under g++.
- The internals_ test FeedForward::load_state_dict spawned a worker thread that
  touches numpy buffer info (py::array) without the GIL while the main thread
  holds it and joins -> deadlock (looked like a kernel hang). Run inline on the
  single-GPU path; release the GIL around the join in the multi-GPU path. Pure
  test-harness fix, not a library bug.
- CMake self-reset loop: find_package(Torch) in the internals subdir finalizes
  CMAKE_HIP_COMPILER, which differs from a -D value by literal path string and
  triggers a cache-deleting reconfigure that drops -DUSE_HIP. Use torch's exact
  compiler path (/opt/rocm/lib/llvm/bin/clang++), set CMAKE_HIP_PLATFORM=amd, and
  honor ZHILIGHT_USE_HIP=1 env so USE_HIP survives the rerun.
- __ldcs -> __ldg, __grid_constant__ -> empty, dim3 braced-init `{32,32}` in a
  launch config -> dim3(32,32) (clang/HIP `<<<>>>` parser rejects the braced list).
- rocPRIM radix sort has no codec for __hip_bfloat16 keys (half and float are
  fine); the bf16 sort (sort.cu, random_util.cu top-p sampling) is gated to a
  runtime error on HIP rather than shipping an order-incorrect bit-cast sort.

## Deferred follow-up (NVIDIA tensor-core / Hopper-only, reimplement-not-port)

These are gated OFF under USE_HIP (throw a clear error or fall back); an AMD-native
pass (rocWMMA / Composable Kernel / MFMA, ck_tile preferred) can re-add them:

- marlin / awq / fp8 / gptq quant kernels: mma.sync, ldmatrix, cp.async,
  cvt.e4m3x2, prmt.b32, lop3, __vsub4. So GPTQ/AWQ/Marlin/fp8 quantized Linear +
  MOE are unsupported on AMD now; fp16/bf16 and cublasLt-int8 Linear are supported.
- wmma tensor-core attention decode path (multiply_q_k_block_wmma etc.) and the
  int8-quantized-KV attention (quant_attention.cuh prmt/sub.f16x2 PTX).
- deep_gemm (deep_gemm_fp8_block_h20_group) + flash_mla H20 prebuilt static libs:
  the DeepSeek-V3 FP8-block and MLA fast paths. MLA falls back to the generic
  attention path. These are the headline 2025 features and the most visible gap.
- flash-attn prefill (.so is a CUDA binary): prefill uses the in-tree attention
  kernels. ROCm flash-attn could be wired later.

## Review 2026-06-02 (reviewer, linux-gfx90a, fork moat-port @ 9bb3e8b vs ee84468)

Verdict: Request Changes (one latent wave64 correctness defect to fix or explicitly de-risk; one inaccurate claim in the commit/notes). The wave64 reduce rewrite, the deferred-path gating, the hipBLASLt GEMM rewrite, the compat header, the library swaps, and commit hygiene are all correct. Findings below are problems only.

### Fault Classes

1. (wave64, latent silent-wrong) src/nn/quant/int8/quant_reduce_kernel.cu:26, :56, :267 call functions::warpReduceMaxB<T>() to reduce one 32-element group, but warpReduceMaxB now spans BM_WARP_SIZE (64 on gfx90a). The launch configs make this wrong on wave64 two different ways:
   - KERNEL_quant_group_32 / KERNEL_quant_group_32_v2 launch dim3(32,32) (quant_reduce_kernel.cu:92): the 64-lane wavefront packs threadIdx.y in {0,1} x threadIdx.x in {0..31}, i.e. TWO rows. warpReduceMaxB with offset 32 then folds the next row's value into this row's abs_max -> cross-row contamination.
   - KERNEL_dequant_sum_quant_g32 launches <<<M, GROUP_SIZE=32>>> (quant_reduce_kernel.cu:312): a 32-thread block is a half-filled wavefront on wave64; warpReduceMaxB at offset 32 reads inactive lanes 32-63.
   The diff only converted the brace-init {32,32} -> dim3(32,32) (a clang/HIP syntax fix) and did NOT convert these reductions to the width-32 form (warpReduceMaxWidth<T,32>) that was added to reduce.cuh precisely for 32-lane-tiled kernels and used correctly in attention_kernel.cu. Severity is reduced because the only caller, ModelContext::reduce_tp_int8 (model_context.cpp:225), is gated on get_compute_capability() == 89 (model_context.cpp:224) -- an NVIDIA Ada SM id; on gfx90a hipDeviceProp major.minor yields cc 90, so reduce_tp_int8 is unreachable today and the 80 single-GPU tests do not exercise it. But the code is silently wrong if ever reached (e.g. a follower relaxing the cc gate), and notes.md line 107 / the commit body list "cublasLt-int8 Linear" and imply the int8 surface is supported. Fix: route these three warpReduceMaxB calls through warpReduceMaxWidth<float,32> (or assert the path is unreachable on HIP), and correct the "supported" wording to scope it to Int8Linear (hipBLASLt) only, excluding the int8-TP all-reduce quant kernels.

### Backward Compatibility (upstream)

2. (inaccurate claim) The commit body states the NVIDIA build is "byte-for-byte unchanged (every edit is behind USE_HIP / __HIP_PLATFORM_AMD__)", but the blockReduce* rewrite in reduce.cuh and attention_kernel.cu changes the CUDA codegen too: the partial-selection guard went from `threadIdx.x < blockDim.x / 32` (original) to `lane < num_warps` with num_warps = ceil(blockDim.x / BM_WARP_SIZE) (reduce.cuh:127,131; attention_kernel.cu:426-440). On CUDA BM_WARP_SIZE is 32, but ceil vs the old floor differ for non-multiple-of-32 block sizes, and `lane <` vs `threadIdx.x <` differ for wid>0. This is a real (and arguably more-correct) change to the NVIDIA path, so it is NOT behind a USE_HIP guard. Either wrap the guard rewrite in `#if defined(USE_HIP)` to keep the CUDA path literally unchanged, or correct the commit/notes wording to "the NVIDIA dispatch and the deferred fast paths are unchanged; the shared block-reduce partial-selection guard is tightened on both paths (a no-op for the 32-multiple block sizes used here)".

### Verified clean (no action)

- reduce.cuh wave rewrite: BM_WARP_SIZE per __GFX9__, full-width no-mask __shfl on HIP, num_warps-driven block reduce, shared[33] safe (max 16 partials on wave64, 32 on wave32). round_up_thread() rounds to 64 on HIP (utils.h). The amax/amin/softmax/sum block-reduce paths (arthmetic.cu reduce_abs_max/amax_kernel/amin_kernel/sum_kernel, all blockReduce* with round_up_thread block sizing) are wave-correct -- these are the test-covered paths.
- Deferred NVIDIA/Hopper paths cleanly gated OFF and fail loudly: marlin/awq/fp8/gptq excluded in CMakeLists; wmma attention path (#if !USE_HIP + use_mma=false forcing scalar); int8-quantized-KV attention (quant_attention.cuh whole-file #if !USE_HIP + BM_EXCEPTION at the launcher); GPTQ/AWQ/Marlin/fp8 Linear + MOE (BM_EXCEPTION in the factories); deep_gemm/flash_mla (CMake NOT USE_HIP + ds_flash_mla_api throws when ENABLE_DS_FLASH_MLA undefined; USE_FLASH_MLA defaults off and falls back to attn_by_gemm); flash-attn prefill (run_mha_fwd redefined to throw). No deferred path runs a broken kernel.
- hipBLASLt GEMM rewrite (gemm.cpp): 32F compute+scale for fp16 I/O (16F rejected on gfx90a), find_algo() rewritten to take the top hipblasLtMatmulAlgoGetHeuristic result; CUDA algo-introspection path kept under #else. attention_kernel.cu uses width-32 logical-warp reductions (warpReduceSumWidth<float,32> / attnShfl32) for the lane*4 128-dim-head tiling -- correct on wave32 and wave64.
- Compat header (zhilight_cuda_to_hip.h): cstring/cstdlib before hip_runtime, bf16 only under __clang__, __ldcs->__ldg, __grid_constant__ dropped, no #endif-without-#if. Forwarding shims map cublas/cublasLt/curand/nccl/cub correctly; CUDA build never sees hip_compat/.
- Library swaps: cuBLAS/cuBLASLt/cuRAND -> hipBLAS/hipBLASLt/hipRAND, NCCL -> RCCL, cub -> hipcub. rocPRIM bf16-key sort gated to a runtime error rather than an order-incorrect bit-cast (sort.cu, random_util.cu).
- Build: USE_HIP opt-in, enable_language(HIP), CMAKE_HIP_ARCHITECTURES defaults gfx90a only when unset (followers need no source edit), .cu marked LANGUAGE HIP (not renamed), CUDA path intact behind NOT USE_HIP, pybind11 NO_EXTRAS + IPO OFF, --offload-compress.
- Commit hygiene: title "[ROCm] Portable HIP build for ZhiLight on AMD (gfx90a first pass)" 53 chars, Claude disclosed, no Co-Authored-By/noreply trailer, no ghstack, no em-dash. Fork main = ee844680 (clean upstream mirror), moat-port = 9bb3e8b, Actions disabled on the fork. No AMD-internal account references.
- Note for the validator: the supported int8 Linear (Int8Linear via hipBLASLt) and the int8-TP all-reduce are not exercised by the 80 standalone kernel tests; test_linear.py covers the fp16 NormalLinear GEMM. Not a review blocker (GPU run is the validator's stage), but the int8 Linear surface is unverified by the listed gate.
