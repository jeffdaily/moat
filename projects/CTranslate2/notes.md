# CTranslate2 notes

## Validation 2026-05-31 (linux-gfx1100, gfx1100/W7800, ROCm 7.2.1) -- PASS

Fork: jeffdaily/CTranslate2 @ moat-port, HEAD dfa0d30dd18c4e65863e091f4ac99d7b936a02f1 (unchanged, no follower fork push)
GPU: 4x AMD Radeon Pro W7800 48GB, gfx1100 (RDNA3, wave32). Tested on HIP_VISIBLE_DEVICES=0.
Compiler: amdclang++ 22.0.0 (roc-7.2.1), cmake 3.31

Build command (gfx1100, adapted from lead):
```
cmake -S projects/CTranslate2/src -B projects/CTranslate2/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=amdclang -DCMAKE_CXX_COMPILER=amdclang++ \
  -DWITH_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DWITH_MKL=OFF -DWITH_OPENBLAS=ON -DOPENMP_RUNTIME=COMP \
  -DBUILD_TESTS=ON -DBUILD_CLI=OFF \
  -DCMAKE_CXX_FLAGS="-Wno-deprecated-literal-operator" \
  -DCMAKE_HIP_FLAGS="-Wno-deprecated-literal-operator"
bash utils/timeit.sh CTranslate2 compile -- cmake --build projects/CTranslate2/build -j 16
```
Result: compile CLEAN in 119.5s. Only -Wnodiscard on benchmark_ops (same as lead).

Code-object arch evidence (roc-obj-ls on libctranslate2.so):
All bundles: `hipv4-amdgcn-amd-amdhsa--gfx1100` (no gfx90a present). Confirmed via roc-obj-ls.

GPU test (CUDA/* filter, HIP_VISIBLE_DEVICES=0):
```
HIP_VISIBLE_DEVICES=0 bash utils/timeit.sh CTranslate2 test -- \
  projects/CTranslate2/build/tests/ctranslate2_test \
  projects/CTranslate2/src/tests/data --gtest_filter='CUDA/*'
```
Result: 164 PASSED, 3 SKIPPED (Conv1DGroupNoBiasQuantized x3 dtypes). MATCHES gfx90a bar.

Full suite (filter out Conv1DGroupNoBiasQuantized CPU abort):
```
HIP_VISIBLE_DEVICES=0 bash utils/timeit.sh CTranslate2 test -- \
  projects/CTranslate2/build/tests/ctranslate2_test \
  projects/CTranslate2/src/tests/data "--gtest_filter=-*Conv1DGroupNoBiasQuantized*"
```
Result: 350 PASSED, 1 SKIPPED (CPU/OpDeviceFPTest.Conv1DDilation). MATCHES gfx90a bar. No non-GPU regressions.

Determinism (wave32, agent_space/ct2_determinism_gfx1100.cc):
softmax/log_softmax/layer_norm/rms_norm across widths {15,32,96,1023,2048} run twice each on GPU: ALL BIT-IDENTICAL (20/20).

State transition: port-ready -> completed. validated_sha = dfa0d30. Fork untouched (no commit, no push).

## Validation 2026-05-31 (linux-gfx90a, gfx90a/MI250X, ROCm 7.2.1) -- PASS

Fork: jeffdaily/CTranslate2 @ moat-port, HEAD dfa0d30dd18c4e65863e091f4ac99d7b936a02f1
GPU arch: gfx90a (MI250X), device 2 (HIP_VISIBLE_DEVICES=2, 0% load at test time)
Compiler: amdclang++ 22.0.0 (roc-7.2.1), cmake 3.31

Compile:
```
cmake --build projects/CTranslate2/build -j 16
```
Result: all targets rebuilt from cache (no source change), compile CLEAN (0 errors, -Wnodiscard only on benchmark_ops, suppressed in build).

GPU test (CUDA/* filter, device 2):
```
HIP_VISIBLE_DEVICES=2 bash utils/timeit.sh CTranslate2 test -- \
  projects/CTranslate2/build/tests/ctranslate2_test \
  projects/CTranslate2/src/tests/data --gtest_filter='CUDA/*'
```
Result: 164 PASSED, 3 SKIPPED (Conv1DGroupNoBiasQuantized x3 dtypes -- intentional feature gap). MATCHES DOCUMENTED BAR.

Full suite (filter out Conv1DGroupNoBiasQuantized CPU abort):
```
HIP_VISIBLE_DEVICES=2 bash utils/timeit.sh CTranslate2 test -- \
  projects/CTranslate2/build/tests/ctranslate2_test \
  projects/CTranslate2/src/tests/data "--gtest_filter=-*Conv1DGroupNoBiasQuantized*"
```
Result: 350 PASSED, 1 SKIPPED (CPU/OpDeviceFPTest.Conv1DDilation -- intentional). MATCHES DOCUMENTED BAR. No non-GPU regressions.

Determinism (wave64 bar, agent_space/ct2_determinism.cc):
softmax/log_softmax/layer_norm/rms_norm/quantize_scale across widths {15,32,96,1023,2048} run twice each on GPU: ALL BIT-IDENTICAL (25/25).

State transition: review-passed -> completed. validated_sha = dfa0d30.

## Disposition: VERIFY/MODERNIZE -- mature upstream HIP backend, first CDNA2 (gfx90a/wave64) GPU validation

CTranslate2 ships a mature Strategy-A ROCm/HIP backend (WITH_HIP=ON: enable_language(HIP) +
LANGUAGE HIP source props + CT2_USE_HIP compat-header aliasing in src/cuda/{utils.h,primitives.cu,
allocator.cc,random.h,helpers.h}). Every upstream arch list targets RDNA only
(gfx1030;gfx1100;...;gfx1201); gfx90a/CDNA2 (wave64) was absent everywhere, and upstream CI runs
zero GPU tests on ROCm (CPU-container wheel builds only). The MOAT deliverable is the first CDNA2
GPU validation + any wave64 fix.

Result: the stock HIP backend builds for gfx90a with NO source change and the full GPU gtest suite
passes + is bit-deterministic on MI250X. No wave64 source fix was required. The fork change is the
minimal build-config enablement: add gfx90a to the two Linux ROCm arch lists.

## Lead platform: linux-gfx90a (MI250X, ROCm 7.2.1) -- VALIDATED

- Box: 4x MI250X GCDs (all gfx90a, devices 0-3). Pin a free GCD with HIP_VISIBLE_DEVICES (do not
  hardcode); validation ran on device 1.
- Build: amdclang/amdclang++ 22.0.0 (roc-7.2.1), cmake 3.31, OpenBLAS 0.3.26 (apt libopenblas-dev).
- Build out-of-the-box: configure succeeded with -DCMAKE_HIP_ARCHITECTURES=gfx90a (the library
  CMakeLists reads the cache var; no hardcoded literal arch list in CMakeLists.txt -- the RDNA lists
  live only in docker/Dockerfile_rocm + python/tools/prepare_build_environment_*_rocm.sh). Compile
  clean (158s, -j16); only -Wnodiscard warnings on benchmark_ops. hipBLAS enum/algo coverage
  (cublasGemmEx/StridedBatchedEx -> hipblas*, HIPBLAS_COMPUTE_16F/32F/32I) resolved at compile with
  no edit -- the plan's risk 3 is a non-issue on 7.2.1.

## Build script (gfx90a, BUILD_TESTS, OpenBLAS CPU backend)

```
cmake -S projects/CTranslate2/src -B projects/CTranslate2/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=amdclang -DCMAKE_CXX_COMPILER=amdclang++ \
  -DWITH_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DWITH_MKL=OFF -DWITH_OPENBLAS=ON -DOPENMP_RUNTIME=COMP \
  -DBUILD_TESTS=ON -DBUILD_CLI=OFF \
  -DCMAKE_CXX_FLAGS="-Wno-deprecated-literal-operator" \
  -DCMAKE_HIP_FLAGS="-Wno-deprecated-literal-operator"
cmake --build projects/CTranslate2/build -j 16
```

Prereq: `git -C projects/CTranslate2/src submodule update --init --recursive` (the depth-1 clone did
not init submodules; googletest is needed for tests). third_party/{thrust,cutlass} are fetched but
unused on the HIP build. A follower (gfx1100/gfx1151) builds the same source by changing only
-DCMAKE_HIP_ARCHITECTURES (upstream already targets those RDNA arches).

## GPU validation (the gate) -- PASS

Test binary: `ctranslate2_test` (gtest), takes the data dir as argv[1]:
`./tests/ctranslate2_test <src>/tests/data`. Device-parametrized: CPU/* run on CPU, CUDA/* run on
the AMD GPU on a HIP build (the build defines CT2_WITH_CUDA so the CUDA instantiations compile).

- GPU-only set `--gtest_filter='CUDA/*'`: 164/167 PASS, 3 SKIPPED. The 167 CUDA cases cover the full
  wave64 reduction surface: SoftMax/LogSoftMax/MaskedSoftMax/MaskedSoftMaxTriangular, LayerNorm/
  LayerNormAxis/RMSNorm, TopK/TopPMask/Multinomial, QuantizeINT8, Gemm/GemmBias/GemmInt8/GemmGELU,
  Conv1D, plus CUDA/LayerDeviceFPTest (full transformer layers: attention/FFN/embedding, FP32+FP16+
  BF16 end-to-end on GPU), CUDA/PrimitiveTest (hipBLAS GEMM + device memory), CUDA/BiasedDecoding,
  CUDA/StorageViewDeviceTest. Real GPU execution confirmed via AMD_LOG_LEVEL=3 (15 hipLaunchKernel ->
  hipSuccess per SoftMax test).
- Full suite (CPU+GPU+inference), `--gtest_filter='-*Conv1DGroupNoBiasQuantized*'`: 350/351 PASS, 1
  SKIPPED (CPU dilated-conv, intentional). Includes TranslatorTest/ModelTest/DecodingTest/
  BatchingTest/CrossAttentionTest on the bundled tiny models under tests/data.

### Determinism (PORTING_GUIDE wave64 bar) -- PASS, bit-identical

The gtest pass/skip set is identical across two runs. Beyond that, a standalone harness
(agent_space/ct2_determinism.cc) runs softmax/log_softmax/layer_norm/rms_norm/quantize on the GPU
twice with fixed input over row widths {15,32,96,1023,2048} (sub-warp, exact-warp, multi-warp,
odd-ILP-epilogue, 32-partition) and memcmp's the raw output bytes: ALL BIT-IDENTICAL. The
__syncwarp(mask) tail in block_reduce introduces no wave64 race on gfx90a.

### Intentional skips (not regressions)

- CUDA/OpDeviceFPTest.Conv1DGroupNoBiasQuantized (x3 dtypes): the test itself GTEST_SKIPs for
  non-CPU devices ("Grouped quantized convolution is not implemented for CUDA", ops_test.cc:1325).
  This is a CUDA-path feature gap that predates the port; the HIP build inherits it. Not a defect.
- CPU/OpDeviceFPTest.Conv1DDilation: "Dilated convolution is not implemented for CPU" (intentional).

### Known CPU-backend artifact (orthogonal to the port)

`CPU/OpDeviceFPTest.Conv1DGroupNoBiasQuantized/float32` THROWS ("No INT8 GEMM backend for CPU") and
aborts the whole binary because this validation builds OpenBLAS-only (-DWITH_OPENBLAS=ON, no
oneDNN/MKL). The CPU int8 GEMM backend comes from MKL/DNNL, which we omit for a correctness build.
This would also throw on an x86 CUDA build with the same OpenBLAS-only CPU config; it is not a
GPU/HIP regression and the GPU variants of this test skip on their own. Run the suite with
`--gtest_filter='-*Conv1DGroupNoBiasQuantized*'` for a clean pass, or build with -DWITH_DNNL=ON to
get the CPU int8 path (not needed to validate the GPU port).

## Wave64 analysis: src/cuda/helpers.h C10_WARP_SIZE 32 -- left UNCHANGED (correct on wave64)

helpers.h:398 `#define C10_WARP_SIZE 32` is unconditional and drives the host `get_block_size()`
(:407, computes the launch dim3) and the device `block_reduce()` (:411-455) used by softmax_gpu.cu
and quantize_gpu.cu. The plan flagged this as the PRIMARY wave64 risk. Conclusion after analysis +
hardware validation: leave it at 32.

Why 32 is correct here (it is a logical shared-memory partition size, NOT the hardware warp size):
block_reduce writes all blockDim.x values to __shared__, __syncthreads(), then the first
C10_WARP_SIZE threads each reduce a contiguous C10_WARP_SIZE-stride group from shared memory
(smem[lane*32 + i]), write smem[lane], __syncthreads(), and thread 0 reduces smem[0..blockDim.x/32).
The coverage is exact for ANY partition size W that divides blockDim.x with blockDim.x/W <= W,
regardless of the hardware wavefront width, because every value is read from the shared memory the
whole block wrote (not from lane registers). get_block_size returns powers of two >= 32, so
blockDim.x is always a multiple of 32 and blockDim.x/32 <= 1024/32 = 32. So the reduced VALUE is
correct on wave64 unchanged.

Critically, do NOT key C10_WARP_SIZE on __GFX9__ here: that is the raft/cuvs HOST/DEVICE WarpSize
split trap (PORTING_GUIDE 2026-05-31). get_block_size is HOST code where __GFX9__ is undefined, so
it would read 32 and launch a 32-thread block, while the device block_reduce would partition by 64
and compute blockDim.x/64 = 0 warps -> the reduction loop never runs -> zero/garbage output. Host
and device MUST use the same W; 32 is the value that is simultaneously correct on CUDA, RDNA, and
CDNA, so the single unconditional define is the right abstraction.

The only genuine wave64 hazard was the `__syncwarp(mask)` tail (helpers.h:435, mask =
(1<<(blockDim.x/32))-1) executed by only the first blockDim.x/32 lanes of a 64-lane wavefront -- the
MPPI-Generic partial-mask warp-sync class. But here the mask explicitly names exactly the active
lanes (all within the first wavefront, since blockDim.x/32 <= 32 < 64) and the WaR hazard it guards
(lane 0 reading smem[1] vs lane 1 writing smem[1]) is among lanes that are all in that one
wavefront and all named in the mask. HIP/gfx90a handles this correctly: 164/167 GPU tests pass and
the determinism harness is bit-identical. So no MPPI-style "drop the tail" fix is needed (and it
would be unsafe here -- a bare drop reintroduces the inter-lane WaR race, and a __syncthreads inside
the divergent `if (threadIdx.x < C10_WARP_SIZE)` branch would hang).

## Out of scope on HIP (expected-unsupported, not defects)

- AWQ-int4 (src/ops/awq/*_gpu.cu) and FlashAttention: REMOVED from the HIP build (CMakeLists.txt
  708-723). These are the CUTLASS/sm80 kernels; CUTLASS does not port to ROCm. No tests reference
  them. The gemv_gpu.cu `#define WARP_SIZE 32` + __shfl_down_sync(0xffffffff) and the
  flash-attention utils.h __shfl_xor_sync(uint32_t(-1)) are in these excluded files -- moot.
- cuDNN conv1d (conv1d_cudnn_gpu.cu, WITH_CUDNN) is CUDA-only; HIP uses the generic im2col+GEMM
  conv1d_gpu.cu (validated: CUDA/OpDeviceFPTest.Conv1D* pass).
- NCCL/RCCL tensor parallelism (CT2_WITH_TENSOR_PARALLEL) is incompatible with WITH_HIP
  (CMakeLists.txt:687 FATAL_ERROR) -- out of scope.

## Fork change (minimal, NVIDIA-safe, RDNA-safe)

Library source is byte-identical to upstream (the wave64 fix was validated unnecessary). The only
change adds gfx90a to the two Linux ROCm arch lists so CDNA2 is a documented, wheel-buildable target:
- docker/Dockerfile_rocm:70 HIP_ARCHITECTURES default
- python/tools/prepare_build_environment_linux_rocm.sh:43 PYTORCH_ROCM_ARCH
Both prepend gfx90a; existing RDNA wheels and the NVIDIA/CUDA build are untouched. The Windows ROCm
prepare script is left alone (gfx90a is a Linux CDNA datacenter part; Windows targets RDNA gfx1151).

## Followers

gfx1100 (RDNA3, wave32) and gfx1151 (RDNA3.5, wave32) are arches upstream ALREADY targets and ships
wheels for. They build the same source with only -DCMAKE_HIP_ARCHITECTURES changed. On wave32 the
C10_WARP_SIZE 32 partition matches the hardware warp exactly (the historically-tested path), so no
new wave64 concern. Validate on their own hardware per MOAT follower flow.

## Review 2026-05-31 (reviewer, lead linux-gfx90a) -- PASS, no changes requested

Reviewed moat-port HEAD dfa0d30 vs upstream base 5dfc5d8 via /pr-review (local-branch mode).
No problems found. Verdict: review-passed; safe to proceed to GPU validation toward completed.

Verification trail (all load-bearing claims independently re-derived from the fork tree):
- Minimal footprint CONFIRMED: `git diff 5dfc5d8...HEAD` touches exactly 2 files, 1 line each --
  docker/Dockerfile_rocm:70 and python/tools/prepare_build_environment_linux_rocm.sh:43, each
  prepending `gfx90a;` to an RDNA-only list. src/ include/ CMakeLists.txt cmake/ third_party/ diff
  is empty: library source byte-identical to upstream.
- NVIDIA + RDNA builds unperturbed: docker `ARG HIP_ARCHITECTURES` (line 69) precedes the `ENV ...:-`
  default (line 70), so the new value is only a fallback default; any build passing --build-arg or a
  CUDA build (separate CMAKE_CUDA_ARCHITECTURES path) is unaffected. Change is additive (prepend), no
  prior target dropped. Windows ROCm prepare script (line 22) correctly left RDNA-only.
- block_reduce/C10_WARP_SIZE=32 wave64 analysis VALID. Only two callers of the hand-rolled
  cuda::block_reduce: softmax_gpu.cu:222/228 and quantize_gpu.cu:70, both launching blockDim =
  get_block_size() (power of two, >=32, <=max_threads=1024 -> blockDim.x/32 in [1,32]) with smem
  sized blockDim.x*sizeof(float). First-warp loop covers smem[0,blockDim.x) exactly (no gap/overlap/
  OOB); all inputs read from shared memory after the line-422 __syncthreads (no __shfl in the
  function) so the VALUE is wave-width-independent. The one genuine intra-wavefront WaR hazard (lane
  0 reading smem[1..N-1] at :433 vs lanes 1..N-1 writing smem[lane] at :436, all within the first 32
  lanes = first half of one wave64 wavefront) IS correctly ordered by __syncwarp(mask) at :435 with
  mask=(1<<(blockDim.x/32))-1 naming exactly those active lanes. So the tail must be KEPT (an
  MPPI-style "drop the __syncwarp" fix would reintroduce that race); leaving it is correct. On HIP
  __syncwarp(MaskT) (amd_warp_sync_functions.h:176) takes the uint64_t mask and syncs the wavefront,
  so it cannot under-synchronize. Keying C10_WARP_SIZE on __GFX9__ would break the host get_block_size
  (host has no __GFX9__) -> host/device split; the unconditional 32 is the correct shared abstraction.
- Other reduction-surface ops (rms_norm/layer_norm/topk/multinomial/mean) use cub::BlockReduce, and
  utils.h:15 `#define cub hipcub` routes them to rocPRIM (wave-size-aware by construction) -- no
  hardcoded-32 or hand-rolled shuffle path there. Library swaps correct (find_package hipblas/hiprand/
  rocprim/rocthrust/hipcub; Strategy A `set_source_files_properties(... LANGUAGE HIP)`).
- Out-of-scope exclusions are upstream's existing HIP CMake (unchanged by the fork): WITH_TENSOR_PARALLEL
  and WITH_FLASH_ATTN -> FATAL_ERROR, AWQ/CUTLASS sources REMOVE_ITEM'd. Consistent with the notes.
- Commit hygiene CLEAN: title "[ROCm] Add gfx90a (CDNA2/MI250X) to the ROCm build arch lists" = 61
  chars; body discloses Claude (Anthropic); Test Plan with literal commands; no Co-Authored-By/noreply,
  no ghstack, no AMD-internal account ref (author jeff.daily@amd.com is the user's own public commit
  identity, permitted). Fork Actions disabled ("enabled":false).
- GPU run not re-executed at review (validator's stage, per PORTING_GUIDE 2026-05-30); porter recorded
  CUDA/* 164/167 +3 skip, full 350/351 +1 skip, determinism bit-identical. Not a blocker for review.
