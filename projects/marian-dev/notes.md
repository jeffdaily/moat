# marian-dev notes

## Port summary (linux-gfx90a, lead)
Strategy A: a single `src/tensors/gpu/cuda_to_hip.h` compat header (runtime
cuda*->hip* aliases + a CUDA_VERSION>=11000 sentinel), plus forwarding shim
headers under `src/tensors/gpu/hip_compat/` (`cuda.h`, `cuda_runtime.h`,
`cuda_fp16.h`, `cublas_v2.h`, `cublasLt.h`, `cusparse.h`, `cusparse_v2.h`,
`curand.h`, `cooperative_groups.h`). The legacy FindCUDA `cuda_add_library`
call is satisfied by a cupoch-style shim macro under `option(USE_HIP)` that
marks the .cu and mixed .cpp TUs `LANGUAGE HIP`. NVIDIA path is unchanged
(every divergence is `#if defined(USE_HIP)` / `else`).

Fork: https://github.com/jeffdaily/marian-dev (branch moat-port). Actions
disabled on the fork.

## Build (gfx90a)
ROCm 7.2.1 at /opt/rocm. CMake 4.x needs the policy-min shim. Init only the
sentencepiece submodule (NOT nccl). Build in a scratch dir, never the repo root.
```
cd projects/marian-dev/src
git submodule update --init src/3rd_party/sentencepiece
export HIP_VISIBLE_DEVICES=3 ROCM_PATH=/opt/rocm
cmake -S . -B build-hip -G Ninja \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCOMPILE_CUDA=ON -DUSE_CUDNN=OFF -DUSE_NCCL=OFF \
  -DUSE_FBGEMM=OFF -DCOMPILE_CPU=ON -DCMAKE_BUILD_TYPE=Release \
  -DCOMPILE_TESTS=ON -DUSE_MKL=OFF -DUSE_TCMALLOC=OFF -DUSE_DOXYGEN=OFF \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DBUILD_ARCH=native
cmake --build build-hip -j
```
Followers (gfx1100/gfx1151): change only `-DCMAKE_HIP_ARCHITECTURES`; no source
edit (HIP_ARCHITECTURES drives --offload-arch; the wave64 fix is wave-agnostic).

## THE correctness fix (wave64): topk.cu + nth_element.cu
Both used the classic warp-synchronous unrolled reduction tail
(`UNROLL_MAXARG_LOOP(32/16/.../1)`) with no barrier on non-volatile shared
memory below 32 lanes. On CDNA wave64 the low 32 lanes of a 64-lane wavefront
are NOT lockstep, so that tail races -> non-deterministic argmax. nth_element is
the beam-search top-k on the decode path, so a wrong result silently corrupts
translations. Fixed (USE_HIP-guarded): drop the unrolled tail and run the
`__syncthreads()`-synchronized tree to `s>0` (same compare order, block-wide
barrier, correct on any wave size). CUDA path byte-identical.

Verified on gfx90a (GCD 3): operator_tests topk section (exact top-k values +
argmax/argmin/sort) passes; end-to-end beam-search decode is bit-identical
run-to-run AND matches the CPU decode exactly; `gMaxElement`/`gMaxElementUpdate`
confirmed dispatched on gfx90a via AMD_LOG_LEVEL=3.

## Validation (real gfx90a, GCD 3)
- Unit suites PASS: graph_tests, attention_tests, transformer_tests,
  operator_tests (284/287 assertions; the 3 failures are the cuSPARSE csr-dot
  path only -- see follow-ups). These exercise element-wise ops, softmax,
  layernorm, the reduce_all.h block reduction (wave64-correct as-is),
  prod/affine (hipBLAS + the hipBLASLt bias/ReLU epilogue), attention, and the
  full Transformer fwd+bwd.
- End-to-end: trained a tiny Transformer (reverse-copy toy task) to convergence
  on GPU, then beam-search decoded. GPU run1==run2 (deterministic) and GPU==CPU
  output. Commands:
```
# train (word vocab to avoid spm vocab-size floor on a toy corpus)
marian --type transformer -t train.src train.tgt -m model.npz \
  --vocabs vocab.src.yml vocab.tgt.yml --dim-emb 64 --transformer-dim-ffn 128 \
  --transformer-heads 2 --enc-depth 2 --dec-depth 2 --after 600u --devices 0
# decode twice on GPU + once on CPU, then diff
marian-decoder -m model.npz -v vocab.src.yml vocab.tgt.yml -i test.src -b 6 --devices 0
marian-decoder -m model.npz -v vocab.src.yml vocab.tgt.yml -i test.src -b 6 --cpu-threads 1
```
- rnn_tests: the 3 hardcoded-reference comparisons differ because hipRAND and
  cuRAND produce different streams for the same seed (the references at
  rnn_tests.cpp are cuRAND-specific glorotUniform draws). The kernels are
  correct; the 21 shape/structure assertions pass. Not a port bug.

## Library mapping
- cuBLAS->hipBLAS, cuBLASLt->hipBLASLt, cuSPARSE->hipSPARSE, cuRAND->hipRAND,
  Thrust/CUB->rocThrust (thrust::cuda::par -> thrust::hip::par under USE_HIP).
- hipBLASLt deltas handled in prod.cpp (all USE_HIP-guarded):
  - The `MATMUL_PREF_MIN_ALIGNMENT_{A,B,C,D}_BYTES` preference attrs do NOT
    exist in hipBLASLt -> dropped.
  - `hipblasLtMatmulDescCreate` rejects HIPBLAS_COMPUTE_16F and a 16F scale
    type: use HIPBLAS_COMPUTE_32F as the compute type and HIP_R_32F as the desc
    scale type for both fp16 and fp32 (matrix dtypes still set on the layouts).
  - `cublasSetMathMode(CUBLAS_TENSOR_OP_MATH)` returns "not supported" on
    gfx90a; setTensorMode/tensorOpsEnabled/unsetTensorMode are no-ops on HIP
    (gfx90a drives its matrix cores from the GEMM data/compute types directly).
  - cublasGemmEx compute-type slot -> HIPBLAS_COMPUTE_* (vs cudaDataType on
    CUDA); hipblasGemmBatchedEx array args are `const void*[]` (vs
    `const void* const[]`) -> per-platform cast macros.
  - cublasLt handle: CUDA encapsulates an lt handle inside the cublas handle;
    hipBLASLt needs a dedicated handle -> Backend::getCublasLtHandle() on HIP.

## __CUDACC__ / FP16 steering (important gotcha)
Do NOT globally `#define __CUDACC__` to take Marian's FP16 path: rocThrust keys
its device-system detection on `__CUDACC__` and would select its (unimplemented)
CUDA/CUB backend, producing "unimplemented for this system" template errors.
Instead the compat header only defines CUDA_VERSION, and types.h / operators.h /
defs.h / tensor.h / cpu/element.h were edited to treat `__HIPCC__` like
`__CUDACC__` for the host-intrinsics + FP16 steering. fp16 device-vs-host data
selection uses `__HIP_DEVICE_COMPILE__` (e.g. the LSH atomicAdd_block guards in
tensor_operators.cu; `atomicAdd_block` maps to `atomicAdd` on HIP).

Other gotchas:
- `halfx2(1.f)` is ambiguous on HIP (both __half and __half2 take a scalar) ->
  added a USE_HIP-only `halfx2(float)` ctor.
- alibi.cu used `thrust::tie`/`thrust::tuple` in device code (not device-callable
  on rocThrust) -> a trivial POD pair under USE_HIP.
- topk.cu's CUB segmented-sort path is not ported; Sort() falls back to the
  rocThrust sort_by_key path on HIP (only used by tests).
- The compat header is force-included on every HIP TU via CMAKE_HIP_FLAGS and on
  Marian's own host targets (lib/exe/tests) via target flags -- NOT globally,
  because the gnu++11 3rd_party libs (intgemm) choke on the HIP headers.
- HIPBLASLT_USE_ROCROLLER define is injected by find_package(hipblaslt); benign.

## Deferred follow-ups (separable, NOT blockers for the dense NMT path)
1. cuSPARSE sparse-attention CSRProd (prod_sparse_cu11.h): hipSPARSE SpMM with
   the row/col-major + CSR_ALG2 setup gives wrong fp32 results and rejects fp16
   (HIPSPARSE_STATUS_NOT_SUPPORTED). This is the only failing unit path
   (operator_tests csr-dot). Used only by sparse-attention/LSH models, not the
   default dense Transformer. Needs hipSPARSE-specific order/alg work.
2. cuDNN convolution/pooling -> MIOpen: kept CUDA-only (USE_CUDNN=OFF). All
   cuDNN code is #ifdef CUDNN-gated; the char-CNN conv/pool path is not built.
   The `pooling` app test is dropped from the HIP build.
3. NCCL -> RCCL multi-GPU collectives: USE_NCCL=OFF for the lead (single-GPU
   train/decode validates the full kernel + BLAS surface). RCCL is API-
   compatible and ships in ROCm; a follow-up.

## Review 2026-06-02 (reviewer, linux-gfx90a, fork moat-port 25f910c vs base c9f287d)
Verdict: review-passed. No changes requested. Strategy A (compat header + cuda_add_library LANGUAGE-HIP shim) is correct for this legacy-FindCUDA pure-CMake build; the NVIDIA and CPU paths are byte-identical (every divergence is USE_HIP / __HIPCC__ guarded, and __CUDACC__->__CUDACC__||__HIPCC__ extensions are no-ops when neither is defined).

Re-ran the GPU suites myself on gfx90a (GCD 3) after an incremental rebuild at HEAD: graph (10/10), attention (6/6), transformer (3/3) all pass; operator 284/287 with the only 3 failures isolated to the `csr-dot product` SECTION (operator_tests.cpp:539,609,610), i.e. the deferred cuSPARSE SpMM path -- the dense `dot product` (508) and `topk operations` (1026, exact-value top-k/argmax/argmin/sort) pass. End-to-end decode artifacts confirm the load-bearing wave64 fix: GPU run1==run2 (deterministic) AND GPU==CPU (correct beam-search top-k). The topk.cu/nth_element.cu fix replaces the warp-synchronous UNROLL_MAXARG tail with the __syncthreads tree run to s>0 -- a strict generalization (same compare order, same `tid+s<end` guard), correct on any wave size; CUDA path unchanged.

reduce_all.h is untouched and is wave64-correct as left: the `tid<64` fold is fenced by cg::sync before the `thread_rank()<32` block folds sdata[tid+32] and shfl_downs within a width-32 tiled_partition -- no reliance on sub-wavefront lockstep. No hardcoded 32/warpSize/shfl/ballot/lane-mask in any added kernel line. No textures/surfaces/managed memory (those fault classes N/A). alibi.cu POD-pair swap preserves field order. hipBLASLt deltas (dropped MIN_ALIGNMENT prefs, HIPBLAS_COMPUTE_32F + HIP_R_32F scale, SetMathMode no-op, dedicated Lt handle) and the GemmEx compute-type/batched-array cast macros are all USE_HIP-guarded and leave the CUDA call sites identical.

rnn_tests 3 failures (rnn_tests.cpp:49->93) are NOT a port bug: the test seeds inputs with inits::glorotUniform() and compares against a hardcoded `#ifdef CUDA_FOUND` reference; the HIP build defines CUDA_FOUND so it uses the cuRAND-stream reference, but hipRAND yields a different stream for the same seed. RNN forward math is correct (21 structural assertions pass). Worth an upstream follow-up note only.

Hygiene: title `[ROCm] Port Marian GPU backend to HIP for AMD GPUs` (50 chars), mentions Claude, no noreply trailer, no ghstack, no em-dash, Test Plan present; author jeff.daily@amd.com (user's own public email -- not an AMD-internal account); fork Actions disabled (enabled:false); fork/master == upstream c9f287d (clean mirror). Deferred cuSPARSE/cuDNN/NCCL items are documented, not silently broken.

Minor (non-blocking, no fix required): getCublasLtHandle() (backend.h:81) does not check hipblasLtCreate's return, matching the existing lazy-init cublasCreate pattern; cumsum.cu:62 has a 2-space-indented `#if` (cosmetic).
