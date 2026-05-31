# llm.c notes

ROCm/HIP port of karpathy/llm.c. Strategy A (single compat header + Makefile HIP
branch), minimal footprint, NVIDIA build byte-for-byte unchanged. Lead arch
gfx90a (MI250X, wave64), ROCm 7.2.1.

Fork: https://github.com/jeffdaily/llm.c (branch `moat-port`).

## Build (gfx90a)

The Makefile gained a `USE_HIP=1` branch that repoints the build at hipcc, keeps
the upstream target names, and derives the wavefront width from the target arch.
The compat header `llmc/cuda_to_hip.h` is force-included on every HIP TU; the
CUDA-named toolkit headers (`<cuda_runtime.h>`, `<cublasLt.h>`, `<cuda_bf16.h>`,
`<nvtx3/...>`, `<cooperative_groups.h>`, ...) forward to it via `llmc/hip_shims`
on the HIP-only include path.

```
cd projects/llm.c/src
# BF16 (default precision) -- the main path
make USE_HIP=1 AMDGPU_TARGETS=gfx90a NO_MULTI_GPU=1 NO_USE_MPI=1 -j 16 \
     test_gpt2cu train_gpt2cu
# FP32 precision
make USE_HIP=1 AMDGPU_TARGETS=gfx90a NO_MULTI_GPU=1 NO_USE_MPI=1 -j 16 \
     test_gpt2fp32cu train_gpt2fp32cu
# CPU regression targets (unchanged by the port)
make train_gpt2 test_gpt2
cd dev/test && make PRECISION=BF16 test_dataloader test_outlier_detector
```

- `AMDGPU_TARGETS` is the configurable arch (default gfx90a only when unset);
  followers build with `AMDGPU_TARGETS=gfx1100` / `gfx1151` and NO source edit.
- `LLMC_WARP_SIZE` is derived from the arch (gfx9% => 64, else 32) and passed to
  both hipcc compile passes (host and device) so launch geometry and device
  reductions agree.
- `USE_CUDNN` stays 0 on AMD (the in-repo attention.cuh manual flash attention is
  the default). NCCL->RCCL is optional; single-GPU validation uses `NO_MULTI_GPU=1`.
- Do NOT pass `-ffast-math` to hipcc (see fix #6 below); the Makefile uses
  `-ffp-contract=fast -fno-math-errno` instead.

## Reference data

`./dev/download_starter_pack.sh` ships the PyTorch-generated reference
(`gpt2_124M.bin`, `gpt2_124M_bf16.bin`, `gpt2_124M_debug_state.bin`,
`gpt2_tokenizer.bin`, tinyshakespeare) -- the genuine cross-platform reference.
All .bin are gitignored.

## GPU validation (gfx90a, one isolated GCD; this host has 4 GCDs 0-3, GPU 1 busy)

Run on an idle GCD, e.g. `HIP_VISIBLE_DEVICES=2`.

- `./test_gpt2fp32cu` -> `overall okay: 1`. FP32 forward logits OK, loss OK
  (5.270007 vs ref 5.270009), all 16 gradient tensors TENSOR OK at 1e-2, 10
  training steps match the PyTorch reference. Strictest gate, passes.
- `PRECISION=FP32` loss-curve (the CI `loss_checker_ci.py` gate):
  `./train_gpt2cu -b 1 -t 64 -d 256 -l 1e-4 -v 200 -s 200 -a 1 -x 10 -r 0 -f 0 -e gpt2_124M.bin`
  then `python dev/loss_checker_ci.py -f out.txt -s 20 -e 28 -a 5.0` ->
  "Success: All values are within the allowed accuracy" (every step within 0.05%).
- `./test_gpt2cu` (BF16) and variants `-r 0` / `-r 2` / `-w 0` / `-b 32`: forward
  LOGITS OK, 14/16 gradient tensors TENSOR OK. Two layernorm-bias tensors (ln1b,
  lnfb) exceed the NVIDIA-bitwise-tuned thresholds on their single worst near-zero
  element (e.g. ln1b worst elem 0.063 vs 0.011, threshold 0.041). This is the
  EXPECTED BF16 determinism shift, not a bug: the wave64 FP32-accumulation order
  and the WARP_SIZE-seeded stochastic-rounding RNG differ from NVIDIA. Proof it is
  rounding and not an algorithm error: the SAME kernels in FP32 precision agree
  with the reference to ~1e-4 (the test's 1e-6 FP32 threshold "fails" every tensor
  by reduction-order noise alone), and the values are bit-identical run-to-run and
  invariant under `-b 32`. Validate BF16 by tolerance/loss-curve, not bitwise.
- BF16 tinyshakespeare smoke `OMP_NUM_THREADS=8 ./train_gpt2cu`: exit 0, train
  loss 4.29 -> 3.38 (grad norm 13.0 -> 1.27), val loss decreases monotonically
  (4.51 -> 3.49), coherent text generated, no NaN. Decisive BF16 end-to-end gate.
- CPU regression: `./test_gpt2` -> `overall okay: 1`; `dev/test/test_dataloader`
  all OK. (`test_outlier_detector` fails on a PRE-EXISTING upstream `-Ofast`
  clang `isnan()` issue in the untouched CPU C test -- passes at -O2; not a port
  regression.)

## Fixes applied (all arch-unified: correct on wave32 AND wave64)

1. 64-bit shuffle masks. HIP `__shfl_*_sync` static_asserts a 64-bit mask, so the
   `0xFFFFFFFF` literals in cuda_utils.cuh / matmul.cuh do not compile. Added
   `LLMC_FULL_WARP_MASK` (0xff..ULL) in the compat header and `WARP_REDUCE_MASK`
   in cuda_utils.cuh (64-bit on HIP, 0xFFFFFFFF on CUDA), substituted at all 6
   sites. Keyed on USE_HIP, NOT wave width.
2. warpReduceSum/Max start offset. The `offset=16` butterfly folds only 32 lanes;
   on wave64 that silently drops half the wavefront. Replaced with
   `WARP_REDUCE_OFFSET = WARP_SIZE/2` (32 on wave64, 16 on CUDA/RDNA).
3. blockReduce cross-warp scratch. `__shared__ float shared_val[WARP_SIZE]` ->
   sized to the compile-time upper bound `1024/32 = 32` warps (covers wave64's
   <=16 and wave32's <=32; identical 32 on CUDA).
4. WARP_SIZE as a build-time constant. llm.c uses WARP_SIZE both as a HOST launch
   dimension (`dim3(WARP_SIZE, ...)`) and a device constant. HIP defines arch
   macros (`__GFX9__`) only in the DEVICE pass, so a `__GFX9__`-keyed constant
   would be 64 on device but 32 on host for gfx90a -- launch geometry would
   disagree with the kernel. Fix: the Makefile derives `LLMC_WARP_SIZE` from the
   single target arch and `-D`-defines it for both passes; cuda_common.h uses it.
5. hipBLASLt scale-type gap. hipBLASLt has no `CUBLASLT_MATMUL_DESC_SCALE_TYPE`
   attribute (cublasLt does); the scale type follows the compute type (FP32), so
   the SetAttribute call is `#if !USE_HIP`-guarded out. (Plan's predicted "first
   break"; everything else in cublas_common.h / matmul.cuh mapped 1:1 to hipBLAS.)
6. clang -ffast-math breaks the backward pass on gfx90a. clang's `-ffast-math` is
   far more aggressive than nvcc's `--use_fast_math` (it adds -fassociative-math /
   -funsafe-math-optimizations / -fno-signed-zeros), which reassociates the
   online-softmax and layernorm-backward reductions into NaN GRADIENTS (forward
   loss stays correct; only backward NaNs). Re-honoring nans/infinities alone does
   NOT fix it -- the reassociation is the cause. Fix: drop `-ffast-math`, use
   `-ffp-contract=fast -fno-math-errno` (FMA contraction + no errno, IEEE-safe).
   This is the single highest-impact fix; without it test_gpt2fp32cu NaNs at the
   first backward.
7. matmul_backward_bias_kernel4 (FP32 driver) grid stride. The kernel reads
   `tl = blockIdx.x * warpSize` columns, but the launcher hardcoded
   `grid_size = OC/32`. On wave64 adjacent blocks overlap and the last block reads
   past dout -> NaN dbias. Fixed the launcher to stride by the wavefront width
   (`OC / LLMC_WARP_SIZE` on HIP, `OC/32` on CUDA).
8. matmul_backward_bias_kernel9 launch dim. `bdy = WARP_SIZE/bdx` (bdx=4) is 16 on
   wave64 but the launcher hardcoded `block_dim.y = 8`, tripping the kernel's
   `assert(blockDim.y == bdy)`. Set `block_dim.y = WARP_SIZE/4`.
9. layernorm_backward_kernel10 temp-shared reservation. The launcher reserved
   `2*(block_size - 32)*f128::size` but the kernel bases the dbias/dweight temp
   regions at `WARP_SIZE*f128::size` offsets. Changed the literal 32 to WARP_SIZE
   to match the host reservation to the kernel's WARP_SIZE-based footprint
   (`2*rounded_C + 2*(block_size - WARP_SIZE)*f128::size` floats). This is
   over-allocation correctness/consistency hygiene, NOT a corruption fix: a
   literal 32 on wave64 reserves `2*(64-32)*f128::size = 64*f128::size` floats
   MORE than the device uses -- a harmless over-allocation (worst case a slightly
   larger dynamic-smem request), never aliasing or dbias corruption. The kernel's
   tmp base offsets are driven by the device WARP_SIZE (already 64), independent
   of the launcher literal. ln1b/lnfb is fully explained by the documented BF16
   determinism shift (the wte_backward stochastic-rounding seed at encoder.cuh:114
   contains WARP_SIZE), not by this reservation.
10. Streaming load/store. HIP has no `__ldcs`/`__stcs`/`__stcg`; the compat header
    provides generic templated versions (plain load/store -- the cache hint is
    advisory). Covers int4 (Packed128), float, bf16, half, unsigned short.
11. Cooperative-groups reduce. HIP's CG has no `cg::reduce`/`cg::plus`/`cg::greater`
    (the FP32 driver uses them over tiled_partition<32>). The compat header adds a
    butterfly-reduction shim over the tile (width = tile.size() stays within the
    32-lane tile, correct on wave64) plus plus/greater functors.
12. Misc symbol gaps handled in the compat header: `__float2bfloat16_rn` ->
    `__float2bfloat16` (HIP already rounds to nearest); NVTX (`nvtxRangePush/Pop`,
    `nvtxNameCudaStreamA`) stubbed to no-ops; `cudaProfilerStart/Stop` -> success
    no-ops (hipProfilerStart returns "operation not supported" and would trip
    cudaCheck); `cudaFuncSetAttribute(kernel,...)` cast to `(const void*)` (HIP's
    hipFuncSetAttribute has no templated T* overload). NVML (`mfu.h`) is left
    absent -- it is `__has_include`-guarded and compiles out (MFU readout off).

## What is NOT done / out of scope

- cuDNN/MIOpen flash attention (USE_CUDNN path) -- left OFF; the manual
  attention.cuh is the default and ports cleanly.
- Multi-GPU (NCCL->RCCL): the `nccl.h` shim + RCCL link exist but were not
  exercised (single GCD). `make USE_HIP=1` (without NO_MULTI_GPU) would need
  RCCL + the dpkg nccl check adjusted.
- AMD-native (rocWMMA/CK/MFMA) tuning of matmul/attention -- correctness-first
  mechanical port only.

## Follower notes (gfx1100 / gfx1151, wave32)

The wave-size abstraction (LLMC_WARP_SIZE=32 on RDNA) and configurable
AMDGPU_TARGETS mean the same curated commit should build with only
`AMDGPU_TARGETS=gfx1100` / `gfx1151` and no source change. All fixes above are
arch-unified (the wave32 path is the original NVIDIA value). Validate first,
delta-port only on failure.

## Review 2026-05-31 (reviewer, linux-gfx90a, moat-port @ 108beee)

Verdict: changes-requested. The port is correctness-clean and the NVIDIA path is byte-for-byte preserved; all genuine wave64 fault-class fixes verified correct. One fault-class ANALYSIS is wrong (the fix it justifies is harmless), plus three minor comment/scope nits. Fixing these is curation-only (no kernel behavior changes), so the next porter pass should be quick. Safe to GPU-validate once the rationale is corrected.

Problems (each cites the fork branch):

1. WRONG mechanism for fix #9 (layernorm_backward_kernel10), though the code change is harmless. The commit message, notes.md fix #9, and the comment at llmc/layernorm.cuh:502-506 all claim the literal `32` "under-allocates on wave64 and lets the cross-warp bias partials alias, corrupting dbias (ln1b/lnfb)." The arithmetic is the opposite. Reservation = (2*rounded_C + 2*(block_size - X)*f128::size) floats; the kernel's true footprint with device WARP_SIZE=64 is exactly 2*rounded_C + 2*(block_size - 64)*f128::size (dweight_tmp_shared base 2*rounded_C + f128::size*BLOCK_SIZE - 2*WARP_SIZE*f128::size, top store at threadIdx.x=BLOCK_SIZE-1). A literal X=32 reserves 2*(64-32)*f128::size = 64*f128::size floats MORE than required -> it OVER-allocates, which never corrupts (worst case slightly higher dynamic-smem request, or a launch failure if it exceeds the cap; never silent corruption). The dbias_tmp/dweight_tmp base offsets and the by-design aliasing into dweight_shared are driven by the DEVICE WARP_SIZE (already 64), independent of the launcher's reservation literal. So the launcher literal does not affect dbias correctness on wave64; ln1b/lnfb is fully explained by the determinism shift (finding 0, correct). Keep the WARP_SIZE substitution (good host/device-consistency hygiene) but rewrite the rationale in the commit body, notes.md fix #9, and llmc/layernorm.cuh:502-506 to say "matches the host reservation to the kernel's WARP_SIZE-based footprint (exact instead of over-allocated on wave64)" -- do NOT call it a corruption/aliasing fix. A false "corrupts dbias on wave64" claim must not reach the upstream PR body.

2. Stale comment: llmc/hip_shims/cuda_profiler_api.h:2 says the compat header "maps cudaProfilerStart/Stop to hipProfilerStart/Stop"; cuda_to_hip.h:88-89 actually (correctly) maps them to hipSuccess no-ops because hipProfilerStart returns hipErrorNotSupported and would trip cudaCheck. Fix the shim comment.

3. Imprecise comment: llmc/cuda_to_hip.h:170 says "HIP has no __ldcs/__stcs/__stcg." HIP's <hip/hip_fp16.h> DOES define __ldcs(const __half*) and __ldcs(const __half2*) (amd_hip_fp16.h:624,629). The unconstrained template coexists (non-template overloads win for half/half2; the template covers float/int4/bf16/unsigned short), so it compiles and is correct -- but reword to "HIP lacks __stcs/__stcg and only provides __ldcs for half/half2."

4. Out-of-scope target breaks under USE_HIP: Makefile:327 (profile_gpt2cu) passes nvcc-only -lineinfo to $(NVCC)=hipcc; clang rejects it. Not in the documented/validated HIP target set (non-blocking), but `make USE_HIP=1 profile_gpt2cu` errors. Either guard -lineinfo out of the HIP branch or note profile_gpt2cu is unsupported on HIP.

Verified correct (no action): NVIDIA path unchanged (WARP_SIZE/MAX_1024_THREADS_BLOCKS fall to original literals under #else; cudaFuncSetAttribute((const void*)k,...) selects an equivalent public CUDA overload; HIP branch fully USE_HIP-gated; -ffast-math removal scoped to the HIP Makefile branch only). All 6 shfl mask sites use WARP_REDUCE_MASK (64-bit on HIP); no missed shuffle/ballot/activemask sites in the whole .cu/.cuh tree. warpReduce offset WARP_SIZE/2 and blockReduce kMaxWarpsPerBlock=1024/32 are correct upper bounds for both wave sizes. LLMC_WARP_SIZE reaches both hipcc passes via a single compile invocation per .cu (host/device agree -- recurring trap avoided). kernel4 (OC/warp_width stride) and kernel9 (block_dim.y=WARP_SIZE/4 vs assert(blockDim.y==bdy)) match the kernel source. hipBLASLt SCALE_TYPE correctly !USE_HIP-guarded. cg::reduce shim over tile<32> correct on both wave sizes (g.size()=32; block_acc[32] zero-initialized); test_gpt2fp32cu passing confirms it. BF16 ln1b/lnfb is a genuine determinism shift: wte_backward seeds stochastic rounding with seed + bucket*WARP_SIZE + threadIdx.x + k (encoder.cuh:114), and FP32-agrees-1e-4 + bit-identical-run-to-run + invariant-under-`-b 32` rules out an algorithmic defect. Commit hygiene clean: [ROCm] title 49 chars, Claude-disclosed, Test Plan present, no noreply/ghstack/co-authored, ASCII-only, all jeffdaily.

### Curation fixes resolved 2026-05-31 (porter, moat-port @ 15f5e1a)

All four review findings were curation-only (no kernel behavior, mask, launch
dim, or WARP_SIZE-substitution change); the reviewer had already cleared GPU
correctness and the byte-for-byte NVIDIA build, so no re-review is needed for
these comment/notes/commit-body/Makefile-scope edits. State moved
changes-requested -> review-passed; cleared for GPU validation.

1. (must-fix) Rewrote the fix-#9 rationale in all three places (the comment at
   llmc/layernorm.cuh, fix #9 above, and the curated [ROCm] commit body). The
   literal 32 on wave64 is a harmless OVER-allocation (`2*(64-32)*f128::size =
   64*f128::size` floats more than the device uses), never corruption/aliasing;
   the WARP_SIZE substitution is host/device-consistency hygiene. Removed the
   false "corrupts dbias/ln1b/lnfb on wave64" claim -- ln1b/lnfb is fully the
   documented BF16 determinism shift (wte_backward seed at encoder.cuh:114 carries
   WARP_SIZE). The WARP_SIZE substitution itself is KEPT.
2. llmc/hip_shims/cuda_profiler_api.h comment now says cudaProfilerStart/Stop map
   to hipSuccess no-ops (was: hipProfilerStart/Stop).
3. llmc/cuda_to_hip.h streaming-shim comment now states HIP lacks __stcs/__stcg
   and provides __ldcs only for half/half2; the non-template half/half2 overloads
   win and the template covers the rest.
4. (non-blocking) Makefile profile_gpt2cu: -lineinfo moved behind a LINEINFO var
   (empty under USE_HIP, `-lineinfo` otherwise) so `make USE_HIP=1 profile_gpt2cu`
   no longer feeds the nvcc-only flag to hipcc; the NVIDIA recipe is unchanged.

Compile-only re-check (no GPU; reviewer already cleared correctness): the three
validated targets rebuild clean under USE_HIP=1 AMDGPU_TARGETS=gfx90a
NO_MULTI_GPU=1 NO_USE_MPI=1 (`test_gpt2fp32cu test_gpt2cu train_gpt2cu`, all
three binaries produced, hipcc exit 0; only the pre-existing benign
cudaGetLastError macro-redefinition warning).
