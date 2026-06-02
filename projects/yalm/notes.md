# yalm notes

## Fork / branch
- Fork: https://github.com/jeffdaily/yalm (Actions disabled)
- Port branch: `moat-port` (off upstream `main`); fork `main` stays a clean mirror.
- HEAD: 311e1c39ebf5cfad02ca13c70aaf7f9942f39101

## Build (gfx90a)
The project builds with a hand-written `src/Makefile` driving nvcc/hipcc directly; there is no CMake. The HIP path is a `USE_HIP=1` branch added to the Makefile.

```
cd projects/yalm/src
export HIP_VISIBLE_DEVICES=2          # this host: GCD0/1 busy, use GCD2 only
make clean
make test USE_HIP=1 HIPARCH=gfx90a    # -> build/test (GPU validation binary)
make      USE_HIP=1 HIPARCH=gfx90a    # -> build/main (full model)
```
Followers reuse the same source with `HIPARCH=gfx1100` / `HIPARCH=gfx1151`; the arch is a make variable, no source edit and no new fork commit.

## Validation (real gfx90a, MI250X)
```
make test USE_HIP=1 HIPARCH=gfx90a && ./build/test   # prints "All tests passed"
```
`test_cuda_kernels()` compares matmul / mha (attn_dot, attn_softmax, att_mix) / ffn against a CPU gold at epsilon 1e-4, launching kernels directly (no hipGraph). `test_attn()` is the CPU regression guard. `AMD_LOG_LEVEL=3 ./build/test` confirms dispatch on gfx90a with 64-wide blocks.

## Wave64 fixes (USE_HIP-guarded; CUDA path byte-identical)
1. matmul_wide / fused_matmul_add_residuals launched `<<<rows/32, warpSize*32>>>`. The literal `32` is warps-per-block, not warp width; on wave64 the block is 64*32 = 2048 > 1024 cap -> launch fails. Fixed with `MATMUL_WPB` (16 on HIP, 32 on CUDA), used consistently in grid divisor and block dim. WPB must divide the model dims (4096, 32000) -- 16 does.
2. att_mix `__shared__ float shared0/1[32]` indexed by threadIdx.x, launched with tpb.x = warpSize (64) -> OOB on wave64. Sized arrays to 64 (max wavefront width).
3. The three standalone test entry points (mha_cuda/matmul_cuda/ffn_cuda) hardcoded `warp_size = 32` / `max_threads_per_block = 1024` and never call set_cuda_device; now query device attributes lazily (`query_warp_size`/`query_max_threads_per_block`).

## Gotchas / traps for followers
- ROCm 7.2.x `__shfl_*_sync` static_assert that the mask is a 64-bit integer; `FULL_MASK 0xffffffff` (32-bit) fails to compile. Use `0xffffffffffffffffULL` under USE_HIP (correct for wave64 anyway). CUDA keeps the 32-bit mask.
- Host C++ TUs include model.h -> cuda_to_hip.h -> `<hip/hip_runtime_api.h>`. g++ needs `-D__HIP_PLATFORM_AMD__` (hipcc sets it automatically for the .cu); both `USE_HIP` and `__HIP_PLATFORM_AMD__` are added to host CFLAGS under USE_HIP.
- hipcc-compiled .cu host stub is not PIE; g++ host objects are. Link with `-no-pie` (added to LDFLAGS under USE_HIP) to avoid `R_X86_64_32 ... can not be used when making a PIE object`.
- `hipStreamLegacy` exists in ROCm 7.2.1 (`(hipStream_t)1`); no fallback needed. The test path uses it; the full-model path uses hipGraph stream capture (not exercised by the test gate).
- `kernel_bench("matmul-wide")` is misleadingly named: it calls `matmul_cuda` (the `matmul` kernel), NOT `matmul_wide`, and asserts nothing -- "All tests passed" there is the unconditional main() print. The real WPB launch path is only hit by the full-model `_forward_cuda`; it was validated separately with a standalone probe (agent_space, throwaway) at `<<<d/16, 64*16>>>` matching a CPU reference.

## Secondary / not the gate
Full-model `./build/main model.yalm -d cuda -m perplexity` needs a converted model (offline `convert.py`, torch) and exercises hipGraph capture/replay. Not run here (no model staged); not required for the gate. If hipGraph replay misbehaves on a follower, a non-graph dispatch fallback is low effort since every kernel is already directly launchable (the test path proves it).

## Validation 2026-06-02 (validator, linux-gfx90a)

Result: PASS -- linux-gfx90a completed, validated_sha=311e1c39ebf5cfad02ca13c70aaf7f9942f39101.

GPU: AMD Instinct MI250X (GFX Version gfx90a), GCD2 (HIP_VISIBLE_DEVICES=2). ROCm/hipcc 7.2.53211.

Commands run (from /var/lib/jenkins/moat/projects/yalm/src):

```
export HIP_VISIBLE_DEVICES=2
make clean && make test USE_HIP=1 HIPARCH=gfx90a   # clean build; exit 0
./build/test                                         # run 1 -> "All tests passed"
./build/test                                         # run 2 -> "All tests passed" (deterministic)
```

Wrapped with `utils/timeit.sh yalm compile` and `utils/timeit.sh yalm test`.

gfx90a device dispatch confirmed via AMD_LOG_LEVEL=3:
```
hip_fatbin.cpp: Using native code object for device: amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack- co: amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-
```

Pass/fail counts: matmul/mha/ffn CPU-vs-GPU tests PASS (eps 1e-4); test_attn() CPU regression guard PASS. Total: 1 binary, all subtests pass, both runs identical.

Caveat carried from review: the WPB launch path (matmul_wide/fused_matmul_add_residuals) is not exercised by ./build/test -- it runs only in the full-model _forward_cuda, which requires a staged model (none downloaded). The porter/reviewer verified it via an agent_space probe at <<<d/16, 64*16>>> vs CPU reference. The formal gate is ./build/test + this documented caveat.

## Review 2026-06-02 (reviewer, linux-gfx90a)
Verdict: review-passed. Reviewed `git diff 6cd1ef6...311e1c3` on moat-port. Built clean from scratch (`make clean && make test USE_HIP=1 HIPARCH=gfx90a`, hipcc 7.2.53211, gfx90a) and ran `./build/test` on GCD2 -> "All tests passed". Confirmed gfx90a device code embedded (`roc-obj-ls`: hipv4-amdgcn-amd-amdhsa--gfx90a, 86888 bytes). Wave64 fixes verified as correct and the CUDA path preserved.

Verified correct:
- WPB self-consistency (infer.cu:29-31, 297-309, 314-329, 963/1003/1236/1242): `<<<rows/MATMUL_WPB, warp_size*MATMUL_WPB>>>` with WPB=16 on HIP gives 64*16=1024 <= cap; kernel-internal `blockDim.x/warpSize` == MATMUL_WPB == grid divisor, so blocktranspose store `block_start_i = blockIdx.x*blockDim.x/warpSize` tiles every row; 16 divides 4096 and 32000. CUDA WPB=32 keeps the original /32,*32 exactly.
- att_mix shared sizing (infer.cu:468-470, 557-565): shared0/1 indexed by threadIdx.x in [0,warpSize), sized to WARP_SIZE_MAX=64; head_dim=128 covered by `i=2*threadIdx.x; i<head_dim; i+=2*warpSize` (one pass at warpSize=64). blocktranspose sm[32] and block_all_reduce shared[32] hold <=16 wavefronts safely.
- Test entry points (infer.cu:1017/1061/1087) now use lazy query_warp_size()/query_max_threads_per_block() instead of hardcoded 32/1024; matches device-side warpSize=64.
- FULL_MASK widened to 0xffffffffffffffffULL only under USE_HIP (infer.cu:15-19); CUDA keeps 0xffffffff.
- Compat header (cuda_to_hip.h): libc-before-hip, host-safe (__HIPCC__ gates hip_runtime.h vs hip_runtime_api.h), aliases only used symbols; NVIDIA branch unchanged. model.h reroute correct.
- Makefile USE_HIP branch: hipcc -x hip --offload-arch, -lamdhip64 -no-pie, host CFLAGS get -DUSE_HIP -D__HIP_PLATFORM_AMD__; CUDA path untouched.
- Commit hygiene: title 60 chars `[ROCm] ...`, no noreply/Co-Authored/ghstack, mentions Claude, no em-dash. Fork main == upstream 6cd1ef6 (clean mirror); fork/moat-port == HEAD. Actions disabled (api enabled=false).
- No texture/surface, no library swaps, no per-arch hack in shared code (changes are wave-agnostic rewrites or USE_HIP-guarded).

Minor (non-blocking; porter may address opportunistically, not required to re-validate):
- att_mix shared array size changed 32 -> 64 unconditionally (infer.cu:468), so it is NOT byte-identical on the NVIDIA path (extra 256B shared, indices 32..63 unused, behavior unchanged). The commit message says the wave64 fixes are "guarded by USE_HIP only where the value genuinely differs" -- here the value differs but is unguarded. Harmless on CUDA; the claim is slightly imprecise.
- hipcc command places `-x hip` after the input file (Makefile CUFLAGS), so clang emits `warning: '-x hip' after last input file has no effect`. The .cu still compiles as HIP (verified gfx90a code object present) because clang treats .cu as HIP by default, so this is cosmetic; could move `-x hip` ahead of `$<` or drop it.
- hipGraph capture/replay path (_forward_cuda, add_or_update_kernel_node) is unverified by the gate (full-model only, no model staged). Already documented as secondary; flagging that gfx90a graph parity remains unproven for the eventual end-to-end check.
