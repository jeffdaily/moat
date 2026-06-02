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
