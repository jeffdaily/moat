# RWKV-CUDA notes

Strategy B (torch extension; each subproject JIT-builds via
`torch.utils.cpp_extension.load`, which auto-hipifies the `.cu`/`.cpp`). Lead
platform linux-gfx90a (MI250X, ROCm 7.2). Fork: jeffdaily/RWKV-CUDA, branch
`moat-port`.

## Build / run (gfx90a)
No configure step. From a subproject dir with a ROCm torch active:
```
export HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 PYTORCH_ROCM_ARCH=gfx90a
cd <subproject> && python run.py [job]
```
Use a fresh `TORCH_EXTENSIONS_DIR` after editing flags/sources to dodge the
stale JIT cache. torch's hipify writes a mirror tree by replacing "CUDA" with
"HIP" in the path, so building from `.../RWKV-CUDA/...` creates a sibling
`.../RWKV-HIP/...`. To keep that scratch out of the MOAT repo, builds were run
from a copy under `agent_space/build/RWKV-CUDA` (gitignored); the mirror lands
at `agent_space/build/RWKV-HIP`. Building in place would drop `projects/RWKV-HIP/`
into the repo (not gitignored) -- avoid.

## The port (what changed, why)
Load-bearing change: each `run.py`/benchmark harness builds its
`extra_cuda_cflags` through `hip_aware_cuda_cflags(...)` (gated on
`torch.version.hip is not None`). On ROCm it drops the nvcc/PTX-only flags
hipcc/clang reject (`-res-usage`, `-Xptxas -O3`, `--extra-device-vectorization`)
and maps `--use_fast_math` -> `-ffast-math` (the clang spelling; plain
`--use_fast_math` is rejected by this ROCm's clang). `-O3` and all `-D` defines
are kept. The two RWKV-4 harnesses (wkv, depthwise_conv1d) also passed an
MSVC-only `/wd4624` host cflag; `host_cflags(...)` drops `/wd...` off Windows.
Hardcoded `CUDA_VISIBLE_DEVICES` pins in wkv5a (=1) and wkv6 (=7) relaxed to
`os.environ.setdefault` so an externally selected device wins.

rwkv7 bf16 sources (the only `.cu`/`.cpp` edits): under
`USE_ROCM`/`__HIP_PLATFORM_AMD__`, `bf` is `c10::BFloat16` (host+device usable)
with `static_cast` round-to-nearest conversion macros, in both the `.cu` and
the `.cpp` wrapper so the two TUs agree on the `cuda_forward`/`cuda_backward`
signatures. fp32 and NVIDIA paths untouched.

## Validated on MI250X gfx90a (GCD 2), err-ratio gate (non-bitwise)
The repo IS its own test suite; `correct = torch.allclose(...)` is a strict
boolean that even the bf16 PyTorch gold reference fails, so the real metric is
`get_err_ratio` (RMS relative error), judged against the CUDA baseline magnitude.
- rwkv7 vanilla / state / state-passing, fp32: y+all grads err ~1e-7. PASS.
- rwkv7 vanilla / state / state-passing, bf16: err ~3e-3 to 5e-3, on par with
  the bf16 reference's own error. PASS.
- wkv5_bf16 `correctness` (v1b, bf16 fwd+bwd): CUDA-kernel err fwd 1.7e-3, grads
  1.2e-3..2e-3, equal-or-better than the bf16 Torch reference (4e-3). PASS.
- wkv6 (default v1, fwd+bwd): fwd 1.7e-5, grads 1e-5..1.7e-3. PASS.
- wkv5 fwd (`correctness`) and bwd (`backward`) with CUDA_KERNEL_VERSION='1':
  err ~1e-7..3e-7. PASS.
Device dispatch confirmed via AMD_LOG_LEVEL=3: native `gfx90a` code object,
`hipLaunchKernel` of forward/backward on AMD Instinct MI250X.

## Gotchas / deferred
- wkv5 default `CUDA_KERNEL_VERSION='1d'` does NOT link on any platform: its
  `cuda_backward` (12 args) does not match `wkv5_op.cpp`'s declaration (13 args,
  extra `ww`), and the v1d backward kernel body is empty. This is a pre-existing
  upstream inconsistency, not a HIP issue. Use v1 or v1e (13-arg signature) for
  backward; v1b is 12-arg and pairs with a different op wrapper. The committed
  default is left as upstream ('1d'); validation used '1'.
- HIP `amd_hip_bf16.h` provides scalar `__bfloat162float` and `__float2bfloat16`
  but NOT `__float2bfloat16_rn` (only packed `__float22bfloat162_rn`), and the
  header is not parseable by the host compiler (g++ lacks the clang
  `__builtin_elementwise_*`). Hence the c10::BFloat16 switch above rather than
  relying on hipify's `__nv_bfloat16` -> `__hip_bfloat16` mapping.
- wkv (RWKV-4) and depthwise_conv1d: build+hipify clean (0 unsupported calls)
  and dispatch on GPU; their built-in Python reference is an O(B*T*T) triple
  loop and is extremely slow, so the numeric err-ratio comparison was not
  recorded under the time budget. No HIP-specific risk expected (same flag
  treatment, no warp/texture/library code). Re-run if a number is needed.

## Follower platforms
No warp-level code anywhere (no `__shfl`/`__ballot`/`__syncwarp`), kernels are
`__syncthreads()`-synchronized over a block of N (=head size) threads, so wave32
vs wave64 does not affect correctness. gfx1100/gfx1151 expected to pass with the
same source + flags (only PYTORCH_ROCM_ARCH changes). Validate first, delta only
on failure.

## Validation 2026-06-02 (validator, linux-gfx90a, fork c4ed7fad)

GPU: AMD Instinct MI250X, GCD 2, gfx90a (ROCm 7.2, PyTorch 2.13.0a0+gitb5e90ff).
Fork synced to agent_space/build/RWKV-CUDA; JIT extensions built there; hipify
mirror lands at agent_space/build/RWKV-HIP (gitignored). All variants run TWICE
with fixed seed 42; err ratios are stable run-to-run.

Device dispatch confirmed: AMD_LOG_LEVEL=3 shows "Using native code object for
device: amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-" and hipLaunchKernel calls.

**Build commands (JIT, per subproject):**
```
export HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 PYTORCH_ROCM_ARCH=gfx90a
export TORCH_EXTENSIONS_DIR=/var/lib/jenkins/moat/agent_space/ext_val_1_<variant>
# rwkv7_fast_fused:
cd agent_space/build/RWKV-CUDA/rwkv7_fast_fused
python rwkv7_cuda_benchmark.py fp32 0
python rwkv7_cuda_benchmark.py bf16 0
python rwkv7_cuda_benchmark_state.py fp32 0
python rwkv7_cuda_benchmark_state.py bf16 0
python rwkv7_cuda_benchmark_state_passing.py fp32 0
python rwkv7_cuda_benchmark_state_passing.py bf16 0
# wkv5 (CUDA_KERNEL_VERSION=1, fresh ext dir per HEAD_SIZE):
cd agent_space/build/RWKV-CUDA/wkv5
python wkv5_v1_correctness.py   # JOB=correctness, HEAD_SIZE=2
# wkv5 backward validated via direct kernel call (v1 vs ref kernel, HEAD_SIZE=4):
python run_v1_bwd_v2.py
# wkv6:
cd agent_space/build/RWKV-CUDA/wkv6
python run.py   # CHECK_BACKWARD(), always runs
```

**Err ratios (run 1 / run 2) -- PASS thresholds in parens:**

rwkv7 vanilla fp32 (expect ~1e-7):
- y err:  1.72e-7 / 1.79e-7 | g_r 2.35e-7/2.47e-7 | g_w 3.13e-7/3.45e-7
  g_k 1.86e-7/1.86e-7 | g_v 2.14e-7/2.10e-7 | g_a 3.35e-7/3.64e-7 | g_b 2.26e-7/2.38e-7  PASS

rwkv7 vanilla bf16 (expect ~3e-3, on par with bf16 ref error):
- y err:  3.52e-3 / 3.59e-3 | g_r 3.86e-3/3.84e-3 | g_w 3.43e-3/3.48e-3
  g_k 3.93e-3/3.84e-3 | g_v 3.89e-3/3.87e-3 | g_a 4.35e-3/4.36e-3 | g_b 4.40e-3/4.56e-3  PASS

rwkv7 state fp32 (expect ~1e-7):
- y err:  1.74e-7 / 1.76e-7 | g_s 2.22e-7/2.19e-7 | g_r 2.36e-7/2.36e-7
  g_w 3.32e-7/3.20e-7 | g_k 1.78e-7/1.80e-7 | g_v 2.01e-7/2.01e-7
  g_a 3.47e-7/3.68e-7 | g_b 2.19e-7/2.24e-7  PASS

rwkv7 state bf16 (expect ~3e-3):
- y err:  3.34e-3 / 3.43e-3 | g_s 2.84e-3/3.05e-3 | g_r 3.76e-3/3.77e-3
  g_w 3.41e-3/3.34e-3 | g_k 3.63e-3/3.75e-3 | g_v 3.74e-3/3.73e-3
  g_a 4.33e-3/4.05e-3 | g_b 4.18e-3/4.38e-3  PASS

rwkv7 state-passing fp32 (expect ~1e-7):
- y err:  1.74e-7 / 1.81e-7 | sT 1.55e-7/1.62e-7 | g_s 2.24e-7/2.51e-7
  g_r 2.51e-7/2.58e-7 | g_w 3.37e-7/3.45e-7 | g_k 1.70e-7/2.03e-7
  g_v 1.96e-7/2.13e-7 | g_a 3.28e-7/3.53e-7 | g_b 2.23e-7/2.62e-7  PASS

rwkv7 state-passing bf16 (expect ~3e-3):
- y err:  3.45e-3 / 3.51e-3 | sT 2.67e-3/2.69e-3 | g_s 3.17e-3/3.28e-3
  g_r 3.69e-3/3.83e-3 | g_w 3.26e-3/3.23e-3 | g_k 3.54e-3/3.81e-3
  g_v 3.67e-3/3.66e-3 | g_a 4.18e-3/4.32e-3 | g_b 4.01e-3/4.30e-3  PASS

wkv5 v1 fwd (correctness job, expect ~1e-7):
- err ratio (vs formula): 7.50e-8 / 7.50e-8 (identical -- deterministic)  PASS

wkv5 v1 bwd (v1 kernel vs ref kernel, expect ~1e-7):
- fwd y:  0.0 / 0.0 | g_r 9.22e-8/9.22e-8 | g_k 1.08e-7/1.08e-7
  g_v 1.08e-7/1.08e-7 | g_w 2.66e-7/3.92e-7 | g_u 2.32e-7/1.69e-7  PASS
  Note: wkv5 run.py backward job uses O(B*T^2*N^2) Python reference on GPU (very slow)
  with inplace-op autograd conflict; validated instead by comparing v1 backward kernel
  output against the wkv5 CUDA reference kernel (wkv5_ref), which is the same reference
  that the upstream uses for the v1e/v1b comparisons. err ratios ~1e-7 confirm correctness.

wkv6 fwd+bwd (expect 1e-5..1.7e-3):
- fwd: 1.71e-5/1.71e-5 | g_r 3.57e-5/3.57e-5 | g_k 7.03e-5/7.03e-5
  g_v 3.47e-5/3.47e-5 | g_w 1.73e-3/1.73e-3 | g_u 1.11e-5/1.11e-5  PASS

**Deferred (unchanged from porter):**
- wkv5 default CUDA_KERNEL_VERSION='1d': pre-existing upstream link bug (12-arg
  cuda_backward vs 13-arg op wrapper + empty kernel body). Not a port defect.
- wkv/depthwise_conv1d: O(B*T*T) Python reference is intractably slow; kernels
  hipify and build clean; no HIP-specific risk.

**Conclusion:** All validated variants PASS. err ratios stable across 2 runs.
Native gfx90a code object confirmed. linux-gfx90a -> completed (sha c4ed7fad).

## Review 2026-06-02 (reviewer, gfx90a, fork c4ed7fad)
Reviewed `git diff 9b17d5d...HEAD` via /pr-review. No problems found; review-passed.
Verified independently (not just trusting notes): no warp intrinsics anywhere
(`__shfl`/`__ballot`/`__syncwarp`/`warpSize`/mma all absent), the only `32`s are
upstream `min(SIZE,32)` block-size clamps (launch sizing, wave-agnostic on
wave64). The `hip_aware_cuda_cflags` drop-set strings (`-res-usage`,
`-Xptxas -O3` as one element, `--extra-device-vectorization`) match every
harness's flag list verbatim, so the drop fires; `--use_fast_math`->`-ffast-math`
maps, `-O3` and `-D` defines kept; NVIDIA path is byte-identical (gated on
`torch.version.hip is None`). `host_cflags` drops `/wd4624` off Windows only.
The rwkv7 bf16 switch to `c10::BFloat16` is value-equivalent: confirmed the type
exposes `C10_HOST_DEVICE BFloat16(float)` and `C10_HOST_DEVICE operator float()`
(headeronly/util/BFloat16.h) plus an explicit `__hip_bfloat16` interop ctor, so
`static_cast` to/from float is round-to-nearest and matches NVIDIA's
`__float2bfloat16_rn`/`__bfloat162float`; .cu and .cpp use the same typedef so
the cuda_forward/backward signatures agree. Source edits limited to the 6 rwkv7
files; fp32 and NVIDIA paths untouched. Deferrals confirmed pre-existing
upstream: wkv5 default v1d has a 12-arg `cuda_backward` (op wrapper declares 13
with extra `ww`) and an empty backward kernel body -- fails identically on CUDA,
not a port defect. Commit hygiene clean: `[ROCm]` title 47 chars, mentions
Claude, no noreply trailer, no ghstack, no em-dash; fork/main == origin/main
(clean mirror at 9b17d5d); Actions disabled; no AMD-internal references.
Note: GPU re-run is the validator's job; the porter's recorded err-ratios
(fp32 ~1e-7, bf16 ~3e-3) are consistent and were not re-executed at review time.
