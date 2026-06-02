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
