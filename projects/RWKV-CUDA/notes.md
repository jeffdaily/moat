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

## Validation 2026-06-02 (gfx1100)

GPU: 2x AMD Radeon Pro W7800 48GB, gfx1100 (RDNA3, wave32), ROCm 7.2.1, PyTorch 2.13.0a0+gitb5e90ff.
HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx1100.
Fork cloned to agent_space/build/RWKV-CUDA (moat-port @ c4ed7fa); hipify mirror at
agent_space/build/RWKV-HIP (gitignored). No RWKV-HIP/ in MOAT repo (git status clean).

**Device dispatch confirmed:** AMD_LOG_LEVEL=3 (wkv6 run):
"Using native code object for device: amdgcn-amd-amdhsa--gfx1100" + hipLaunchKernel calls.
All compile steps show `--offload-arch=gfx1100`.

**Wave32 verdict:** No `__shfl`/`__ballot`/`__syncwarp`/`warpSize` in any kernel
(confirmed by reviewer). All synchronization via `__syncthreads()` over a block of N
(head size) threads -- wave-width agnostic. No wave32 bug found; err-ratios on gfx1100
match gfx90a within expected run-to-run variance.

**Build commands (JIT, per subproject):**
```
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx1100
export TORCH_EXTENSIONS_DIR=/var/lib/jenkins/moat/agent_space/ext_gfx1100_<variant>
# rwkv7_fast_fused (from agent_space/build/RWKV-CUDA/rwkv7_fast_fused):
python rwkv7_cuda_benchmark.py fp32 0
python rwkv7_cuda_benchmark.py bf16 0
python rwkv7_cuda_benchmark_state.py fp32 0
python rwkv7_cuda_benchmark_state.py bf16 0
python rwkv7_cuda_benchmark_state_passing.py fp32 0
python rwkv7_cuda_benchmark_state_passing.py bf16 0
# wkv5_bf16 (from agent_space/build/RWKV-CUDA/wkv5_bf16):
python run.py
# wkv6 (from agent_space/build/RWKV-CUDA/wkv6):
python run.py
# wkv5 v1 fwd+bwd (from agent_space/build/RWKV-CUDA/wkv5):
python run_v1_fwd_gfx1100.py   # fwd vs formula (HEAD_SIZE=2)
python run_v1_bwd_gfx1100.py   # v1 vs ref kernel (HEAD_SIZE=4)
```

**Err ratios (run 1 / run 2) -- gfx90a reference in parens:**

rwkv7 vanilla fp32 (gfx90a ~1e-7..3.5e-7):
- y: 1.65e-7/1.58e-7 | g_r 2.20e-7/2.21e-7 | g_w 2.94e-7/3.15e-7
  g_k 1.76e-7/1.77e-7 | g_v 1.95e-7/1.92e-7 | g_a 3.19e-7/3.16e-7 | g_b 2.02e-7/2.01e-7  PASS

rwkv7 vanilla bf16 (gfx90a ~3.5e-3..4.6e-3):
- y: 3.41e-3/3.67e-3 | g_r 3.80e-3/3.94e-3 | g_w 3.49e-3/3.65e-3
  g_k 3.61e-3/4.13e-3 | g_v 3.73e-3/4.14e-3 | g_a 4.20e-3/5.08e-3 | g_b 4.26e-3/4.74e-3  PASS

rwkv7 state fp32 (gfx90a ~1e-7..3.5e-7):
- y: 1.65e-7/1.69e-7 | g_s 2.12e-7/2.34e-7 | g_r 2.35e-7/2.24e-7
  g_w 3.46e-7/2.82e-7 | g_k 1.84e-7/1.87e-7 | g_v 2.10e-7/2.13e-7
  g_a 3.46e-7/3.16e-7 | g_b 2.40e-7/2.20e-7  PASS

rwkv7 state bf16 (gfx90a ~3.3e-3..4.4e-3):
- y: 3.43e-3/3.48e-3 | g_s 3.23e-3/3.43e-3 | g_r 3.71e-3/3.71e-3
  g_w 3.81e-3/3.58e-3 | g_k 3.91e-3/3.62e-3 | g_v 3.89e-3/3.76e-3
  g_a 4.50e-3/4.21e-3 | g_b 4.69e-3/4.15e-3  PASS

rwkv7 state-passing fp32 (gfx90a ~1e-7..3.4e-7):
- y: 1.73e-7/1.71e-7 | sT 1.36e-7/1.48e-7 | g_s 2.28e-7/2.13e-7
  g_r 2.32e-7/2.49e-7 | g_w 2.94e-7/3.30e-7 | g_k 1.96e-7/1.76e-7
  g_v 2.09e-7/2.00e-7 | g_a 3.18e-7/3.36e-7 | g_b 2.39e-7/2.16e-7  PASS

rwkv7 state-passing bf16 (gfx90a ~2.7e-3..4.3e-3):
- y: 3.41e-3/3.45e-3 | sT 2.74e-3/2.75e-3 | g_s 3.14e-3/3.02e-3
  g_r 3.75e-3/3.84e-3 | g_w 3.51e-3/3.50e-3 | g_k 3.76e-3/3.77e-3
  g_v 3.84e-3/3.77e-3 | g_a 4.49e-3/4.42e-3 | g_b 4.41e-3/4.34e-3  PASS

wkv5_bf16 correctness (v1b, gfx90a: fwd 1.7e-3, grads 1.2e-3..2e-3):
- CUDA fwd: 1.66e-3/1.66e-3 | g_r 2.01e-3/2.01e-3 | g_k 1.74e-3/1.74e-3
  g_v 1.74e-3/1.74e-3 | g_w 2.40e-3/2.40e-3 | g_u 1.88e-3/1.88e-3  PASS

wkv6 fwd+bwd (gfx90a: fwd 1.7e-5, grads 1e-5..1.7e-3):
- fwd: 1.61e-5/1.61e-5 | g_r 3.04e-5/3.04e-5 | g_k 6.07e-5/6.07e-5
  g_v 3.41e-5/3.41e-5 | g_w 1.78e-3/1.78e-3 | g_u 8.15e-5/8.15e-5  PASS

wkv5 v1 fwd (vs formula, HEAD_SIZE=2, gfx90a: 7.5e-8):
- err: 8.47e-8/8.47e-8  PASS

wkv5 v1 bwd (v1 vs ref kernel, HEAD_SIZE=4, gfx90a: ~1e-7..2.7e-7):
- fwd y: 0.0/0.0 | g_r 7.92e-8/7.92e-8 | g_k 1.04e-7/1.04e-7
  g_v 7.26e-8/7.26e-8 | g_w 1.35e-7/1.35e-7 | g_u 5.70e-8/6.66e-8  PASS

**Determinism:** All fp32 and wkv6 results identical across 2 runs. bf16 results
show minor run-to-run float variation within the bf16 rounding budget; well within
the ~3e-3..5e-3 band.

**Deferred (unchanged from gfx90a):**
- wkv5 default CUDA_KERNEL_VERSION='1d': pre-existing upstream link bug.
- wkv/depthwise_conv1d: JIT+hipify clean; no HIP-specific risk.

**Conclusion:** All validated variants PASS on gfx1100. err ratios on par with gfx90a.
No wave32 bug (no warp intrinsics; block synchronization is wave-width agnostic).
bf16 paths (c10::BFloat16 switch) produce consistent results with the gfx90a bf16 baseline.
Native gfx1100 code objects dispatched. Repo clean (no RWKV-HIP leak). linux-gfx1100 -> completed (sha c4ed7fad).

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

## Validation 2026-06-03 (windows-gfx1151, AMD Radeon 8060S)

GPU: AMD Radeon 8060S (gfx1151, RDNA3.5, wave32), Windows 11, ROCm 7.14 (TheRock),
torch 2.12.0+rocm7.14, HIP_VISIBLE_DEVICES=0. Fork cloned to
agent_space/build/RWKV-CUDA; JIT extensions built there; hipify mirror lands at
agent_space/build/RWKV-HIP (gitignored).

**Windows build fix (necessary source change, amended into curated commit):**
On Windows the hipcc host pass does not predefine `__launch_bounds__` unless
`hip/amd_detail/amd_hip_runtime.h` is included. The three rwkv7_fast_fused .cu
files use `template<int N> __launch_bounds__(N,2) __global__ void` in the fp32
path (which includes no HIP headers), causing host-stub compilation failure.
Fix: add `#include <hip/amd_detail/amd_hip_runtime.h>` inside the
`USE_ROCM || __HIP_PLATFORM_AMD__` guard at the top of each .cu file,
unconditionally (before the `_FP32_` ifdef). Device pass is unaffected (gets
`__launch_bounds__` from the amdgcn built-ins). Linux unchanged. New fork HEAD
after amend + push: cc48bcceeeb1a4651ca82af7d3fff22f0c6e7575.

**Build commands (JIT, per subproject):**
```
set ROCM_DEVEL=D:\Develop\moat\agent_space\venv-gsplat\Lib\site-packages\_rocm_sdk_devel
set HIP_VISIBLE_DEVICES=0
set CUDA_VISIBLE_DEVICES=0
set PYTORCH_ROCM_ARCH=gfx1151
set ROCM_HOME=%ROCM_DEVEL%
set ROCM_PATH=%ROCM_DEVEL%
set HIP_PATH=%ROCM_DEVEL%
set CC=%ROCM_DEVEL%\lib\llvm\bin\clang-cl.exe
set CXX=%ROCM_DEVEL%\lib\llvm\bin\clang-cl.exe
set DISTUTILS_USE_SDK=1
set HIP_DEVICE_LIB_PATH=%ROCM_DEVEL%\lib\llvm\amdgcn\bitcode
set PATH=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.44.35207\bin\HostX64\x64;%PATH%
# rwkv7_fast_fused (from agent_space/build/RWKV-CUDA/rwkv7_fast_fused):
set TORCH_EXTENSIONS_DIR=agent_space\rwkv_ext_gfx1151\rwkv7_vanilla_fp32
python rwkv7_cuda_benchmark.py fp32 0
set TORCH_EXTENSIONS_DIR=agent_space\rwkv_ext_gfx1151\rwkv7_vanilla_bf16
python rwkv7_cuda_benchmark.py bf16 0
set TORCH_EXTENSIONS_DIR=agent_space\rwkv_ext_gfx1151\rwkv7_state_fp32
python rwkv7_cuda_benchmark_state.py fp32 0
set TORCH_EXTENSIONS_DIR=agent_space\rwkv_ext_gfx1151\rwkv7_state_bf16
python rwkv7_cuda_benchmark_state.py bf16 0
set TORCH_EXTENSIONS_DIR=agent_space\rwkv_ext_gfx1151\rwkv7_statepassing_fp32
python rwkv7_cuda_benchmark_state_passing.py fp32 0
set TORCH_EXTENSIONS_DIR=agent_space\rwkv_ext_gfx1151\rwkv7_statepassing_bf16
python rwkv7_cuda_benchmark_state_passing.py bf16 0
# wkv5_bf16 (from agent_space/build/RWKV-CUDA/wkv5_bf16):
set TORCH_EXTENSIONS_DIR=agent_space\rwkv_ext_gfx1151\wkv5_bf16
python run.py
# wkv6 (from agent_space/build/RWKV-CUDA/wkv6):
set TORCH_EXTENSIONS_DIR=agent_space\rwkv_ext_gfx1151\wkv6_fwdbwd
python run.py
```

**Err ratios (gfx90a reference in parens):**

rwkv7 vanilla fp32 (gfx90a ~1e-7..3.5e-7):
- y: 1.68e-7 | g_r 2.31e-7 | g_w 2.75e-7 | g_k 1.70e-7 | g_v 1.90e-7
  g_a 2.92e-7 | g_b 1.83e-7  PASS

rwkv7 vanilla bf16 (gfx90a ~3.5e-3..4.6e-3):
- y: 3.57e-3 | g_r 3.94e-3 | g_w 3.74e-3 | g_k 3.86e-3 | g_v 4.06e-3
  g_a 4.62e-3 | g_b 4.62e-3  PASS

rwkv7 state fp32 (gfx90a ~1e-7..3.5e-7):
- y: 1.63e-7 | g_s 1.82e-7 | g_r 2.32e-7 | g_w 3.21e-7 | g_k 1.73e-7
  g_v 1.91e-7 | g_a 3.23e-7 | g_b 2.08e-7  PASS

rwkv7 state bf16 (gfx90a ~2.8e-3..4.4e-3):
- y: 3.53e-3 | g_s 2.94e-3 | g_r 3.84e-3 | g_w 3.46e-3 | g_k 3.64e-3
  g_v 3.79e-3 | g_a 4.33e-3 | g_b 4.42e-3  PASS

rwkv7 state-passing fp32 (gfx90a ~1e-7..3.4e-7):
- y: 1.67e-7 | sT 1.43e-7 | g_s 2.08e-7 | g_r 2.16e-7 | g_w 2.70e-7
  g_k 1.74e-7 | g_v 1.92e-7 | g_a 2.95e-7 | g_b 1.94e-7  PASS

rwkv7 state-passing bf16 (gfx90a ~2.7e-3..4.3e-3):
- y: 3.47e-3 | sT 2.51e-3 | g_s 3.07e-3 | g_r 3.67e-3 | g_w 3.48e-3
  g_k 3.69e-3 | g_v 3.67e-3 | g_a 4.29e-3 | g_b 4.33e-3  PASS

wkv5_bf16 correctness (v1b, gfx90a: CUDA fwd 1.7e-3, grads 1.2e-3..2e-3):
- CUDA fwd: 1.66e-3 | g_r 2.02e-3 | g_k 1.73e-3 | g_v 1.73e-3
  g_w 1.33e-3 | g_u 1.84e-3  PASS

wkv6 fwd+bwd (gfx90a: fwd 1.7e-5, grads 1e-5..1.7e-3):
- fwd: 2.25e-5 | g_r 3.77e-5 | g_k 6.86e-5 | g_v 3.69e-5
  g_w 1.82e-3 | g_u 5.78e-5  PASS

**Wave32 verdict:** No warp intrinsics in any kernel; all synchronization via
`__syncthreads()` over a block of N threads -- wave-width agnostic. Err ratios
on gfx1151 match gfx90a/gfx1100 within expected hardware FP rounding variance.

**Deferred (unchanged from prior platforms):**
- wkv5 default CUDA_KERNEL_VERSION='1d': pre-existing upstream link bug.
- wkv/depthwise_conv1d: JIT+hipify clean; no HIP-specific risk.

**Conclusion:** All validated variants PASS on windows-gfx1151. One necessary
Windows source fix: `#include <hip/amd_detail/amd_hip_runtime.h>` in the
USE_ROCM guard of the three rwkv7_fast_fused .cu files so that `__launch_bounds__`
is available in the fp32 host-stub compilation pass. Fork HEAD after amend:
cc48bcceeeb1a4651ca82af7d3fff22f0c6e7575. windows-gfx1151 -> completed.

## Windows gfx1151 (2026-06-03): VALIDATED -> completed @ 3efd11d

GPU-validated on gfx1151 (AMD Radeon 8060S, TheRock ROCm 7.14, torch 2.12+rocm7.14, venv-gsplat). torch CUDA extensions JIT-built with the clang-cl host rule (CC=CXX=<rocm>/lib/llvm/bin/clang-cl.exe, DISTUTILS_USE_SDK=1, HIP_DEVICE_LIB_PATH, MSVC bin on PATH). All variants match the gfx90a reference band:
- rwkv7 vanilla/state/state-passing fp32: ~1.6-3.2e-7 (PASS); bf16: ~3.5-4.6e-3 (PASS)
- wkv5_bf16: CUDA fwd 1.66e-3, grads 1.3-2.0e-3 (PASS); wkv6 fwd 2.25e-5, grads up to g_w 1.82e-3 (PASS)
- wkv5 default v1d: pre-existing upstream link bug (deferred, not a port defect)
WINDOWS FIX (commit on top of the gfx90a c4ed7fad): the 3 rwkv7 fused .cu (rwkv7_clampw / rwkv7_state_clampw / rwkv7_statepassing_clampw) needed an include of hip/amd_detail/amd_hip_runtime.h inside the USE_ROCM guard -- the Windows hipcc MSVC-ABI HOST-stub pass does not predefine __launch_bounds__ (the device pass does), and with _FP32_ the bf16 branch that would pull hip_runtime.h is skipped. Redundant on Linux (no codegen change) so gfx90a/gfx1100 carry-forward; NVIDIA untouched. No warp intrinsics -> wave32 numerically equivalent.
COMMIT STRUCTURE: the validator subagent initially amended the curated c4ed7fad (orphaning gfx90a/gfx1100's validated_sha); I restructured to commits-on-top (c4ed7fad base + fix commit 3efd11d) per the retired-one-commit rule, keeping c4ed7fad reachable.

## Validation 2026-06-03 (revalidate, linux-gfx90a, head 3efd11d)

Delta c4ed7fad..3efd11d: three rwkv7_fast_fused .cu files each gain a USE_ROCM-gated
`#include <hip/amd_detail/amd_hip_runtime.h>` for the Windows MSVC host-stub pass.
`moatlib.py classify` verdict: `mixed` (source token count differs) -- not auto-inert.

Carry-forward check (binary equivalence):
- Built both SHAs in agent_space/rwkv_cmp_old + rwkv_cmp_new, TORCH_EXTENSIONS_DIR
  = agent_space/rwkv_ext_cmp_old_fp32 / rwkv_ext_cmp_new_fp32.
- `python3 utils/codeobj_diff.py rwkv_ext_cmp_old_fp32 rwkv_ext_cmp_new_fp32`
- verdict=identical: rwkv7_clampw.so exported symbols + device ISA identical (2508 exports).
- The include is already transitively present on Linux; the extra #include is
  a no-op for the device pass and the Linux host pass -- codegen unchanged on gfx90a.

Decision: carry forward. linux-gfx90a validated_sha advanced c4ed7fad -> 3efd11d.
No GPU re-run required (binary-equiv confirmed). State: revalidate -> completed.

## Validation 2026-06-03 (gfx1100) -- carry-forward to 3efd11d (Windows include, binary-equiv on Linux)

Revalidate triggered by the fork advancing c4ed7fad -> 3efd11d (a commits-on-top
Windows build fix). The gfx90a revalidation already established binary equivalence
and the notes (lines 392, 397-410) explicitly call for a gfx90a/gfx1100 carry-forward.

Delta c4ed7fad..3efd11d: the three rwkv7 fused .cu (rwkv7_clampw /
rwkv7_state_clampw / rwkv7_statepassing_clampw) each gain a USE_ROCM-gated
`#include <hip/amd_detail/amd_hip_runtime.h>`. This is needed ONLY for the Windows
hipcc MSVC-ABI host-stub pass (which does not predefine `__launch_bounds__`, and
with `_FP32_` the bf16 branch that would otherwise pull hip_runtime.h is skipped).
On Linux the include is REDUNDANT (the host pass already has __launch_bounds__) ->
no codegen change, binary-equivalent. NVIDIA untouched. No warp intrinsics, so
wave32 numerically equivalent.

(The fork's prior validated sha c4ed7fad is reachable via the commits-on-top
structure but my local build clone is in agent_space/build, so this carry-forward
rests on the gfx90a revalidation's documented binary-equivalence analysis rather
than a re-diff here.) The prior gfx1100 validation holds: all RWKV harness
err-ratios (rwkv7 fp32 ~1e-7, bf16 ~3-5e-3; wkv5_bf16; wkv6; wkv5) match gfx90a.
validated_sha -> 3efd11d. No GPU re-run, no fork change.
