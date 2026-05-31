# lc0 notes

ROCm/HIP port of lc0's native CUDA backend (the `network_cuda` / cuBLAS + custom-kernel
backend). New first-class `hip` / `hip-fp16` / `hip-auto` backends that compile lc0's own
`.cu` kernels with hipcc behind a single compat header. NVIDIA path is byte-identical
(every source edit is behind `USE_HIP` or the new `-Dhip` meson option). cuDNN->MIOpen and
CUTLASS are out of scope (cuDNN is opt-in; CUTLASS does not port to ROCm -- the cuBLAS
attention fallback runs instead).

## Toolchain (lead platform linux-gfx90a)
- ROCm 7.2.1 (/opt/rocm), hipcc (clang 19), hipBLAS 3.x (v2 API), meson 1.11.1, ninja.
- 4x gfx90a (MI250X). Validate on ONE free GCD: `rocm-smi --showuse` then `HIP_VISIBLE_DEVICES=<n>`.

## Build (gfx90a)
```
cd projects/lc0/src           # the jeffdaily fork clone, branch moat-port
meson setup build-hip \
  -Dhip=true -Damd_gfx=gfx90a \
  -Dplain_cuda=false -Dcudnn=false -Dcutlass=false -Dnvcc=false \
  -Dgtest=true -Dblas=true -Dopencl=false -Donnx=false \
  -Db_lto=false -Dnative_arch=false \
  -Dhip_libdirs=/opt/rocm/lib -Dhip_include=/opt/rocm/include
ninja -C build-hip -j 16
```
- `-Db_lto=false` is REQUIRED: lc0 defaults `b_lto=true`; the hipcc(clang) `.o` cannot be
  LTO-linked into the g++-built executable. Disable LTO for the HIP build.
- `-Dnvcc=false -Dplain_cuda=false` so meson does not require nvcc on a ROCm-only host.
- `-Dnative_arch=false` avoids `-march=native` issues; not needed.
- Same commit builds gfx1100/gfx1151 with only `-Damd_gfx=gfx1100|gfx1151` (configurable arch,
  baked from `-Damd_gfx` / autodetected via rocm_agent_enumerator, default gfx90a if unset).
- The `.cu` host pass and the host `.cc` both compile clean under c++20 g++ with the HIP
  headers (only benign nodiscard warnings, same as lc0's CUDA host pass). The two `.cu`
  (common_kernels, fp16_kernels) go through hipcc custom_targets; layers.cc + network_cuda.cc
  are ordinary g++ TUs that include the compat header via cuda_common.h.

## How the HIP backend builds (meson)
Parallel `if get_option('hip')` block in meson.build (after the cuda block), reusing the
`-Dsycl=amd` hipBLAS/amdhip64/amd_gfx discovery. Two hipcc `custom_target`s compile
common_kernels.cu + fp16_kernels.cu (CUDA spelling preserved) with `-x hip -std=c++17 -fPIC
-O3 -DUSE_HIP -D__HIP_PLATFORM_AMD__ -include .../hip_compat.h`; network_cuda.cc + layers.cc
are added to the host `files`. network_cudnn.cc and cutlass_kernels.cu are NOT built on HIP.

## The compat header (src/neural/backends/cuda/hip_compat.h)
Single file that knows HIP. Force-included into every HIP `.cu`, and pulled in by
cuda_common.h under USE_HIP so all backend TUs inherit the renames. Key non-obvious parts:
- `<cstring>`/`<cstdlib>` before `<hip/hip_runtime.h>` (host memcpy/memset overload lesson).
- `CUDART_VERSION` is DEFINED to a low value (10020): every `>= 11000` / `>= 11010` block
  (NVIDIA L2-persistence cache hints, CUDA-graph external-event flags, the >= 13000 clock
  path) then compiles OUT for free, while the plain arithmetic uses in showInfo() still work.
  Do NOT leave CUDART_VERSION undefined -- showInfo() uses it as a number.
- GEMM compute-type shim: hipBLAS v2 `hipblasGemmStridedBatchedEx` takes `hipblasComputeType_t`
  for the compute slot but `hipDataType` for the data slots. lc0 passes `CUDA_R_16F`/`CUDA_R_32F`
  in BOTH. Map `CUDA_R_* -> HIP_R_*` (correct for data slots) and route the call through a shim
  `lc0HipGemmStridedBatchedEx` that translates the compute `hipDataType` -> `HIPBLAS_COMPUTE_*`.
- `cublasHgemm` / `cublasHgemmBatched` shims: hipBLAS types fp16 GEMM on `hipblasHalf`
  (uint16_t), not `__half`; the shims accept the `__half*` the call sites cast to and
  reinterpret_cast. (cublasSgemm/SgemmStridedBatched/SgemmBatched are 1:1, no shim.)
- `__trap -> __builtin_trap` (HIP device runtime has no __trap()).
- `cudaHostAlloc/cudaHostAllocMapped/cudaFreeHost` map 1:1 to hipHostAlloc/.../hipFreeHost.
- `CUBLAS_STATUS_LICENSE_ERROR` has no hipBLAS peer -> folded into `HIPBLAS_STATUS_UNKNOWN`.

## Fault classes hit + fixes (all validated on GPU)
1. `__shfl_*_sync` mask (winograd_helper.inc warpReduce/warpMax/subgroupBroadcast0, the
   globalAvgPool down-shuffle). HIP static_asserts `sizeof(mask)==8` AND asserts at runtime
   that `mask == __ballot(true)` (the mask must EXACTLY equal the active lanes). A literal full
   64-bit mask faults whenever a block is not a whole multiple of 64 lanes. FIX: on HIP set the
   shuffle mask to `__activemask()` (exactly the active set on any wave size / divergence state),
   keyed on USE_HIP. CUDA keeps its 0xFFFFFFFF literal. This was the *actual* cause of the
   "GPU coredump" crash on ODD batch sizes >= 5 (layer_norm launches (32,1,z) blocks whose last
   wavefront is half-populated; the full mask then names 64 lanes but only 32 are active).
2. WAVE64 SOFTMAX BROADCAST (softmax_opt_64_kernel). A 32-lane warpMax/warpReduce followed by a
   broadcast of lane 0 of the WHOLE wavefront gives lanes 32-63 (a different row) the wrong
   row's max/sum on wave64. FIX: `subgroupBroadcast0` = `__shfl_sync(mask, v, 0, 32)` (read lane
   0 of the 32-lane subgroup; width 32). No-op on wave32. Verified by the blas-vs-hip policy
   match (this kernel feeds the attention-policy softmax).
3. layer_norm divergent `__syncthreads()` (common_kernels.cu shared_sum_for_layer_norm). The
   `if (n >= N) return;` early-return left padding z-rows out of the block-wide barrier; on
   wave64 a padding z-row shares a wavefront with a valid row -> partial-wavefront barrier.
   FIX (arch-unified, UNCONDITIONAL): fold `n>=N` into `oobThread` (skips every guarded
   load/store) and clamp `n=N-1` for index math, so all threads reach both barriers. Padding
   rows own their own `sum[threadIdx.z]` slot, so valid rows are never corrupted; identical
   result on NVIDIA. (This is a latent bug for z-padded launches; the activemask fix (1) was
   what fixed the observed crash, but both are needed and correct.)
4. SE / globalAvgPool / shared_sum_for_layer_norm / promotion_logits 32-lane data layout: all
   index with `&0x1F` / `>>5` / `/32` consistently (NOT hardware warpSize) and reduce with the
   32-lane warpReduce, so each 32-lane half of a wave64 wavefront is self-contained and these
   are wave64-correct as-is (verified by the blas cross-check, not by inspection).
5. FP16 / tensor-core capability gating (network_cuda.cc): the SM-number checks misfire on HIP
   (gfx90a major=9 -> would wrongly enable cublasSetMathMode, and the `< 7` path throws "doesn't
   support FP16"). FIX: USE_HIP branch sets `has_tensor_cores_ = true` (gfx9 has MFMA) and skips
   the throw; `cublasSetMathMode` (both call sites: network_cuda.cc ctor + inputs_outputs.h
   multi_stream) is guarded out on HIP (no hipBLAS math-mode peer; defaults are fine).
6. CUTLASS fused-MHA forced off on HIP (`use_fused_mha=false`); the cuBLAS attention fallback
   (`#ifdef USE_CUTLASS` else branch) runs. No MIOpen (cuDNN backend not built).
- atomicMaxFloat (winograd_helper.inc): int/uint atomicMax/Min on __shared__ memory -- works on
  gfx90a (device-local, not coarse-grained), confirmed by the softmax_kernel path matching blas.
- NO textures / surfaces / managed memory / Thrust in this backend.

## Validation (real gfx90a, GPU 3, T1-256x10 attention net from lczero.org)
- fp32 `hip` backendbench: clean sweep batch 1..32 (all sizes incl. odd).
- fp32 `hip` vs CPU `blas` cross-check (`--backend=check mode=check atol=1e-3 rtol=1e-2`):
  148/148 "Check passed", 0 failures over many batch sizes -- proves policy + value match.
- fp16 `hip-fp16` vs `blas`: 100% pass at fp16-appropriate absolute tol (2.5e-2); policy always
  correct (the rtol metric trips only on near-zero WDL components -- meaningless fp16 noise).
- Determinism: identical eval / PV run-to-run (rules out a wave64 reduction race).
- Device dispatch confirmed via AMD_LOG_LEVEL=3 (named lc0 kernels + rocBLAS interleaved).
- CPU gtest suite (the non-GPU regression set): 8/8 OK (`meson test -C build-hip`).

## Gotchas
- rocBLAS prints `:1:... Cannot find the function: Cijk_...` chatter at AMD_LOG_LEVEL>=1 -- these
  are Tensile solution-selection fallbacks (a tuned kernel variant not in the deployed library),
  NON-fatal. Filter with `grep -vE "Cijk|Cannot find|hip_code|hip_module"`.
- To pin a GPU fault to a kernel: `AMD_SERIALIZE_KERNEL=3` (sync+check each launch) then rocgdb;
  the SIGABRT backtrace names the faulting kernel + the __shfl_xor_sync mask/width and the host
  call site (this is how the activemask root cause was found). HIP_VISIBLE_DEVICES isolates a GCD.
- nps in search (~370 fp32) and raw (~7400 at batch 32) are correctness-first numbers; perf is a
  later pass (fp16 is faster; rocBLAS warms up Tensile on first use; CUTLASS->ck_tile fused-MHA
  is a future optimization). Not a correctness signal.

## Review 2026-05-31 (reviewer, linux-gfx90a) -- CHANGES REQUESTED

Verdict: Request Changes. One genuine, default-reachable correctness defect in the fp16 path (plus a compounding facet from the same root cause). The fp32 path, the meson build branch, the hip_compat.h shims (CUDART_VERSION=10020, GEMM compute-type translation, Hgemm reinterpret shims), the capability gating, the CUTLASS-off fallback, the three documented wave64 fixes, and commit hygiene are all correct and verified. The defect was masked because validation used only an attention-body net, which never launches the affected kernels.

### BLOCKER -- fp16 conv-SE kernels compile to EMPTY no-ops on HIP (__CUDA_ARCH__ undefined)
- fork f966255, src/neural/backends/cuda/fp16_kernels.cu:60 and :231: the entire bodies of `SE_Layer_NHWC` and `OutputInputTransformKernel_fp16_shmem_board` are wrapped in `#if __CUDA_ARCH__ >= 530`. hipcc does NOT define `__CUDA_ARCH__` (it defines `__HIP_DEVICE_COMPILE__`); verified empirically on this host: `/opt/rocm/bin/hipcc --offload-arch=gfx90a -dM -E` emits `__HIP_DEVICE_COMPILE__ 1` and NO `__CUDA_ARCH__`, and a device-pass probe prints CUDA_ARCH_UNDEFINED. So `#if __CUDA_ARCH__ >= 530` is `0 >= 530` = false in both passes and these kernels become empty -> they launch, touch no memory, and leave SE-scaled output uninitialized.
- These are default-reachable, not dead code: `kUseFusedSELayer` is a compile-time `true` (src/neural/backends/cuda/layers.cc:102), `SELayer::Eval` routes fp16 (`nhwc_`) SE through `Se_Fp16_NHWC` (layers.cc:379/449/461/526 -> fp16_kernels.cu:145-205 -> `SE_Layer_NHWC`), and `hip-fp16` is a registered user-selectable backend (network_cuda.cc:1374). Any fp16 convolutional-residual net with squeeze-excitation (the dominant historical Leela architecture) silently produces wrong evals on `hip-fp16`.
- Compounding facet, same root cause: src/neural/backends/cuda/fp16_kernels.cu:33 `#if __CUDA_ARCH__ < 530` -> `#define SKIP_FP16_BITS 1`. Because `0 < 530` is true on HIP, SKIP_FP16_BITS is defined for the winograd_helper.inc include that follows at line 36, so the four `#ifndef SKIP_FP16_BITS` bodies (winograd_helper.inc:84,308,514,724) are ALSO skipped when instantiated from fp16_kernels.cu. (common_kernels.cu does not define SKIP_FP16_BITS, so its fp32 instantiations are fine -- which is why fp32 passed 148/148.)
- Fix (single root cause, established PORTING_GUIDE lesson -- cudaKDTree 2026-05-30, MPPI-Generic 2026-05-30): define `__CUDA_ARCH__` on HIP in hip_compat.h (force-included before fp16_kernels.cu:33). Pattern: `#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__` -> `#define __CUDA_ARCH__ 1` (>=530, e.g. 900 to be unambiguous) for the device pass; note the SKIP_FP16_BITS gate at line 33 is read at file scope in BOTH passes, so confirm the host pass of the .cu still sees the template declarations consistently (define a fixed `__CUDA_ARCH__` value unconditionally if the split-pass macro causes a host/device template-availability mismatch). gfx90a has native fp16 + MFMA, so the >=530 bodies are correct to compile in.
- Re-validation must exercise a fp16 CONV-RESIDUAL-SE net (not only the attention-body net) through the blas-vs-hip-fp16 cross-check, since that is the configuration this defect breaks. The current fp16 100%-pass evidence does not cover it.

### Minor (non-blocking, note for the porter/validator)
- CUDA-graph capture runs on HIP by default (`enable_graph_capture_` defaults true, network_cuda.cc:211; the `#else` GraphLaunch path at network_cuda.cc:710-721 is taken since CUDA_GRAPH_SUPPORTS_EXTERNAL_EVENTS=0). `hipGraphInstantiate/Launch/Destroy/Upload` are mapped and the 5-arg `hipGraphInstantiate` signature matches, so it compiles and (per the passing fp32 cross-check) works on gfx90a -- but it relies on HIP graph capture being correct and is a latent follower risk (gfx1100/gfx1151). Not a gfx90a defect; flag for follower validation.
- meson.build:657: the `message('HIP target architecture: ...')` is inside the `if hip_gfx == ''` autodetect branch, so it never prints when `-Damd_gfx` is set explicitly (the common case). Cosmetic.
- meson.build:624: `add_project_arguments('-DUSE_HIP', '-D__HIP_PLATFORM_AMD__', language:'cpp')` applies globally to all C++ TUs, slightly broader than necessary. Harmless here (no active non-backend TU keys on USE_HIP; the sycl/ files that do are only built under -Dsycl), but a tighter scope would be the backend files only.

## Review fix 2026-05-31 (porter, linux-gfx90a) -- BLOCKER resolved

### Fix: define __CUDA_ARCH__ for the HIP device pass (gated blanket, not per-site)
hip_compat.h, right after the `<hip/...>` includes:
```
#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__
#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 800
#endif
#endif
```
This makes the `#if __CUDA_ARCH__ >= 530` fp16 bodies (SE_Layer_NHWC fp16_kernels.cu:60, OutputInputTransformKernel_fp16_shmem_board :231) and the four `#ifndef SKIP_FP16_BITS` winograd bodies (the `< 530` gate at :33) compile in on the device pass. gfx90a has native fp16 + MFMA so they are correct to compile. Object proof: fp16_kernels.hip.o 564KB -> 2.2MB; `nm` now shows non-empty `SE_Layer_NHWC<C,K>` instantiations.

### Why gated-blanket and not per-site, and the cascade check (the cudf/MPPI lesson)
Cascade grep `grep -rn __CUDA_ARCH__ src/` = exactly THREE sites, all in fp16_kernels.cu, all the intended fp16-capability gate (the `<530` SKIP and the two `>=530` bodies). No PTX, no sm-specific intrinsic, no other arch branch reads __CUDA_ARCH__ anywhere reachable from the two HIP-compiled .cu (common_kernels.cu, fp16_kernels.cu) or their includes (cuda_common.h, winograd_helper.inc, tables/*). So a single define cannot wrongly activate any NVIDIA-only path -> blanket is clean and beats 7 per-site edits.

CRITICAL ordering/gating, learned from the HIP headers: `/opt/rocm/include/hip/hip_common.h:52-55` does `#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 0) -> #define __HIP_DEVICE_COMPILE__ 1`. So an UNCONDITIONAL or pre-`<hip/hip_runtime.h>` define of __CUDA_ARCH__ would make the HOST pass think it is a device compile and break HIP's whole host/device dispatch. The define is therefore (a) placed AFTER the HIP runtime include (HIP has already set __HIP_DEVICE_COMPILE__ by then, so we cannot retroactively flip it) and (b) gated on `__HIP_DEVICE_COMPILE__` so it exists ONLY in the device pass. Probe (agent_space/probe_arch.hip) confirmed: device pass sees __CUDA_ARCH__==800 and compiles the >=530 body; host pass sees __CUDA_ARCH__ undefined (no cascade). The body-only guard structure (every `#ifndef SKIP_FP16_BITS` / `#if __CUDA_ARCH__>=530` wraps only the function BODY, the `__global__`/`__device__` signature is always present) means the host pass still emits every launch stub -> no host/device template-availability mismatch.

### Conv-residual-SE fp16 re-validation (the gate the prior validation missed)
Net: maia-1100 (CSSLab/maia-chess), NETWORK_SE_WITH_HEADFORMAT, 6 blocks / 6 SE blocks / 64 filters / POLICY_CONVOLUTION -- a classic conv-residual-SE net that drives the previously-empty SE_Layer_NHWC. agent_space/maia1100.pb.gz. GPU 3 (isolated via HIP_VISIBLE_DEVICES, rocm-smi showed 0,3 free).
- fp32 `hip` vs `blas`, atol=1e-3 rtol=1e-2, freq=1.0: 100% "Check passed", 0 ERROR across all batch sizes incl. the large ones (32/53/55) -- the SAME conv-SE kernels in fp32 are EXACT vs blas, proving no logic/wave64 defect, only that the bodies now run.
- fp16 `hip-fp16` vs `blas`: value always within fp16 envelope (display mode: value abs err <= ~2.2e-2; the large *relative* value figures are near-zero-Q artifacts). Policy: clean pass at fp16-appropriate tol (atol 1.1e-1 / rtol 2e-1 on softmaxed probabilities; max policy abs err ~1-4e-2, one 9.1e-2 outlier). At the tighter attention-net tol (2.5e-2/1e-1) some large batches trip on near-zero policy entries -- pure fp16 rounding, NOT divergence: fp32 is exact and the fp16 bestmoves match fp32 (f3g4/f4f5/h4h3). Same fp16-noise class the reviewer already accepted for the attention net.

### Regression (unchanged paths)
- Attention-body (testnet.pb.gz, NETWORK_ATTENTIONBODY 10 enc) fp32 hip vs blas: 375 passed, 0 ERROR.
- Attention-body fp16 hip-fp16 vs blas (2.5e-2/1e-1): clean.
- CPU gtest `meson test`: 8/8 OK.

### Incremental-build gotcha (cost me several cycles)
1. The hipcc `custom_target`s do NOT track the force-included `-include hip_compat.h` as a ninja dependency, so editing hip_compat.h does NOT rebuild common_kernels.hip.o / fp16_kernels.hip.o. After a compat-header edit you MUST `rm -f build-hip/common_kernels.hip.o build-hip/fp16_kernels.hip.o` then ninja. (The host TUs layers.cc/network_cuda.cc DO track it via the g++ depfile and rebuild on their own.)
2. timeit.sh does `cd $(dirname $0)/..` to the MOAT repo root, so any RELATIVE `ninja -C build-hip` / `./build-hip/lc0` inside the wrapped command resolves against the MOAT root (there is a stray unrelated build-hip there). Always pass ABSOLUTE paths to ninja `-C` and to the lc0 binary when wrapping with timeit.sh.

## Review fix 2026-05-31 (reviewer, linux-gfx90a) -- APPROVED (focused re-review)

Verdict: Approve. Focused re-review of the single blocker fix; the rest of the diff was approved in the 2026-05-31 review and is unchanged. The only delta since changes-requested is the +22-line `__CUDA_ARCH__` block in hip_compat.h. No problems found.

The `__CUDA_ARCH__` fix is cascade-clean and host/device-safe (empirically verified on this host, ROCm 7.2.1 hipcc):
- Placement/gating correct: hip_compat.h:53-57 sits AFTER the hip includes (hip_fp16.h/hip_runtime.h/library_types.h/hipblas.h at :32-35), is gated `#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__`, uses `#ifndef __CUDA_ARCH__`, and is nested inside the file-level `#if defined(USE_HIP)` (so NVIDIA never sees it -- BC-clean/additive).
- The subtlety holds: /opt/rocm/include/hip/hip_common.h:51-54 derives `__HIP_DEVICE_COMPILE__` from `(defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 0)`. A `#pragma message` probe (agent_space/rev_probe2.hip) compiled with hipcc confirms: DEVICE pass sees `__CUDA_ARCH__` defined (==800 -> compiles the `>=530` bodies); HOST pass sees it UNDEFINED (no wrongful device-compile flip, host/device dispatch intact). Native hipcc (no header) defines `__HIP_DEVICE_COMPILE__ 1` and no `__CUDA_ARCH__` -- the root cause the fix addresses.
- Cascade clean: `grep -rn __CUDA_ARCH__ src/` = exactly 3 functional sites, all in fp16_kernels.cu (:33 `<530` SKIP gate, :60 + :231 `>=530` body guards); all other hits are hip_compat.h comments. No PTX/asm/sm_/__nv_ intrinsics and no `>=700/800/900`/Ampere arch gate anywhere in the two HIP-compiled .cu or their includes, so 800 crosses only the 530 fp16 threshold and activates no NVIDIA-only or Ampere-only path. winograd_helper.inc reads only SKIP_FP16_BITS (4 sites), never `__CUDA_ARCH__` directly; with 800 the `<530` gate is false so SKIP_FP16_BITS is undefined and the 4 winograd fp16 bodies compile in. kMaxResBlockFusingSeKFp16Ampere is a plain constexpr (cuda_common.h:58), not arch-gated. Host headers (cuda_common.h/layers.h/inputs_outputs.h) do not read `__CUDA_ARCH__`, so the host g++ TUs are unperturbed.
- Body-only guards: at fp16_kernels.cu:60 (->:135) and :231 the `__global__` template signature is OUTSIDE the `#if`; only the function BODY is guarded, so both passes emit the launch stub -> no host/device template-availability mismatch.
- Object proof reproduced: build-hip/fp16_kernels.hip.o is 2.2MB and `nm -C` shows non-empty SE_Layer_NHWC<C,K> instantiations across template params.

Conv-SE coverage now adequate: the maia-1100 conv-residual-SE net (NETWORK_SE, 6 SE blocks, 64 filters, conv policy) drives the previously-empty SE_Layer_NHWC / OutputInputTransformKernel_fp16_shmem_board. fp32 hip-vs-blas 100% pass proves the now-compiled bodies are CORRECT (same code path, exact in fp32), not merely present; fp16 within envelope with matching bestmoves (same fp16-noise class already accepted). This exercises the path the original attention-only validation bypassed. (Porter-reported evidence; the validator re-runs on real GPU next, which is expected at review time.)

Commit hygiene (HEAD 1a6c3e3): title 53 chars, `[ROCm]` prefix, body mentions Claude, has a Test Plan, no noreply trailer, no em-dash, no AMD-internal account. Body updated to describe the `__CUDA_ARCH__` fix accurately.

Safe to proceed to GPU validation.
