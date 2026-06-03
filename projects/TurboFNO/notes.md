# TurboFNO notes

## Port summary (linux-gfx90a lead)
- Strategy A: one `utils/cuda_to_hip.h` compat shim + a `USE_HIP` CMake path.
  No AMD-native rewrite; the GEMM and FFT are hand-written portable HIP/CUDA C++.
- Fork: https://github.com/jeffdaily/TurboFNO branch `moat-port`.
- Submodule TurboFFT: fork https://github.com/jeffdaily/TurboFFT pinned at the
  upstream sha e28570417284b8e66c124880e8805b677075076b. NO source edits needed
  in the submodule -- its generated float2 headers are pure device code (no CUDA
  includes, no shuffles); the `TurboFFT/Common` NVIDIA CUDA-samples bundle is on
  the include path but no built TU references it. `.gitmodules` URL repointed to
  the jeffdaily fork for reproducibility.

## What changed
- NEW `utils/cuda_to_hip.h`: on ROCm includes hip_runtime + hipfft + hipblas and
  aliases the exact symbol set used (cudaMalloc/Memcpy/DeviceSynchronize/Event*,
  cudaDeviceProp, cufftCreate/Plan1d/PlanMany/ExecC2C/Destroy, CUFFT_C2C/FORWARD,
  cublasCreate/Cgemm/Destroy, CUBLAS_OP_N, cuFloatComplex). On NVIDIA: plain CUDA
  includes. cuFloatComplex -> hipFloatComplex (both float2), so the `(cuFloatComplex*)`
  casts at the hipblas/hipfft call sites are correct with no buffer change.
- `utils/utils.cuh`: include the shim; DROP `helper_functions.h`/`helper_cuda.h`
  (NVIDIA CUDA-samples, not on ROCm, unused).
- All built `.cu`: CUDA includes routed through the shim. All kernel `.cuh`:
  dead `#include <mma.h>` guarded out under HIP.
- NEW `cmake/turbofno_targets.cmake`: `turbofno_configure_target()` selects the
  CUDA-vs-HIP toolchain. All 10 `fusion_variants/*/CMakeLists.txt` rewritten to a
  uniform shape: `option(USE_HIP)`, `project(... LANGUAGES CXX HIP|CUDA)`,
  `set_source_files_properties(... LANGUAGE HIP)`, link `hip::hipfft roc::hipblas`
  (HIP) or `CUDA::cublas CUDA::cufft` (CUDA), `-ffp-contract=on` on HIP.
  `CMAKE_HIP_ARCHITECTURES` left to the caller (no hardcoded gfx / warp width).
- `install.sh`: `USE_HIP=1` switch (optional `CMAKE_HIP_ARCHITECTURES`).

## Build (gfx90a)
```
cd projects/TurboFNO/src && export PROJECT_ROOT=$(pwd)
USE_HIP=1 ./install.sh                 # all 10 variants
# or per variant:
cmake -S fusion_variants/1D_D_exp_fused_fft_cgemm_ifft -B build \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ && cmake --build build -j16
```
Followers: `-DCMAKE_HIP_ARCHITECTURES=gfx1100` (or gfx1151). A
`gfx90a;gfx1100` fat binary emits both code objects (`llvm-objdump --offloading`).
All 10 variants configure + build clean on gfx90a, ROCm 7.2.1.

## Validation method + results (gfx90a, wavefront 64, ROCm 7.2.1)
The shipped `main()` in every variant is TIMING-ONLY (verify_vector/verify_matrix
exist in utils.cu but call sites are commented out). Added a standalone harness
(agent_space/turbofno_validate.cu) that exercises the two hand-written device
primitives the fused kernels compose, against an independent ROCm-library gold:
- cgemm (the logical-32 register-tiled complex GEMM) vs hipblasCgemm:
  max_rel_diff ~9.3e-6, 0 outliers -> PASS.
- fft_8 (hand-rolled radix-2 256-point FFT) vs hipfftExecC2C FORWARD, batch 1024:
  max_rel_diff ~1.4e-5, 0 outliers -> PASS.
Runtime smoke: TurboFNO_1D_D (fused) and TurboFNO_1D_E (hipFFT/hipBLAS baseline)
both run a trimmed config sweep to completion with no CHECK_CUDA_KERNEL errors.

Note on the shipped full default config sweep: bs_list goes up to 32768 while the
buffers are sized to the initial bs=128 and reused -- the upstream main reads OOB
for the large bs entries (a pre-existing upstream issue, not a port concern). Use
a small config for runtime smoke.

## Logical-32 GEMM tiling on wave64: CORRECT, unchanged
The only wave-coupled idiom is the GEMM register tiling (utils/TurboFNO.h:
WARP_M 32, WID=threadIdx.x/32; cgemm/fused index with TID%32). It is a logical
32-lane subgroup fenced by __syncthreads, NOT a warp-collective op. There are no
__shfl/__ballot/warpSize/__syncwarp/__activemask in the COMPILED surface (the
shuffle-using code is only in TurboFFT's fft_codegen.py generator script, which
emits a non-float2 path that this build does not include). The GEMM result
matches hipBLAS to ~1e-5 on wave64 -> the tiling is correct as-is. NO wave64
hardcode introduced; the device width is never baked into a build constant.

## Fault classes hit
- Library swap: cuFFT->hipFFT, cuBLAS->hipBLAS (hipBLAS v3 here: hipblasCgemm
  takes `hipComplex*` == float2; hip::hipfft + roc::hipblas CMake targets).
- Dead-header removal: `<mma.h>` (no tensor-core code) and the two CUDA-samples
  headers, both unresolvable / unused on ROCm.
- fp fast-math drift handled by pinning `-ffp-contract=on`; tolerances are
  relative and fast-math aware.
- NOT triggered: textures/surfaces, rule-of-five, OOB neighbor reads, 256B pitch,
  __smid, layered arrays, managed memory, streams (default only).

## Follower delta (gfx1100 / gfx1151)
Reuse this branch, rebuild with the target arch. Expected no source change: the
only wave dependency is the logical-32 tiling, which is wave-agnostic; RDNA is
wave32 so each logical warp is exactly one wavefront (the safer width). Validate
by the same cgemm-vs-hipBLAS and fft_8-vs-hipFFT harness on the follower host,
and/or a cross-arch output diff against the gfx90a result (deterministic).
gfx1151: confirm hipFFT/hipBLAS presence in the Windows HIP SDK.

## Review 2026-06-03 (reviewer, linux-gfx90a)
Verdict: review-passed. Diff upstream c83a74b..e100b3d on jeffdaily/TurboFNO
moat-port. No blocking findings. Confirmations and minor (non-blocking) items below.

Confirmed sound:
- Logical-32 GEMM tiling on wave64 is CORRECT and unchanged (the key item).
  cgemm.cuh:58-114 reads operands only from __shared__ (sA/sB) indexed by WID/
  WARP_M/(TID%32), shared via __syncthreads (cgemm.cuh:58), with explicit
  per-thread float2 FMA (cgemm.cuh:102-103). No __shfl/__ballot/warpSize/
  __syncwarp anywhere in the compiled surface (repo + the actually-included
  float2 generated FFT headers). So WID=threadIdx.x/32 / WARP_M 32 (TurboFNO.h:
  5,13) is a logical 32-lane subgroup, not a warp-collective; on wave64 a
  wavefront holds two such tiles and correctness is width-independent. No wave64
  hardcode introduced. Followers (wave32) need only a cross-arch output diff.
- hipblasCgemm signature: this host's hipBLAS takes hipComplex* (hipblas.h:14694),
  and hipComplex==hipFloatComplex==float2 (amd_hip_complex.h:46,135). The shim
  alias cuFloatComplex->hipFloatComplex makes the (cuFloatComplex*) call-site
  casts type-correct. Validated numerically (~9.3e-6 vs hipblasCgemm).
- cmake targets hip::hipfft and roc::hipblas both exist and match the ROCm config
  packages (/opt/rocm/lib/cmake/{hipfft,hipblas}); CMAKE_HIP_ARCHITECTURES left to
  the caller (turbofno_targets.cmake:12-14), no hardcoded gfx/warp width; CUDA
  path preserved under else() with CUDA::cublas CUDA::cufft.
- Dead headers: <mma.h> guarded out on HIP in all kernel .cuh that had it; the two
  CUDA-samples headers dropped from utils.cuh; nothing built references them.
- Commit message: [ROCm] prefix, <=72 char title, Claude disclosure, Test Plan,
  no noreply trailer, no MOAT jargon, ASCII-only. CUDA build path intact.
- Submodule TurboFFT clean at e285704, no source edits, .gitmodules repointed.

Minor (non-blocking) cleanup, optional before upstream PR:
- The 10 .cu drivers each carry three redundant `#include "cuda_to_hip.h"` lines
  (e.g. 1D_A/fused.cu:2-3 and :20) -- the porter replaced three distinct CUDA
  headers with the same shim include rather than collapsing to one. Harmless
  (#pragma once) but untidy; collapse to a single include.
- cuda_to_hip.h aliases two symbols never used anywhere: cublasDestroy (line 64)
  and CUFFT_INVERSE (line 51). Plan said "alias the exact set used"; drop them or
  leave -- harmless dead #defines, no effect on the CUDA path.
- Pre-existing upstream (NOT port concerns, do not fix here): E baseline calls
  cublasCreate without a matching cublasDestroy (handle leak); the iFFT exec uses
  CUFFT_FORWARD on iplan (fused.cu:190,212); the default config sweep reads OOB
  for large bs (notes already record this).
