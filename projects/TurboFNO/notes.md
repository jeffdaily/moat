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

## Validation 2026-06-07 (windows-gfx1201, RDNA4 wave32)

GPU: AMD Radeon RX 9070 XT, gfx1201, warpSize=32, 32 CUs. ROCm 7.14.0a20260604 (TheRock nightly).
Fork: jeffdaily/TurboFNO moat-port @ e100b3d (submodule jeffdaily/TurboFFT @ e285704).

Build: all 10 fusion variants configured and compiled clean for gfx1201 via Ninja + all-clang
(clang++ 23.0.0 from _rocm_sdk_devel). Warnings only, no errors.

Commands:
```
# Set up environment
export PROJECT_ROOT=B:\develop\moat\projects\TurboFNO\src
export HIP_VISIBLE_DEVICES=0   # gfx1201 is device 0 (V710 offline this session)
ROCM=B:\develop\TheRock\external-builds\pytorch\.venv\Lib\site-packages\_rocm_sdk_devel
CLANG=$ROCM\lib\llvm\bin\clang++.exe

# Build all 10 variants (Ninja, -j32 shared with another concurrent build)
# python3 agent_space/turbofno_build_win_gfx1201.py --arch gfx1201 --jobs 32

# Build numerical harness directly
$CLANG -x hip --offload-arch=gfx1201 -DUSE_HIP -std=c++17 -O2 -ffp-contract=on \
    -I$PROJECT_ROOT/utils \
    -I$PROJECT_ROOT/fusion_variants/1D_A_exp_fft+cgemm+ifft \
    -I$PROJECT_ROOT/TurboFFT/TurboFFT/include/code_gen/generated/float2 \
    -I$ROCM/include \
    agent_space/turbofno_validate_win.cu \
    $PROJECT_ROOT/utils/utils.cu \
    -L$ROCM/lib -lhipfft -lhipblas \
    -o agent_space/turbofno_validate_win.exe

# Run (DLLs from _rocm_sdk_core/bin and _rocm_sdk_devel/bin on PATH;
#      ROCBLAS_TENSILE_LIBPATH for rocBLAS Tensile kernels)
# HIP_VISIBLE_DEVICES=0 agent_space/turbofno_validate_win.exe
```

Windows-specific note: harness uses utils.cuh include before the generated FFT headers
(turboFFT_ZADD/ZSUB/ZMUL macros must be visible when fft_radix_2_logN_8_upload_0.cuh is
included). Harness compilation is agent_space/turbofno_validate_win.cu.
Runtime: DLLs (amdhip64_7.dll, hipfft.dll, hipblas.dll, rocfft.dll) copied next to exe
or on PATH from _rocm_sdk_core/bin + _rocm_sdk_devel/bin. ROCBLAS_TENSILE_LIBPATH=
_rocm_sdk_libraries/bin/rocblas/library. The rocblaslt stderr errors (cannot read
TensileLibrary_lazy_gfx1201.dat) are non-fatal; they do not affect hipblasCgemm path.

Numerical results (agent_space/turbofno_validate_win.cu on GPU 0):
- Device: AMD Radeon RX 9070 XT, warpSize=32, CUs=32
- GEMM: cgemm (logical-32 tiling, wave32) vs hipblasCgemm M=256 N=256 K=128 (col-maj):
  outlier_cnt=0, outlier_perct=0.000000%, max_rel_diff=3.958292e-05 -> PASS
- FFT: fft_8 (hand-rolled radix-2 256pt) vs hipfftExecC2C FORWARD, batch=1024:
  outlier_cnt=0, outlier_perct=0.000000%, max_rel_diff=3.576712e-04 -> PASS

Runtime smoke (RC=0, no CHECK_CUDA_KERNEL errors):
- TurboFNO_1D_D (fused FFT-GEMM-iFFT): ran to completion, RC=0.
- TurboFNO_1D_E (hipFFT+hipBLAS baseline): ran to completion, RC=0.

Result: windows-gfx1201 COMPLETED at e100b3d.

## Validation 2026-06-04 (linux-gfx1100, RDNA3 wave32)

GPU: AMD Radeon Pro W7800 48GB, gfx1100, warpSize=32, 35 CUs. ROCm 7.2.1.
Fork: jeffdaily/TurboFNO moat-port @ e100b3d (submodule jeffdaily/TurboFFT @ e285704).

Note on host GPU accessibility: this host has 4 W7800 GPUs (gfx1100). GPUs 0, 1, 3
were clock-gated at 0 MHz and the ROCm 7.2.1 runtime hung during queue creation on
them (acquireQueue spin-wait; a known ROCm/driver issue with deep GFXOFF on RDNA3
when orphaned KFD contexts are present). GPU 2 was clocked up (2948 MHz, active
from another session) and was responsive. All validation ran on GPU 2 (HIP_VISIBLE_DEVICES=2).

Build: all 10 fusion variants configured and compiled clean for gfx1100 (~45s, warnings only, no errors).
gfx1100 code objects confirmed via llvm-objdump --offloading for 1D_A, 1D_D, 1D_E, 2D_D, 2D_E
(all show `hipv4-amdgcn-amd-amdhsa--gfx1100`).

Commands:
```
# Build all 10 variants
cd projects/TurboFNO/src && export PROJECT_ROOT=$(pwd)
USE_HIP=1 CMAKE_HIP_ARCHITECTURES=gfx1100 bash install.sh

# Build + run numerical harness (column-major matrices; cgemm.cuh is col-major throughout)
/opt/rocm/llvm/bin/clang++ -x hip --offload-arch=gfx1100 -DUSE_HIP -std=c++17 -O2 -ffp-contract=on \
    -I${PROJECT_ROOT}/utils -I${PROJECT_ROOT}/fusion_variants/1D_A_exp_fft+cgemm+ifft \
    -I${PROJECT_ROOT}/TurboFFT/TurboFFT/include/code_gen/generated/float2 \
    ${PROJECT_ROOT}/utils/utils.cu agent_space/turbofno_validate.cu \
    -L/opt/rocm/lib -lhipfft -lhipblas -o /tmp/turbofno_validate_gfx1100
HIP_VISIBLE_DEVICES=2 /tmp/turbofno_validate_gfx1100

# Runtime smoke
HIP_VISIBLE_DEVICES=2 fusion_variants/1D_D_exp_fused_fft_cgemm_ifft/build/TurboFNO_1D_D | head -30
HIP_VISIBLE_DEVICES=2 fusion_variants/1D_E_baseline/build/TurboFNO_1D_E | head -30
```

Numerical results (agent_space/turbofno_validate.cu on GPU 2):
- Device: AMD Radeon Pro W7800 48GB, warpSize=32, CUs=35
- GEMM: cgemm (logical-32 tiling, wave32) vs hipblasCgemm M=256 N=256 K=128 (col-maj):
  outlier_cnt=0, outlier_perct=0.000000%, max_rel_diff=3.487317e-05 -> PASS
- FFT: fft_8 (hand-rolled radix-2 256pt) vs hipfftExecC2C FORWARD, batch=1024:
  outlier_cnt=0, outlier_perct=0.000000%, max_rel_diff=8.255286e-05 -> PASS

Wave32 GEMM tiling confirmation: the cgemm kernel uses WID=threadIdx.x/32, WARP_M=32,
and __syncthreads for synchronization -- a block-level fence that is wave-size-agnostic.
On wave32 each logical tile (32 threads) is exactly one wavefront. No warp-collective
ops (no __shfl/__ballot/__syncwarp). The numerical PASS (max_rel_diff ~3.5e-5, 0 outliers)
on wave32 confirms the tiling is correct as-is.

Runtime smoke (first 30 lines, no CHECK_CUDA_KERNEL errors):
- TurboFNO_1D_D (fused FFT-GEMM-iFFT): bs=1, dimX=1, DY=128, N=64..128, K=8..112 timing lines printed cleanly.
- TurboFNO_1D_E (hipFFT+hipBLAS baseline): bs=1, dimX=1, DY=128, N=64..128, K=8..112 timing lines printed cleanly.

Result: linux-gfx1100 COMPLETED at e100b3d.

## Validation 2026-06-04 (linux-gfx90a)

GPU: AMD Instinct MI250X / MI250, gfx90a, warpSize=64, 104 SMs. ROCm 7.2.1.
Fork: jeffdaily/TurboFNO moat-port @ e100b3d (submodule jeffdaily/TurboFFT @ e285704).

Build: all 10 fusion variants configured and compiled clean for gfx90a (warnings only, no errors).
gfx90a code objects confirmed via llvm-objdump --offloading for 1D_A, 1D_D, 1D_E, 2D_D, 2D_E.

Commands:
```
# Build all 10 variants
cd projects/TurboFNO/src && export PROJECT_ROOT=$(pwd) && USE_HIP=1 CMAKE_HIP_ARCHITECTURES=gfx90a bash install.sh

# Build + run numerical harness
/opt/rocm/llvm/bin/clang++ -x hip --offload-arch=gfx90a -DUSE_HIP -std=c++17 -O2 -ffp-contract=on \
    -I${PROJECT_ROOT}/utils -I${PROJECT_ROOT}/fusion_variants/1D_A_exp_fft+cgemm+ifft \
    -I${PROJECT_ROOT}/TurboFFT/TurboFFT/include/code_gen/generated/float2 \
    ${PROJECT_ROOT}/utils/utils.cu agent_space/turbofno_validate.cu \
    -L/opt/rocm/lib -lhipfft -lhipblas -o /tmp/turbofno_validate
HIP_VISIBLE_DEVICES=0 /tmp/turbofno_validate

# Runtime smoke
HIP_VISIBLE_DEVICES=0 fusion_variants/1D_D_exp_fused_fft_cgemm_ifft/build/TurboFNO_1D_D | head -30
HIP_VISIBLE_DEVICES=0 fusion_variants/1D_E_baseline/build/TurboFNO_1D_E | head -30
```

Numerical results (agent_space/turbofno_validate.cu on GPU 0):
- Device: AMD Instinct MI250X / MI250, warpSize=64, SMs=104
- GEMM: cgemm (logical-32 tiling, wave64) vs hipblasCgemm, M=256 N=256 K=128:
  outlier_cnt=0, outlier_perct=0.000000%, max_rel_diff=9.313226e-06 -> PASS
- FFT: fft_8 (hand-rolled radix-2 256pt) vs hipfftExecC2C FORWARD, batch=1024:
  outlier_cnt=0, outlier_perct=0.000000%, max_rel_diff=1.415610e-05 -> PASS

Runtime smoke (first 30 lines, no CHECK_CUDA_KERNEL errors):
- TurboFNO_1D_D (fused FFT-GEMM-iFFT): bs=1..128 timing lines printed cleanly.
- TurboFNO_1D_E (hipFFT+hipBLAS baseline): bs=1..2 timing lines printed cleanly.
  (Large bs values, e.g. 32768, trigger the pre-existing upstream OOB allocation;
  not a port regression -- documented in notes above.)

Result: linux-gfx90a COMPLETED at e100b3d.

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
