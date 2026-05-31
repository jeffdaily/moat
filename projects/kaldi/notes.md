# kaldi notes

## Summary
kaldi already ships a mature upstream ROCm/HIP backend (AMD `sfantao`, 2022-2023, written vs ROCm 5.0.2-5.2.x):
`--use-rocm` in src/configure, src/makefiles/hip_64bit.mk, src/hip/hipify.h (colmap-style compat header),
52 files across cudamatrix/cudafeat/cudadecoder/chain. Disposition: FINISH/IMPROVE (bitrotted vs ROCm 7.2.1),
not a skip. See plan.md.

## Build (Strategy A, Makefile/configure -- NOT pytorch, NOT primarily CMake)
Host prereqs: C++17 g++, then build tools/ (OpenFST + a BLAS):
  cd src && cd tools && make -j$(nproc)
GPU configure + build:
  cd src/src && ./configure --use-rocm --rocm-dir=/opt/rocm --rocm-targets=gfx90a --use-cuda=no
  make -j$(nproc) depend && make -j$(nproc)
Arch is configurable via --rocm-targets (followers: --rocm-targets=gfx1100, no source edit).

## CONFIRMED blocker on ROCm 7.2.1
hip_64bit.mk ROCM_FLAGS hardcodes -std=c++14; rocPRIM (pulled by hipCUB) #errors "requires C++17"
(rocprim/config.hpp:68). Probe: `hipcc -x hip -std=c++14 -c <#include hipcub/hipcub.hpp>` -> exit 141;
-std=c++17 -> exit 0. Fix: bump the HIP build to -std=c++17 (HIP TUs only; CPU/CUDA unchanged).

## GPU tests (validator)
Primary: `cd src/src/cudamatrix && make test` (12 device unit tests; run serially on one GPU).
cudafeat/cudadecoder have empty TESTFILES -> build their bins + optional tiny egs smoke-run.

## No inter-project MOAT deps
Only ROCm system libs (hipBLAS/hipSPARSE/hipSOLVER/hipRAND/hipFFT/hipCUB/roctx) + OpenFST/BLAS.

## Fault-class scan: clean except warp size
No textures/surfaces, no __shfl/__ballot, no managed-atomic, no sub-word atomics. Warp size abstracted as
GPU_WARP_SIZE (64 HIP / 32 CUDA) -- but hipify.h hardcodes 64 (wrong for RDNA followers; per-arch fix in the
follower delta-plan). Each .cu does `#define __CUDA_ARCH__ 800` to activate device-intrinsic branches (by design).
The only atomicCAS sites (cu-kernels.cu, chain-kernels.cu) are the classic 64-bit-word myAtomicAdd(double*)
emulation inside `#if __CUDA_ARCH__ < 600` -- dead because __CUDA_ARCH__ is forced to 800, so native fp64
atomicAdd (+ -munsafe-fp-atomics) is used. Full-word CAS, wave-agnostic; the cuCollections sub-word CAS bug
does NOT apply.

## Porter bringup on gfx90a / ROCm 7.2.1 (the actual reproducible build)
Host build deps installed via apt (standard packages): `liblapacke-dev` (kaldi's OpenBLAS path includes
lapacke.h), `sox` (kaldi tools/extras/check_dependencies.sh exits 1 without it, which blocks the tools build).
System OpenBLAS 0.3.26 is used (no in-tree OpenBLAS build needed); it exports the full Fortran LAPACK
(dpotrf_/dgetrf_/dgesvd_/...), and kaldi's matrix/cblas-wrappers.h OpenBLAS branch (`#if !defined(HAVE_ATLAS)`)
calls those raw Fortran symbols, so `-lopenblas` alone satisfies CBLAS + LAPACK (no LAPACKE_* needed).

1. tools/ (OpenFST 1.8.4): the openslr.org mirror 404s for 1.8.4; fetch from openfst.org instead:
   `curl -fSL -o tools/openfst-1.8.4.tar.gz http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.8.4.tar.gz`
   Then `cd tools && make -j12 openfst` (the tools Makefile gates every target on check_required_programs, so
   sox + a `tools/python/` dir must exist first: `mkdir -p tools/python` and install sox). Only `openfst` is
   needed for the GPU libs; the full `make` (sctk/sph2pipe) is unnecessary for cudamatrix validation.
2. configure: system OpenBLAS lives in the multiarch path /usr/lib/x86_64-linux-gnu, which `--openblas-root`
   does NOT probe (it checks $ROOT/lib and $ROOT/lib64 only). Build a tiny shim prefix of symlinks
   (lib/libopenblas.so -> system .so; include/cblas.h -> cblas-openblas.h; include/lapacke.h -> system) and
   point configure at it:
   `cd src/src && ./configure --use-rocm --rocm-dir=/opt/rocm --rocm-targets=gfx90a --use-cuda=no \
        --mathlib=OPENBLAS --openblas-root=<shim_prefix>`
   configure's ROCm >=5.2 gate and `hipconfig -v` parsing (7.2.53211 -> major 7, minor 2) are fine on 7.2.1.
3. build: `make -j12 depend && make -j12` from src/src (cap -j12; concurrent heavy builds on the host).

## ROCm 7.2.1 fixes applied (minimal diff; NVIDIA path byte-identical)
- src/makefiles/hip_64bit.mk: ROCM_FLAGS `-std=c++14` -> `-std=c++17` (rocPRIM/hipCUB #error on <C++17).
- src/hip/hipify.h hipBLAS v2 (ROCm >=6) GemmEx compute-type drift: `hipblasDatatype_t` was REMOVED in 7.2.1.
  `cublasComputeType_t` alias -> `hipblasComputeType_t` (was hipblasDatatype_t); `CUBLAS_COMPUTE_32F*` ->
  `HIPBLAS_COMPUTE_32F*` (was HIPBLAS_R_32F, an operand hipDataType). GemmEx operand types stay HIP_R_32F via
  the existing CUBLAS_R_32F/CUDA_R_32F aliases. hipblasComputeType_t enum is in hipblas-common/hipblas-common.h.
- src/cudafeat/feature-online-batched-ivector-cuda.cc: that file locally redefines CUDA_R_32F to HIPBLAS_R_32F
  and passed CUDA_R_32F in the GemmStridedBatchedEx COMPUTE-type slot (v2 wants hipblasComputeType_t there).
  Added a HIP-only KALDI_GEMMEX_COMPUTE_32F = HIPBLAS_COMPUTE_32F (else CUDA_R_32F on NVIDIA) for that one slot.
- src/hip/hipify.h __syncwarp ambiguity: ROCm 7.2.1 HIP now ships a native wave-correct __syncwarp
  (amd_detail/amd_warp_sync_functions.h: fence + __builtin_amdgcn_wave_barrier, no-arg + 64-bit-mask template),
  so hipify.h's own `__syncwarp(unsigned mask=0xffffffff)` makes a bare __syncwarp() call ambiguous. Guarded the
  shim with `#include <hip/hip_version.h>` + `#if HIP_VERSION < 60200000` (native version arrived in ROCm 6.2);
  on 7.2.1 HIP's builtin is used (stronger semantics than the old s_nop, and correct on wave32 AND wave64).
- rocPRIM arch-detection vs `#define __CUDA_ARCH__ 800`: on ROCm 7.2.1 rocprim/config.hpp keys ROCPRIM_TARGET_CDNA2
  on the compiler's __gfx90a__ macro. cu-kernels.cu / feature-online-cmvn-cuda.cu defined __CUDA_ARCH__ 800 BEFORE
  including <hipcub/hipcub.hpp>, which suppresses __gfx90a__ in the device pass -> rocprim falls through to its
  "support for 128-bit atomics not implemented for current architecture" #error (atomic.hpp). Fix: move the
  `#define __CUDA_ARCH__ 800` to AFTER the hipcub/hipify includes in those two files (verified empirically: a
  3-line hipcub TU compiles with __CUDA_ARCH__ defined-after, errors with defined-before). kaldi's own
  `#if __CUDA_ARCH__ >= 600` native-fp64-atomicAdd branches still fire. chain-kernels.cu pulls no hipcub so its
  define is left in place. ROCm 5.x rocPRIM had no such arch gate, so this only bites modern ROCm.

## WAVE64 CORRECTNESS BUG fixed (cu-kernels.cu _diff_lstm_nonlinearity self_repair_sum_out)
Surfaced by cu-math-test's UnitTestBackpropLstmNonlinearity (the ONLY cudamatrix test that failed pre-fix; the
other 11 passed). Root cause: the kernel wrote the 5 self-repair-sum rows with `if (i0 < 5)` where
i0 = blockIdx.y*blockDim.y + threadIdx.y (gridDim.y==1, so == threadIdx.y), requiring blockDim.y >= 5. The host
launch (cu-math.cc:821-822) sets `dimBlock(GPU_WARP_SIZE, CU1DBLOCK/GPU_WARP_SIZE)`, i.e. blockDim.y =
256/warpSize = 8 on a 32-lane warp (NVIDIA, RDNA) but only 4 on a 64-lane wavefront (CDNA gfx90a). So row 4
(c_t self-repair) had NO thread to write it and was read back stale -> ~50% Frobenius error on that one row
(input/params/value/deriv derivs were all ~1e-7, only self_repair was 0.4969). FIX (arch-unified, NVIDIA-safe):
update_sr[] depends only on the column j, not on threadIdx.y, so the threadIdx.y==0 lane writes all 5 rows in a
loop. Correct on wave32 (was 8 lanes) and wave64 (was 4 lanes); identical values on NVIDIA. This is the
canonical warp-size fault class: a `blockDim.y = blockSize/warpSize` derivation that silently shrinks the y-extent
below a hardcoded count on wave64.

## GPU validation (gfx90a / MI250X / ROCm 7.2.1) -- PASSED
Ran on one isolated GCD via HIP_VISIBLE_DEVICES (HIP enumerates 4 gfx90a devices 0-3 on this host, each warpSize 64).
- cudamatrix: `make -j12 test_compile` then each test run serially. ALL 12 correctness tests PASS:
  cu-vector/matrix/math/test/sp-matrix/packed-matrix/tp-matrix/block-matrix/array/sparse-matrix/device/compressed-matrix.
  Covers hipBLAS L1/L3, hipSPARSE CSR SpMM + csr2csc, hipSOLVER Cholesky, hipRAND, hipCUB BlockReduce, bespoke kernels.
  The 4 speed tests (cu-{matrix,vector,sp-matrix,rand}-speed-test) run without error.
- cudafeat: exercised on-device -- cudafeatbin/compute-mfcc-feats-cuda computed MFCC features from a 16kHz wav,
  selected the MI250X, "Done 1 out of 1 utterances", valid numeric output.
- cudadecoder: batched-wav-nnet3-cuda{,2,-online} build and load the HIP runtime cleanly.
Fork HEAD pushed: jeffdaily/kaldi @ moat-port = cdc8d2f907f00b8fc2dc7459f3e3cb092afc0aa6.

## Repeatable build/test scripts
agent_space/kaldi_build/{build_kaldi.sh (configure|depend|build), run_cudamatrix_tests.sh <gcd>}. OpenBLAS in-tree
install at tools/OpenBLAS/install (built by tools/extras/install_openblas.sh). OpenFst 1.8.4 fetched from
openfst.org (the openslr mirror 404s).
