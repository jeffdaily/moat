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

## Review 2026-05-31 (linux-gfx90a, moat-port cdc8d2f vs e02e35f) -- reviewer
Verdict: review-passed. No defects found; safe to proceed to GPU validation. Diff is 5 files +47/-15, every functional edit inside __IS_HIP_COMPILE__ / HIP_VERSION / the #else of a HIP guard / the HIP makefile, or the arch-unified kernel fix. NVIDIA/CPU path byte-identical (verified line-by-line). Findings: none (problems-only per skill philosophy). Verifications recorded so a re-review need not redo them:
- wave64 LSTM fix (cu-kernels.cu:3806-3816): CU1DBLOCK=256 (cu-matrixdim.h:57), launch dimBlock.y=CU1DBLOCK/GPU_WARP_SIZE (cu-math.cc:821-822) = 8 wave32 / 4 wave64; gridDim is 1D (cu-math.cc:828) so i0==threadIdx.y. update_sr[] (cu-kernels.cu:3675-3680) is a function of column j only (deriv_sum_in[i*stride+j] < sr_config[i]*count), lane-independent. Old `i0<5` had lane s write row s; new `threadIdx.y==0` writes all 5 in an unrolled loop -- identical bytes on wave32/NVIDIA, fixes the starved row 4 on wave64. self_repair write touches no smem and is bracketed by __syncthreads, so no new race. value_sum_out/deriv_sum_out rows use the tid<warpSize reduction-leader pattern (not i0<5), so they were never starved -- self_repair was the only defective site; fix is complete.
- __CUDA_ARCH__ moved after hipcub includes (cu-kernels.cu:44, feature-online-cmvn-cuda.cu:25): headers between old/new position (hip_runtime, hipcub, block_reduce, cu-kernels-ansi.h, hipify.h) consume no __CUDA_ARCH__ (grep clean); value stays 800; only guard in cu-kernels.cu body is `#if __CUDA_ARCH__ < 600` at 988 (dead, below define); cmvn `#if __CUDA_ARCH__ == 750` at line 50 is below define at 25 -> 800==750 false -> __launch_bounds__(1024,2), same as before. Behavior preserved.
- hipBLAS v2 drift: on installed ROCm 7.2.1 hipblasDatatype_t is REMOVED (grep -rl empty), HIPBLAS_COMPUTE_32F/_FAST_16F/_FAST_TF32 + hipblasComputeType_t exist in hipblas-common/hipblas-common.h (hipblas.h:40 includes it), HIPBLAS_R_32F still = HIP_R_32F. hipify.h aliases now resolve; ivector .cc KALDI_GEMMEX_COMPUTE_32F = HIPBLAS_COMPUTE_32F (HIP) / CUDA_R_32F (NVIDIA, genuine cudaDataType -- the local CUDA_R_32F redefine is inside the HIP branch only) -> NVIDIA GemmStridedBatchedEx call byte-identical.
- OPENFST_VER: OPENFSTVER=10804 (kaldi.mk:21); host CXXFLAGS already pass -DOPENFST_VER (linux_openblas.mk:25) and nvcc does too (cuda_64bit.mk:11); kaldi-types.h:42 `#if OPENFST_VER >= 10800` else <fst/types.h> (dropped in OpenFst 1.8.x). HIP compile lacked it -> took the dead branch. Fix adds it to ROCM_FLAGS, matching host/nvcc. Sound.
- __syncwarp shim guarded `#if HIP_VERSION < 60200000` (hipify.h, inside #ifdef __HIPCC__, with #include <hip/hip_version.h>); native wave-correct builtin used on 7.2.1. Preexisting Unicode in the untouched shim comment correctly left alone.
- C++17 bump HIP-only (hip_64bit.mk ROCM_FLAGS); host/CUDA std untouched.
- Hygiene: title 68 chars [ROCm]-prefixed; Claude-disclosed; Test Plan with fenced commands; no Co-Authored-By/noreply/ghstack; ASCII, no em-dash; author jeff.daily@amd.com (user's own public email, jeffdaily fork -- not an AMD-internal account). kaldi.mk.bak is gitignored (.gitignore:80), untracked, NOT in the commit.
Note for validator: validated_sha is null; this review does not gate on the GPU run (expected to be pending at review time). The decisive gate remains the 12 cudamatrix tests incl. cu-math-test BackpropLstmNonlinearity on gfx90a.

## Validation 2026-05-31 (linux-gfx90a, gfx90a / MI250X, ROCm 7.2.1)

Platform: linux-gfx90a. Fork: jeffdaily/kaldi @ moat-port = cdc8d2f907f00b8fc2dc7459f3e3cb092afc0aa6. GPU: gfx90a (MI250X), HIP_VISIBLE_DEVICES=2 (selected as idle GCD; 4 GCDs enumerated 0-3, GCD 2 chosen at 0% utilization, 0% VRAM).

Commands run (all from /var/lib/jenkins/moat):

```
# Incremental configure (near-no-op):
bash utils/timeit.sh kaldi compile -- bash agent_space/kaldi_build/build_kaldi.sh configure

# Incremental build (cudamatrix only):
bash utils/timeit.sh kaldi compile -- make -j16 -C projects/kaldi/src/src/cudamatrix

# Build test binaries:
bash utils/timeit.sh kaldi compile -- make -j16 -C projects/kaldi/src/src/cudamatrix test_compile

# Run 12 correctness tests serially on GCD 2:
bash utils/timeit.sh kaldi test -- bash agent_space/kaldi_build/run_cudamatrix_tests.sh 2
```

Results: PASS=12 FAIL=0. All 12 cudamatrix correctness tests passed:
cu-vector-test, cu-matrix-test, cu-math-test, cu-test, cu-sp-matrix-test, cu-packed-matrix-test, cu-tp-matrix-test, cu-block-matrix-test, cu-array-test, cu-sparse-matrix-test, cu-device-test, cu-compressed-matrix-test.

Decisive gate -- cu-math-test BackpropLstmNonlinearity: PASSED. The wave64 _diff_lstm_nonlinearity fix (threadIdx.y==0 writes all 5 self-repair rows) is verified correct on gfx90a wavefront-64. UnitTestBackpropLstmNonlinearity ran float+double at dim 16-1024 without error.

Timing: incremental configure ~0.6s, cudamatrix build ~5s, test_compile ~6.5s, 12 tests ~64s.

Verdict: COMPLETED. validated_sha = cdc8d2f907f00b8fc2dc7459f3e3cb092afc0aa6. Followers linux-gfx1100 and windows-gfx1151 unblocked to port-ready.

## Validation 2026-05-31 (gfx1100, ROCm 7.2.1)

Platform: linux-gfx1100. GPU: 2x AMD Radeon Pro W7800 48GB (gfx1100 / RDNA3 / wave32), HIP_VISIBLE_DEVICES=0. ROCm 7.2.1. Fork before fix: jeffdaily/kaldi @ moat-port = cdc8d2f907f00b8fc2dc7459f3e3cb092afc0aa6; after fix (amended): e8c5613b789eefb6b0a251d8b15867bb53f1a01d.

### GPU_WARP_SIZE wave32 fix

hipify.h hardcoded GPU_WARP_SIZE to 64 (CDNA/wave64), which is wrong on gfx1100 (RDNA3/wave32). The fix is a 3-file change, all inside the HIP build path:

- src/configure: in the configure_rocm ROCM_TARGETS loop, default ROCM_WARP_SIZE=32, set to 64 for any gfx9* target, write `ROCM_WARP_SIZE = <N>` to kaldi.mk.
- src/makefiles/hip_64bit.mk: add `-DHIP_WARP_SIZE=$(ROCM_WARP_SIZE)` to both CXXFLAGS (host g++ path) and ROCM_FLAGS (hipcc device path).
- src/hip/hipify.h: replace `#define GPU_WARP_SIZE 64` with arch-conditional:
  - Device code (inside `#ifdef __HIP_DEVICE_COMPILE__`): `#if defined(__GFX9__) -> 64 else 32` -- `__GFX9__` is set by hipcc for gfx9x CDNA targets and absent on RDNA gfx11xx.
  - Host code (else branch): `#define GPU_WARP_SIZE HIP_WARP_SIZE` -- uses the configure-injected define.
  GPU_MAX_WARPS_PER_BLOCK derives from GPU_WARP_SIZE, so all host block-dim calculations are correct on both arches.

Verified correct with compile-time static_assert: GPU_WARP_SIZE==32 for both host and device on gfx1100; GPU_WARP_SIZE==64 for both on gfx90a (no error). gfx90a backward compatibility confirmed -- the configure case matches `gfx9*`, ROCM_WARP_SIZE=64, hipify.h __GFX9__ is defined in device code.

### Build commands

```
# Install deps
sudo apt-get install -y liblapacke-dev sox

# OpenFST (curl from openfst.org; openslr mirror 404s for 1.8.4)
cd projects/kaldi/src/tools
curl -fSL -o openfst-1.8.4.tar.gz http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.8.4.tar.gz
mkdir -p python && make -j12 openfst

# In-tree OpenBLAS (system lapacke.h includes lapack.h with LAPACK_FORTRAN_STRLEN_END, 
# adding hidden-length args to Fortran function decls that conflict with kaldi's call sites;
# the in-tree OpenBLAS headers do not have this issue)
bash extras/install_openblas.sh

# Configure and build for gfx1100
cd projects/kaldi/src/src
./configure --use-rocm --rocm-dir=/opt/rocm --rocm-targets=gfx1100 --use-cuda=no \
    --mathlib=OPENBLAS --openblas-root=../tools/OpenBLAS/install

# Compile (wrapped)
bash utils/timeit.sh kaldi compile -- make -j12 depend -C projects/kaldi/src/src/cudamatrix
bash utils/timeit.sh kaldi compile -- make -j12 -C projects/kaldi/src/src/cudamatrix
bash utils/timeit.sh kaldi compile -- make -j12 -C projects/kaldi/src/src/cudamatrix test_compile
```

### gfx1100 code-object evidence and GPU_WARP_SIZE=32 confirmation

roc-obj-ls on cu-kernels.o:
  `hipv4-amdgcn-amd-amdhsa--gfx1100` (961 KB device code; no gfx90a).

kaldi.mk after configure: `ROCM_WARP_SIZE = 32`.
Compiler invocations confirmed: `-DHIP_WARP_SIZE=32` in both the host g++ CXXFLAGS line and the hipcc ROCM_FLAGS line.
Compile-time static_assert (external probe): `static_assert(GPU_WARP_SIZE == 32)` passes for both host and device on gfx1100.

### Validation results

```
# Run 12 cudamatrix device tests serially on HIP_VISIBLE_DEVICES=0
bash utils/timeit.sh kaldi test -- bash agent_space/run_cudamatrix_gfx1100.sh 0

# Second run (determinism)
bash utils/timeit.sh kaldi test -- bash agent_space/run_cudamatrix_gfx1100.sh 0
```

Run 1: PASS=12 FAIL=0. Run 2: PASS=12 FAIL=0 (deterministic).

All 12 tests passed:
cu-vector-test, cu-matrix-test, cu-math-test, cu-test, cu-sp-matrix-test, cu-packed-matrix-test, cu-tp-matrix-test, cu-block-matrix-test, cu-array-test, cu-sparse-matrix-test, cu-device-test, cu-compressed-matrix-test.

Decisive gate -- cu-math-test BackpropLstmNonlinearity: PASSED (exit 0, both runs). The test exercises float+double at dim 16-2048, covering the warp-reduction-heavy _diff_lstm_nonlinearity kernel. With GPU_WARP_SIZE=32, blockDim.y = CU1DBLOCK/GPU_WARP_SIZE = 256/32 = 8 (as on NVIDIA), which gives all 5 self-repair rows (0-4) a thread to write them; no stale row 4, all rows correct. This is the definitive wave32 proof: the kernel that previously failed on wave64 (blockDim.y=4, row 4 starved) now also has blockDim.y=8 on wave32 with the correct warp size, and passes.

Matches gfx90a bar: 12/12 (identical pass count, same test names).

### Notes

The gfx1100 configure needs the in-tree OpenBLAS (tools/extras/install_openblas.sh), not the system OpenBLAS shim. The system lapacke.h (liblapacke-dev 3.12.0) transitively includes lapack.h which defines LAPACK_FORTRAN_STRLEN_END unconditionally, adding hidden-length size_t args to Fortran function declarations (stptri_, sgesvd_, etc.) that conflict with kaldi's direct Fortran call sites in matrix/cblas-wrappers.h. The in-tree OpenBLAS 0.3.13 lapacke.h uses only LAPACKE_* wrapper prototypes and does not include lapack.h.

Verdict: COMPLETED. validated_sha = e8c5613b789eefb6b0a251d8b15867bb53f1a01d. Wave32 verdict: CORRECT -- BackpropLstmNonlinearity passes on gfx1100 with GPU_WARP_SIZE=32, confirming the warp-reduction launch geometry is correct for RDNA3.

## Windows-gfx1151 attempt 2026-06-03

Platform: windows-gfx1151 (AMD Radeon 8060S, gfx1151, RDNA3.5, wave32, Windows 11, TheRock ROCm 7).
Fork cloned: jeffdaily/kaldi @ moat-port = e8c5613b789eefb6b0a251d8b15867bb53f1a01d (confirmed with git ls-remote + git rev-parse HEAD).

### Verdict: BLOCKED -- POSIX-host-build, not ROCm-port scope

The kaldi ROCm build path is gated exclusively on Linux at the configure script level. `src/configure` `configure_rocm()` function (lines 308-316):

```bash
# 64bit/32bit? Not Linux? We do not support cross compilation with ROCm so,
# use direct calls to uname -m here
if [ "`uname -m`" == "x86_64" ] && [ "`uname`" == "Linux" ] ; then
    cat makefiles/hip_64bit.mk >> kaldi.mk
else
    echo "WARNING: ROCm will not be used!
         ROCm is only supported with 64-bit Linux builds."
    exit 1;
fi
```

On Windows (including MSYS2/git-bash where `uname` returns `MINGW64_NT-*` not `Linux`), the configure script exits 1 before emitting any HIP build rules. This is a deliberate, designed exclusion in the upstream build system.

The full build chain is POSIX-only:

- `tools/Makefile`: OpenFST 1.8.4 via autoconf `./configure && make`; sox; OpenBLAS via POSIX `make`.
- `src/configure`: a bash script using `uname`, `sed`, `awk`, POSIX path discovery.
- `src/makefiles/hip_64bit.mk`: GNU make rules, `$(HIPCC)` invocations, POSIX linker flags.
- `src/cudamatrix/Makefile` and all peer Makefiles: GNU make, recursive `$(CXX)` rules, no MSVC support.

Alternative build paths investigated:

- `cmake/INSTALL.md` + `CMakeLists.txt`: No ROCm/HIP option anywhere in the CMake tree (grep clean for rocm/ROCm/HIP/hip). The CMake path is CPU-only.
- `windows/INSTALL.md`: Generates MSVC `.sln` files (Windows/.props); explicitly states "For now (20171121), we do not support CUDA. We might add the support again in the future, but for now we do not express any commitment to do so." No ROCm support added since.

Porting kaldi's autotools+Makefile system to Windows would require rewriting `src/configure` (bash, POSIX) as a Windows-compatible build system, porting OpenFST's autoconf build to MSVC/CMake, providing sox and OpenBLAS on Windows in a form the Makefile can consume, and creating a new Windows HIP build path in `src/makefiles/`. This is a host-build-system port of a POSIX-native project, not the ROCm port (the HIP device code and `hipify.h` are sound and validated on gfx90a + gfx1100). Out of scope per MOAT SCOPE rule.

State: windows-gfx1151 blocked=True, blocked_reason: configure_rocm() explicitly requires uname==Linux; exits 1 on Windows; no Windows-native ROCm build path exists.

## Validation 2026-05-31 (linux-gfx90a revalidate, moat-port e8c5613b)

Platform: linux-gfx90a. Prior validated_sha: cdc8d2f. New HEAD: e8c5613b789eefb6b0a251d8b15867bb53f1a01d (gfx1100 delta: per-arch GPU_WARP_SIZE + in-tree OpenBLAS sourcing). GPU: 4x AMD Instinct MI250X (gfx90a, wave64), HIP_VISIBLE_DEVICES=0. ROCm 7.2.1.

Purpose: confirm the gfx1100 follower delta did not regress gfx90a.

### Gfx90a invariants verified at e8c5613b

Tree at new HEAD: `git reset --hard e8c5613b` confirmed (`git rev-parse HEAD == e8c5613b789eefb6b0a251d8b15867bb53f1a01d`).

OpenBLAS path: in-tree OpenBLAS (tools/OpenBLAS/install), same as gfx1100. The configure call is `./configure --use-rocm --rocm-dir=/opt/rocm --rocm-targets=gfx90a --use-cuda=no --mathlib=OPENBLAS --openblas-root=.../tools/OpenBLAS/install`. The system lapacke.h conflict (LAPACK_FORTRAN_STRLEN_END) that forced in-tree OpenBLAS on gfx1100 applies equally on gfx90a at this HEAD; in-tree OpenBLAS is now universal for both arches.

ROCM_WARP_SIZE: configure sets `ROCM_WARP_SIZE = 64` for gfx90a (the `gfx9*` case in the configure_rocm loop). kaldi.mk confirmed: `ROCM_WARP_SIZE = 64`. Compiler flags confirmed in both host CXXFLAGS and ROCM_FLAGS: `-DHIP_WARP_SIZE=64`.

Device GPU_WARP_SIZE: hipify.h `#ifdef __HIP_DEVICE_COMPILE__ / #if defined(__GFX9__) #define GPU_WARP_SIZE 64` -- __GFX9__ is set by hipcc for gfx90a, so device code sees 64. Static_assert-equivalent: confirmed by the gfx1100 delta notes and per-arch compile logic.

Code object: `roc-obj-ls cu-kernels.o` -> `hipv4-amdgcn-amd-amdhsa--gfx90a` (1018 KB device code; no gfx1100 object). Correct.

### Commands

```
cd /var/lib/jenkins/moat/projects/kaldi/src
git fetch jeffdaily moat-port
git reset --hard e8c5613b789eefb6b0a251d8b15867bb53f1a01d

# Configure (regenerates kaldi.mk with ROCM_WARP_SIZE = 64)
bash utils/timeit.sh kaldi compile -- bash agent_space/kaldi_build/build_kaldi.sh configure

# Depend + build cudamatrix + test_compile
cd projects/kaldi/src/src && make -j12 depend
bash utils/timeit.sh kaldi compile -- make -j12 -C projects/kaldi/src/src/cudamatrix
bash utils/timeit.sh kaldi compile -- make -j12 -C projects/kaldi/src/src/cudamatrix test_compile

# Run 1
bash utils/timeit.sh kaldi test -- bash agent_space/kaldi_build/run_cudamatrix_tests.sh 0

# Run 2 (determinism)
bash utils/timeit.sh kaldi test -- bash agent_space/kaldi_build/run_cudamatrix_tests.sh 0
```

### Results

Run 1: PASS=12 FAIL=0. Run 2: PASS=12 FAIL=0 (deterministic).

All 12 tests passed both runs: cu-vector-test, cu-matrix-test, cu-math-test, cu-test, cu-sp-matrix-test, cu-packed-matrix-test, cu-tp-matrix-test, cu-block-matrix-test, cu-array-test, cu-sparse-matrix-test, cu-device-test, cu-compressed-matrix-test.

Decisive gate -- cu-math-test BackpropLstmNonlinearity: PASSED (exit 0, both runs). The wave64 _diff_lstm_nonlinearity fix (threadIdx.y==0 writes all 5 self-repair rows) continues to be correct on gfx90a wave64 at e8c5613b. The gfx1100 delta (warp-size conditionalization in hipify.h/__GFX9__) does not change device code for gfx90a: __GFX9__ is set, GPU_WARP_SIZE=64, blockDim.y=CU1DBLOCK/64=4 -- same as before the delta, same fix, same correctness.

No regression introduced by the gfx1100 follower delta.

Verdict: COMPLETED. validated_sha = e8c5613b789eefb6b0a251d8b15867bb53f1a01d.
