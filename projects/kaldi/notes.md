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
