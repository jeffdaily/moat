# 3P-ADMM-PC2 notes

Privacy-preserving distributed ADMM for LASSO. CPU does the optimization
(Paillier homomorphic encryption + quantized ADMM, gmpy2). The GPU's only job
is to accelerate the Paillier modular exponentiation `g^m mod n` over a batch
of big integers, via cuFFT-based large-integer polynomial multiplication +
Barrett reduction.

## Build classification
ext_type `nvcc-shared`: ad-hoc nvcc commands in README.md sec.1 (no CMake, no
Makefile, no setup.py). The `.cu` is compiled + a generated C wrapper, linked
with `g++ -shared` into `/tmp/lib_cufft.so`. The Python side
(`crypto/paillier_gpu.py`, `protocol/edge_crt_helper.py`) loads that .so by
absolute path via `ctypes.CDLL` and calls `init_gpu(int)` / `run_modexp(...)`.

## Port scope (LIVE GPU path only)
Ported ONLY `gpu/cufft_modexp.cu` (cuFFT Z2Z + scalar bigint kernels). This is
the only GPU code wired into the protocol.

Deliberately EXCLUDED:
- Dead register/warp-NTT files: `gpu/reg_ntt_modexp*.cu`, `gpu/ntt_modexp*.cu`,
  `gpu/ntt_funcs.cu`, `gpu/ntt_kernel2.cu`, `gpu/ntt_wrapper*.cu`,
  `gpu/modexp_final.cu`. Referenced by nothing in the repo (no Python import,
  no launch site, not built by the README). The `reg_ntt_*` and `ntt_funcs`
  files DO use `__shfl_xor_sync(0xffffffff, ..., delta)` warp-NTT butterflies
  with blockDim=32 (one warp/task) -- classic wave64 hazards -- but they are
  not on any live path, so porting them would expose the wave64 fault class for
  no validatable benefit. A future contributor reviving the register-NTT path
  on AMD would need the `__GFX9__`->64 / RDNA->32 + width-aware ballot
  treatment.
- Standalone PyCUDA self-test: `gpu/gpu_modexp.py` (JITs `gpu/modexp.cu` /
  `gpu/modexp_v2.cu`). PyCUDA is not in requirements.txt and this is a
  `__main__` self-test, not called by the protocol. Out of scope.

## What changed
- ADD `gpu/cuda_to_hip.h`: single compat shim. Under `__HIP__`/
  `__HIP_PLATFORM_AMD__`/`USE_HIP` it includes `<hip/hip_runtime.h>` +
  `<hipfft/hipfft.h>` and aliases the exact CUDA symbols the kernel uses
  (`cudaMalloc/Free/Memcpy` + dirs; `cufftHandle`/`cufftDoubleComplex`,
  `cufftPlanMany`/`cufftExecZ2Z`/`cufftDestroy`, `CUFFT_Z2Z`/`CUFFT_FORWARD`,
  `CUFFT_INVERSE`->`HIPFFT_BACKWARD`); else it includes the original CUDA
  headers. NVIDIA build stays byte-identical.
  - Gotcha: hipcc compiling `-x hip` defines `__HIP__`, NOT
    `__HIP_PLATFORM_AMD__`, at preprocess time. Key the `#if` off `__HIP__`
    too, or the shim falls through to `<cuda_runtime.h>` and fails to find it.
  - libc headers (`<cstdlib><cstring><cmath>`) are included BEFORE the HIP
    runtime (gpuRIR lesson): host `calloc`/`free`/`round` in the .cu would
    otherwise risk binding to HIP `__device__` overloads.
- EDIT `gpu/cufft_modexp.cu`: replaced `<cuda_runtime.h>`/`<cufft.h>`/`<math.h>`
  with `#include "cuda_to_hip.h"`. No body changes; all cuda*/cufft* spellings
  preserved.
- ADD `gpu/build_hip.sh`: hipcc recipe mirroring the README nvcc steps,
  producing `/tmp/lib_cufft.so` with the SAME `init_gpu`/`run_modexp` C ABI
  (Python untouched), linking `-lhipfft`. Arch from `HIP_ARCH` (default
  `gfx90a`, never a literal). hipcc does host+device link in one step (no
  separate `-dlink`/`-fgpu-rdc` needed for this non-RDC build).

## Build (gfx90a)
```
HIP_VISIBLE_DEVICES=1 \
BUILD=<scratch> OUT=/tmp/lib_cufft.so \
  bash gpu/build_hip.sh
```
Followers: `HIP_ARCH=gfx1100` / `gfx1151`. Multi-arch fat binary:
`HIP_ARCH="gfx90a,gfx1100"` builds clean (verified). The only warnings are
benign `-Wunused-value` on the cuda*Memcpy nodiscard return (present in the
NVIDIA build too). hipcc/ROCm 7.2.1, hipfft present at /opt/rocm.

## hipFFT rounding note (the one real risk -- PASSED)
The bigint multiply is a double-precision Z2Z FFT then `round()` to recover
exact integer limbs (`norm_round` + `carry_prop`), relying on a < 0.5 rounding
margin. hipFFT twiddles differ bit-for-bit from cuFFT, so a thin margin could
in principle round a limb the wrong way. Validated statistically: 6144 random
modexp cases across moduli up to 2046-bit n^2 -- all exact-match gmpy2. The
double-FFT margin holds on gfx90a (full-rate fp64). No code change was needed.

## Validation (real gfx90a, GCD 1, HIP_VISIBLE_DEVICES=1)
- modexp-vs-gmpy2 gold: 6144/6144 exact match. `gpu_batch_modexp(g,m,n)` vs
  `gmpy2.powmod(g,m,n)`, n at 255/510/511/1022/768/1535/1023/2046 bits, varied
  m_bits, 256/batch. Harness: agent_space/3p-admm/validate.py (not committed).
- ALGORITHM PRECONDITION: the kernel assumes base g < modulus n (no initial
  reduction). A test with g >= n mismatches gmpy2 -- this is expected (contract
  violation), NOT a HIP bug. Real Paillier always has g < modulus. Reducing
  g %= n before the call makes it match exactly.
- encrypt round-trip: `crypto/paillier_gpu.encrypt_batch_gpu` (GPU g^m mod n^2)
  -> CPU `crypto/paillier.decrypt` = 64/64 plaintexts recovered.
- CPU regression (no GPU touched): `crypto/test_paillier.py`,
  `test_quantization.py`, `test_full_chain.py` all pass.
- gfx90a dispatch confirmed via AMD_LOG_LEVEL=3 (hipFFT `twiddle_gen_radices_dp`
  + our kernels on device; rocminfo Name gfx90a).

## Wave size
Not exposed in the live path: no `__shfl`/`__ballot`/`warpSize`/`__syncwarp`,
no cub/thrust/curand, no textures/atomics. The `<<<N,32>>>` launches
(`cond_copy`, `load_complex` grid.x) are independent-thread strided copies, not
warp collectives. Multi-arch is trivially fine; gfx1100/gfx1151 deltas are
expected to be no-op revalidations (same .so rebuilt for the arch). hipFFT is
arch-agnostic.

## Upstream warts (NOT touched -- out of port scope)
- Hardcoded `/mnt/3p-admm-pc2/...` absolute paths in Python (lib + kernel source
  paths) and `sys.path.append('/mnt/3p-admm-pc2')` in the test files. For our
  build+validate we point the loader at the real /tmp/lib_cufft.so and set
  PYTHONPATH. We do not refactor upstream's path scheme.
- `/tmp` is non-persistent (README notes "recompile each boot").
- Distributed end-to-end tests (`experiments/*.py`, `protocol/master_node.py`)
  need multi-node SSH / matpool cloud; not runnable here. The modexp gold gate
  + encrypt round-trip are the representative validatable proxy for the GPU work
  those drive.
