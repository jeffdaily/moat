# sppark notes

## Validation Summary (linux-gfx90a)

sppark has authoritative upstream HIP support from Supranational. Validation on gfx90a (MI250X, ROCm 7.2.1) shows **partial** functionality:

### What Works: NTT (Number Theoretic Transform)

All 7 field types pass correctness tests against arkworks reference:

```
NVCC=off HIPCC=/opt/rocm/bin/hipcc cargo test --release --features=<field> -- --nocapture
```

| Field | Test | Status |
|-------|------|--------|
| bls12_381 | test_against_arkworks | PASS |
| bls12_377 | test_against_arkworks | PASS |
| bn254 | test_against_arkworks | PASS |
| pallas | test_against_arkworks | PASS |
| vesta | test_against_arkworks | PASS |
| gl64 (Goldilocks) | gl64_self_consistency | PASS |
| bb31 (Baby Bear) | bb31_self_consistency | PASS |

### What Does Not Work: MSM (Multi-Scalar Multiplication)

The MSM POC does not compile on HIP due to missing infrastructure:

1. **Build system gap**: poc/msm-cuda/build.rs only detects nvcc, not hipcc (unlike sppark crate's build.rs which detects both)

2. **mont_t.hip incompleteness**: Only ~25% of mont_t.cuh is ported to mont_t.hip:
   - mont_t.hip: 311 lines
   - mont_t.cuh: 1221 lines
   - Missing critical functions: `shfl_xor`, `cneg`, `vt_inverse_mod_x`, and warp-shuffle-based Montgomery arithmetic

3. **fp2 extension field not ported**: bls12-381-fp2.hpp, bls12-377-fp2.hpp, and alt_bn128-fp2.hpp are guarded by `#ifdef __CUDA_ARCH__` with no HIP equivalent. The fp2 device code (required for G2 MSM) uses mont_t functions not available in mont_t.hip.

4. **MSM kernel headers not HIP-ready**: pippenger.cuh, batch_addition.cuh use `#ifdef __CUDA_ARCH__` guards and include `<cuda.h>`, `<cooperative_groups.h>` without HIP alternatives.

### Root Cause

Supranational's HIP support focuses on NTT which uses basic Montgomery field operations. The MSM algorithm requires more advanced warp-shuffle-based routines (inverse modulo, vectorized shifts with ballot/shuffle) that were not ported to mont_t.hip. This is likely intentional -- MSM has CUDA-specific optimizations (PTX inline assembly, cooperative groups) that would require significant rewrite for HIP.

### Recommendation

The project is **already-supported for NTT**. For MSM:
- Completing the mont_t.hip port requires ~900 lines of complex code with GCN inline assembly
- The fp2 device code needs full implementation for HIP
- MSM kernels need cooperative_groups -> hip/hip_cooperative_groups.h conversion

This is beyond validation scope. Options:
1. **Document as partial support**: NTT works, MSM requires additional work
2. **Upstream issue/discussion**: Report findings to Supranational
3. **Full port**: Complete mont_t.hip and fp2 HIP paths (substantial effort)

## MSM HIP Port Progress (2026-06-05)

Work in progress to enable MSM on HIP/ROCm.

### Completed Changes

1. **poc/msm-cuda/build.rs**: Added hipcc detection (NVCC=off pattern from rust/build.rs)
2. **ff/bls12-381-fp2.hpp, ff/bls12-377-fp2.hpp, ff/alt_bn128-fp2.hpp**: Changed `#ifdef __CUDA_ARCH__` to `#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)`
3. **ff/mont_t.hip**: Extended significantly (~500 lines added):
   - `operator<<=` and `operator>>=` (bit shifts)
   - `cneg` (conditional negation)
   - `abs` (sign handling)
   - `is_one`, `is_zero` checks
   - `one(int or_zero)` selector
   - `shfl` member function
   - `shfl_xor` template
   - `dot_product` (simple a*b + c*d form)
   - `vt_inverse_mod_x` (modular inverse using Euclidean algorithm)
   - `reciprocal` and division operators
   - Supporting infrastructure: `approx_t`, `factorx_t`, `sfp_t` classes
   - Helper functions: `ab_approximation_n`, `inner_loop_x`, `cneg_v`, `smul_n_shift_x`
4. **msm/pippenger.cuh, msm/batch_addition.cuh**: Added HIP header guards, PTX-to-HIP conversions for laneid and shuffle operations, cooperative_groups polyfill
5. **msm/sort.cuh**: Added HIP versions for `pack` (lop3 alternative) and `sum_up` (shuffle-based sum)
6. **util/cuda2hip.hpp**: Added `HIP_VERSION_MAJOR < 7` guard for `__ballot_sync` polyfill (ROCm 7+ has native)
7. **rust/src/build.rs**: Added `-x hip` flag for proper HIP compilation

### Blocking Issue

ROCm 7.2.x's hipcc performs dual-pass compilation: first a host pass, then a device pass. During the HOST pass, `__HIP_DEVICE_COMPILE__` is NOT defined, but the file is still parsed and template functions are instantiated. This causes `__device__` inline functions from HIP headers (like `__shfl_up`, `__syncthreads`, `atomicAdd`) to fail with "call to __device__ function from __host__ function" errors.

This is NOT a cc-rs issue or a `-x hip` flag issue -- it's fundamental to hipcc's architecture.

Confirmed behavior:
```
/opt/rocm/bin/hipcc -E --offload-arch=gfx90a -x hip test.hip
```
produces BOTH `HIP_DEVICE_COMPILE_IS_DEFINED` and `HIP_DEVICE_COMPILE_IS_NOT_DEFINED` because preprocessor runs twice.

Error example:
```
/var/lib/jenkins/moat/projects/sppark/src/msm/sort.cuh:61:22: error: no matching function for call to '__shfl_up'
    61 |         uint32_t v = __shfl_up(sum, off, WARP_SZ);
       |                      ^~~~~~~~~
note: candidate function not viable: call to __device__ function from __host__ function
```

### Possible Solutions

1. **Make functions `__host__ __device__`**: But HIP intrinsics only work on device, need stubs for host
2. **Use `__HIPCC_RTC__` or other undocumented macros**: May have better host/device discrimination
3. **Restructure code**: Separate host-visible declarations from device-only implementations
4. **Use `.hip` extension with header-only guards**: May change hipcc's parsing behavior
5. **Report upstream to Supranational**: They may have solved this for other projects

### Files Modified (not yet committed)

- ff/mont_t.hip (+502 lines)
- msm/pippenger.cuh, msm/batch_addition.cuh, msm/sort.cuh
- poc/msm-cuda/build.rs, poc/msm-cuda/Cargo.toml, poc/msm-cuda/cuda/pippenger_inf.cu
- ff/bls12-381-fp2.hpp, ff/bls12-377-fp2.hpp, ff/alt_bn128-fp2.hpp
- rust/src/build.rs, util/cuda2hip.hpp
