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
