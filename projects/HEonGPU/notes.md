# HEonGPU notes

## Build

Library and tests build successfully on linux-gfx90a:

```bash
cd projects/HEonGPU/src
mkdir build && cd build
cmake -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_BUILD_TYPE=Release -DHEonGPU_BUILD_TESTS=ON ..
cmake --build . -j$(nproc)
```

Output: `src/libheongpu.a`, test executables in `bin/test/`, plus dependencies (`libntt-1.0.a`, `libfft-1.0.a`, `librngongpu-1.0.a`)

## Submodule Updates

GPU-FFT, GPU-NTT, and RNGonGPU (all by the same author, Alisah Ozcan) were updated to HIP-compatible commits. These submodules contain CUDA-specific code that needed adaptation:

1. **GPU-NTT** (abbb2c3): PTX inline assembly in `modular_arith.cuh` replaced with HIP-compatible `__umul64hi` intrinsic for 128-bit multiplication. Chained comparison syntax fixed for clang.

2. **GPU-FFT** (00d3f8b): CMake HIP support added (USE_HIP option, HIP language, hipcc compilation).

3. **RNGonGPU** (50558fe): CUDA->HIP compatibility header added. `hipPointerAttribute_t.type` member used (differs from CUDA's `.memoryType`). Links hiprand instead of curand.

The submodules link `hip::host` (not `hip::device`) to avoid propagating HIP compile flags to downstream consumers. This allows pure C++ test executables to link against the HIP library without requiring HIP compilation themselves -- though they still need HIP compilation to use rocThrust headers.

## Port Details

### Key adaptations:

1. **cuda_to_hip.h header**: Central compatibility header mapping CUDA runtime symbols to HIP equivalents, including cuRAND->hipRAND mappings and warp size abstraction.

2. **rmm_hip_stub/**: Minimal RMM implementation for HIP since the real RMM does not support HIP. Implements `device_uvector`, `device_buffer`, `pool_memory_resource`, `statistics_resource_adaptor`, `pinned_memory_resource`.

3. **hip_compat/**: Shim headers for thirdparty code that includes `cuda_runtime.h` and `curand_kernel.h` directly.

4. **PTX inline assembly**: Replaced in `bigintegerarith.cuh` and `GPU-NTT/modular_arith.cuh` with portable C++ using `__umul64hi` intrinsic.

5. **Device function linking**: Made `SmallForwardNTT`/`SmallInverseNTT` inline in header to avoid cross-TU device linking issues (HIP doesn't support CUDA's CUDA_SEPARABLE_COMPILATION the same way).

6. **Warp shuffles**: HIP uses `__shfl_down(val, offset)` without mask; CUDA uses `__shfl_down_sync(mask, val, offset)`.

7. **Warp size**: Changed hardcoded `32` to runtime `warpSize` for gfx90a wave64 compatibility.

8. **HostVector**: Added explicit copy/move assignment operators (clang stricter than NVCC about std::vector inheritance).

9. **hipPointerAttribute_t**: HIP uses `.type` member, not `.memoryType`.

10. **CMake HIP flag propagation fix**: Changed `hip::device` to `hip::host` in GPU-NTT, GPU-FFT, and rmm_hip_stub CMakeLists.txt. The `hip::device` target propagates HIP compile flags (`-x hip --offload-arch=gfx90a`) via INTERFACE properties to all downstream targets, causing g++ to fail on pure C++ files. The `hip::host` target provides the HIP runtime library without compile-time flags.

11. **Test compilation**: Test .cpp files are compiled as HIP sources (`set_source_files_properties(... LANGUAGE HIP)`) because they transitively include rocThrust headers via heongpu.hpp. rocThrust requires HIP compilation context.

12. **RMM HIP stub error checking**: Added `hipError_t` return value checking to all allocation functions in the RMM stub. Throws `std::runtime_error` on allocation failure. Deallocation ignores errors (cannot throw in destructors).

## Review 2026-06-05

### Test Results
- Pass: bfv_encoding, ckks_encoding (2/20)
- Fail: All other tests (18/20) -- encryption, decryption, addition, multiplication, relinearization, rotation, TFHE

### Root Cause Analysis

The porter suspected warp size handling. However:

1. **GPU-NTT `<< 5` patterns are NOT warp-size bugs**: The `<< 5` shifts in `thirdparty/GPU-NTT/src/lib/ntt_4step/ntt_4step.cu` are algorithmic indexing into 32-element shared memory rows (`__shared__ T sharedmemorys[32][32+1]`), not warp-lane operations. The NTT algorithm inherently uses 32-element blocks regardless of hardware warp width.

2. **Warp-level fixes look correct**: The `warp_reduce` function in `src/include/heongpu/util/util.cuh:312` correctly uses runtime `warpSize`. The warp index calculations in encryption/decryption/keygeneration kernels (`wid = idx / warpSize`) are also correct.

3. **Encoding passes because it only uses NTT/INTT**: The encoding tests call `GPU_NTT` and `GPU_INTT` which work correctly. All failing tests additionally involve:
   - Random number generation (AES-based, not curand/hiprand)
   - Key generation
   - Modular arithmetic with biginteger operations

### Potential Issues Requiring Investigation

1. **bigintegerarith.cuh carry/borrow logic** (`src/include/heongpu/util/bigintegerarith.cuh:77-169`): The HIP path uses manual overflow detection. Example at line 85:
   ```cpp
   carry = (sum < a || (carry && sum == a)) ? 1 : 0;
   ```
   This logic appears correct on paper but should be verified against the PTX version for edge cases.

2. **AES RNG differences**: HEonGPU uses `rngongpu::RNG<rngongpu::Mode::AES>` for random number generation, not curand/hiprand. The AES implementation should produce identical results on CUDA and HIP, but this needs verification.

3. **Missing hypothesis**: The fact that ALL cryptographic operations fail (encryption, addition, multiplication, relinearization, rotation across BFV, CKKS, and TFHE schemes) while encoding works suggests a common code path issue, likely in either:
   - Random polynomial generation
   - NTT-domain modular multiplication
   - Key generation

### Verdict

**changes-requested**: The port builds and encoding works, but cryptographic operations produce incorrect results. This requires debugging before validation. The warp-size hypothesis is unlikely; the issue is more subtle and requires:
1. Unit testing the biginteger arithmetic against known values
2. Comparing RNG output between CUDA and HIP builds
3. Step-by-step comparison of a simple encrypt/decrypt operation

## Porter Debug Attempt 1 (2026-06-05)

### Components Verified Working

1. **Barrett modular multiplication**: Tested with 1024 random 36-bit inputs, all correct including edge cases (max values, zero, one).

2. **128-bit subtraction**: Tested 8 cases including borrow propagation, all correct.

3. **128-bit multiplication** (`__umul64hi`): Implicitly verified through Barrett test passing.

4. **Ternary RNG**: `modular_ternary_random_number` produces valid ternary values (-1, 0, 1 mod q). Distribution is skewed (50% -1, 25% 0, 25% 1) but this matches the original algorithm logic and doesn't cause complete failure.

5. **hiprand device API**: `hiprand_init`/`hiprand` produce reasonable random values with good distribution.

6. **Encode/decode roundtrip**: Works perfectly, confirming NTT/INTT with the library's own tables is functional.

### Key Finding

Tested `pk[0] + pk[1]*sk` (pointwise in NTT domain): values are NOT small. Expected: small Gaussian error (|e| < 50 for std_dev=3.2). Actual: values like 34 billion (half of 68B modulus).

This means the public key relationship `pk[0] = -a*s + e, pk[1] = a` is NOT being satisfied. The bug is in either:
- Public key generation kernel (`publickey_gen_kernel`)
- NTT transformation of the involved polynomials
- Data layout mismatch (wrong RNS component being used)

Note: Cannot directly verify pk[0]+pk[1]*sk=e in NTT domain because e in NTT domain is NOT small (NTT spreads energy). The relationship only holds algebraically for polynomial multiplication. However, the observed values are so large (50% of modulus) that even accounting for this, something is fundamentally wrong.

### What the Error Pattern Tells Us

Encryption of ALL-ZEROS produces decrypted values of magnitude ~50% of plain_modulus (max error ~515840 out of 1032193). This is NOT noise, it's complete garbage. If only the noise were wrong, we'd see small but incorrect noise. The magnitude indicates the core algebraic relationship is broken.

### Remaining Hypotheses

1. **NTT table generation bug**: Tables are generated on CPU. If the primitive roots or powers are wrong, all NTT-domain operations fail.

2. **Inconsistent RNS indexing**: The various kernels (keygen, encrypt, decrypt) might use different conventions for indexing RNS components.

3. **modulus_->data() ordering**: The array of moduli on GPU might not match what kernels expect.

4. **NTT layout mismatch**: `ntt_rns_configuration.ntt_layout = gpuntt::PerPolynomial` might behave differently than expected.

### Continued Analysis

**Key test**: Verified that `pk[0] + pk[1]*sk` in NTT domain produces values that are ~50% of the modulus (should be small Gaussian error). This means the fundamental relationship is broken.

The public key formula is `pk[0] = -(s*a + e), pk[1] = a` (verified by reading publickey_gen_kernel). The relationship `pk[0] + pk[1]*sk = -e` should hold in NTT domain.

**Verified components**:
1. CPU Barrett multiplication (used in NTT table generation): works
2. GPU Barrett multiplication: works
3. 128-bit subtraction/multiplication: works
4. hiprand device API: works
5. Encode/decode (plaintext NTT): works

**Open question**: The GPU-NTT RNS API call semantics. The keygenerator calls `GPU_NTT_Inplace(errors_a.data(), ..., Q_prime_size, Q_prime_size)` where errors_a has 2*Q_prime_size*n elements but the call pattern suggests only Q_prime_size*n elements are processed (only error_poly, not a_poly). This is the ORIGINAL code pattern -- if this were wrong, upstream CUDA would also fail. Need to understand the batch_size/mod_count semantics to confirm.

**Hypotheses**:
1. NTT table generation produces wrong values on HIP (despite CPU Barrett working)
2. There's a subtle difference in GPU kernel execution (memory layout, register usage) between CUDA and HIP that corrupts NTT results
3. The `__umul64hi` intrinsic or 128-bit subtraction has edge cases we haven't tested

### Next Steps (for attempt 2)

1. Create a direct NTT roundtrip test using the coefficient moduli tables to verify NTT/INTT work correctly

2. Add debug output to publickey_gen_kernel to print intermediate values on GPU

3. Test with a simpler configuration (smaller polynomial, single modulus) to isolate the bug

4. Binary comparison of GPU memory after NTT between CUDA (if accessible) and HIP builds
