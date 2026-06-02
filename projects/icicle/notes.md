# icicle notes

## Fork / port
- Fork: https://github.com/jeffdaily/icicle (Actions disabled), branch `moat-port`.
- Base sha: 625532a624e5aaa6e9d31a1c92587f1fcc30dc76 (upstream main, 2025-08-06).
- Port head: d66e92a7733d958f2c5ec118cfa98e0e6de4c683.
- Strategy A, additive sibling backend: new `icicle/backend/hip_pqc/` registering device "HIP-PQC", mirroring `backend/cuda_pqc/`. NVIDIA "CUDA-PQC" path untouched.

## Out of scope (do NOT attempt from this repo)
- ICICLE's main compute backends -- MSM, NTT, ECNTT, EC, big-integer field arithmetic (the PTX carry-chain field math) -- are CLOSED-SOURCE. The top CMake pulls CUDA_BACKEND / METAL_BACKEND / VULKAN_BACKEND from PRIVATE repos at configure time; none of that source is in the tree. Only `backend/cuda_pqc/` (post-quantum ML-KEM) is open and portable. cuda_pqc's field type Zq (q=3329) is pure portable C++ (no inline asm), so no PTX reimplementation arises.

## Build (gfx90a, ROCm 7.2, GCD 0)
```
export HIP_VISIBLE_DEVICES=0
cmake -S icicle -B build -DCMAKE_BUILD_TYPE=Release \
  -DCPU_BACKEND=ON -DHIP_PQC_BACKEND=ON -DCUDA_PQC_BACKEND=OFF \
  -DPQC=ON -DBUILD_TESTS=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build -j
```
Followers: re-run with `-DCMAKE_HIP_ARCHITECTURES=gfx1100` / `gfx1151`, no source change (kernels are wave-agnostic). Build scratch was kept under agent_space/, not in the repo.

## Validation (real gfx90a, GCD 0) -- all PASS
- Backend gtests: `ctest --test-dir build/backend/hip_pqc/tests` -> 58/58 pass, including the bit-exact NIST KAT vectors (test_ml_kem_keygen compares ek/dk against ml_kem_{512,768,1024} reference files; encaps/decaps/pke also KAT-checked). Crypto is exact, not tolerance-based.
- Frontend dispatch: `ctest --test-dir build/tests -R PqcTest` -> 6/6 pass on "HIP-PQC" (shared-secret consistency + on-device path, batch 4096) through the full device abstraction.
- Dispatch confirmed: AMD_LOG_LEVEL=3 shows ml_kem_{keygen,encaps,decaps}_kernel launching on gfx90a via HIP-PQC.

## The wave64 work (the make-or-break)
The ML-KEM kernels run 128-thread blocks organized as four logical 32-thread warps via threadIdx.x/32 and threadIdx.x%32. On gfx90a (wave64) one 64-lane wavefront carries two logical warps. Fixes are localized to the hip_pqc copies; the cuda_pqc tree is untouched.
- `cuda_to_hip.h` provides wave-agnostic 32-lane-subgroup primitives and maps the CUDA warp-sync spellings onto them:
  - `__ballot_sync` -> per-32-lane ballot: take the full-wave `__ballot`, shift by `32 * ((tid & (warpSize-1)) >> 5)`, keep the low 32 bits.
  - `__shfl_sync` -> `__shfl(val, src, 32)` (width-32 confines to the caller's own subgroup; src is a within-group lane 0..31).
  - `__any_sync` -> nonzero per-32-lane ballot.
  - `__reduce_add_sync` -> width-32 `__shfl_xor` butterfly (avoids HIP's `__reduce_add_sync`, which reduces over the FULL 64-lane wavefront and would break MASK28 semantics, not just width).
  - `__syncwarp` -> left as HIP's native (wavefront barrier with memory ordering; a safe superset of a 32-lane fence). Do NOT replace it with a bare `__builtin_amdgcn_fence` -- a release-only fence is too weak for the cross-lane shared reads in keccakf / generate_k_r and produces wrong KATs for k=3/4.
- SHAKE coefficient counter (`load_coeffs_shake`): lanes 28..31 must now CONTRIBUTE A ZERO COUNT and participate in the width-32 reduction instead of `return`-ing early; the early stores are guarded with `if(!active) return;` AFTER the reduce.

## Gotchas (record for followers / reviewers)
- Two latent races that 32-lane lockstep hid but wave64 timing exposed. Both fixes are wave-agnostic (also correct on NVIDIA):
  1. encaps/decaps: warp 0 produces `k_r`, but `r = k_r + 32` is read by ALL warps inside `pke::encrypt`. Added a `__syncthreads()` between the producer and `pke::encrypt`. Symptom without it: K mismatch for k=2 (512).
  2. `generate_k_r`: `H(ek)` writes shared `hashed_ek` with lanes 0-3, `G_m_ek` reads it with lanes 4-7. Added `__syncwarp()` between them. Symptom without it: K mismatch for k=3/4 (768/1024) only -- 512's shorter H happened to win the race.
- `Zq` needs a trivial default constructor (`Zq() = default;`) under clang HIP: a value-initializing ctor makes `__shared__ Zq arr[...]` an error ("initialization is not supported for __shared__ variables"). All shared Zq buffers are written before read, so uninitialized is safe.
- `cuda::barrier<thread_scope_block>` in pke decrypt (init to 128, only warp 0 waits) -> replaced with `__syncthreads()` (equivalent: all 128 arrive, warp 0 then encodes).
- Missing HIP intrinsics: `__ldcs`/`__stcs`/`__ldlu` -> shimmed with `__builtin_nontemporal_load/store` (streaming/last-use hints; plain access is correct). `__restrict__` placed before the type on array params (`const __restrict__ uint8_t x[...]`) is rejected by clang HIP; rewrite as `const uint8_t* __restrict__ x`.
- Registration: the upstream `REGISTER_ML_KEM{512,768,1024}_BACKEND` macros each emit an anonymous-namespace `[]()->bool` static-init lambda; expanding all three in one HIP TU gives them colliding closure manglings under clang, so the static initializer invokes one variant's registrar twice and throws a duplicate-registration error at load. Replaced the three macros with one named `register_hip_pqc_ml_kem_backends()` calling the `register_*` entry points directly.
- Linking: the shared HIP-PQC backend is consumed via runtime dlopen (icicle_load_backend_from_env_or_default, RTLD_LOCAL), NOT linked into icicle_pqc -- linking it as a NEEDED dep would load it a second time and double-register. For ICICLE_STATIC_LINK the interface lib whole-archives it. Tests likewise rely on the runtime load (BACKEND_BUILD_DIR) and do not link the backend directly.
- `.gitignore` ignores `icicle/backend/*` except cpu/cuda_pqc; added `!icicle/backend/hip_pqc` so the new backend is tracked.
- `test_pqc_api.cpp` was hard-coded to "CUDA-PQC"; changed to `IcicleTestBase::main_device()` so it runs on the registered GPU backend on both NVIDIA and ROCm.
- error_translation.h: HIP lacks distinct `hipErrorInvalidHostPointer` / a `SyncDepthExceeded` analog; the hip copy maps hipError_t -> eIcicleError directly (do not `#define` cuda error enums to colliding hip values -> duplicate case labels).

## Validation 2026-06-02 (validator, linux-gfx90a, fork moat-port d66e92a7733d958f2c5ec118cfa98e0e6de4c683)

GPU: AMD Instinct MI250X / MI250 (gfx90a, amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-), GCD 0 (HIP_VISIBLE_DEVICES=0). ROCm 7.2.

### Build

Configure (fresh, includes Taskflow + GTest fetch via FetchContent):
```
export HIP_VISIBLE_DEVICES=0
cmake -S projects/icicle/src/icicle -B agent_space/icicle_val_build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCPU_BACKEND=ON -DHIP_PQC_BACKEND=ON -DCUDA_PQC_BACKEND=OFF \
  -DPQC=ON -DBUILD_TESTS=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_PREFIX_PATH=/opt/rocm
# Configuring done (369.1s -- includes network fetch)
```

Compile (via timeit.sh):
```
cmake --build agent_space/icicle_val_build -j128
# [100%] Built target test_ml_kem  -- exit 0, ~25 s
```

### Backend gtests (58 KAT-exact tests, run twice)

Run 1:
```
ctest --test-dir agent_space/icicle_val_build/backend/hip_pqc/tests -V --output-on-failure
# 100% tests passed, 0 tests failed out of 58 -- Total Test time: 20.04 s
```

Run 2:
```
ctest --test-dir agent_space/icicle_val_build/backend/hip_pqc/tests -V --output-on-failure
# 100% tests passed, 0 tests failed out of 58 -- Total Test time: 20.27 s
```

Both runs: 58/58 PASS. Includes bit-exact NIST KAT vectors for ML-KEM-512/768/1024 keygen, encaps, decaps, and PKE encrypt/decrypt. Main-device=HIP-PQC, Reference-device=CPU printed in each test.

### Frontend dispatch tests (6 tests, run twice)

Run 1:
```
ctest --test-dir agent_space/icicle_val_build/tests -R PqcTest -V --output-on-failure
# 100% tests passed, 0 tests failed out of 6 -- Total Test time: 1.40 s
```

Run 2:
```
ctest --test-dir agent_space/icicle_val_build/tests -R PqcTest -V --output-on-failure
# 100% tests passed, 0 tests failed out of 6 -- Total Test time: 1.45 s
```

Both runs: 6/6 PASS (MLkemSharedSecretConsistencyTest + MLkemSharedSecretConsistencyTestOnDevice for Kyber512/768/1024, batch 4096). Main-device=HIP-PQC confirmed.

### Kernel dispatch on gfx90a (AMD_LOG_LEVEL=3)

```
export HIP_VISIBLE_DEVICES=0
AMD_LOG_LEVEL=3 ./test_ml_kem_keygen --gtest_filter="MLKemKeygenTest.ML_KEM_Internal_Keygen512"
```
Output:
```
Using native code object for device: amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-
ShaderName : void icicle::pqc::ml_kem::ml_kem_keygen_kernel<...>  hipLaunchKernel: Returned hipSuccess
```

```
AMD_LOG_LEVEL=3 ./test_ml_kem_encaps --gtest_filter="MLKemEncapsTest.MlKemEncaps512Batch"
```
Output:
```
Using native code object for device: amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-
ShaderName : void icicle::pqc::ml_kem::ml_kem_encaps_kernel<...>
```

```
AMD_LOG_LEVEL=3 ./test_ml_kem_decaps --gtest_filter="MLKemDecapsTest.MlKemDecaps512Batch"
```
Output:
```
Using native code object for device: amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-
ShaderName : void icicle::pqc::ml_kem::ml_kem_decaps_kernel<...>
```

All three ML-KEM kernels (keygen, encaps, decaps) confirmed launching native code on gfx90a. Deterministic across both runs.

### Verdict

PASS. 58/58 backend KAT gtests + 6/6 frontend dispatch tests, both deterministic. ML-KEM kernels run native gfx90a code. No non-GPU regressions (CPU backend gtests are not built in this configuration; the CPU backend itself is unmodified and the closed-source MSM/NTT/EC backends are out of scope and not built).

## Validation 2026-06-02 (gfx1100) -- FAIL: 12/58 backend tests, wave32 race exposed

GPU: AMD Radeon Pro W7800 48GB (gfx1100, wave32, RDNA3). ROCm 7.2.1. HIP_VISIBLE_DEVICES=0.

### Build (gfx1100)

Clone fork moat-port @ d66e92a7733d:
```
git clone --branch moat-port --single-branch https://github.com/jeffdaily/icicle /var/lib/jenkins/moat/agent_space/icicle_src
```

Configure:
```
cmake -S agent_space/icicle_src/icicle -B agent_space/icicle_gfx1100_build \
  -DCMAKE_BUILD_TYPE=Release -DCPU_BACKEND=ON -DHIP_PQC_BACKEND=ON -DCUDA_PQC_BACKEND=OFF \
  -DPQC=ON -DBUILD_TESTS=ON -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_PREFIX_PATH=/opt/rocm
# Configuring done (10.6s -- FetchContent deps already cached)
```

Compile (timeit.sh):
```
cmake --build agent_space/icicle_gfx1100_build -j128
# [100%] Built target test_ml_kem -- exit 0, 17.7s
```

Architecture verified:
```
roc-obj-ls agent_space/icicle_gfx1100_build/backend/hip_pqc/libicicle_backend_hip_pqc.so
# hipv4-amdgcn-amd-amdhsa--gfx1100  (473032 bytes)
```
No gfx90a code present. Correct.

### Backend gtests (58 tests) -- FAIL 12/58

```
ctest --test-dir agent_space/icicle_gfx1100_build/backend/hip_pqc/tests
# 79% tests passed, 12 tests failed out of 58 -- Total Test time: 18.9s
```

Failing tests (consistent across runs):
- 16 - KyberTest.PkeKeygen768
- 24 - KyberTest.ML_KEM_Internal_Keygen768
- 31 - MLKemTest.KeyCheckTest768Batch
- 32 - MLKemTest.KeyCheckTest1024Batch
- 38 - MLKemKeygenTest.ML_KEM_Internal_Keygen768
- 41 - MLKemKeygenTest.ML_KEM_Internal_Keygen768_Batch
- 45 - MLKemEncapsTest.MlKemEncaps768Batch
- 46 - MLKemEncapsTest.MlKemEncaps1024Batch
- 48 - MLKemDecapsTest.MlKemDecaps768Batch
- 49 - MLKemDecapsTest.MlKemDecaps1024Batch
- 54 - PkeEncryptTest.Encrypt768Batch
- 55 - PkeEncryptTest.Encrypt1024Batch

All k=512 tests pass. All 1024 single-instance KAT tests pass. The failures are concentrated in k=768 and batch k=1024.

### Wave32 diagnosis

The failures are NON-DETERMINISTIC: `PkeKeygen768` (KAT test with fixed input d="3c53596d...") produces a DIFFERENT wrong dk_pke on every run, even though the input is fixed. This is the definitive signature of a shared-memory race condition. On gfx90a (wave64), wavefront 0 carries threads 0-63 (logical warps 0+1) and wavefront 1 carries threads 64-127 (logical warps 2+3). Intra-wavefront lockstep hides cross-warp shared memory races within a wavefront. On gfx1100 (wave32), each of the 4 logical warps (warp 0..3, threads 0..31, 32..63, 64..95, 96..127) is its own independent wavefront; there is NO automatic lockstep synchronization across warps.

Pattern analysis:
- k=512: `generate_error_vector<2, 4, eta=3, start=1, end=1>` -- ONE warp (warp 1). Passes.
- k=768: `generate_error_vector<3, 6, eta=2, start=0, end=1>` -- TWO warps (0 and 1). FAILS non-deterministically.
- k=1024 single KAT: `generate_error_vector<4, 8, eta=2, start=0, end=0>` -- ONE warp (warp 0). Passes.
- k=1024 batch: fails. Involves encaps/decaps which run `pke::encrypt` with `generate_matrix_A<4,1,3>` (warps 1-3) and `generate_error_vector<4,4,eta=2,0,0>` (warp 0) concurrently without a barrier -- and with 8192 blocks running simultaneously, the timing-dependent race becomes reliable.

The race is a genuine wave32 issue not fixed by the two existing race fixes (encaps `__syncthreads` and `generate_k_r __syncwarp`). Those fixes address cross-warp DATA races in specific functions. The new failure exposes a DIFFERENT class of race: multiple independent wavefronts accessing shared memory within the same block without sufficient ordering.

The porter needs to audit `pke::keygen` (k=768 two-warp error vector), `pke::encrypt` (concurrent matrix_A + error vector generation), and the encaps/decaps kernels for missing `__syncthreads()` between concurrent warp assignments that access or are followed by reads of the same shared memory regions.

### Verdict

FAIL. 46/58 backend gtests pass; 12/58 fail with non-deterministic results indicating a shared-memory race exposed uniquely on wave32 (gfx1100). No HSA faults (0x1016), no hang. The gfx90a wave64 lockstep hid this race. State set to validation-failed; escalated to porter.

## Review 2026-06-02 (reviewer, linux-gfx90a, fork moat-port d66e92a)

Verdict: review-passed. Additive sibling HIP-PQC backend; reviewed `git diff 625532a...HEAD` in full plus the cuda_to_hip shim, all wave64-touched kernels, the two race fixes (against the untouched cuda_pqc originals), registration, CMake wiring, and commit hygiene. No changes requested. The GPU gate (58/58 KAT-exact backend gtests + 6/6 frontend dispatch) is recorded above as porter-run on real gfx90a; the validator re-runs it next.

No defects found. Fault-class analysis verified correct:
- Wave64 subgroup mapping (cuda_to_hip.h): per-32-lane ballot extracted from the 64-bit `__ballot` (shift by `32 * ((tid & (warpSize-1)) >> 5)`), width-32 `__shfl`/`__shfl_xor`, and a width-32 butterfly that replaces HIP `__reduce_add_sync` (which would fold lanes 28..63 into MASK28). Confinement is by hardware lane group, not mask width. Sound; the two logical warps in a wavefront index distinct shared slots (`threadIdx.x/32`) so they stay independent.
- SHAKE coeff counter (cuda_sample_utils.cuh load_coeffs_shake): lanes 28-31 now compute `active=false`, contribute `valid_count=0` to the butterfly, participate in the reduction, then `if(!active) return;` after. Equivalent to the masked 28-lane CUDA reduce. Correct.
- Both races are genuine and pre-existing latently in cuda_pqc (confirmed byte-diff: cuda_pqc encaps has no `__syncthreads` between warp-0 `generate_k_r` and all-warp `pke::encrypt` read of `r=k_r+32`; cuda_pqc `generate_k_r` has no `__syncwarp` between `H` write and `G` read of shared `hashed_ek`). Fixes are wave-agnostic (`__syncthreads`, `__syncwarp`) and correct on NVIDIA too. Port (correctly per its additive strategy) leaves cuda_pqc untouched; the upstream PR ships only HIP-PQC.
- keccakf (cuda_sha3_32threads.cuh) is byte-identical to cuda_pqc; relies on intra-32-lane lockstep, valid within a wavefront on both wave widths; `sha3_state_raw` sized for 4 logical warps (max), fits wave64 and wave32.
- Registration: named `register_hip_pqc_ml_kem_backends()` avoids the three-anonymous-lambda mangling collision; device API uses single `REGISTER_DEVICE_API("HIP-PQC")` in a separate TU. Shared backend consumed via runtime dlopen (RTLD_LOCAL), not NEEDED-linked, so no double-registration; static build whole-archives. Sound.
- Shims `__ldcs`/`__stcs`/`__ldlu` -> `__builtin_nontemporal_load/store`; `cuda::barrier`->`__syncthreads`; `Zq() = default` for `__shared__` arrays. `__funnelshift_l`/`__signbit`/`ROTL1` are HIP builtins, KAT-validated.

Hygiene: cuda_pqc and cpu trees untouched (verified empty diff). `[ROCm]` title 46 chars, mentions Claude, Test Plan present, no noreply trailer, no ghstack, no em-dash, no AMD-internal account refs. Fork main is a clean mirror at base 625532a; moat-port single curated commit at d66e92a. Actions disabled on jeffdaily/icicle (`enabled:false`). Closed MSM/NTT/ECNTT/EC out-of-scope documented in commit, README, plan, notes.

Observational only (no action required):
- icicle/cmake/backend_include.cmake:37 sets `BACKEND_BUILD_DIR` for HIP-PQC for the frontend test loader, matching how the existing CUDA_BACKEND/METAL/VULKAN blocks set it (last-wins). A combined CUDA_PQC_BACKEND+HIP_PQC_BACKEND build would have the HIP value win, but that combo is not a MOAT target and the last-wins pattern is pre-existing upstream; the validated HIP-only build is correct.
