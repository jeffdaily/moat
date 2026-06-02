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
