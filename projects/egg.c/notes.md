# egg.c notes

## Build (lead trainer, linux-gfx90a)
Raw-nvcc project, no CMake. Port = compile the existing `.cu` with hipcc instead
of nvcc. hipcc with `-x hip` defines `__HIP__` (NOT `__HIP_PLATFORM_AMD__`); all
HIP-specific code is guarded on `__HIP__` so the NVIDIA nvcc build is unchanged.

Lead (simplest trainer, no cuBLAS):
```
cd projects/egg.c/src
export HIP_VISIBLE_DEVICES=3
/opt/rocm/bin/hipcc -O3 --offload-arch=gfx90a -x hip full_cuda_train_egg.cu -o egg_hip
```
Multi-arch warp-size gate (one fat binary; confirm BOTH code objects):
```
/opt/rocm/bin/hipcc -O3 --offload-arch=gfx90a --offload-arch=gfx1100 -x hip full_cuda_train_egg.cu -o egg_hip_multi
/opt/rocm/lib/llvm/bin/llvm-objdump --offloading egg_hip_multi | grep -io 'gfx[0-9a-f]*' | sort -u   # -> gfx1100 gfx90a
```

## Compat headers (the whole port surface for the lead)
- `egg_hip_compat.cuh`: under `__HIP__` pulls in `hip/hip_runtime.h` + `hipcub/hipcub.hpp`,
  aliases `namespace cub = hipcub`, and `#define`s the small fixed set of CUDA
  runtime symbols this file uses (cudaMalloc/Free/Memset/Memcpy[To|From]Symbol,
  cudaDeviceSynchronize, cudaGetDeviceProperties, cudaDeviceProp, cudaError_t,
  cudaSuccess, the memcpy-kind enums) to their hip* equivalents. rocThrust already
  exposes the `thrust::` namespace at the same header paths, so thrust needs no
  remap. This ROCm (7.2) has NO `cub/` compat dir and NO automatic cuda->hip symbol
  shim, so the source's `#include <cuda_runtime.h>` / `#include <cub/cub.cuh>` are
  guarded `#if !defined(__HIP__)` and replaced by the compat header.
- `egg_warp_compat.cuh`: the warp-width fix (see below).

## PRIMARY fix: EGG_WARP_SIZE = 32 is a DATA-LAYOUT stride, not launch geometry
egg.c maps ONE perturbation to ONE LOGICAL warp and partitions HIDDEN_DIM across
exactly 32 lanes: loops stride `i += WARP_SIZE`, per-lane arrays are sized
`MAX_STRIDE = 8 = HIDDEN_DIM_max(256)/32` and indexed `[i/WARP_SIZE]`, and the host
launch geometry is `warps_per_block = BLOCK_THREADS/WARP_SIZE`,
`blocks = POPULATION_SIZE/warps_per_block`. WARP_SIZE here is a fixed 32-lane tiling
constant, NOT the hardware wavefront width.

Therefore EGG_WARP_SIZE stays 32 on EVERY arch (wave32 gfx1100 AND wave64 gfx90a).
Do NOT switch it to the runtime warpSize query (64 on CDNA) -- that would corrupt
the data tiling and the launch geometry. This is the "logical-warp" exception to
the physical-warp host-query rule. On wave64 two logical warps share one physical
wavefront, so every shuffle/reduce runs at explicit width 32 to keep them independent:
- `cub::WarpReduce<long long>` -> `cub::WarpReduce<long long, 32>` (`EggWarpReduce<T>`).
  hipCUB defaults LOGICAL_WARP_THREADS to the PHYSICAL warp (64 on gfx90a); without
  the explicit 32 it would sum two perturbations together -> wrong fitness/loss.
- `warpBroadcast` and its `__shfl_sync` now route through width-32 wrappers
  (`__shfl_sync(mask, v, lane, 32)`), with a 64-bit all-ones mask under `__HIP__`
  (`~0ull`) because the physical wavefront is 64 wide; the explicit width=32 confines
  the op to the logical warp.

Arch-unified, no per-arch hack: identical source is correct on wave32 and wave64,
proven by the multi-arch fat binary + the bit-identical pinned-seed run below.

## Determinism harness (added, portable, off by default)
The per-step kernel seed was `time(NULL) ^ (step*0x9e3779b9)`, so runs differed by
wall clock. Added an `EGG_FIXED_SEED` env override: when set, the seed is a pure
function of `(fixed_seed, step)`. Default behavior (unset) is unchanged. This lets
the determinism gate prove the 32-lane masks/reduces are correct: a wrong width-32
partition shows up as a wrong AND non-deterministic loss.
`srand(time(NULL))` at main() seeds rand() but rand() is never used in the loss path.

## Validation (real gfx90a, MI250X, GCD 3) -- PASS
Corpus: 18000-byte repeating-text `input.txt` (70 steps available); each step ~11s
(POPULATION_SIZE=65536), so a 300s run covers ~16 steps -- enough signal.
1. Multi-arch build: fat binary has gfx90a AND gfx1100 code objects.
2. gfx90a run (EGG_FIXED_SEED=12345): native MI250X dispatch; Loss decreases
   monotonically 8.3489 -> 3.3417 over 16 steps; generated sample becomes text-like
   (recognizable letters/spaces, not `.`-only garbage).
3. Determinism: two pinned-seed runs are BIT-IDENTICAL on Loss + Up+/Up- + per-pair
   fitness (only Fwd/Host wall-clock timings differ). Decisive warp-width fingerprint.
4. Non-GPU regression: `d-eggs/test_ternary.cpp` builds (`g++ -O3 -Id-eggs/include`)
   and PASSES (pack/unpack roundtrip verified, 1.60 bits/value). The pure-C
   `full_trained_egg.c` is ARM-NEON-only (`#include <arm_neon.h>`) and was never
   x86-buildable upstream; it is byte-identical (untouched) -- not a regression.

## Out of scope for the lead (deferred)
- cuBLAS transformer variants (`full_cuda_train_egg_transformer*.cu`,
  `full_cuda_train_transformer_adam_mgpu.cu`, `muon_internal.cuh`): hipBLAS swap +
  Newton-Schulz numerics. Not needed for the warp-width correctness gate.
- `d-eggs/` distributed coordinator/worker (hipGraph capture, cudaMallocManaged
  atomicMin/Max audit): multi-node path deferred per task scoping.
These will need the same EGG_WARP_SIZE=32 treatment on their `__shfl_down_sync(...,16)`
reductions and RoPE `__shfl_xor_sync(...,1)` (explicit width 32) when ported.

## Gotchas
- hipcc `-x hip` defines `__HIP__`, NOT `__HIP_PLATFORM_AMD__`. Key all guards on `__HIP__`.
- ROCm 7.2 ships no `cub/` compat dir; use `hipcub/hipcub.hpp` + `namespace cub = hipcub`.
- hipCUB `WarpReduce<T>` default logical width = PHYSICAL warp (64 on gfx90a). Any
  logical-32 reduction MUST template `<T, 32>` explicitly.
- `timeit.sh` runs the wrapped command from the repo root, so pass an ABSOLUTE
  source path to hipcc.

## Review 2026-06-02 (reviewer, linux-gfx90a, fork 0472ed5)
Verdict: review-passed. Independently re-ran the port on real gfx90a (MI250X, GCD 3).

Verified clean:
- Logical-warp-32 fix (load-bearing): EggWarpReduce pins cub::WarpReduce<long long,32>; warpBroadcast routes through width-32 __shfl_sync with a 64-bit all-ones mask under __HIP__; eggWarpBroadcast body is byte-identical to the upstream broadcast (same lo/hi split + (unsigned int)lo low-half mask). Two EGG_FIXED_SEED=12345 runs were BIT-IDENTICAL on Loss + Up+/Up- across all 24 steps -- the decisive wave64 fingerprint (a wrong width-32 partition would diverge). Loss fell monotonically 8.2460 -> 3.2439; sample text-like; native MI250X dispatch.
- Multi-arch fat binary carries both gfx90a and gfx1100 code objects (llvm-objdump --offloading).
- CUDA-path byte-identity: both compat headers guard their bodies on __HIP__; under nvcc egg_hip_compat.cuh expands to nothing and the explicit width-32 shuffle args are no-ops (NVIDIA default warp width is 32). NVIDIA build behavior preserved.
- Non-GPU regression: d-eggs/test_ternary.cpp builds + passes (1.6000 bits/value). All deferred .cu/.cuh and the ARM-NEON full_trained_egg.c are untouched; deferred paths documented in "Out of scope for the lead".
- Commit hygiene: [ROCm] title 59 chars, mentions Claude, no noreply/ghstack/co-author trailer, ASCII clean. fork/moat-port == HEAD; fork/master clean upstream mirror; Actions disabled.

Minor (non-blocking, no fix required for this commit):
- egg_warp_compat.cuh:36 eggShflDownSync and :40 eggShflXorSync are defined but unused in this commit (the lead trainer uses only the WarpReduce + broadcast paths; down/xor shuffles belong to the deferred transformer/RoPE files). They are __device__ __forceinline__ so they emit no unused-function warning under hipcc or nvcc and do not affect the NVIDIA build. They are a deliberate forward-looking part of the warp-compat shim API; acceptable to keep, but if the transformer ports stall they should be removed per the orphan-cleanup rule.
- The 11 -Wunused-value warnings on cudaFree/cudaDeviceSynchronize returns are pre-existing in the upstream source (hipError_t is nodiscard; nvcc is not), not introduced by the port.

## Validation 2026-06-02 (validator, linux-gfx90a, fork 0472ed5)
Verdict: PASS. Real GPU: AMD Instinct MI250X / MI250, gfx90a (GCD 0), ROCm 7.2.

Commands run:

```
# 1. Multi-arch build
utils/timeit.sh egg.c compile -- \
  /opt/rocm/bin/hipcc -O3 --offload-arch=gfx90a --offload-arch=gfx1100 \
  -x hip projects/egg.c/src/full_cuda_train_egg.cu \
  -o agent_space/egg_hip_multi_val

/opt/rocm/lib/llvm/bin/llvm-objdump --offloading agent_space/egg_hip_multi_val \
  | grep -io 'gfx[0-9a-f]*' | sort -u
# -> gfx1100 gfx90a  (BOTH code objects present)

# 2. Two determinism runs (pinned seed, AMD_LOG_LEVEL=3, HIP_VISIBLE_DEVICES=0)
cd agent_space && EGG_FIXED_SEED=12345 AMD_LOG_LEVEL=3 \
  timeout 200 ./egg_hip_multi_val > egg_val_run1.log 2>&1
cd agent_space && EGG_FIXED_SEED=12345 AMD_LOG_LEVEL=3 \
  timeout 200 ./egg_hip_multi_val > egg_val_run2.log 2>&1

# 3. Non-GPU regression
utils/timeit.sh egg.c test -- \
  g++ -O3 -Id-eggs/include d-eggs/test_ternary.cpp -o agent_space/test_ternary_val
./agent_space/test_ternary_val
```

Results:
- Fat binary: both gfx90a and gfx1100 code objects confirmed by llvm-objdump.
- Native gfx90a dispatch: AMD_LOG_LEVEL=3 log shows "Using native code object for device: amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-".
- Loss decreases monotonically across 16 steps: 8.3489 -> 7.6246 -> 6.9552 -> 6.3868 -> 5.8830 -> 5.4134 -> 5.0936 -> 4.7565 -> 4.4101 -> 4.1356 -> 3.8607 -> 3.5874 -> 3.5018 -> 3.4543 -> 3.3883 -> 3.3417.
- Sample text-like by step 6+: "ick brown fox jumps over the l" prompt reproduced correctly; completion increasingly word-like.
- Determinism: two EGG_FIXED_SEED=12345 runs are BIT-IDENTICAL on Loss, Up+, Up- across all 16 steps (diff returned empty). Only Fwd/Host wall-clock timings differ. Decisive warp-width fingerprint: a wrong 64-lane partition would produce different Up+/Up- counts between runs.
- Non-GPU regression: d-eggs/test_ternary.cpp builds + passes (Test 1 Passed, Test 2 Passed, Verification Passed, 1.6000 bits/value).
- GPU count: 1 pass, 0 fail. Non-GPU count: 1 pass, 0 fail.
