# libSGM notes

## Port status
- Strategy A (compat header + CMake HIP-language gating). Pure CUDA, no
  CUTLASS/cuBLAS/CUB/Thrust/textures. Correctness-first mechanical port.
- Fork: https://github.com/jeffdaily/libSGM (default branch `master`).
- Branch: `moat-port` off upstream `master` (e4c669b). Fork HEAD 9ce43fd.
- Actions disabled on the fork.
- linux-gfx90a: ported and validated (67/67 bit-exact, deterministic).

## Build (gfx90a, ROCm 7.2.1)
```
cd projects/libSGM/src
git submodule update --init            # googletest, needed for ENABLE_TESTS
cmake -S . -B build-hip -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DENABLE_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-hip -j
HIP_VISIBLE_DEVICES=3 ./build-hip/test/sgm-test
```
Followers: same command with `-DCMAKE_HIP_ARCHITECTURES=gfx1100` / `gfx1151`,
no source change (arch propagates to the host warp-size derivation too).

## Key changes
- `src/cuda_to_hip.h` (new): on USE_HIP includes hip_runtime + aliases the
  cuda* runtime surface used by host code; else passthrough to cuda_runtime.h.
- `src/constants.h`: WARP_SIZE became the TRUE per-arch wavefront width. Device
  pass keys on `__GFX9__` (64 on CDNA gfx9, 32 on RDNA); host pass uses
  `SGM_HOST_WARP_SIZE` injected by CMake from the target arch. CUDA stays 32.
- `src/median_filter.cu`: USE_HIP-guarded software emulation of __vcmpgtu2/4,
  __vminu2/4, __vmaxu2/4 (absent in ROCm 7.x), bit-identical to the intrinsics.
- `src/winner_takes_all.cu`: WTA inter-lane smem handoff uses __syncwarp() on
  HIP (CUDA_VERSION is undefined under HIP, which otherwise routed to the
  weaker __threadfence_block fallback).
- CMake: USE_HIP option; enable_language(HIP); .cu LANGUAGE HIP; sgm built PIC
  (static lib linked into the PIE test exe); test target gated to CXX-only.

## Gotchas / lessons (worth promoting to PORTING_GUIDE)
- WARP_SIZE host/device split trap (same class as raft select_warpsort): the
  .cu compute their launch geometry (BLOCK_SIZE = WARP_SIZE * N, grid dims) in
  hipcc's HOST pass, where `__GFX9__` and every gfxNNN macro is UNDEFINED. A
  device-only `#if defined(__GFX9__)` constant then reads 64 on device but 32
  on host -> block dim disagrees with the 64-lane kernel. Fix: derive a host
  warp size from CMAKE_HIP_ARCHITECTURES (gfx9* -> 64, else 32) and inject it
  as a compile def; the constant uses `__HIP_DEVICE_COMPILE__` for the device
  branch and the injected def for the host branch. host==device per build.
- This SGM design overloads one WARP_SIZE constant for three roles (subgroup
  partitioning, WTA per-lane layout REDUCTION_PER_THREAD = MAX_DISPARITY /
  WARP_SIZE, block sizing). Making it the true wavefront width keeps all three
  internally consistent on wave64 -- no per-arch hack needed. The DP shuffle
  uses width=WARP_SIZE shuffle-up-by-1 with subgroup boundaries masked by a
  subgroup-relative lane_id check, so it stays correct at width 64; the
  subgroup_min/and reductions use width=SUBGROUP_SIZE (<=32, divides 64) and
  partition cleanly. The legacy bare `__shfl*` branch (CUDA_VERSION undefined
  on HIP) is mask-free, so the generate_mask 32-bit limitation is moot.
- `<hip/hip_runtime.h>` alone does NOT alias the `cuda*` runtime names; alias
  them explicitly in the compat header (cudaMalloc, cudaStream_t, cudaSuccess,
  cudaGetLastError, etc.).
- gfx90a static HIP lib + PIE test exe: relocation R_X86_64_32 link error;
  set POSITION_INDEPENDENT_CODE ON on the library target.

## Install as a dependency
libSGM is a leaf (a stereo-matching application library; nothing in MOAT
consumes it). No dependency-install section needed.
