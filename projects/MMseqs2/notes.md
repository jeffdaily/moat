# MMseqs2 notes

Lead platform: linux-gfx90a (MI250X). ROCm 7.2.1. GPU surface is the vendored
`lib/libmarv` aligner (CUDASW++/cudasw4 derivative), built only when the GPU
option is on. The CPU search/clustering toolkit has no GPU code and is untouched.

## DPX / SIMD strategy: EMULATE (strategy b), not guard-out

The planner's option (a) -- compile-time guard out the DPX short2/int path -- is
NOT viable here, for two independent reasons:

1. The short2 (gapless) and int (Smith-Waterman) kernels are force-instantiated
   in dedicated .cu files (`pssmkernels_gapless_instantiation_dpx.cu`,
   `pssmkernels_smithwaterman_instantiation_dpx.cu`) and their `call_*<short2>` /
   `call_*<int>` symbols are referenced unconditionally from the dispatch switch
   in `cudasw4.cuh` (`case Datatype::Short2`, the `<int>` SW branch). They must
   compile and link even if never selected at runtime.
2. CRITICAL: ROCm reports gfx90a as compute capability 9.0
   (`hipDeviceAttributeComputeCapabilityMajor`=9, Minor=0). The kernel-config
   selectors key on cc: `ccMajor==9 && ccMinor==0` -> the Hopper "sm90" config
   list, which sets `dpx=1` (short2 gapless) and `dpx=1` (int SW). So WITHOUT
   intervention the DPX path would actually RUN on gfx90a, not just compile.

So the ~17 missing intrinsics are emulated AND the config selector is pinned to a
portable path. Both are needed.

### Emulated intrinsics (lib/libmarv/src/marv_simd_amd.cuh)
All scalar, lane-count tied to the SIMD WORD (two 16-bit / four 8-bit lanes),
never to warpSize. Semantics matched to the CUDA Math API:
- 16x2: `__vadd2` (wrap add), `__vmaxs2` (signed max), `__vibmax_u16x2` (unsigned
  max + per-lane a>=b bool), `__vimax3_s16x2`, `__vimax3_s16x2_relu`,
  `__viaddmax_s16x2` (max(a+b,c)), `__viaddmax_s16x2_relu` (max(a+b,c,0)).
- s32 (DPX): `__vibmax_s32` (max + a>=b bool), `__vimax3_s32`, `__viaddmax_s32`,
  `__viaddmax_s32_relu`.
- u8x4: `__vmaxu4`, `__vminu4`, `__vadd4` (wrap), `__vaddus4` (unsigned sat add),
  `__vsubus4` (unsigned sat sub).
- `__hmax2`: per-lane `__hmax` (ROCm has scalar __hmax, not __hmax2); __hmax
  propagates the non-NaN operand, matching CUDA __hmax2.
The `__vibmax_*` boolean is "first operand wins" = (a>=b), which the SW traceback
end-position logic depends on; the relu forms clamp the result at 0.

### Config selector pinned to portable path
`getOptimalKernelConfigs_gapless` and `getOptimalKernelConfigs_SW` get a
`#if defined(USE_HIP)` branch returning the T4-class (sm75) list: smallest tiles,
`datatype=0` (half2 gapless) / `dpx=0` (float SW). This routes runtime through the
portable half2/float SIMT path (no DPX selected) AND keeps dynamic shared memory
within the gfx90a 64 KiB per-block ceiling (probed:
`hipDeviceAttributeMaxSharedMemoryPerBlock`=65536). The emulated DPX intrinsics
still compile and are correct if a future config selects short2/int.

## Wave-size handling (LOW risk, confirmed)
- All cooperative-groups tiles are `cg::tiled_partition<N>`, N in {4,8,16}
  (static_assert <=32). These are logical sub-warp tiles that live within one
  wave64 wavefront; arch-agnostic. `group.shfl_up/down/xor` use the tile width
  (`numThreads`), not the hardware warp.
- `__shfl_*_sync(0xFFFFFFFF, ...)` in the legacy non-PSSM kernels: HIP requires a
  64-bit lane mask and static_asserts on a 32-bit one. Routed to the maskless
  `__shfl_*` builtins via macros in cuda_to_hip.h (the mask is dropped; width is
  always passed explicitly or forced to 32 in kernelhelpers.cuh). Width-32 logical
  semantics preserved on wave64.
- NO host+device-shared warp-width serialized-format constant. `WARPSIZE(32)` in
  hpc_helpers is defined but UNUSED in libmarv. No warp-width geometry is baked
  into any buffer layout. So the serialized-format fault class does not apply.
- `__reduce_max_sync` is behind `#if __CUDA_ARCH__ >= 800`; on HIP __CUDA_ARCH__
  is undefined (=0) so the shfl fallback is taken -- no missing intrinsic call.
- cg::reduce / `<cooperative_groups/reduce.h>` do not exist on ROCm 7.2.1.
  Replaced every reduce_max with a tile-local butterfly over `group.shfl_xor`
  (`marv_tile_reduce` in mathops.cuh), guarded by USE_HIP; correct on wave32 and
  wave64 since it iterates `group.size()/2 .. 1` within the tile.

## Strategy A glue (compat shim)
- NEW `lib/libmarv/src/cuda_to_hip.h` (force `-include`d on the HIP build): maps
  the cuda runtime API (~70 symbols) to hip*, defines `__grid_constant__` away,
  aliases `cuda::std` -> `std` and provides a `cub::SwitchDevice`, adds generic
  `__shfl*` overloads for the packed types libmarv shuffles that ROCm does not
  cover (short2, int2, float2, int3; __half2 is already provided by ROCm),
  redirects the _sync shuffles to maskless, and includes the SIMD emulations.
- NEW `lib/libmarv/src/hip_compat/{cuda_fp16.h,cooperative_groups.h,
  cooperative_groups/reduce.h}`: redirect the source's `#include <cuda_*>` /
  `<cooperative_groups*>` to the HIP headers; reduce.h is just a stub (the
  reduce calls are replaced).
- The vendored hpc_helpers and a few libmarv files key device code on `__CUDACC__`
  / `__NVCC__`, which hipcc does not define. Do NOT `#define __CUDACC__` globally
  -- it makes rocThrust route to the CUDA backend (`cub/detail/...` not found) and
  duplicates HIP's 64-bit atomics. Instead each guard was made HIP-aware
  (`defined(__CUDACC__) || defined(__HIPCC__)`, `defined(__NVCC__) || defined(__HIPCC__)`),
  with the uint64 atomic / ffs overloads and the `mov %%laneid` PTX kept
  NVCC-only (HIP provides those; lane_id uses `__lane_id()`).
- `__constant__` blosum / query arrays in blosum.cu/.hpp and kernels.cuh were
  NVCC-guarded; made HIP-aware so the device symbols exist.

## Build (gfx90a)
libmarv is built as a SHARED library on HIP so the relocatable-device-code link
(`-fgpu-rdc`, `HIP_RESOLVE_DEVICE_SYMBOLS ON`, `--offload-compress`) happens at
marv.so creation; the mmseqs executable then links only host symbols and never
needs the HIP device linker (mmseqs is a plain C++ link). rocThrust/hip::host are
linked PRIVATE on marv so their HIP-language `--offload-arch` flags do not leak to
the C++ consumers. Only the `.cu`/`.cpp` TUs are compiled HIP; the `.cuh`/`.h`
entries in ALIGN_SOURCES must NOT be compiled (that duplicates the kernels.cuh
`__constant__`).

```
cd projects/MMseqs2/src
cmake -S . -B build-hip -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build-hip -j16 --target mmseqs   # HIP_VISIBLE_DEVICES=2 on this host
```
Follower (gfx1100/gfx1151): same, `-DCMAKE_HIP_ARCHITECTURES=gfx1100` (or a
fat-binary `"gfx90a;gfx1100"`). No source change expected.

Non-GPU build (must not regress): `cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release`
then `--build build-cpu --target mmseqs`. Confirmed builds and runs identically to
the GPU build's `--gpu 0` path (12901 hits on the bundled example).

## Validation (real gfx90a, GCD2, zero external downloads)
Bundled `examples/QUERY.fasta` (300 KB) + `examples/DB.fasta` (11 MB):
```
mmseqs createdb examples/DB.fasta targetDB
mmseqs makepaddedseqdb targetDB targetDB_padded
mmseqs easy-search examples/QUERY.fasta targetDB_padded gpu.m8 tmp_gpu --gpu 1
mmseqs easy-search examples/QUERY.fasta targetDB     cpu.m8 tmp_cpu --gpu 0
```
(`--gpu` honors CUDA_VISIBLE_DEVICES; set both CUDA_VISIBLE_DEVICES=2 and
HIP_VISIBLE_DEVICES=2.)

Result: GPU 14482 hit pairs, CPU 12901. Of the 12438 (query,target) pairs present
in BOTH, ZERO differ in bitscore by >0.5 and ZERO differ in pident by >0.5 -- the
GPU Smith-Waterman scores match the CPU oracle EXACTLY. The 463 CPU-only and 2044
GPU-only pairs are all low-score prefilter-boundary hits (median bitscore 70/60 vs
overall median 134, max 6666) -- the documented prefilter-sensitivity difference
between the CPU and GPU recall paths, not a correctness gap. PASS.

## For the foldseek port (it vendors this same libmarv)
Reuse this whole approach verbatim: the cuda_to_hip.h shim, marv_simd_amd.cuh
emulations, the cg::reduce -> butterfly replacement, the config-selector USE_HIP
pin, the SHARED-lib RDC build, and the hpc_helpers `__CUDACC__/__NVCC__` ->
`||__HIPCC__` edits. The deciding facts (gfx90a reports cc 9.0 so DPX would RUN;
.cuh must not be compiled as TUs; do NOT define __CUDACC__) are libmarv-intrinsic
and will recur in foldseek.
