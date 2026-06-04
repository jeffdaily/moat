# foldseek notes

Fork: https://github.com/jeffdaily/foldseek (branch moat-port)
Default branch: master (NOT main).
Port head: e7471b4164e38cbac58b4f2c6c1b592e9bfac330

Scope: the GPU structure-search aligner (`--gpu`), which is the vendored
libmarv (CUDASW++4.0) at `lib/mmseqs/lib/libmarv`. Strategy A (pure CMake +
single compat header). ProstT5's ggml-CUDA LLM backend is OUT OF SCOPE and
stays off (USE_HIP does not turn on GGML_CUDA).

libmarv is the SAME aligner MMseqs2 vendors -- this port and the MMseqs2 port
must converge. See "Cross-reference: MMseqs2" below.

## Build recipe (gfx90a, ROCm 7.2.1)
```
cmake -S projects/foldseek/src -B projects/foldseek/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
cmake --build projects/foldseek/build --target foldseek -j16
```
- `USE_HIP` is a new cache option threaded top-level -> lib/mmseqs -> libmarv.
  It drives the libmarv subdir (parallel to `ENABLE_CUDA`) and the
  `HAVE_CUDA=1` define + marv link in mmseqs/foldseek, WITHOUT enabling the
  ProstT5 ggml-CUDA path (that one is still keyed off `ENABLE_CUDA`).
- libmarv builds as a SHARED library on HIP so the `-fgpu-rdc` device link
  runs inside the HIP toolchain (clang `--hip-link`); the rest of the project
  then links `libmarv.so` as an ordinary host shared object. A static libmarv
  left the device code unlinked and g++ (the project's host linker) cannot do
  the amdgcn device link -> relocation/`R_X86_64_32` PIE failure. SHARED +
  POSITION_INDEPENDENT_CODE fixes it.
- Force-include `cuda_to_hip.h` and `-I hip_compat` are added only on the HIP
  compile via `target_compile_options($<COMPILE_LANGUAGE:HIP>...)`.

## Compat surface (all under lib/mmseqs/lib/libmarv/src)
- `cuda_to_hip.h` (NEW, force-included on HIP): cudaXxx->hipXxx runtime/event/
  stream/mempool/IPC aliases; `__grid_constant__`->nothing;
  `cudaFuncSetAttribute(func,...)`->`hipFuncSetAttribute((const void*)func,...)`;
  `THRUST_DEVICE_PAR_NOSYNC`->`thrust::hip::par_nosync`; a `cub::SwitchDevice`
  RAII shim (hipCUB has none); `WARP_FULL_MASK`=64-bit + `WarpMaskT`; short2/
  int2/float2 `__shfl*` overloads (HIP covers scalars and __half2 only);
  `__hmax2`/`__hmin2` pairwise emulation; and scalar emulations of the DPX
  intrinsics (`__vadd2`, `__vmaxs2`, `__vibmax_u16x2`, `__vimax3_s16x2[_relu]`,
  `__viaddmax_s16x2[_relu]`, `__vibmax_s32`, `__vimax3_s32`,
  `__viaddmax_s32[_relu]`).
- `hip_compat/cooperative_groups.h` + `cooperative_groups/reduce.h` (NEW):
  HIP cg has tiled_partition/thread_block_tile with shfl_* but no free
  `reduce()` and no `greater/less/plus`. Shim adds them; `reduce()` is a
  butterfly all-reduce (`shfl_xor`) that works for any commutative+associative
  op (the int3 endpoint-pack lambda and `greater<half2/float/int>`), and a
  `tile_shfl_xor` that relays arbitrary trivially-copyable T as 32-bit words.
- `hip_compat/cuda_fp16.h` (NEW): `<cuda_fp16.h>` -> `<hip/hip_fp16.h>`.

## DPX strategy (THE key item -- matches the plan)
- libmarv selects DPX vs portable at runtime from a per-device kernel config.
  TRAP: on ROCm gfx90a `hipDeviceAttributeComputeCapabilityMajor` returns
  **9** (== Hopper sm90), so the unmodified selector picks the sm90 config
  whose entries set `dpx=1` -> it would launch the NVIDIA-only DPX short2/int
  kernels. FIX: under `__HIPCC__`, `getOptimalKernelConfigs_gapless/_SW` and
  the benchmark `supportsDPX` are forced to the portable default
  (sm89 config, `dpx=0`). So AMD always runs the half2 gapless + float SW
  SIMT path. (The user-supplied tuning-config-file override that can force
  dpx=1 is an edge case not on the default foldseek path; the runtime
  `if(!config.dpx)` dispatch still selects half2/float there.)
- The DPX `*_instantiation_dpx*.cu` and `*_smithwaterman_instantiation_dpx.cu`
  TUs are KEPT in the HIP build (their kernel templates are referenced via
  `extern template` from the dispatcher, so excluding them would leave
  unresolved device symbols at link). They compile against the scalar DPX
  emulations in cuda_to_hip.h and are simply never launched.
- `__hmax2`/`__hmin2`: ROCm 7.2.x has scalar `__hmax`/`__hmin` and `__hadd2`
  but not the packed max/min; emulated pairwise in cuda_to_hip.h for the
  portable half2 path.

## Wave-size handling (gfx90a wave64)
- PSSM gapless + Smith-Waterman kernels reduce only within
  `cg::tiled_partition<N>` tiles, N in {4,8,16,32} <= 32: wave-agnostic.
- `getPaddedQueryLength` pads the query layout by `sizeof(char4)*32`. This is
  a host+device-SHARED serialized/in-memory FORMAT constant (PORTING_GUIDE
  warp-size-dependent-format class). PINNED at the literal 32 (NOT warpSize),
  so the host-built layout matches device reads on wave64. Left untouched.
- `kernelhelpers.cuh::warp_max_reduce_broadcast` (legacy `kernels.cuh` path,
  not the foldseek `--gpu` path) made warp-width-generic: `__reduce_max_sync`
  guard tightened to `defined(__CUDA_ARCH__) && >= 800`, shuffle reduction
  start changed from literal 16 to `warpSize/2`. Its `0xFFFFFFFF` masks (and
  the kernels.cuh shfl masks) widened to 64-bit `WARP_FULL_MASK`/`WarpMaskT`
  because HIP `__shfl_*_sync` static_assert require a 64-bit mask on wave64.
- Inline PTX `%laneid` (`cuda_helpers.cuh::lane_id`) given a `__HIPCC__`
  branch using `__lane_id()`. (lane_id is actually unused by libmarv.)

## Other guards
- `#ifdef __NVCC__`/`#ifdef __CUDACC__` whole-file/qualifier guards across
  hpc_helpers (cuda_raiiwrappers, simple_allocation, peer_access,
  utility_kernels, custom_thrust_allocators, nvtx_markers, timers), blosum.cu,
  blosum.hpp `extern __constant__`, and convert.cuh qualifiers were widened to
  `defined(__CUDACC__) || defined(__HIPCC__)`. EXCEPTION: the uint64
  atomicCAS/Add/Exch/Min/Max/And/Or + ffs overloads in cuda_helpers.cuh stay
  NVCC-only -- HIP provides those natively and the library's versions collided.
- `convert.cuh::ClampToInvalid::operator()(uint)` device path: ROCm has no
  `__vminu4`; added a scalar per-byte clamp under `__HIP_DEVICE_COMPILE__`
  (the `#else` host path used `std::memcpy`, not callable on device).
- NVTX is compiled out via the pre-existing `NO_NVTOOLSEXT` target define;
  `nvtx::ScopedRange` becomes a no-op stub on HIP.
- `kernels.cuh`/`pssm*.cuh` are headers: marked `HEADER_FILE_ONLY` on HIP so
  they are NOT compiled standalone (doing so duplicated the `__constant__
  constantQuery4` device global and broke the device link). Only the `.cu`
  TUs are LANGUAGE HIP.

## Validation (real gfx90a, HIP_VISIBLE_DEVICES=1, ZERO downloads)
Inputs: bundled `example/` (28 SCOP structures). CPU mode is the oracle.
```
foldseek createdb example/ exDB
foldseek search exDB exDB aln_cpu tmp_cpu --gpu 0 -e 10      # CPU oracle
foldseek makepaddedseqdb exDB exDB_pad                       # GPU needs padded DB
foldseek search exDB exDB_pad aln_gpu tmp_gpu --gpu 1 -e 10  # libmarv half2/float path
foldseek convertalis exDB exDB     aln_cpu cpu.m8
foldseek convertalis exDB exDB_pad aln_gpu gpu.m8
```
RESULT: all 834 CPU hits reproduced on GPU with byte-identical pident /
alnlen / query+target coords / e-value / bitscore (0 differing rows on the 834
common (query,target) pairs, 0 CPU-only pairs). GPU additionally returns 38
borderline low-bitscore pairs (mostly bits 1-10) -- the known GPU-vs-CPU
ungapped-prefilter sensitivity difference, NOT a numeric/wave bug. GPU result
is a strict superset; every shared alignment is exact. Confirms the wave64
half2/float SIMT path is correct.

The GPU search log shows `ungappedprefilter ... --gpu 1` (the libmarv GPU
prefilter) ran, feeding `structurealign`.

The foldseek/MMseqs2 `regression/` submodules pull external data (egress-bound)
and are deferred; the bundled example cross-check is the gate. The `--gpu 0`
CPU run doubles as the non-GPU regression check (unchanged behavior).

## Cross-reference: MMseqs2 (sibling port, same libmarv)
At the time of this port the MMseqs2 sibling notes were still empty (planner
only). This foldseek port establishes the libmarv HIP approach; MMseqs2 should
reuse `cuda_to_hip.h` + `hip_compat/` verbatim and the same DPX-forced-off /
__hmax2-emulation / wave-mask widening / getPaddedQueryLength-pinned-at-32 /
shared-lib-device-link decisions. The only foldseek-specific wiring is the
top-level/foldseek CMake gating; the libmarv changes are identical and should
land identically in both forks.

## Follower platforms (gfx1100 / gfx1151, wave32)
Reuse this moat-port branch and validate first. Wave32 is the natural fit for
libmarv's 32-lane logic. The fixes here are wave-agnostic: getPaddedQueryLength
stays at 32 on every arch; WARP_FULL_MASK/WarpMaskT auto-narrow to 32-bit on
CUDA but on HIP wave32 the 64-bit mask still works (HIP narrows it). Cross-arch
result diff: gfx1100 GPU hits/scores must match gfx90a GPU and CPU on example/.
