# dietgpu notes

Fork: https://github.com/jeffdaily/dietgpu (branch `moat-port`)
Lead platform: linux-gfx90a (ROCm 7.2, gfx90a). Strategy A (compat header + CMake HIP gating).

## Build (gfx90a, the GPU correctness gate)

The codec core builds and validates WITHOUT a ROCm PyTorch -- the gtest
round-trip suite is the gate. Submodules (glog, googletest) must be initialized.

```
cd projects/dietgpu/src
git submodule update --init --recursive
export HIP_VISIBLE_DEVICES=0
cmake -S . -B build-hip -G Ninja \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-hip --target ans_test ans_statistics_test \
  batch_prefix_sum_test float_test gpu_ans gpu_float_compress dietgpu_utils -j 16
```

`-DCMAKE_POLICY_VERSION_MINIMUM=3.5` is required only because the vendored glog
submodule predates CMake 3.5 policy removal; unrelated to the port.

Followers reuse this same commit with only `-DCMAKE_HIP_ARCHITECTURES=gfx1100`
(wave32) -- no source edit. The build derives the wavefront width (and thus
DIETGPU_WARP_SIZE / kWarpSize) from the single target arch.

## Test (run the four gtests directly)

```
export HIP_VISIBLE_DEVICES=0
cd build-hip
./bin/ans_test               # ANSTest: ZeroSized, BatchPointer, BatchPointerLarge, BatchStride
./bin/ans_statistics_test    # Histogram, Normalization{,_NonZero,_EqualWeight}
./bin/batch_prefix_sum_test  # OneLevel, TwoLevel
./bin/float_test             # Batch, LargeBatch, BatchSize1 (fp16/bf16/fp32)
```

gfx90a result: ans_test 4/4, ans_statistics_test 4/4, batch_prefix_sum_test 2/2,
float_test 3/3 -- all PASS, deterministic across repeated runs. AMD_LOG_LEVEL=3
confirms dispatch to the native gfx90a code object.

GOTCHA: `ctest` reports "No tests were found" -- gtest_discover_tests does not
register under the HIP language build here. Run the four executables directly;
that is the real gate. (Do not treat the empty ctest list as a pass/fail.)

## Wave64 archive non-portability (IMPORTANT, by design)

kWarpSize defines the rANS archive geometry, not just a reduction width: the
input is striped across lanes by kWarpSize and one rANS state per lane is
serialized into the archive header (ANSWarpState::warpState[kWarpSize]). On
gfx90a kWarpSize=64, so a gfx90a archive interleaves 64 lane-states and is NOT
byte-compatible with a CUDA / RDNA (wave32, 32-state) archive, and vice versa.
This is a wave-width consequence, not a bug. Round-trip on a single arch is
self-consistent (compress and decompress agree), which is the correctness gate.
Cross-warp-width archive interop is intentionally out of scope. gfx1100/gfx1151
are wave32 and will produce 32-lane archives like CUDA.

kWarpSize is derived by the BUILD from the single target arch
(DIETGPU_WARP_SIZE; gfx9* -> 64 else 32), NOT keyed on __GFX9__. __GFX9__ is
defined only in the device compile pass, so a __GFX9__-keyed constant would make
the host compute archive buffer sizes / launch geometry at 32 lanes while a
gfx90a device ran 64 -- buffer mis-size / corruption. A single-arch build is
required (a compile-time-constant warp size cannot vary per-arch in one fat
binary anyway).

## Deferred: the PyTorch tensor binding

The top-level `dietgpu` target (dietgpu/DietGpu.cpp) is the ONLY thing that
needs find_package(Torch). It is gated off under USE_HIP. DietGpu.cpp has no
.cu and just calls the already-HIP device libraries, so it can be built later
against a ROCm PyTorch with no device changes. The codec libs
(gpu_ans, gpu_float_compress, dietgpu_utils) and the gtest suite are the GPU
gate and need no ROCm PyTorch. The Python tests (ans_test.py, float_test.py,
benchmark.py) require that binding and a ROCm PyTorch; not part of the lead gate.

## Port mechanics (where the HIP-specific code lives)

- dietgpu/utils/cuda_to_hip.h -- single compat shim (HIP runtime include, cuda*
  aliases, cub->hipcub, __thrust_exec_check_disable__ no-op, CUDA_VERSION pin).
  Force-included on every HIP TU via -include.
- hip_compat/{cuda.h,cuda_runtime.h,cuda_profiler_api.h,cub/...} -- name-shim
  forwarders so source include lines are untouched; on the HIP include path only.
- dietgpu/utils/DeviceDefs.cuh -- kWarpSize from DIETGPU_WARP_SIZE.
- dietgpu/utils/PtxUtils.cuh -- HIP branch for bfe/bfi/rotate/funnel-shift,
  laneid/lanemask, ballot/popc/shfl helpers (64-bit masks, maskless builtins).
  NVIDIA branch keeps the original PTX verbatim.
- dietgpu/ans/GpuANS{Encode,Decode}.cuh -- call sites route through
  warpBallot/warpPopc/warpShflIdx helpers.
- dietgpu/float/GpuFloatUtils.cuh -- bf16 join funnel shift via funnelShiftRight.
- CMakeLists.txt (top) -- option(USE_HIP), enable_language(HIP), add_library/
  add_executable override to retag .cu LANGUAGE HIP (subdir CMakeLists mostly
  untouched), DIETGPU_WARP_SIZE derivation, compat include dir + force-include.
- dietgpu/utils/CMakeLists.txt -- DeviceUtils.cpp/StackDeviceMemory.cpp marked
  LANGUAGE HIP (they call the HIP runtime; plain g++ has no hip headers).

## R6 (cub TempStorage reuse) audit result

No __syncthreads() additions needed. Every cub collective in the tree
(BlockScan in BatchPrefixSum / decode table build, BlockRadixSort+BlockScan in
statistics, BlockReduce in checksum) uses its OWN dedicated TempStorage union
used exactly once; there is no back-to-back reuse of a single TempStorage that
would race on a single-wavefront block. The statistics kernel already separates
its sort and scan with a shared-memory round-trip + __syncthreads().

## Review 2026-06-02 (reviewer, linux-gfx90a)

Reviewed git diff a4d70a14...03088ce1 on moat-port. Verdict: review-passed.
Strategy A executed correctly; R1-R6 analysis verified; reproduced the GPU gate
(all four gtests PASS on real gfx90a, DIETGPU_WARP_SIZE=64 baked, deterministic
across repeated runs). Commit hygiene clean. Findings are minor (latent UB in
two unused helpers); none block.

Minor (non-blocking, fix opportunistically):
- dietgpu/utils/PtxUtils.cuh:78 (getLaneMaskLe) and :82 (getLaneMaskGt) compute
  `LaneMaskT(2) << laneId`; at laneId==63 that is a shift of an effective 1<<64,
  i.e. shift-count == type width, which is UB in C++. These two helpers are DEAD
  in the codec (only getLaneMaskLt and getLaneMaskGe are called -- GpuANSEncode
  :69/:116 and GpuANSDecode :95/:149 -- and neither uses the 2<<laneId form), so
  there is no live miscompression today. If kept, rewrite Le as
  `getLaneMaskLt() | (LaneMaskT(1) << laneId)` and Gt as `~getLaneMaskLe()` to
  avoid the lane-63 UB. getLaneMaskLt/Ge use `1<<laneId` and are well-defined at
  lane 63.
- getBitfield/setBitfield are defined in both branches but unused anywhere in
  the codec (pre-existing upstream dead code, not port-introduced); their HIP
  rewrites were not exercised by any test. Left as-is is fine (they mirror the
  NVIDIA branch); no action required.

Verified correct (load-bearing):
- rotateLeft/rotateRight/funnelShiftRight HIP C++ are bit-identical to the PTX
  shf.l/r.clamp.b32 they replace for the shift==1 call sites used in the float
  split/join, and across the full 0..31 rotate range (exhaustive check). bf16
  join operand order (funnelShiftRight(lo,hi,1)) matches the original
  shf.r.clamp.b32 dst,lo,hi,1.
- 64-bit getLaneMaskLt/Ge match PTX lanemask_lt/ge for all lanes 0..63.
- ballot/popc/shfl route through maskless 64-bit HIP builtins; lanes 32-63 are
  not dropped. NVIDIA branch keeps the original PTX + 0xffffffff verbatim.
- kWarpSize derived from build-injected DIETGPU_WARP_SIZE (gfx9*->64 else 32),
  NOT __GFX9__; host and device agree. Build confirmed DIETGPU_WARP_SIZE=64.
- R6: smemSort (BlockRadixSort) and smemScan (BlockScan) are distinct unions
  separated by __syncthreads() (GpuANSStatistics.cuh:325); BatchPrefixSum passes
  are separate kernels; checksum BlockReduce used once. No reused-TempStorage
  race. notes' "no __syncthreads needed" claim verified.
- NVIDIA/CUDA path untouched (compat header no-op; CMake else-branch reproduces
  the original verbatim). Torch binding deferral documented and gated, not
  silently broken.

GPU gate reproduced (HIP_VISIBLE_DEVICES=0, gfx90a, ROCm 7.2.1):
ans_test 4/4, ans_statistics_test 4/4, batch_prefix_sum_test 2/2, float_test 3/3
-- all PASS, deterministic.
