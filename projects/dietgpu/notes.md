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
