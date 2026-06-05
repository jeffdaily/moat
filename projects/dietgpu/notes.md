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
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100" \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-hip --target ans_test ans_statistics_test \
  batch_prefix_sum_test float_test gpu_ans gpu_float_compress dietgpu_utils -j 16
```

A single arch (`-DCMAKE_HIP_ARCHITECTURES=gfx90a`) also works; the multi-arch
form is the new must-pass gate (confirm both code objects:
`llvm-objdump --offloading build-hip/lib/libgpu_ans.so` shows gfx90a AND
gfx1100). `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` is required only because the
vendored glog submodule predates CMake 3.5 policy removal; unrelated to the
port.

Followers reuse this same commit with no source edit. The device wavefront
width resolves per-arch at compile time (__GFX*__) and host code queries the
runtime device warpSize -- there is no build-injected warp constant anymore.

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

## Multi-arch warp size (the fixed-64-slot archive) -- SUPERSEDES the old single-arch model

The original port baked a single build-injected DIETGPU_WARP_SIZE into BOTH
host and device (gfx9* -> 64 else 32), which forced a single-arch build and
mis-sized the archive on the host of a multi-arch binary. That is removed. The
current model builds one fat binary for gfx90a;gfx1100:

- DEVICE width is per-arch compile-time: DeviceDefs.cuh kWarpSize is guarded on
  __HIP_DEVICE_COMPILE__ then __GFX8__||__GFX9__ -> 64 else 32. These __GFX*__
  macros are defined ONLY in the device compile pass and ARE constexpr-usable,
  so a fat binary resolves 64 on gfx90a and 32 on gfx1100. (No
  __AMDGCN_WAVEFRONT_SIZE__ macro exists in ROCm 7.2.) CUDA path stays 32.
- HOST never reads kWarpSize (it would always be 32 in the host pass, and there
  is no single host warp width in a multi-arch build). The one host consumer
  was the encode grid divisor GpuANSEncode.cuh `kThreads / kWarpSize`; it now
  uses `getCurrentDeviceProperties().warpSize` (runtime device query) so the
  grid matches the device kernel's `block = tid / warpSize` per arch.
- ARCHIVE FORMAT is pinned arch-independent: GpuANSUtils.cuh defines
  `kMaxWarpSize = 64` and `ANSWarpState::warpState[kMaxWarpSize]`. kWarpSize is
  a serialized data-format parameter, so the header layout (sizeof(ANSWarpState),
  getWarpStates/getBlockWords offsets, getCompressedOverhead) must NOT depend on
  a per-arch or per-build warp width. Fixing it at 64 keeps the gfx90a archive
  byte-identical to the previously validated format and gives gfx90a and gfx1100
  the SAME 64-slot geometry. A wave32 device writes/reads only lanes 0-31 (its
  device kWarpSize stride); slots 32-63 are unused on wave32. The device kernels
  still stride by the per-arch device kWarpSize -- do NOT change that.
- CMakeLists.txt: DIETGPU_WARP_SIZE and the gfx9* arch-scan loop are deleted;
  only `add_compile_definitions(USE_HIP=1)` remains for the warp path.

Round-trip on a single arch is self-consistent (the correctness gate). The
gfx90a (wave64) archive uses all 64 slots; a gfx1100/CUDA (wave32) archive uses
32 of the 64 -- the header geometry is identical (64 slots) but the live lane
count differs, so cross-warp-width DECODE interop is still not a goal (a wave64
producer fills 64 states that a wave32 consumer would not all read). The win is
that the host/device sizing no longer couples to the warp width and the
multi-arch build compiles and dispatches correctly.

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
- dietgpu/utils/DeviceDefs.cuh -- per-arch DEVICE kWarpSize (__GFX*__ guards);
  GpuANSUtils.cuh kMaxWarpSize=64 pins the archive layout (see multi-arch section).
- dietgpu/utils/PtxUtils.cuh -- HIP branch for bfe/bfi/rotate/funnel-shift,
  laneid/lanemask, ballot/popc/shfl helpers (64-bit masks, maskless builtins).
  NVIDIA branch keeps the original PTX verbatim.
- dietgpu/ans/GpuANS{Encode,Decode}.cuh -- call sites route through
  warpBallot/warpPopc/warpShflIdx helpers.
- dietgpu/float/GpuFloatUtils.cuh -- bf16 join funnel shift via funnelShiftRight.
- CMakeLists.txt (top) -- option(USE_HIP), enable_language(HIP), add_library/
  add_executable override to retag .cu LANGUAGE HIP (subdir CMakeLists mostly
  untouched), compat include dir + force-include. (No DIETGPU_WARP_SIZE anymore.)
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

## Validation 2026-06-02 (gfx1100)

Platform: linux-gfx1100, 2x AMD Radeon Pro W7800 48GB (gfx1100, RDNA3, wave32), ROCm 7.2.1, HIP_VISIBLE_DEVICES=0.
Fork: jeffdaily/dietgpu @ moat-port, SHA 03088ce1542c8413ba1daad3eb5c03a4ece22976.
No source or fork changes; follower validates-first reusing the lead commit unchanged.

### kWarpSize / DIETGPU_WARP_SIZE resolution on gfx1100

CMake configure output: `-- dietgpu HIP arch=gfx1100 warp_size=32`

DIETGPU_WARP_SIZE=32 is baked into the build via the CMakeLists.txt logic: if
CMAKE_HIP_ARCHITECTURES matches gfx9*, warp_size=64, else warp_size=32. For
-DCMAKE_HIP_ARCHITECTURES=gfx1100 (not gfx9*), it resolves to 32. This value
is injected as a compile definition and DeviceDefs.cuh derives kWarpSize from it
at compile time -- host and device both see 32. No source edit required.

### Build

```
cd /var/lib/jenkins/moat/projects/dietgpu/src
git submodule update --init --recursive
cmake -S . -B build-hip -G Ninja \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-hip --target ans_test ans_statistics_test \
  batch_prefix_sum_test float_test gpu_ans gpu_float_compress dietgpu_utils -j 16
```

Build: PASS, 35 targets, 0 errors, 1 pre-existing upstream warning (braces around
scalar initializer in FloatTest.cu:306). Build time: 8.4s (timeit).

### gfx1100 code-object evidence

llvm-objdump --offloading on libgpu_ans.so shows:
`hipv4-amdgcn-amd-amdhsa--gfx1100` bundles -- no gfx90a present. All .so and
executables link the gfx1100 code objects exclusively.

### GPU tests (run directly; ctest still reports no tests under HIP-language build)

```
export HIP_VISIBLE_DEVICES=0
cd build-hip
./bin/ans_test
./bin/ans_statistics_test
./bin/batch_prefix_sum_test
./bin/float_test
```

Run 1 (test time: 22.6s total via timeit):
- ans_test: 4/4 PASSED (ZeroSized, BatchPointer, BatchPointerLarge, BatchStride)
- ans_statistics_test: 4/4 PASSED (Histogram, Normalization_NonZero, Normalization_EqualWeight, Normalization)
- batch_prefix_sum_test: 2/2 PASSED (OneLevel, TwoLevel)
- float_test: 3/3 PASSED (Batch, LargeBatch, BatchSize1; fp16/bf16/fp32)

Run 2 (determinism check -- ans_test + float_test rerun):
- ans_test: 4/4 PASSED
- float_test: 3/3 PASSED

Pass/fail identical across runs: deterministic confirmed. Zero HSA faults (0x1016).

### Tally vs gfx90a @ 03088ce

| Test binary          | gfx90a | gfx1100 |
|----------------------|--------|---------|
| ans_test             | 4/4    | 4/4     |
| ans_statistics_test  | 4/4    | 4/4     |
| batch_prefix_sum_test| 2/2    | 2/2     |
| float_test           | 3/3    | 3/3     |
| **Total**            | 13/13  | 13/13   |

### Wave32 verdict

PASS. 32-lane rANS round-trip is self-consistent on gfx1100: all round-trip tests
(ANSTest, float codec) compress and decompress to reproduce the input exactly with
kWarpSize=32. No wrong round-trip, no 0x1016 HSA fault, no NaN, no hang.

Cross-arch archive non-portability: a gfx90a (wave64, 64-lane) archive is not
byte-compatible with a gfx1100 (wave32, 32-lane) archive -- this is expected by
design and not tested as a failure. gfx1100/gfx1151 produce 32-lane archives
consistent with CUDA behavior.

Transitioning linux-gfx1100 to completed.

## Validation 2026-06-02 (validator, linux-gfx90a)

Platform: linux-gfx90a, AMD Instinct MI250X (gfx90a), ROCm 7.2, HIP_VISIBLE_DEVICES=0.
Fork: jeffdaily/dietgpu @ moat-port, SHA 03088ce1542c8413ba1daad3eb5c03a4ece22976.

### Build

```
cd projects/dietgpu/src
git submodule update --init --recursive
cmake -S . -B build-hip -G Ninja \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-hip --target ans_test ans_statistics_test \
  batch_prefix_sum_test float_test gpu_ans gpu_float_compress dietgpu_utils -j 16
```

CMake configure confirmed: `-- dietgpu HIP arch=gfx90a warp_size=64` (DIETGPU_WARP_SIZE=64).
Build: PASS, 34 targets, 0 errors, 1 pre-existing upstream warning (braces around scalar initializer in FloatTest.cu:306).

### GPU tests (run directly; ctest reports no tests under HIP-language build)

```
export HIP_VISIBLE_DEVICES=0
cd build-hip
./bin/ans_test
./bin/ans_statistics_test
./bin/batch_prefix_sum_test
./bin/float_test
```

Run 1:
- ans_test: 4/4 PASSED (ZeroSized, BatchPointer, BatchPointerLarge, BatchStride)
- ans_statistics_test: 4/4 PASSED (Histogram, Normalization_NonZero, Normalization_EqualWeight, Normalization)
- batch_prefix_sum_test: 2/2 PASSED (OneLevel, TwoLevel)
- float_test: 3/3 PASSED (Batch, LargeBatch, BatchSize1; fp16/bf16/fp32)

Run 2 (determinism check):
- ans_test: 4/4 PASSED
- ans_statistics_test: 4/4 PASSED
- batch_prefix_sum_test: 2/2 PASSED
- float_test: 3/3 PASSED

Pass/fail identical across both runs: deterministic confirmed.

Device dispatch confirmed via AMD_LOG_LEVEL=3:
`Using native code object for device: amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-`

Total: 13/13 PASS. Verdict: PASS. Transitioning linux-gfx90a to completed.

## Multi-arch warp-size fix 2026-06-02 (porter, linux-gfx90a) -- fork b6e0d3f

Re-entered from completed: the 03088ce port baked a single DIETGPU_WARP_SIZE in
host + device, which broke a combined gfx90a;gfx1100 build (the warp width
cannot be one compile-time constant across two archs, and it is an
archive-format parameter). Fixed per the multi-arch warp-size standard:

1. DeviceDefs.cuh kWarpSize -> per-arch DEVICE constexpr guarded on
   __HIP_DEVICE_COMPILE__ + __GFX8__||__GFX9__ -> 64 else 32 (CUDA 32). Fixes
   every device-side kWarpSize consumer per-arch with no other device edit.
2. GpuANSUtils.cuh: ANSWarpState pinned to a fixed 64-slot layout
   (kMaxWarpSize = 64) so the serialized header geometry is arch-independent;
   gfx90a archive byte-identical to before, gfx1100 same 64-slot geometry.
3. GpuANSEncode.cuh:747 host grid divisor now uses
   getCurrentDeviceProperties().warpSize (runtime device query) instead of the
   device-pass kWarpSize; matches the device kernel's block = tid / warpSize.
4. CMakeLists.txt: deleted DIETGPU_WARP_SIZE + gfx9* arch-scan loop + the
   "multi-arch not supported" comments; only USE_HIP=1 remains.

### Validation gate (the new must-pass multi-arch test)

Build `-DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100"`: compiles clean (1
pre-existing FloatTest.cu:306 warning). `llvm-objdump --offloading` on
libgpu_ans.so AND libgpu_float_compress.so shows BOTH gfx90a and gfx1100
offload bundles in the fat binary.

gfx90a run from the multi-arch binary (HIP_VISIBLE_DEVICES=0):
- ans_test 4/4, ans_statistics_test 4/4, batch_prefix_sum_test 2/2,
  float_test 3/3 -- all PASS, deterministic across two runs.
- AMD_LOG_LEVEL=3: "Using native code object for device:
  amdgcn-amd-amdhsa--gfx90a" -- correct per-arch dispatch from the fat binary.
- Fixed-64-slot archive round-trips correctly on gfx90a (all round-trip tests
  pass), so the format change is byte-correct.

gfx1100/gfx1151 RUN re-validates on follower hosts (advance-head flips the
already-completed platforms to revalidate). The clean two-arch build + correct
gfx90a run + runtime-query host path is this porter's gate.

## Review (multi-arch) 2026-06-02 (reviewer, linux-gfx90a)

Reviewed `git diff 03088ce..b6e0d3f` on moat-port (the delta: per-arch device
kWarpSize, fixed-64-slot archive, runtime-query host grid divisor, CMake
DIETGPU_WARP_SIZE removal). Verdict: review-passed. Reproduced the multi-arch
GPU gate on real gfx90a (ROCm 7.2.1, HIP_VISIBLE_DEVICES=0).

Verified correct (load-bearing, the highest-risk fixed-64-slot change):
- DeviceDefs.cuh:32-39 -- device kWarpSize is `__HIP_DEVICE_COMPILE__` then
  `__GFX8__||__GFX9__`->64 else 32, host/CUDA fall-through 32. Resolves per-arch
  in the fat binary; gfx90a dispatch confirmed (AMD_LOG_LEVEL=3 "native code
  object for device: amdgcn-amd-amdhsa--gfx90a"). CUDA path unchanged.
- GpuANSUtils.cuh:72-77 -- `kMaxWarpSize=64`, `ANSWarpState::warpState[64]`.
  ANSStateT=uint32_t so sizeof(ANSWarpState)=256B both before (gfx90a baked
  DIETGPU_WARP_SIZE=64) and now -> gfx90a archive byte-identical to the prior
  validated 64-lane format. getCompressedOverhead (GpuANSUtils.cuh:80-93, the
  __host__ __device__ sizing fn) and the host inline getMaxBlockSizeUnCoalesced
  (GpuANSEncode.cuh:38-41) both size via sizeof(ANSWarpState), now
  arch-independent. getWarpStates/getBlockWords offsets are __device__ and
  stride by sizeof(ANSWarpState) -- arch-independent.
- Device kernels still stride by per-arch kWarpSize: encode writes
  warpState[laneId] / warpState[tid] under `tid < kWarpSize` (GpuANSEncode.cuh
  :207,:588-592), decode reads warpState[laneId] (GpuANSDecode.cuh:374). gfx90a
  fills 0-63, a wave32 device fills 0-31; per-arch self-consistent.
- GpuANSEncode.cuh:752,756 -- host grid divisor now
  `getCurrentDeviceProperties().warpSize`. Confirmed it is the ONLY host
  consumer of the warp width: every other kWarpSize use (GpuANSEncode 165-394,
  GpuANSDecode 174-469, GpuANSStatistics 26-200, GpuFloatCompress 297-301,
  PtxUtils) is inside __global__/__device__ code; the decode launcher grid is
  occupancy-based and never divides by kWarpSize on the host.
- CMakeLists.txt -- DIETGPU_WARP_SIZE + gfx9* arch-scan loop deleted; only
  `add_compile_definitions(USE_HIP=1)` remains.

GPU gate reproduced from a SINGLE multi-arch fat binary
(`-DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100"`): clean build, only the
pre-existing FloatTest.cu:306 braced-scalar-init warning. `llvm-objdump
--offloading` on libgpu_ans.so AND libgpu_float_compress.so shows BOTH gfx90a
and gfx1100 bundles. gfx90a run (HIP_VISIBLE_DEVICES=0): ans_test 4/4,
ans_statistics_test 4/4, batch_prefix_sum_test 2/2, float_test 3/3 -- all PASS,
identical across two runs (deterministic). Suites assert compress->decompress
equality (EXPECT_EQ orig/dec), so the fixed-64-slot archive round-trip is
genuinely verified, not just no-crash.

Commit hygiene clean: title `[ROCm] ...` 68 chars, mentions Claude, no
Co-Authored-By/noreply trailer, no ghstack, no em-dash, no AMD-internal
references. Fork main is a clean upstream mirror at a4d70a1; Actions disabled.

Minor (non-blocking, pre-existing, NOT touched by this delta -- carried from the
prior review): PtxUtils.cuh:78/82 getLaneMaskLe/Gt `2<<laneId` lane-63 UB in
dead helpers; getBitfield/setBitfield unused. No action required for this port.

## Validation (multi-arch) 2026-06-02

Platform: linux-gfx90a, AMD Instinct MI250X (gfx90a), ROCm 7.2.1, HIP_VISIBLE_DEVICES=0.
Fork: jeffdaily/dietgpu @ moat-port, SHA b6e0d3f64558eb7d30cab1573be487318514c7a1.
State: review-passed -> completed (multi-arch warp-size fix).

### Two-arch build

```
cmake -S /var/lib/jenkins/moat/projects/dietgpu/src -B .../build-hip -G Ninja \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100" \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-hip --target ans_test ans_statistics_test \
  batch_prefix_sum_test float_test gpu_ans gpu_float_compress dietgpu_utils -j 16
```

Configure: `-- dietgpu HIP arch=gfx90a;gfx1100` (no DIETGPU_WARP_SIZE -- removed by multi-arch fix).
Build: PASS, 34/34 targets, 0 errors, 1 pre-existing upstream warning (FloatTest.cu:306 braced-scalar-init, appears for each arch pass).

### Offload bundle verification

`llvm-objdump --offloading` on libgpu_ans.so: contains `hipv4-amdgcn-amd-amdhsa--gfx1100` AND `hipv4-amdgcn-amd-amdhsa--gfx90a` bundles (3 .so segments each).
`llvm-objdump --offloading` on libgpu_float_compress.so: same -- both gfx90a and gfx1100 bundles present.
Fat binary confirmed for both codec libraries.

### gfx90a dispatch from fat binary

AMD_LOG_LEVEL=3: `Using native code object for device: amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-`
Correct per-arch dispatch from the fat binary confirmed.

### GPU tests (run directly; ctest reports no tests under HIP-language build)

```
export HIP_VISIBLE_DEVICES=0
cd build-hip
./bin/ans_test
./bin/ans_statistics_test
./bin/batch_prefix_sum_test
./bin/float_test
```

Run 1:
- ans_test: 4/4 PASSED (ZeroSized, BatchPointer, BatchPointerLarge, BatchStride)
- ans_statistics_test: 4/4 PASSED (Histogram, Normalization_NonZero, Normalization_EqualWeight, Normalization)
- batch_prefix_sum_test: 2/2 PASSED (OneLevel, TwoLevel)
- float_test: 3/3 PASSED (Batch, LargeBatch, BatchSize1; fp16/bf16/fp32)

Run 2 (determinism check):
- ans_test: 4/4 PASSED
- ans_statistics_test: 4/4 PASSED
- batch_prefix_sum_test: 2/2 PASSED
- float_test: 3/3 PASSED

Total: 13/13 PASS both runs. Pass/fail identical across runs: deterministic confirmed.

### Verdict

PASS. Clean two-arch fat binary (both gfx90a and gfx1100 bundles), native gfx90a dispatch confirmed, fixed-64-slot archive round-trip correct (all round-trip tests assert EXPECT_EQ orig/dec and pass), deterministic x2. Transitioning linux-gfx90a to completed (validated_sha=b6e0d3f).

## Validation 2026-06-02 (gfx1100) -- revalidate at b6e0d3f (multi-arch fixed-64-slot archive)

Platform: linux-gfx1100, AMD Radeon Pro W7800 48GB (gfx1100, RDNA3, wave32), ROCm 7.2.1, HIP_VISIBLE_DEVICES=0.
Fork: jeffdaily/dietgpu @ moat-port, SHA b6e0d3f64558eb7d30cab1573be487318514c7a1.
State: revalidate (was completed at 03088ce) -> completed at b6e0d3f.

### DeviceDefs.cuh fixed-64-slot resolution on gfx1100

DeviceDefs.cuh:32-40 resolves kWarpSize per-arch in the DEVICE compile pass:
- `#if defined(__HIP_DEVICE_COMPILE__)` then `__GFX8__||__GFX9__` -> 64 else 32.
- On gfx1100 (not GFX9): device kWarpSize = 32. On gfx90a (GFX9): device kWarpSize = 64.
- The host fall-through (neither __HIP_DEVICE_COMPILE__ nor CUDA) resolves to 32 as well,
  but host code no longer consumes kWarpSize for archive sizing -- that was the old bug.

The ARCHIVE GEOMETRY is pinned by GpuANSUtils.cuh kMaxWarpSize=64 (ANSWarpState::warpState[64]).
This is separate from the device kWarpSize. On gfx1100 (wave32) the device kernel:
- Writes/reads lanes 0-31 (its kWarpSize stride in warpState[laneId]).
- Slots 32-63 in ANSWarpState are unused but present in every block header.
- sizeof(ANSWarpState) = 64 * 4 = 256 bytes -- identical on gfx90a and gfx1100.
- The host grid divisor uses getCurrentDeviceProperties().warpSize (runtime query = 32 on gfx1100),
  so the launch geometry matches the wave32 device kernel exactly.

Wave32 encode semantics: 32 active lanes each independently encode a stripe of input
into one of the 32 live rANS states (laneId 0..31). The 64-slot header fits this with
32 live entries; the remaining 32 slots are zero-initialized. Decode reverses exactly.
Self-consistent round-trip on gfx1100 with the fixed-64-slot header confirmed by all tests.

### Build

```
cd /var/lib/jenkins/moat/projects/dietgpu/src
git submodule update --init --recursive
cmake -S . -B build-hip -G Ninja \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100" \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-hip --target ans_test ans_statistics_test \
  batch_prefix_sum_test float_test gpu_ans gpu_float_compress dietgpu_utils -j 16
```

Configure: `-- dietgpu HIP arch=gfx90a;gfx1100` (no DIETGPU_WARP_SIZE -- deleted by multi-arch fix).
Build: PASS, 35/35 targets, 0 errors, 1 pre-existing upstream warning (FloatTest.cu:306
braced-scalar-init, appears for each of the 3 compile passes: gfx1100, gfx90a, host).
Build time: 23.7s (timeit, fresh two-arch build).

### Fat binary / gfx1100 code-object evidence

`llvm-objdump --offloading` on libgpu_ans.so: 3 segments, each containing:
  `hipv4-amdgcn-amd-amdhsa--gfx1100` AND `hipv4-amdgcn-amd-amdhsa--gfx90a` bundles.
`llvm-objdump --offloading` on libgpu_float_compress.so: same -- both gfx90a and gfx1100 bundles.
Fat binary with both code objects confirmed for all codec libraries.

AMD_LOG_LEVEL=3 dispatch: `Using native code object for device: amdgcn-amd-amdhsa--gfx1100
co: amdgcn-amd-amdhsa--gfx1100` -- correct per-arch dispatch from the fat binary on gfx1100.

### GPU tests (run directly; ctest still reports no tests under HIP-language build)

```
export HIP_VISIBLE_DEVICES=0
cd build-hip
./bin/ans_test
./bin/ans_statistics_test
./bin/batch_prefix_sum_test
./bin/float_test
```

Run 1 (test time: 22.7s total via timeit):
- ans_test: 4/4 PASSED (ZeroSized, BatchPointer, BatchPointerLarge, BatchStride)
- ans_statistics_test: 4/4 PASSED (Histogram, Normalization_NonZero, Normalization_EqualWeight, Normalization)
- batch_prefix_sum_test: 2/2 PASSED (OneLevel, TwoLevel)
- float_test: 3/3 PASSED (Batch, LargeBatch, BatchSize1; fp16/bf16/fp32)

Run 2 (determinism check -- ans_test + float_test rerun):
- ans_test: 4/4 PASSED
- float_test: 3/3 PASSED

Pass/fail identical across runs: deterministic confirmed. Zero HSA faults (0x1016).

### Tally vs gfx90a @ b6e0d3f

| Test binary           | gfx90a@b6e0d3f | gfx1100@b6e0d3f |
|-----------------------|----------------|-----------------|
| ans_test              | 4/4            | 4/4             |
| ans_statistics_test   | 4/4            | 4/4             |
| batch_prefix_sum_test | 2/2            | 2/2             |
| float_test            | 3/3            | 3/3             |
| **Total**             | 13/13          | 13/13           |

### Wave32 verdict (fixed-64-slot archive)

PASS. The fixed-64-slot archive (kMaxWarpSize=64, ANSWarpState[64]) on gfx1100 (wave32):
- Device kWarpSize=32 (per-arch compile-time, GFX9->64 else 32 in device pass).
- Archive header is 256 bytes regardless of arch: sizeof(ANSWarpState)=64*4=256B.
- Encoder writes 32 lane-states into slots 0-31; slots 32-63 unused but header fits.
- Host grid divisor uses runtime warpSize=32, matching the 32-lane device kernel.
- All round-trip tests (ANSTest, FloatTest) assert EXPECT_EQ(orig, dec) and pass.
- Deterministic across 2 runs, zero HSA faults.

Cross-arch portability status: this re-port makes the archive geometry arch-independent
(fixed 64-slot header, same sizeof on all arches). However, a wave32 producer (gfx1100)
writes only 32 live lane-states into a 64-slot structure, while a wave64 producer (gfx90a)
writes 64. Cross-warp-width decode interop (gfx90a archive -> gfx1100 decode) is NOT a
goal and not tested: the slot counts (32 vs 64 live) differ and the decode kernel reads
exactly kWarpSize states. Same-arch round-trip is the correctness gate and is confirmed.
On this host (gfx1100 only), only gfx1100->gfx1100 round-trip is exercisable; no
gfx90a-produced archive fixture exists in-tree, and the fat binary does not allow targeting
gfx90a from a gfx1100 host (HIP_VISIBLE_DEVICES=0 selects the gfx1100 device). The win
from the fixed-64-slot layout is that sizeof(ANSWarpState) and all header offsets are
compile-time-constant and arch-independent in a multi-arch build -- not write-interop.

Transitioning linux-gfx1100 to completed (validated_sha=b6e0d3f).

## Validation 2026-06-04 (windows-gfx1151)

Platform: windows-gfx1151, AMD Radeon 8060S (gfx1151, RDNA3.5, wave32), Windows 11,
TheRock ROCm (pip wheel), clang++ 23.0 gcc-driver, HIP_VISIBLE_DEVICES=0.
Fork: jeffdaily/dietgpu @ moat-port, SHA 64c792d3da5eaa048b3cf5aac6f6632cbc5c22f2.
State: port-ready (follower) -> completed.

### Windows delta (one CMake line)

The Linux build uses `clang++ -shared` with an ELF linker, which exports all symbols by
default. The Windows build uses lld-link (MSVC ABI), which requires `__declspec(dllexport)`
or `CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS` -- without it the DLL import libs are empty and
`gpu_ans.dll` / `gpu_float_compress.dll` fail to link against `dietgpu_utils.dll`.

Fix: added `if(WIN32) set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON) endif()` to the USE_HIP
block in CMakeLists.txt. This is inside the existing `if(USE_HIP)` and the `if(WIN32)`
guard keeps it out of the Linux path entirely. The Linux platforms' compiled code objects
are identical (the CMake change does not touch any compiled source or compiler flags on
Linux); they are in `revalidate` for the binary-equiv check, which is expected to
carry-forward.

Committed as a new commit on top of b6e0d3f (64c792d), not amending the validated b6e0d3f.

### Build

```
ROCM_ROOT=D:/Develop/TheRock/.venv/Lib/site-packages/_rocm_sdk_devel
cd D:/Develop/moat/projects/dietgpu/src
git submodule update --init --recursive
cmake -S . -B build-hip -G Ninja \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_HIP_COMPILER=$ROCM_ROOT/lib/llvm/bin/clang++.exe \
  -DCMAKE_C_COMPILER=$ROCM_ROOT/lib/llvm/bin/clang.exe \
  -DCMAKE_CXX_COMPILER=$ROCM_ROOT/lib/llvm/bin/clang++.exe \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-hip --target ans_test ans_statistics_test \
  batch_prefix_sum_test float_test gpu_ans gpu_float_compress dietgpu_utils -j6
# Copy TheRock runtime DLLs next to test executables (System32 amdhip64_7.dll is
# the broken Adrenalin driver; TheRock's must be in the same dir as the .exe):
cp $ROCM_ROOT/bin/amdhip64_7.dll build-hip/bin/
cp $ROCM_ROOT/bin/amd_comgr0713.dll build-hip/bin/
cp $ROCM_ROOT/bin/rocm_kpack.dll build-hip/bin/
```

Build: PASS, 16/16 targets, 0 errors, 1 pre-existing upstream warning (FloatTest.cu:306
braced-scalar-init). Build time: ~11s (timeit).

### gfx1151 code-object evidence

`strings build-hip/bin/gpu_ans.dll | grep hipv4` shows:
  `hipv4-amdgcn-amd-amdhsa--gfx1151` -- gfx1151 code object present.
(Windows COFF: llvm-objdump --offloading does not support COFF; use strings on .hipFatB
section confirmed by llvm-readobj showing `.hipFatB` and `.hip_fat` sections.)

### GPU tests (run directly from build-hip/bin/)

```
export HIP_VISIBLE_DEVICES=0
export HIP_DEVICE_LIB_PATH=$ROCM_ROOT/lib/llvm/amdgcn/bitcode
cd build-hip
./bin/ans_test.exe
./bin/ans_statistics_test.exe
./bin/batch_prefix_sum_test.exe
./bin/float_test.exe
```

Results:
- ans_test.exe: 4/4 PASSED (ZeroSized, BatchPointer, BatchPointerLarge, BatchStride)
- ans_statistics_test.exe: 4/4 PASSED (Histogram, Normalization_NonZero, Normalization_EqualWeight, Normalization)
- batch_prefix_sum_test.exe: 2/2 PASSED (OneLevel, TwoLevel)
- float_test.exe: 3/3 PASSED (Batch, LargeBatch, BatchSize1; fp16/bf16/fp32)

Total: 13/13 PASS. Test time: ~11s (timeit, includes LargeBatch 6s).

### Tally vs gfx90a and gfx1100 @ b6e0d3f

| Test binary           | gfx90a | gfx1100 | gfx1151 |
|-----------------------|--------|---------|---------|
| ans_test              | 4/4    | 4/4     | 4/4     |
| ans_statistics_test   | 4/4    | 4/4     | 4/4     |
| batch_prefix_sum_test | 2/2    | 2/2     | 2/2     |
| float_test            | 3/3    | 3/3     | 3/3     |
| **Total**             | 13/13  | 13/13   | 13/13   |

### Wave32 verdict (gfx1151)

PASS. Same semantics as gfx1100 (both wave32, both RDNA3-family): device kWarpSize=32
(GFX9->64 else 32 guard in DeviceDefs.cuh, device-side only), fixed-64-slot archive
header (kMaxWarpSize=64), host grid divisor via runtime warpSize query. All round-trip
tests assert EXPECT_EQ(orig, dec) and pass. No HIP faults, clean exit.

Transitioning windows-gfx1151 to completed (validated_sha=64c792d).

## Revalidation 2026-06-04 (linux-gfx1100, binary-equivalence carry-forward)

Platform: linux-gfx1100, AMD Radeon Pro W7800 48GB (gfx1100, RDNA3, wave32), ROCm 7.2.1.
Previous validated_sha: b6e0d3f. New HEAD: 64c792d.
Delta: one commit -- "Enable symbol export for Windows DLL builds". The change adds
`if(WIN32) set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON) endif()` inside the existing
`if(USE_HIP)` block in CMakeLists.txt. On Linux, WIN32 is false; this block never
executes, so no compiler flags, no source files, and no device code are affected.

### Binary-equivalence check

Built both b6e0d3f and 64c792d for gfx1100 (`-DCMAKE_HIP_ARCHITECTURES=gfx1100`), each
in an isolated build dir. Ran `utils/codeobj_diff.py` on the two trees:

- libgpu_ans.so: **identical** (exported symbols + device ISA identical, 162 exports)
- libgpu_float_compress.so: **identical** (exported symbols + device ISA identical, 332 exports)
- libdietgpu_utils.so, libglog.so.0.6.0, libgtest*.so: host-only (no device code); roc-obj-ls
  exits non-zero ("No kernel section found") causing codeobj_diff to report indeterminate for
  these -- but manual llvm-nm confirms all three have zero symbol diffs (78/158/872/1 exports
  each, identical between the two builds). These libs carry no GPU ISA to compare.

The GPU device code libs (libgpu_ans.so, libgpu_float_compress.so) are byte-identical across
the two builds. No GPU re-run required; carrying validation forward.

Carried forward with: `moatlib.py carry-forward dietgpu linux-gfx1100 64c792d binary-equiv`
Transitioning linux-gfx1100 to completed (validated_sha=64c792d).

## Revalidation 2026-06-04 (linux-gfx90a, binary-equivalence carry-forward)

Platform: linux-gfx90a, AMD Instinct MI250X (gfx90a), ROCm 7.2.1.
Previous validated_sha: b6e0d3f. New HEAD: 64c792d.
Delta: one commit -- "Enable symbol export for Windows DLL builds". The change adds
`if(WIN32) set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON) endif()` inside the existing
`if(USE_HIP)` block in CMakeLists.txt. On Linux, WIN32 is false; this block never
executes, so no compiler flags, no source files, and no device code are affected.

### Binary-equivalence check

Built both b6e0d3f and 64c792d for gfx90a (`-DCMAKE_HIP_ARCHITECTURES=gfx90a`), each
in an isolated build dir. Ran `utils/codeobj_diff.py` on the two trees:

- libgpu_ans.so: identical (exported symbols + device ISA identical, 162 exports)
- libgpu_float_compress.so: identical (exported symbols + device ISA identical, 332 exports)
- libdietgpu_utils.so, libglog.so.0.6.0, libgtest*.so: host-only (no device code); roc-obj-ls
  exits non-zero causing codeobj_diff to report indeterminate -- same as gfx1100 observed;
  these libs carry no GPU ISA to compare.

The two GPU device code libs are device-ISA-identical across the two builds. No GPU re-run required.

Carried forward with: `moatlib.py carry-forward dietgpu linux-gfx90a 64c792d binary-equiv`
Transitioning linux-gfx90a to completed (validated_sha=64c792d).

## Revalidation 2026-06-05 (linux-gfx90a) -- squash regression fix

Platform: linux-gfx90a, AMD Instinct MI250X (gfx90a), ROCm 7.2.1, HIP_VISIBLE_DEVICES=0.
Fork: jeffdaily/dietgpu @ moat-port, SHA 8b0aec3.
State: revalidate (squashed commit 03005c7 introduced build regression) -> completed.

### Regression analysis

The squashed commit 03005c7 introduced a CMake regression: dietgpu/utils/CMakeLists.txt
referenced `hip::host` without calling `find_package(hip)`, breaking the configure step
with "Target dietgpu_utils links to: hip::host but the target was not found."

The original working commits (b6e0d3f, 64c792d) did NOT use `hip::host`. They:
1. Marked DeviceUtils.cpp and StackDeviceMemory.cpp as LANGUAGE HIP (these files call
   the HIP runtime API via the compat header)
2. Linked against ${CUDA_LIBRARIES} for both CUDA and HIP builds (HIP provides
   compatible symbols)

Fix: restored the working CMakeLists.txt pattern (set_source_files_properties LANGUAGE
HIP, single target_link_libraries using ${CUDA_LIBRARIES}). Committed as 8b0aec3 on top
of 03005c7.

### Build

```
cd /var/lib/jenkins/moat/projects/dietgpu/src
git submodule update --init --recursive
cmake -S . -B build-hip -G Ninja \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100" \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-hip --target ans_test ans_statistics_test \
  batch_prefix_sum_test float_test gpu_ans gpu_float_compress dietgpu_utils -j 16
```

Build: PASS, 34/34 targets, 0 errors, 1 pre-existing upstream warning (FloatTest.cu:306
braced-scalar-init).

### Fat binary verification

`llvm-objdump --offloading` on libgpu_ans.so shows both gfx1100 and gfx90a bundles:
- hipv4-amdgcn-amd-amdhsa--gfx1100 (3 segments)
- hipv4-amdgcn-amd-amdhsa--gfx90a (3 segments)

Multi-arch fat binary confirmed.

### GPU tests

```
export HIP_VISIBLE_DEVICES=0
cd build-hip
./bin/ans_test
./bin/ans_statistics_test
./bin/batch_prefix_sum_test
./bin/float_test
```

Results:
- ans_test: 4/4 PASSED (ZeroSized, BatchPointer, BatchPointerLarge, BatchStride)
- ans_statistics_test: 4/4 PASSED (Histogram, Normalization_NonZero, Normalization_EqualWeight, Normalization)
- batch_prefix_sum_test: 2/2 PASSED (OneLevel, TwoLevel)
- float_test: 3/3 PASSED (Batch, LargeBatch, BatchSize1; fp16/bf16/fp32)

Total: 13/13 PASS. All tests deterministic, zero HSA faults.

Transitioning linux-gfx90a to completed (validated_sha=8b0aec3).

## Validation 2026-06-04 (windows-gfx1101 + windows-gfx1201, one FAT binary) -- follower, NO source change

validated_sha: 64c792d (zero-churn followers; the CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS Windows
fix is already in this HEAD from the gfx1151 work). Host = dual-GPU Windows workstation
(memory windows-gfx1101-gfx1201-host). ROCm 7.14 / TheRock pip SDK.

### Multi-arch fat build (one binary, both GPUs)
dietgpu's CMake reads CMAKE_HIP_ARCHITECTURES; one configure with a LIST emits both archs.
Script: agent_space/dietgpu-win/build.sh.
```
ROCM=.../_rocm_sdk_devel
git submodule update --init --recursive   # glog, googletest
cmake -S . -B build-win -G Ninja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES="gfx1101;gfx1201" \
  -DCMAKE_C/CXX/HIP_COMPILER=$ROCM/lib/llvm/bin/clang(++).exe -DCMAKE_PREFIX_PATH=$ROCM \
  -DCMAKE_HIP_STANDARD=17 -DCMAKE_CXX_STANDARD=17 -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_CXX_FLAGS="-DNOMINMAX -DWIN32_LEAN_AND_MEAN" -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-win --target ans_test ans_statistics_test batch_prefix_sum_test \
  float_test gpu_ans gpu_float_compress dietgpu_utils -j64
```

### RUNTIME DLL TRAP (important, generalizable)
dietgpu's tests CRASH (silent, after `[ RUN ]`, no gtest output) when the process loads
`C:\WINDOWS\SYSTEM32\amdhip64_7.dll` (the Adrenalin driver -- present on this host). Putting
the TheRock bin dirs on PATH does NOT help: the Windows loader searches the EXE's own dir,
then System32, then PATH -- so System32's amdhip64 wins over PATH. FIX: copy TheRock's
runtime DLLs INTO the exe dir (build-win/bin) so dir-search #1 beats System32:
amdhip64_7.dll, amd_comgr.dll, rocm_kpack.dll, hiprtc0714.dll, hiprtc-builtins0714.dll
(from _rocm_sdk_core/bin). (GPUMD/cudaKDTree happened to run correctly on the System32
amdhip64 -- their AOT kernels launched fine and were oracle-verified -- but dietgpu's
runtime path does not tolerate it.) See memory windows-gfx1101-gfx1201-host.

### GPU tests (HIP_DEVICE_LIB_PATH=$ROCM/lib/llvm/amdgcn/bitcode; run per device)
| suite | gfx1101 (dev0) | gfx1201 (dev1) |
|-------|----------------|----------------|
| ans_test (ZeroSized/BatchPointer/BatchPointerLarge/BatchStride) | 4/4 | 4/4 |
| ans_statistics_test (Histogram/Normalization x3) | 4/4 | 4/4 |
| batch_prefix_sum_test (OneLevel/TwoLevel) | 2/2 | 2/2 |
| float_test (Batch/LargeBatch/BatchSize1; fp16/bf16/fp32) | 3/3 | 3/3 |
| TOTAL | 13/13 | 13/13 |

All exit 0, 0 FAILED. Matches gfx90a/gfx1100/gfx1151 (13/13). One fat binary ran on both
GPUs. State: windows-gfx1101 + windows-gfx1201 port-ready -> completed (validated_sha
64c792d, fork unchanged). All five platforms terminal -> PR-ready.
