# visionaray notes

## Build (HIP/ROCm)

The library is header-only. Build the test with:

```bash
cd projects/visionaray/src
git submodule update --init --recursive
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DVSNRAY_ENABLE_HIP=ON \
  -DVSNRAY_ENABLE_CUDA=OFF \
  -DVSNRAY_ENABLE_UNITTESTS=OFF \
  -DVSNRAY_ENABLE_COMMON=ON \
  -DVSNRAY_ENABLE_VIEWER=OFF \
  -DVSNRAY_ENABLE_EXAMPLES=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
cmake --build build -j$(nproc)
```

Dependencies: boost, glew, freeglut, opengl, rocthrust, hipcub

## Test

```bash
HIP_VISIBLE_DEVICES=2 ./build/test/hip_test
# Expected output:
# Testing visionaray HIP support...
# Device: AMD Instinct MI250X / MI250
# Warp size: 64
# PASS: Basic HIP test succeeded
```

## Port details

- Upstream has experimental HIP support (v0.4.2) with existing hip/ headers
- Added VSNRAY_ENABLE_HIP CMake option
- Created hip_sched.h/inl mirroring cuda_sched
- Extended LBVH builder to support HIP via __HIPCC__ guards and hipCUB
- Fixed VSNRAY_GPU_MODE to detect __HIP_DEVICE_COMPILE__
- Added missing hip/managed_allocator.h and hip/managed_vector.h
- No warp intrinsics in the codebase -- no wave64/wave32 risk
- The unit tests use undefined visionaray_* CMake macros -- skipped for now

## Review 2026-06-05

**Port Review: visionaray (moat-port vs upstream/master)**

### Summary
The port extends visionaray's experimental HIP support by adding CMake HIP build option, HIP scheduler (hip_sched.h/inl), HIP LBVH builder with hipCUB, and missing managed memory headers. Strategy is correct (Strategy A variant -- extending existing HIP support). No blocking issues found; approved for validation.

### Port Correctness
No issues. The port correctly mirrors the CUDA implementations:
- hip_sched.h/inl parallels cuda_sched.h/inl with HIP API calls
- LBVH builder HIP section mirrors the CUDA section with hipCUB
- managed_allocator/managed_vector match CUDA versions

### Fault Classes
No issues. Verified:
- No warp intrinsics (`__shfl`, `__ballot`, `__activemask`) in the port -- no wave64/wave32 risk
- hipCUB DeviceMergeSort::StableSortKeys is a 1:1 API mapping from CUB
- No hardcoded warp size of 32
- No layered arrays or linear-filter float textures in the new code
- The lbvh_builder lacks rule-of-five (no move constructor/assignment), but this matches upstream CUDA version (pre-existing upstream defect, not introduced by port)

### Build System
No issues:
- enable_language(HIP) correctly used
- HIP libraries found via find_package (hip, rocthrust, hipcub)
- CMAKE_HIP_ARCHITECTURES defaults to gfx90a only when unset (correct pattern)
- ROCm deps gated behind VSNRAY_ENABLE_HIP

### Minimal Footprint
No issues:
- CUDA headers (`include/visionaray/cuda/`) unchanged
- Changes are additive and HIP-guarded (`#ifdef __HIPCC__`, `if(VSNRAY_ENABLE_HIP)`)
- macros.h change is a correct generalization (adds HIP device compile check)

### Backward Compatibility
No issues. The CUDA and CPU paths are unchanged.

### Commit Hygiene
No issues:
- Title: `[ROCm] Add HIP/ROCm support for AMD GPUs` (41 chars, under 72)
- Body mentions Claude, has Test Plan section
- No noreply trailer, no AMD-internal account references

### Testing
Note for validator: The hip_test exercises hip::device_vector and a basic kernel launch. The new hip_sched, LBVH builder, and managed_allocator are NOT directly exercised by the test. The validator should confirm the build succeeds and hip_test passes; more comprehensive tests would require porting the upstream unit test infrastructure (which uses custom visionaray_* CMake macros).

### Recommendation
**Approve** -- proceed to validation.

## Validation 2026-06-05 (linux-gfx90a)

**Build**: Configured and built successfully with CMake HIP support enabled.

**Environment**:
- GPU: AMD Instinct MI250X / MI250 (gfx90a)
- HIP_VISIBLE_DEVICES=1
- ROCm: /opt/rocm
- Warp size: 64

**Build command**:
```bash
cd /var/lib/jenkins/moat/projects/visionaray/src
git submodule update --init --recursive
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DVSNRAY_ENABLE_HIP=ON \
  -DVSNRAY_ENABLE_CUDA=OFF \
  -DVSNRAY_ENABLE_UNITTESTS=OFF \
  -DVSNRAY_ENABLE_COMMON=ON \
  -DVSNRAY_ENABLE_VIEWER=OFF \
  -DVSNRAY_ENABLE_EXAMPLES=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
cmake --build build -j$(nproc)
```

**Test command**:
```bash
HIP_VISIBLE_DEVICES=1 ./build/test/hip_test
```

**Test output**:
```
Testing visionaray HIP support...
Device: AMD Instinct MI250X / MI250
Warp size: 64
PASS: Basic HIP test succeeded
```

**Result**: PASS

The hip_test successfully runs on real GPU hardware (gfx90a), verifying:
- HIP kernel compilation and launch
- hip::device_vector allocation and data transfer
- Basic GPU computation correctness

The LBVH builder, hip_sched, and managed_allocator are not directly exercised by this test but compile successfully with hipCUB integration. Comprehensive testing of these components would require porting upstream's unit test infrastructure.

**Validated at**: 38aa60a4232970e6c0b092dbc77cd7197749f620

## Validation 2026-06-05 (linux-gfx1100)

**Build**: Configured and built successfully with CMake HIP support enabled for gfx1100.

**Environment**:
- GPU: AMD Radeon Pro W7800 48GB (gfx1100, RDNA3)
- HIP_VISIBLE_DEVICES=0
- ROCm: /opt/rocm-7.2.1
- Warp size: 32

**Build command**:
```bash
cd /var/lib/jenkins/moat/projects/visionaray/src
git submodule update --init --recursive
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DVSNRAY_ENABLE_HIP=ON \
  -DVSNRAY_ENABLE_CUDA=OFF \
  -DVSNRAY_ENABLE_UNITTESTS=OFF \
  -DVSNRAY_ENABLE_COMMON=ON \
  -DVSNRAY_ENABLE_VIEWER=OFF \
  -DVSNRAY_ENABLE_EXAMPLES=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
cmake --build build -j$(nproc)
```

**Test command**:
```bash
HIP_VISIBLE_DEVICES=0 ./build/test/hip_test
```

**Test output**:
```
Testing visionaray HIP support...
Device: AMD Radeon Pro W7800 48GB
Warp size: 32
PASS: Basic HIP test succeeded
```

**Result**: PASS

The hip_test successfully runs on real GPU hardware (gfx1100, RDNA3), verifying:
- HIP kernel compilation and launch for gfx1100
- hip::device_vector allocation and data transfer
- Basic GPU computation correctness
- Wave32 support (RDNA3 warp size is 32, vs wave64 on CDNA gfx90a)

The port correctly handles the warp size difference between CDNA (64) and RDNA3 (32), demonstrating portability across AMD architectures.

**Validated at**: 38aa60a4232970e6c0b092dbc77cd7197749f620

## Device RNG fix 2026-06-07 (audit finding visionaray-hip-device-rng)

`include/visionaray/random_generator.h` selected the thrust-backed device RNG
only under `__CUDACC__`. Under hipcc (`__HIPCC__`) `rand_engine` fell back to
`std::default_random_engine` (host-only, not device-callable), so the GPU RNG
was unavailable on ROCm: the jittered / basic_jittered_blend pixel samplers
(via make_generator) and the hip_sched render kernel construct
`random_generator<T>` on the device and could not use it.

Fix: extend BOTH guards (the `thrust/random.h` include and the
engine/distribution typedefs) to `#if defined(__CUDACC__) || defined(__HIPCC__)`,
matching the existing pattern in macros.h / math.h / lbvh.h. Used `__HIPCC__`
(not `__HIP_DEVICE_COMPILE__`) deliberately: both guards must be the identical
condition so the include and the typedef stay consistent across the host AND
device compiler passes; `random_generator` appears in host code too, so a
device-only guard would make the type differ between passes (compile error).
The CUDA path is byte-identical (NVCC still sees `__CUDACC__`).

New GPU proof: `test/hip_random_test.hip` constructs `random_generator<float>`
inside a HIP kernel, draws per-thread samples, and asserts finite + in [0,1) +
varied. Built and ran on MI250X (gfx90a, wave64), ROCm 7.2.1,
HIP_VISIBLE_DEVICES=0:

```
samples=32768 min=1.2144e-05 max=0.99997 mean=0.499811
PASS: device random_generator produced finite, varied values
```

The original hip_test still PASSes (no regression). Functional device-code
change, so advance_head flipped both completed Linux platforms to revalidate
(expected); validated_sha stays at 38aa60a4 for the regression guard.

New head_sha: d904b8b02cb386ac6f97be9774ede7fe8314a3ed

### Gotcha: porter handoff lands in `revalidate`, not `ported`

This was an audit-fix on an already-`completed` platform, not a from-scratch
port. `advance_head` correctly classifies a functional delta on a completed
platform as `revalidate` (validator re-confirms on GPU). The state machine
forbids `revalidate -> ported` by design (ported routes back through the
reviewer; revalidate routes straight to the validator). So the correct porter
handoff state here is `revalidate`, NOT `ported` -- the validator picks it up
and marks `completed` after re-running the GPU test. Do not try to force
`ported` on a post-completion functional fix.

## Validation 2026-06-07 (linux-gfx90a, revalidate at d904b8b0)

**Purpose**: Re-confirm functional device-code change -- `random_generator.h` guards extended to `|| defined(__HIPCC__)`, new `hip_random_test.hip` added.

**GPU**: AMD Instinct MI250X / MI250 (gfx90a, wave64), HIP_VISIBLE_DEVICES=0

**Build**: Incremental rebuild at d904b8b0 (CMakeLists.txt updated for hip_random_test, header change compiled into new test).

**Tests run**:

```bash
HIP_VISIBLE_DEVICES=0 ./build/test/hip_test
HIP_VISIBLE_DEVICES=0 ./build/test/hip_random_test
```

**Output**:
```
Testing visionaray HIP support...
Device: AMD Instinct MI250X / MI250
Warp size: 64
PASS: Basic HIP test succeeded
---
Testing visionaray device random_generator (HIP)...
Device: AMD Instinct MI250X / MI250
Warp size: 64
samples=32768 min=1.2144e-05 max=0.99997 mean=0.499811
PASS: device random_generator produced finite, varied values
```

**Result**: PASS (both tests). Device RNG produces finite, varied, in-range values (min~1.2e-5, max~0.99997, mean~0.500). Original hip_test shows no regression.

**Validated at**: d904b8b02cb386ac6f97be9774ede7fe8314a3ed

## Validation 2026-06-07 (linux-gfx1100, revalidate at d904b8b0)

**Purpose**: Full GPU revalidation of functional device-code change -- `random_generator.h` guards extended to `|| defined(__HIPCC__)`, new `hip_random_test.hip` added.

**GPU**: AMD Radeon Pro W7800 48GB (gfx1100, RDNA3), HIP_VISIBLE_DEVICES=0, warp size: 32

**Build**: Incremental rebuild at d904b8b0 (new hip_random_test.hip compiled for gfx1100).

```bash
bash utils/timeit.sh visionaray compile -- cmake --build /var/lib/jenkins/moat/projects/visionaray/src/build -j$(nproc)
```

**Tests run**:

```bash
bash utils/timeit.sh visionaray test -- bash -c "HIP_VISIBLE_DEVICES=0 /var/lib/jenkins/moat/projects/visionaray/src/build/test/hip_test && HIP_VISIBLE_DEVICES=0 /var/lib/jenkins/moat/projects/visionaray/src/build/test/hip_random_test"
```

**Output**:
```
Testing visionaray HIP support...
Device: AMD Radeon Pro W7800 48GB
Warp size: 32
PASS: Basic HIP test succeeded
Testing visionaray device random_generator (HIP)...
Device: AMD Radeon Pro W7800 48GB
Warp size: 32
samples=32768 min=1.2144e-05 max=0.99997 mean=0.499811
PASS: device random_generator produced finite, varied values
```

**Result**: PASS (both tests). Device RNG produces finite, varied, in-range values on gfx1100 (RDNA3 wave32), identical statistics to gfx90a. Original hip_test shows no regression.

**Validated at**: d904b8b02cb386ac6f97be9774ede7fe8314a3ed

## Validation 2026-06-08 (windows-gfx1201)

**Purpose**: First-time GPU validation for windows-gfx1201 at d904b8b0.

**GPU**: AMD Radeon RX 9070 XT (gfx1201, RDNA4, wave32), HIP_VISIBLE_DEVICES=0

**Build command**:
```bash
ROCM="B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel"
cd projects/visionaray/src && git submodule update --init --recursive
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DVSNRAY_ENABLE_HIP=ON \
  -DVSNRAY_ENABLE_CUDA=OFF \
  -DVSNRAY_ENABLE_UNITTESTS=OFF \
  -DVSNRAY_ENABLE_COMMON=OFF \
  -DVSNRAY_ENABLE_VIEWER=OFF \
  -DVSNRAY_ENABLE_EXAMPLES=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DCMAKE_HIP_COMPILER="$ROCM/lib/llvm/bin/clang++.exe" \
  -DCMAKE_C_COMPILER="$ROCM/lib/llvm/bin/clang.exe" \
  -DCMAKE_CXX_COMPILER="$ROCM/lib/llvm/bin/clang++.exe" \
  -DCMAKE_MAKE_PROGRAM="/c/Strawberry/c/bin/ninja" \
  -DCMAKE_PREFIX_PATH="$ROCM"
bash utils/timeit.sh visionaray compile -- cmake --build build -j24
```

Notes:
- Used Ninja + all-clang (clang.exe for C, clang++.exe for C++/HIP); MSVC generator rejects HIP.
- VSNRAY_ENABLE_COMMON=OFF (Boost not installed on this host; common lib not needed by test targets).
- TheRock runtime DLLs (amdhip64_7.dll, amd_comgr.dll, rocm_kpack.dll, hiprtc*.dll) copied to test/ dir to override System32 amdhip64.

**Test command**:
```bash
# TheRock runtime DLLs copied to build/test/ to override System32
bash utils/timeit.sh visionaray test -- bash -c "HIP_VISIBLE_DEVICES=0 ./build/test/hip_test.exe && HIP_VISIBLE_DEVICES=0 ./build/test/hip_random_test.exe"
```

**Test output**:
```
Testing visionaray HIP support...
Device: AMD Radeon RX 9070 XT
Warp size: 32
PASS: Basic HIP test succeeded
Testing visionaray device random_generator (HIP)...
Device: AMD Radeon RX 9070 XT
Warp size: 32
samples=32768 min=1.2144e-05 max=0.99997 mean=0.499811
PASS: device random_generator produced finite, varied values
```

**Result**: PASS (both tests). Device RNG produces identical statistics to gfx90a/gfx1100 (min~1.2e-5, max~0.99997, mean~0.500). RDNA4 wave32 handled correctly.

**Validated at**: d904b8b02cb386ac6f97be9774ede7fe8314a3ed
