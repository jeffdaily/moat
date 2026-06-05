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
