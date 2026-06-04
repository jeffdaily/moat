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
