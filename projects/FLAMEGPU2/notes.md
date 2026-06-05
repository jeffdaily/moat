# FLAMEGPU2 notes

## Status

The jeffdaily fork already contains a mature AMD/ROCm/HIP port on the `amdgpu` branch, 77 commits ahead of upstream master. This port was validated without modifications -- the `moat-port` branch is based directly on `fork/amdgpu`.

## Build instructions (gfx90a)

IMPORTANT: Must use amdclang++ as the CXX compiler, not GCC. The hip::device target includes `-x hip` in INTERFACE_COMPILE_OPTIONS which does not work with GCC.

```bash
cd projects/FLAMEGPU2/src

# Configure
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DFLAMEGPU_GPU=HIP \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_CXX_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
  -DCMAKE_C_COMPILER=/opt/rocm/lib/llvm/bin/clang \
  -DFLAMEGPU_BUILD_TESTS=ON

# Build
cmake --build build --target flamegpu boids_bruteforce tests -j$(nproc)
```

## Test results (gfx90a, ROCm 7.2.1)

- Non-RTC tests: 1069/1069 PASSED, 8 skipped (RTC-related)
- Examples: boids_bruteforce, game_of_life, circles_spatial3D all run successfully
- RTC (Runtime Compilation) is NOT supported on AMD -- marked as skipped

## Known limitations (documented in README)

| Feature               | NVIDIA GPUs  | AMD GPUs         |
|:----------------------|:-------------|:-----------------|
| Linux                 | Supported    | Supported        |
| Windows               | Supported    | Not Supported    |
| C++ AoT               | Supported    | Supported        |
| C++ RTC               | Supported    | Not Supported    |
| Python (pyflamegpu)   | Supported    | Not Supported    |
| Visualisation         | Supported    | Not Supported    |
| GLM                   | Supported    | Supported        |
| MPI                   | Supported    | Not Supported    |
