# cuPDLP-C notes

## Build (HIP/ROCm)

Dependencies:
- HiGHS 1.6.0 (NOT 1.10.0 -- API mismatch causes segfault)
- ROCm with hipBLAS, hipSPARSE

```bash
# Build HiGHS 1.6.0
git clone --depth 1 --branch v1.6.0 https://github.com/ERGO-Code/HiGHS.git
cd HiGHS && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PWD/../install -DBUILD_SHARED_LIBS=ON
make -j$(nproc) && make install
export HIGHS_HOME=$PWD/../install

# Build cuPDLP-C
cd cuPDLP-C
mkdir build && cd build
cmake .. -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Test

```bash
export LD_LIBRARY_PATH=$HIGHS_HOME/lib:$PWD/lib:$LD_LIBRARY_PATH
./bin/plc -fname ../example/afiro.mps -nIterLim 5000
```

Expected output: "Solving information: Optimal current solution" with primal objective ~-464.75.

## Warp-size fix

The original code assumed warpSize=32 everywhere:
- Explicit exit(1) if warpSize != 32 in cupdlp_movement_interaction_cuda()
- `int lane = threadIdx.x % 32` and `int wid = threadIdx.x / 32` in reduction kernels
- `__shared__ cupdlp_float shared[32]` sized for 256/32=8 warps

Fixed with arch-dependent macros:
- `__GFX9__` -> wave64 (CDNA), else wave32 (RDNA/CUDA)
- `__shfl_down` for HIP (no sync needed), `__shfl_down_sync` for CUDA
- Shared memory sized to kMaxWarpsPerBlock=16 (upper bound)
- Warp-level reductions expanded to include offset 32 for wave64

## Known issues

- BUILD_APPS=ON fails due to API mismatch in onlinelp.cpp (upstream issue, not port-related)
- Requires HiGHS 1.6.0 specifically; 1.10.0 causes segfault in Init_Scaling (API incompatibility)

## Validation 2026-06-05

### Platform: linux-gfx90a (AMD Instinct MI250X / MI250)
### Arch: gfx90a (wave64, CDNA2)
### ROCm: 7.2.53211

Build:
```bash
cd /var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PWD/../install -DBUILD_SHARED_LIBS=ON
make -j$(nproc) && make install
export HIGHS_HOME=/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/install

cd /var/lib/jenkins/moat/projects/cuPDLP-C/src/build
HIP_VISIBLE_DEVICES=2 cmake .. -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Tests (HIP_VISIBLE_DEVICES=2):
```bash
LD_LIBRARY_PATH=$HIGHS_HOME/lib:$PWD/lib ./bin/testcudalin
# Output: 0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0 (PASS)

LD_LIBRARY_PATH=$HIGHS_HOME/lib:$PWD/lib ./bin/testcublas
# Output: 2-norm is :0.000000 (PASS)

LD_LIBRARY_PATH=$HIGHS_HOME/lib:$PWD/lib ./bin/plc -fname ../example/afiro.mps -nIterLim 5000
# Solving information: Optimal current solution
# Primal objective: -464.75088
# Dual objective: -464.83139
# Primal infeas: 2.69e-02 / 3.21e-05 (relative)
# Dual infeas: 3.97e-05 / 3.59e-06 (relative)
# Duality gap: 8.05e-02 / 8.65e-05 (relative)
# 200 iterations, all tolerances < 1e-4 (PASS)
```

Result: ALL TESTS PASS
- testcudalin: element-wise GPU operations correct
- testcublas: hipBLAS nrm2 correct
- plc afiro.mps: LP solver converges to OPTIMAL solution with correct objective -464.75

Wave64 reduction kernels (movement_1_kernel, movement_2_kernel, sum_kernel) work correctly on gfx90a CDNA2 with the __GFX9__ guards and warp-size-independent code.

## Validation 2026-06-05 (gfx1100)

### Platform: linux-gfx1100 (AMD Radeon Pro W7800 48GB)
### Arch: gfx1100 (wave32, RDNA3)
### ROCm: 7.0.53211

Build:
```bash
cd /var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PWD/../install -DBUILD_SHARED_LIBS=ON
make -j$(nproc) && make install
export HIGHS_HOME=/var/lib/jenkins/moat/projects/cuPDLP-C/HiGHS/install

cd /var/lib/jenkins/moat/projects/cuPDLP-C/src/build
cmake .. -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Tests:
```bash
export LD_LIBRARY_PATH=$HIGHS_HOME/lib:$PWD/lib:$LD_LIBRARY_PATH
./bin/testcudalin
# Output: 0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0 (PASS)

./bin/testcublas
# Output: 2-norm is :0.000000 (PASS)

./bin/plc -fname ../example/afiro.mps -nIterLim 5000
# Solving information: Optimal current solution
# Primal objective: -464.75089
# Dual objective: -464.83139
# Primal infeas: 2.69e-02 / 3.21e-05 (relative)
# Dual infeas: 3.97e-05 / 3.60e-06 (relative)
# Duality gap: 8.05e-02 / 8.65e-05 (relative)
# 200 iterations, all tolerances < 1e-4 (PASS)
```

Result: ALL TESTS PASS
- testcudalin: element-wise GPU operations correct
- testcublas: hipBLAS nrm2 correct
- plc afiro.mps: LP solver converges to OPTIMAL solution with correct objective -464.75

Wave32 reduction kernels work correctly on gfx1100 RDNA3. The arch-dependent warp-size macros successfully handle both wave64 (gfx90a) and wave32 (gfx1100) in a single codebase.

## Validation 2026-06-08 (windows-gfx1201)

### Platform: windows-gfx1201 (AMD Radeon RX 9070 XT, gfx1201, RDNA4, wave32)
### ROCm: 7.14.0a20260604 (TheRock pip SDK)
### GPU index: HIP_VISIBLE_DEVICES=0 (only GPU enumerated this session)

Two Windows-specific build fixes were required and committed on top of the Linux port:

1. commit ee2f874 `[ROCm] Fix Windows build: skip -lm on Windows`
   - Changed `-lm` linker flag in 3 CMakeLists.txt files to `$<$<NOT:$<PLATFORM_ID:Windows>>:m>`
   - On Windows there is no m.lib; math functions are in the C runtime

2. commit 503569a `[ROCm] Fix Windows DLL export and HIP link for shared libraries`
   - Added `WINDOWS_EXPORT_ALL_SYMBOLS ON` to cudalin, cupdlp, wrapper_lp, wrapper_highs targets
   - Added explicit `target_link_libraries(wrapper_lp/wrapper_highs PRIVATE ${HIP_LIBRARY})` since mps_lp.c uses hipMalloc via the cuda_to_hip.h shim and cupdlp does not re-export those symbols on Windows

Build (using amdclang++ toolchain via TheRock venv, all-clang, no MSVC):

```powershell
# Activate TheRock venv
source B:/develop/TheRock/external-builds/pytorch/.venv/Scripts/activate

# Build HiGHS 1.6.0
cd B:/develop/moat/projects/cuPDLP-C/HiGHS
mkdir build && cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=B:/develop/moat/projects/cuPDLP-C/HiGHS/install \
  -DBUILD_SHARED_LIBS=ON \
  -DFAST_BUILD=OFF -DBUILD_EXAMPLES=OFF \
  -DCMAKE_C_COMPILER=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel/lib/llvm/bin/clang.exe \
  -DCMAKE_CXX_COMPILER=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel/lib/llvm/bin/clang++.exe
ninja -j24 && ninja install

# Build cuPDLP-C
cd B:/develop/moat/projects/cuPDLP-C/src
mkdir build && cd build
ROCM_PATH="B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel"
cmake .. -G Ninja \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_FLAGS="--rocm-device-lib-path=${ROCM_PATH}/lib/llvm/amdgcn/bitcode" \
  -DHIGHS_HOME=B:/develop/moat/projects/cuPDLP-C/HiGHS/install \
  -DCMAKE_C_COMPILER=${ROCM_PATH}/lib/llvm/bin/clang.exe \
  -DCMAKE_CXX_COMPILER=${ROCM_PATH}/lib/llvm/bin/clang++.exe
ninja -j24

# Copy ROCm runtime DLLs into build/bin for DLL search
# (from _rocm_sdk_core and _rocm_sdk_libraries)
# Also create build/.kpack/blas_lib_gfx1201.kpack (rocsparse.dll resolves kpack at ../kpack relative to its DLL dir)
mkdir -p build/.kpack
cp _rocm_sdk_libraries/.kpack/blas_lib_gfx1201.kpack build/.kpack/
```

Tests (HIP_VISIBLE_DEVICES=0):
```python
# Run from build/bin with PATH including BINDIR + _rocm_sdk_core/bin + _rocm_sdk_libraries/bin
# ROCBLAS_TENSILE_LIBPATH=_rocm_sdk_libraries/bin/rocblas/library

testcudalin.exe
# Output: 0.000000, 1.000000, 4.000000, 9.000000, 16.000000, 25.000000, 36.000000, 49.000000, 64.000000, 81.000000 (PASS)

testcublas.exe
# Output: 2-norm is :0.000000 (PASS)

plc.exe -fname B:\...\example\afiro.mps -nIterLim 5000
# GPU device: AMD Radeon RX 9070 XT (gfx1201, RDNA4)
# Solving information: Optimal current solution
# Primal objective: -4.64750896e+02 (PASS, matches reference -464.75)
# Dual objective: -4.64831392e+02
# 200 iterations in 0.141s
```

Key Windows runtime requirement: rocsparse.dll references `../.kpack/blas_lib_gfx1201.kpack` relative to its own directory. The build/bin/ rocsparse.dll requires build/.kpack/blas_lib_gfx1201.kpack to exist. Copy from `_rocm_sdk_libraries/.kpack/blas_lib_gfx1201.kpack`.

Result: 3/3 PASS on AMD Radeon RX 9070 XT (gfx1201, RDNA4, wave32).
validated_sha = 503569a57a7a8b5ab2d9f0a0c4e6375696cc6750
