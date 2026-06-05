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
