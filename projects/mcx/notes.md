# mcx notes

## Build (gfx90a)

```bash
cd projects/mcx/src
mkdir build && cd build
cmake ../src -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
    -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
    -DBUILD_MEX=OFF -DBUILD_PYTHON=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

## Test

```bash
HIP_VISIBLE_DEVICES=0 ./bin/mcx -L
HIP_VISIBLE_DEVICES=0 ./bin/mcx --bench cube60 -n 1e6
```

## Validation notes

The core simulation works. Test suite shows 11/40+ tests fail:
- cube60 (no reflection): absorbed 17.85% -- expected ~17% (PASS)
- cube60b (reflection): absorbed 18.46% -- expected ~27% (FAIL)

The reflection boundary tests fail -- this needs investigation. It may be
a numerical/physics difference in how reflections are computed on AMD
hardware vs NVIDIA, or a bug in the port.

## ABI alignment gotcha

The Config struct uses float4/uint3/float3 types. HIP's float4 is 16-byte
aligned, but a simple C struct `{float x,y,z,w}` is 4-byte aligned. This
causes the Config struct to have different sizes when compiled with gcc
vs hipcc, leading to field offset mismatches (e.g., flog at offset 520 vs
536). The fix is to add `__align__(16)` to float4/uint4/int4 definitions
in mcx_vector_types.h.
