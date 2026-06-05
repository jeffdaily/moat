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

Core simulation validates correctly. Test suite: 29/40 tests pass.

Working benchmarks (verified physics results):
- cube60 (no reflection): absorbed 17.72% @ 1e7 photons -- expected ~17%
- spherebox: absorbed 10.98% @ 1e7 photons -- expected ~11%

Failing benchmarks:
- cube60b (DoMismatch=true): absorbed 18.27% @ 1e7 -- expected ~27%
  The "mismatch" flag enables internal refractive index mismatches and
  boundary reflections. With reflections, photons should bounce back into
  the medium instead of escaping, increasing total absorption. The HIP
  port shows absorption similar to the non-reflecting case, suggesting
  reflections are not being applied correctly.

The reflection logic in mcx_core.cu is complex (see the isreflect template
parameter and gcfg->doreflect paths). This needs investigation to find
where the HIP port diverges from CUDA behavior.

Other failing tests (related to reflection): cube60 -b 1, cube60 -B flags,
photon detection, saving photon seeds, photon replay.

## ABI alignment gotcha

The Config struct uses float4/uint3/float3 types. HIP's float4 is 16-byte
aligned, but a simple C struct `{float x,y,z,w}` is 4-byte aligned. This
causes the Config struct to have different sizes when compiled with gcc
vs hipcc, leading to field offset mismatches (e.g., flog at offset 520 vs
536). The fix is to add `__align__(16)` to float4/uint4/int4 definitions
in mcx_vector_types.h.
