# gRASPA notes

## Build Instructions (HIP/ROCm)

```bash
cd projects/gRASPA/src
mkdir build && cd build
cmake .. -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a
cmake --build . -j$(nproc)
```

The executable is at `build/src_clean/graspa`.

## Validation

Tested on gfx90a (MI250X) with CO2-MFI example:
- 288 seconds runtime
- ENERGY DRIFT: 0.00000 (all components)
- 26 CO2 molecules adsorbed

## Porting Notes

### HIP vector type operators
HIP's `HIP_vector_type` (double3/int3/etc.) has built-in operators that differ from CUDA's bare struct types:
- HIP provides member operators (+=, -=, *=, /=) and friend operators (*, /)
- HIP does NOT provide free-standing operator+ or operator- 
- Custom operators in maths.cuh are conditionally compiled with `#if !defined(__HIP_PLATFORM_AMD__)` to avoid ambiguity
- Scalar operators (vector * scalar) are provided for both CUDA and HIP

### Shared memory
- HIP disallows `__shared__ bool var = false;` -- use thread-0 initialization
- Dynamic shared memory requires `extern __shared__` (same as CUDA but was missing in one kernel)

## Review 2026-06-05

**Verdict: APPROVE**

Port implements Strategy A correctly:
- Single `cuda_to_hip.h` compat header with necessary symbol mappings
- `.cu` files marked `LANGUAGE HIP` via CMake, not renamed
- HIP vector operator ambiguity correctly resolved via `#if !defined(__HIP_PLATFORM_AMD__)` guards in maths.cuh and VDW_Coulomb.cuh
- Shared memory initialization fixed correctly (thread-0 init + `__syncthreads()` in VDW_Coulomb.cu:1055-1057)
- Dynamic shared memory `extern` keyword added where missing (Ewald_Energy_Functions.h:1046)

No fault class concerns:
- No warp primitives -- all reductions use `__syncthreads()` tree reduction (wave-size agnostic)
- No textures/surfaces
- No rule-of-five concerns
- `cudaMallocManaged` usage is minimal and HIP supports it

Build system correct:
- `enable_language(HIP)` + `USE_HIP` option (default OFF)
- `CMAKE_HIP_ARCHITECTURES` parameterized (not hardcoded)
- `find_package(hip)` + `hip::host` linkage

Commit hygiene clean:
- Title follows `[ROCm]` prefix, 36 chars
- Body mentions Claude, no noreply trailer, no MOAT jargon

Ready for gfx90a validation.
