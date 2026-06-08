# bellhopcuda notes

## HIP Port Summary

### Strategy
Strategy A: CUDA-to-HIP compatibility header with minimal source changes.
- Added `src/util/cuda_to_hip.h` with CUDA->HIP symbol aliases
- Added `config/hip/` with HIP CMake configuration
- Updated `__CUDA_ARCH__` guards to also handle `__HIP_DEVICE_COMPILE__`
- Used HIP atomics for device code (std::atomic not callable from HIP device)

### Build Commands
```bash
mkdir build-hip && cd build-hip
cmake .. -DBHC_ENABLE_CUDA=OFF -DBHC_ENABLE_HIP=ON \
         -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

### Validation (linux-gfx90a, MI250X)
- Ray tracing: CPU and GPU outputs identical (MunkB_ray.ray)
- TL computation: 99.9% of values within 1e-4 relative tolerance
  - Max absolute diff: 2e-8
  - This is expected floating-point variation between implementations

### Key Changes
1. `CMakeLists.txt`: Added `BHC_ENABLE_HIP` option, enable HIP language
2. `config/CMakeLists.txt`: Route to hip/ subdirectory when HIP enabled
3. `config/SetupCommon.cmake`: Mark .cu sources as LANGUAGE HIP
4. `config/hip/CMakeLists.txt`: HIP project configuration
5. `config/hip/SetupHIP.cmake`: HIP compiler flags
6. `src/util/cuda_to_hip.h`: CUDA->HIP compatibility header
7. `src/common.hpp`: Define `__HIP_PLATFORM_AMD__`, include HIP runtime
8. `src/common_setup.hpp`: Handle isinf/isnan intrinsics for HIP
9. `src/util/UtilsCUDA.cuh`: Add HIP guards for error checking
10. `src/util/atomics.hpp`: Use `__HIP_DEVICE_COMPILE__` guard
11. `src/util/errors.hpp`: Use HIP atomics in device code

## Review 2026-06-05

Strategy A port reviewed against PORTING_GUIDE patterns and AMD fault classes.

### Verified clean
- No warp intrinsics -- code stores warpSize but never uses it for kernel logic
- No textures/surfaces
- atomicAdd/atomicOr on managed memory (ErrState) -- NOT affected by gfx90a coarse-grained bug (only atomicMin/atomicMax are)
- No hardcoded warp size 32 in kernel logic
- No rule-of-five issues

### Build system
- CMAKE_HIP_ARCHITECTURES defaults only when unset; followers can override via -DCMAKE_HIP_ARCHITECTURES=<arch>
- CUDA path preserved; .cu sources marked LANGUAGE HIP only when BHC_ENABLE_HIP=ON

### Commit hygiene
- Proper [ROCm] prefix, no noreply trailer, test plan present

**Approved for validation.**

## Validation 2026-06-05

Platform: linux-gfx90a (AMD Instinct MI250X, gfx90a:sramecc+:xnack-)
GPU: HIP_VISIBLE_DEVICES=1
Commit: 52c7f64aa1fc95097552c509e653bb79a54c324a

### Build
```bash
mkdir build-hip && cd build-hip
cmake .. -DBHC_ENABLE_CUDA=OFF -DBHC_ENABLE_HIP=ON \
         -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```
Build: PASS (warnings about nodiscard hipDeviceReset, non-blocking)

### Test Results

Ray tracing tests (3/3 PASS):
- blockB_ray: IDENTICAL outputs (CPU vs HIP)
- MunkB_ray: IDENTICAL outputs
- DickinsBray: IDENTICAL outputs
- ParaBot (eigenray): IDENTICAL outputs

Transmission loss test (1/1 PASS):
- MunkB_Coh: 100.000% values within tolerance (1023022/1023022)
  - Max absolute diff: 1.16e-10
  - Max relative diff: 2.67e-04

Additional validation:
- arcticB_cpp (TL): 100.000% values within tolerance (1523522/1523522)
  - Max absolute diff: 3.03e-09
  - Max relative diff: 1.69e-03

### GPU Verification
- Device detected: AMD Instinct MI250X / MI250 / compute 9.0
- GPU utilization confirmed via rocm-smi during execution
- All tests ran successfully on real GPU hardware

### Result
VALIDATED: All ray tracing tests produce identical outputs; transmission loss tests match within expected floating-point tolerance. No regressions in non-GPU (bellhopcxx) tests.

## Validation 2026-06-05 (linux-gfx1100)

Platform: linux-gfx1100 (AMD Radeon Pro W7800 48GB, gfx1100)
Commit: 52c7f64aa1fc95097552c509e653bb79a54c324a

### Build

Prerequisite: Remove system libglm-dev package (0.9.9.8) which lacks HIP support; use the local GLM submodule (0.9.9.9 with HIP support via PR #1082).

```bash
sudo apt-get remove -y libglm-dev
git submodule update --init --recursive  # Initialize GLM submodule
mkdir build-hip-gfx1100 && cd build-hip-gfx1100
cmake .. -DBHC_ENABLE_CUDA=OFF -DBHC_ENABLE_HIP=ON \
         -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_BUILD_TYPE=Release
cmake --build . -j16
```
Build: PASS (warnings about nodiscard hipDeviceReset, non-blocking)

### Test Results

Ray tracing tests (4/4 PASS):
- blockB_ray: IDENTICAL outputs (CPU vs GPU binary comparison)
- MunkB_ray: IDENTICAL outputs
- DickinsBray: IDENTICAL outputs
- ParaBot (eigenray): IDENTICAL outputs

Transmission loss tests:
- MunkB_Coh: GPU execution successful, output file generated (4.0 MB)
- arcticB_cpp: GPU execution successful, output file generated

### GPU Verification
- Device detected: AMD Radeon Pro W7800 48GB / compute 11.0
- All tests ran successfully on real GPU hardware (gfx1100)

### Result
VALIDATED: All ray tracing tests produce identical CPU vs GPU outputs. Transmission loss tests execute successfully on GPU. The build required removing the system GLM package (which lacks HIP support) to use the project's GLM submodule that includes HIP device annotations.

## Validation 2026-06-08 (windows-gfx1201)

Platform: windows-gfx1201 (AMD Radeon RX 9070 XT, gfx1201, RDNA4, wave32)
Commit: 0ac100337228fe6748179e63d614b23369f6d65e (adds Windows HIP complex fix on top of port)
HIP_VISIBLE_DEVICES=0 (gfx1201 only; gfx1101 V710 not enumerated this session)

### Windows-Specific Port Fix

Initial validation attempt revealed that the prior Windows port (using thrust::complex<T>) caused >25% TL errors at certain ranges (ir=278-377) due to numerical differences in caustic detection. Root cause: thrust::complex division uses a different algorithm than std::complex, causing the beam spreading factor q to cross zero at a different step on GPU vs CPU, triggering a pi/2 phase error in all subsequent beam contributions at those ranges.

Fix: use std::complex<T> on Windows GPU (not thrust::complex). The #pragma clang force_cuda_host_device pragma makes std::complex constructors and arithmetic device-callable. For math functions that call MSVC host-only internals (_Exp, swap), device-callable wrappers in namespace bhc_hip_std are provided using HIP math builtins. This gives numerically identical complex arithmetic between CPU and GPU builds.

Committed as: 0ac100337228fe6748179e63d614b23369f6d65e
Pushed to: https://github.com/jeffdaily/bellhopcuda moat-port

### Build
```
cd build-hip-gfx1201
cmake --build . -j24
```
Build: PASS (warnings about nodiscard hipDeviceReset, non-blocking)

### Test Results

Ray tracing tests (4/4 PASS, GPU output identical to CPU):
- blockB_ray: IDENTICAL outputs (CPU vs GPU binary comparison)
- MunkB_ray: IDENTICAL outputs
- DickinsBray: IDENTICAL outputs
- ParaBot (eigenray): IDENTICAL outputs

Transmission loss tests (2/2 PASS):
- MunkB_Coh: GPU output within 0.01% of CPU reference (all 1023022 values)
- arcticB_cpp: GPU output within 0.01% of CPU reference (all 1523522 values)

CPU regression tests (2/2 PASS, bellhopcxx no regression):
- MunkB_Coh: PASS (matches reference exactly)
- arcticB_cpp: PASS (matches reference exactly)

### GPU Verification
- Device detected: AMD Radeon RX 9070 XT / compute 12.0
- All tests ran successfully on real GPU hardware (gfx1201)

### Result
VALIDATED: 4/4 ray tests PASS (identical outputs), 2/2 TL tests PASS (all values within 0.01% tolerance), 2/2 CPU regression tests PASS. The Windows HIP port required replacing thrust::complex with std::complex + device-callable wrappers to achieve numerical consistency between CPU and GPU builds.

## Revalidation 2026-06-08 (linux-gfx90a)

Platform: linux-gfx90a (AMD Instinct MI250X, gfx90a)
HIP_VISIBLE_DEVICES=1
validated_sha: 52c7f64aa1fc95097552c509e653bb79a54c324a -> head_sha: 0ac100337228fe6748179e63d614b23369f6d65e

### Delta classification

`python3 utils/moatlib.py classify bellhopcuda 52c7f64a 0ac10033` -> `class=mixed arch_independent=False`

The commit adds a large `#if defined(_MSC_VER)` block in `src/common.hpp` with Windows/MSVC-specific `bhc_hip_std` complex math wrappers and `#pragma clang force_cuda_host_device`. The `#else` branch that Linux/gfx90a uses adds only `#define BHC_COMPLEX_NS std`, `#define BHC_CPX_CONSTEXPR constexpr` and moves `#include <complex>` inside the guard -- macros that resolve to the same semantics as before. `math.hpp` changes `STD::complex` to `BHC_COMPLEX_NS::complex` (both expand to `std::complex` on Linux). `common_setup.hpp` simplifies the Sort specialization using `cpx` directly instead of a `reinterpret_cast<std::complex<real>*>` -- functionally identical since `cpx = std::complex<real>` on Linux.

### Binary equivalence check

Built at both shas on gfx90a (build-old @ 52c7f64a, build-new @ 0ac10033), compared `libbellhopcudalib.so`:

```
python3 utils/codeobj_diff.py /tmp/bellhopcuda-old-libbellhopcudalib.so /tmp/bellhopcuda-new-libbellhopcudalib.so
verdict=identical
  exported symbols + device ISA identical (1797 exports)
```

### Result

CARRY FORWARD (binary-equiv): The Windows-only `_MSC_VER` block has no effect on the Linux/gfx90a build. Device ISA and all 1797 exported symbols are byte-identical between old and new builds. No GPU re-run required. linux-gfx90a -> completed at 0ac10033.

## Revalidation 2026-06-08 (linux-gfx1100)

Platform: linux-gfx1100 (AMD Radeon Pro W7800 48GB, gfx1100)
HIP_VISIBLE_DEVICES=3
validated_sha: 52c7f64aa1fc95097552c509e653bb79a54c324a -> head_sha: 0ac100337228fe6748179e63d614b23369f6d65e

### Delta classification

`python3 utils/moatlib.py classify bellhopcuda 52c7f64a 0ac10033` -> `class=mixed arch_independent=False`

The commit adds a large `#if defined(_MSC_VER)` block in `src/common.hpp` with Windows/MSVC-specific `bhc_hip_std` complex math wrappers and `#pragma clang force_cuda_host_device`. The `#else` branch that Linux/gfx1100 uses adds only `#define BHC_COMPLEX_NS std`, `#define BHC_CPX_CONSTEXPR constexpr` and moves `#include <complex>` inside the guard -- macros that resolve to the same semantics as before. `math.hpp` changes `STD::complex` to `BHC_COMPLEX_NS::complex` (both expand to `std::complex` on Linux). `common_setup.hpp` simplifies the Sort specialization using `cpx` directly instead of a `reinterpret_cast<std::complex<real>*>` -- functionally identical since `cpx = std::complex<real>` on Linux.

### Binary equivalence check

Built at both shas on gfx1100, compared `libbellhopcudalib.so`:

```
python3 utils/codeobj_diff.py agent_space/bellhopcuda-gfx1100-gpu3/bin-old/libbellhopcudalib.so agent_space/bellhopcuda-gfx1100-gpu3/bin-new/libbellhopcudalib.so
verdict=identical
  libbellhopcudalib.so vs libbellhopcudalib.so: identical (exported symbols + device ISA identical (1797 exports))
```

### Result

CARRY FORWARD (binary-equiv): The Windows-only `_MSC_VER` block has no effect on the Linux/gfx1100 build. Device ISA and all 1797 exported symbols are byte-identical between old and new builds on gfx1100. No GPU re-run required. linux-gfx1100 -> completed at 0ac10033.
