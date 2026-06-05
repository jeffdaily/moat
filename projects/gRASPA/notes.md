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

## Validation 2026-06-05 (linux-gfx90a, MI250X gfx90a)

**Build**: Clean build successful with HIP_VISIBLE_DEVICES=1
```bash
cd /var/lib/jenkins/moat/projects/gRASPA/src
mkdir build && cd build
cmake .. -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a
cmake --build . -j$(nproc)
```
Executable: `build/src_clean/graspa` (2.2MB)

**Test Results**:

CO2-MFI GCMC benchmark (Examples/CO2-MFI):
- PASS: 26 CO2 molecules adsorbed
- PASS: Zero energy drift (all components 0.00000)
- Completed successfully without errors

Methane-TMMC (Examples/Methane-TMMC):
- FAIL: Memory access fault during final energy check
- Crash at "CHECKING FINAL ENERGY FOR SYSTEM [0]" after successful simulation completion
- Error: "Memory access fault by GPU node-3 on address 0x73fd2f6bd000"
- Exit code 141 (SIGPIPE)

Tail-Correction (Examples/Tail-Correction):
- FAIL: Same memory access fault during final energy check
- Simulation completed successfully (52 seconds), crash only at final validation
- Error: "Memory access fault by GPU node-3 on address 0x7187094c6000"

**Analysis**:
The core Monte Carlo simulation works correctly. CO2-MFI completes all phases including final energy check. Larger simulations (Methane-TMMC with 63 molecules, Tail-Correction with 1300+ molecules) crash during the post-simulation final energy validation step, immediately after printing component energies in VDW_Coulomb.cu:222. The crash is consistent across GPU devices (tested on both HIP_VISIBLE_DEVICES=0 and 1, both gfx90a MI250X).

**Verdict**: VALIDATION FAILED

Bug in final energy check code for simulations with larger molecule counts. The primary CO2-MFI benchmark passes, but the broader test suite fails. Needs porter investigation of the final energy calculation routine.

## Fix 2026-06-05 (OOB memory access in TotalVDWRealCoulomb)

**Root cause**: Out-of-bounds memory access in `TotalVDWRealCoulomb` kernel (VDW_Coulomb.cu:1544-1546). The code accessed `System[compA].MolID[AtomA]` BEFORE checking if AtomA was within bounds. When thread counts exceed valid interactions (larger simulations with more molecules spawn more threads than valid atom pairs), the bounds check came too late and the OOB access triggered a memory fault on HIP/AMD.

**Fix**: Move the bounds check to before the MolID array access:
```cpp
// Before (OOB access happens before check):
MolA = System[compA].MolID[AtomA];
MolB = System[compB].MolID[AtomB];
if(AtomA >= System[compA].size || AtomB >= System[compB].size) continue;

// After (check first, then access):
if(AtomA >= System[compA].size || AtomB >= System[compB].size) continue;
MolA = System[compA].MolID[AtomA];
MolB = System[compB].MolID[AtomB];
```

This is a pre-existing bug in upstream code that happens to not crash on CUDA (likely due to different memory access fault handling) but manifests on HIP/AMD.

**Validation after fix**:
- CO2-MFI (26 molecules): PASS, zero energy drift
- Methane-TMMC (58 molecules): PASS, completes without fault
- Tail-Correction (1327 molecules): PASS, completes without fault

Commit ddf08ad pushed to jeffdaily/gRASPA moat-port branch.

## Re-Review 2026-06-05

**Scope**: Re-review following porter fix for OOB memory access in TotalVDWRealCoulomb kernel.

**Verification of OOB Fix (VDW_Coulomb.cu:1544-1546)**:
The fix is correct. The bounds check `if(AtomA >= System[compA].size || AtomB >= System[compB].size) continue;` now occurs BEFORE the MolID array accesses `MolA = System[compA].MolID[AtomA];` and `MolB = System[compB].MolID[AtomB];`. In the original code, the array access happened first, causing memory faults on larger simulations where thread counts exceed valid atom pairs.

**Fault Class Review**:
- No warp primitives (wave-size agnostic)
- No textures/surfaces
- No rule-of-five concerns
- cudaMallocManaged usage is for small control structures, no atomicMin/atomicMax on managed memory
- All reductions use __syncthreads() tree reduction

**Build System**: Correct (enable_language(HIP), USE_HIP option, parameterized CMAKE_HIP_ARCHITECTURES)

**Commit Hygiene**: Clean (36-char title, Claude disclosure, Test Plan, no noreply trailer)

**Verdict**: APPROVE -- ready for gfx90a validation.

## Re-Validation 2026-06-05 (linux-gfx90a, MI250X gfx90a, HIP_VISIBLE_DEVICES=1)

**Build**: Clean build from scratch at commit ddf08ad4208fc4a426bbf0897d6f7186878bc48e
```bash
cd /var/lib/jenkins/moat/projects/gRASPA/src/build
HIP_VISIBLE_DEVICES=1 cmake .. -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a
HIP_VISIBLE_DEVICES=1 cmake --build . -j$(nproc)
```
Executable: `build/src_clean/graspa` (2.2MB)
ROCm: 7.2.1, HIP: 7.2.53211

**Test Results** (all with HIP_VISIBLE_DEVICES=1):

1. CO2-MFI GCMC benchmark (Examples/CO2-MFI):
   - PASS: Completed successfully
   - 18 CO2 molecules adsorbed (C_co2 pseudoatoms: 18)
   - Zero energy drift (all components 0.00000)

2. Methane-TMMC (Examples/Methane-TMMC):
   - PASS: Completed successfully without crashes
   - Previously crashed with memory access fault during final energy check
   - Now completes cleanly after OOB fix

3. Tail-Correction (Examples/Tail-Correction):
   - PASS: Completed successfully without crashes
   - 1327 Argon molecules (Ar[20] pseudoatoms: 1327)
   - Previously crashed with memory access fault during final energy check
   - Now completes cleanly after OOB fix

**Verdict**: VALIDATED -- All three benchmark simulations pass. The OOB memory access fix in TotalVDWRealCoulomb kernel resolved the crashes in larger molecule count simulations. The port correctly implements HIP support with Strategy A (cuda_to_hip.h header, .cu files marked LANGUAGE HIP).

## Validation 2026-06-05 (linux-gfx1100, RDNA3 gfx1100, HIP_VISIBLE_DEVICES=2)

**Build**: Clean build from scratch at commit ddf08ad4208fc4a426bbf0897d6f7186878bc48e
```bash
cd /var/lib/jenkins/moat/projects/gRASPA/src
rm -rf build && mkdir build && cd build
HIP_VISIBLE_DEVICES=2 cmake .. -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100
HIP_VISIBLE_DEVICES=2 cmake --build . -j$(nproc)
```
Executable: `build/src_clean/graspa` (2.1MB)

**Test Results** (all with HIP_VISIBLE_DEVICES=2):

1. CO2-MFI GCMC benchmark (Examples/CO2-MFI):
   - PASS: Completed successfully
   - 18 CO2 molecules adsorbed (C_co2 pseudoatoms: 18)
   - Zero energy drift (all components 0.00000)

2. Methane-TMMC (Examples/Methane-TMMC):
   - PASS: Completed successfully
   - 58 methane molecules (CH4_sp3 pseudoatoms: 58)
   - Zero energy drift (all components 0.00000)

3. Tail-Correction (Examples/Tail-Correction):
   - PASS: Completed successfully
   - 1327 Argon molecules (Ar[20] pseudoatoms: 1327)
   - Zero energy drift (all components 0.00000)

**Verdict**: VALIDATED -- All three benchmark simulations pass on gfx1100 with identical results to gfx90a. Port works correctly across AMD architectures.
