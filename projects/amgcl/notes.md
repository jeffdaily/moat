# amgcl notes

## Disposition

PORT (not already-supported). amgcl is header-only AMG with several backends.
Grep of the whole tree (`hip|rocsparse|rocthrust|rocprim|hipsparse`) returns
ZERO hits -- no native HIP/rocSPARSE backend exists upstream. AMD GPUs are
reachable today only through the OpenCL-based vexcl/viennacl backends, which per
PORTING_GUIDE does NOT count as already-supported. Value-add: a native
HIP/hipSPARSE backend mirroring the existing cuSPARSE+Thrust CUDA backend.

## Build classification

Pure CMake (no find_package(Torch), no torch.utils.cpp_extension) -> Strategy A
family. But the backend is header-only template code, so the "build" is just
compiling a TU that includes the new backend header with the HIP toolchain. No
.cu files to re-language; the example wiring copies `solver.cpp` -> `solver.hip`
and marks it `LANGUAGE HIP` (mirrors the existing `.cpp` -> `.cu` cuda example).

## Strategy: hipSPARSE shim, mirror cuda.hpp (minimal diff)

The CUDA backend uses the cuSPARSE GENERIC api (cusparseCreateCsr /
cusparseSpMV* / cusparseSpSV*) + Thrust. hipSPARSE (installed, ROCm 7.2.1)
exposes that exact api name-for-name (hipsparseCreateCsr, hipsparseSpMV*,
hipsparseSpSV*, hipsparseXcsrilu02*), so the port mirrors cuda.hpp line for line
with a mechanical cusparse->hipsparse / CUSPARSE_->HIPSPARSE_ / cudaDataType->
hipDataType / CUDA_R_*->HIP_R_* swap. Chose hipSPARSE over the lower-level
rocSPARSE because rocSPARSE's generic SpSV has no persistent descriptor (it is
stage-based), whereas hipSPARSE keeps cuSPARSE's `hipsparseSpSVDescr_t`, so the
ilu0 triangular-solve code ports unchanged in shape. Thrust stays `thrust::`
(rocThrust is a header-path-compatible drop-in under /opt/rocm/include).

## Files

New (NVIDIA path untouched):
- `amgcl/backend/hip.hpp`            -- mirror of backend/cuda.hpp
- `amgcl/relaxation/rocsparse_ilu0.hpp` -- mirror of relaxation/cusparse_ilu0.hpp

Modified (additive, guarded):
- `CMakeLists.txt`          -- `option(USE_HIP)`; enable_language(HIP),
  find_package(hipsparse/rocsparse), `hip_target` INTERFACE lib (mirrors
  cuda_target). Default lead arch only when CMAKE_HIP_ARCHITECTURES unset; never
  hardcode gfx90a literally (follower-revalidation churn rule).
- `examples/CMakeLists.txt` -- `add_hip_example` (copy .cpp -> .hip, LANGUAGE
  HIP, -DSOLVER_BACKEND_HIP); wires `solver`.
- `examples/solver.cpp`     -- `#elif defined(SOLVER_BACKEND_HIP)` arms
  (include+typedef, handle setup, two thrust copy-out branches folded into the
  CUDA branch).

## cuSPARSE -> hipSPARSE quirks hit (the real port surface)

1. hipsparseMatDescr_t AND hipsparseDnVecDescr_t are BOTH `typedef void*` in
   hipSPARSE (hipsparse-types.h:62, hipsparse-generic-types.h:52), whereas in
   cuSPARSE they are distinct opaque struct pointers. cuda.hpp's single
   `cuda_deleter` overloads `operator()` on those two types; under hipSPARSE
   those two overloads collide ("class member cannot be redeclared"). Fix:
   the backend `hip_deleter` keeps only the void* (DnVec) overload and drops the
   MatDescr one; the legacy CSR matrix descriptor (used only in the ilu0
   relaxation) gets its own `hip_mat_descr_deleter` in that header. The shared_ptr
   is `shared_ptr<void>` for both, so distinct deleter callables at .reset() are
   what disambiguate, not the type.

2. No `hipsparseXcsrilu02_bufferSize`/`_analysis`/`hipsparseXcsrilu02` generic-
   dispatch names exist (only the type-suffixed hipsparseDcsrilu02* /
   hipsparseScsrilu02*). cuSPARSE is the same; the upstream cuda code defines
   static `cusparseXcsrilu02_*` D/S overload wrappers for exactly this. Those
   wrappers were re-added (as `hipsparseXcsrilu02_*`) inside the HIP ilu0 class.
   The TYPE-agnostic `hipsparseXcsrilu02_zeroPivot` DOES exist (matches cuSPARSE).

3. The pre-CUDA-11 fallbacks in cuda.hpp / cusparse_ilu0.hpp (Hyb SpMV under
   `CUDART_VERSION < 11000`, csrsv2 triangular solves) are dropped in the HIP
   mirror: ROCm always ships the generic SpMV/SpSV api, so only the modern path
   is kept. Smaller, cleaner header.

4. SpMV algorithm: cuSPARSE `CUSPARSE_SPMV_CSR_ALG1` -> hipSPARSE
   `HIPSPARSE_SPMV_CSR_ALG1` (= rocSPARSE rowsplit, which is DETERMINISTIC per
   the rocsparse_spmv.h algorithm table). This is what gives bitwise run-to-run
   determinism in validation.

5. `hipsparseSpMV_preprocess` exists unconditionally in hipSPARSE; cuda.hpp only
   calls cusparseSpMV_preprocess under `CUDART_VERSION >= 12040`. The HIP mirror
   calls it always (valid, improves SpMV setup). No CUDART_VERSION on HIP.

6. hipEvent_t timing calls (hipEventCreate/Record/Synchronize/ElapsedTime) are
   `nodiscard` in the HIP headers (CUDA's are not), giving warnings if ignored
   like cuda.hpp does. Wrapped them in AMGCL_CALL_HIP (the header's existing
   error-check macro) instead of casting to void.

7. C++ standard: rocThrust/rocPRIM hard-#error below C++17 (general ROCm gotcha).
   `hip_target` sets `cxx_std_17`.

## Warp size

Not a factor here. The backend has NO custom device kernels with lane
assumptions -- all GPU work is hipSPARSE SpMV/SpSV + Thrust per-element
zip-iterator functors. grep of the new headers for warpSize/__shfl/__ballot/
hardcoded-32 = none. So the wave64 (gfx90a CDNA) fault class does not apply; the
port is arch-agnostic and the same headers should work on RDNA (gfx1100/gfx1151)
followers without change.

## Validation (REAL GPU, gfx90a / GPU 1, HIP_VISIBLE_DEVICES=1)

System: 7-point 3D Poisson on an n^3 grid (SPD), amgcl's own generator
(tests/sample_problem.hpp; the example's built-in problem when no -A given).
Device confirmed: "AMD Instinct MI250X / MI250".

Two validation vehicles, both pass:

A) Standalone driver (agent_space/amgcl_hip_validate.cpp, NOT in the fork):
   builds AMG(smoothed_aggregation, spai0)+CG on BOTH the HIP backend and the
   builtin CPU backend for the SAME system, n=48 (110592 unk) and n=64 (262144
   unk). Results (n=64):
     - builtin:   17 iters, resid 3.36e-11
     - hip spai0: 17 iters, resid 3.36e-11  (IDENTICAL iters + residual)
     - determinism: ||x_run1 - x_run2||inf = 0  (bitwise)
     - hip vs cpu solution: ||x_hip - x_cpu||inf / ||x_cpu||inf = 1.44e-15 (eps)
     - hip ilu0 (rocsparse_ilu0.hpp): 11 iters, resid 9.07e-11 (converges)
   Verdict line: VALIDATION PASS.

B) The project's own example built through its CMake (proves the deliverable
   build path): configure with -DUSE_HIP=ON and -DAMGCL_BUILD_EXAMPLES=ON,
   build target solver_hip. `./build-hip/examples/solver_hip` on the built-in
   Poisson:
     - spai0 relaxation: 3-level AMG, BiCGStab, 7 iters, error 5.24e-09
     - ilu0  relaxation: 3-level AMG, BiCGStab, 5 iters, error 1.40e-09
   Both converge on GPU 1.

Compile/lint was NOT treated as validation; the above are real GPU solves whose
convergence, residual, cross-backend agreement and determinism were checked.

## Build commands (recorded)

Standalone driver:
```
export HIP_VISIBLE_DEVICES=1
cd projects/amgcl/src
/opt/rocm/bin/hipcc -std=c++17 --offload-arch=gfx90a -O2 -I . -I tests \
  -Wno-deprecated-declarations \
  ../../../agent_space/amgcl_hip_validate.cpp \
  -o ../../../agent_space/amgcl_hip_validate \
  -L/opt/rocm/lib -lhipsparse -lrocsparse
../../../agent_space/amgcl_hip_validate 64
```

Project example:
```
export HIP_VISIBLE_DEVICES=1
cd projects/amgcl/src
cmake -S . -B build-hip -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
      -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
      -DCMAKE_PREFIX_PATH=/opt/rocm -DAMGCL_BUILD_EXAMPLES=ON
cmake --build build-hip --target solver_hip -j
./build-hip/examples/solver_hip -p precond.relax.type=ilu0
```

Toolchain: ROCm 7.2.1, hipSPARSE 4.2.0 / rocSPARSE 4.2.0, gfx90a (MI250X).
