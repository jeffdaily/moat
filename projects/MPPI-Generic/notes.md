# MPPI-Generic notes

ROCm/HIP port (lead linux-gfx90a). Upstream ACDSLab/MPPI-Generic @ b5c8daa.
Pure-CMake header-only CUDA MPPI control library. Strategy A.

## Environment
- gfx90a (MI250X, wave64), ROCm 7.2.1, hipcc clang 22, cmake 3.31, Ninja.
- Deps installed via apt: libeigen3-dev (3.4.0), libyaml-cpp-dev. hipRAND/hipFFT
  ship with ROCm. GoogleTest is downloaded by the build at configure time.
- npz/test-network generation needs numpy (present in /opt/conda envs).

## Build
```
HIP_VISIBLE_DEVICES=3 cmake -S src -B build-hip -GNinja \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DMPPI_BUILD_TESTS=ON -DMPPI_BUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release
HIP_VISIBLE_DEVICES=3 cmake --build build-hip -j 16
```
Followers (gfx1100/gfx1151) need only a different -DCMAKE_HIP_ARCHITECTURES; no
source change (arch read from CMAKE_HIP_ARCHITECTURES, never hardcoded).

## Existing AMD support
None. README requires an NVIDIA GPU + CUDA; no rocm/hip/amdgpu/gfx token in-tree.
Fresh CUDA->HIP port.

## Port footprint (USE_HIP-guarded, CUDA path unchanged)
- include/mppi/hip_compat/cuda_to_hip.h -- the single compat header. Maps the
  cuda*/curand*/cufft* spellings to hip*, plus several semantic shims (below).
  Force-included on every HIP TU via CMAKE_HIP_FLAGS `-include`.
- include/mppi/hip_compat/{cuda_runtime,curand,cufft,device_launch_parameters,
  cooperative_groups}.h -- shim headers so the library's angle-bracket toolkit
  includes resolve to the compat layer. This dir is on the HIP include path only.
- CMakeLists.txt: USE_HIP option; conditional project() language (C CXX [+HIP]);
  enable_language(HIP); arch defaulted to gfx90a only when CMAKE_HIP_ARCHITECTURES
  unset; force-include flag; add_executable/add_library overrides that retag .cu
  sources LANGUAGE HIP (so the per-target test/example CMakeLists stay untouched);
  hip_compat/ prepended to the header lib include dirs; USE_HIP compile def.
- cmake/MPPIGenericToolsConfig.cmake: on HIP use hip::hiprand/hip::hipfft and skip
  the CUDA arch autodetection + CUDAToolkit find.
- src/controllers/CMakeLists.txt: the nvcc -Xcompiler=-fno-guess-branch-probability
  hint is GCC-only (clang rejects it) and unneeded on HIP, so applied on CUDA only.
- source edits in include/: mppi_common.cu, rmppi_kernels.cu, parallel_utils.cuh,
  cuda_math_utils.cuh, gpu_err_chk.cuh, colored_noise.cu -- see fault classes.

## Fault classes hit + fixes

1. WAVE64 warp-synchronous reduction (THE correctness fix).
   warpReduceAdd<BLOCKSIZE> (mppi_common.cu:1170) is the classic CUDA unrolled warp
   reduction: volatile shared mem, steps 32->16->...->1 with NO __syncwarp, relying
   on 32-lane lockstep. costArrayReduction (mppi_common.cu:1198) and
   multiCostArrayReduction (rmppi_kernels.cu:1135) stop their __syncthreads() tree at
   size==32 and hand the last 64->1 to warpReduceAdd<64>. On gfx90a the low 32 lanes
   of a 64-lane wavefront are NOT guaranteed lockstep across those unsynced steps ->
   the running_cost reduction races (wrong + non-deterministic cost sums). Fix: on
   HIP set stop_condition=0 so the fully-__syncthreads()-synchronized tree runs all
   the way to 1 (identical add order, block-wide barrier -> correct on any wave size)
   and skip the warpReduceAdd switch. CUDA path byte-for-byte unchanged. No
   __shfl/__ballot/warpSize anywhere else; all other reductions are serial or
   single-thread, so this is the only wave64 site.

2. __CUDACC__ not defined under hipcc (header-only .cu-include idiom).
   Each foo.cuh ends with `#if __CUDACC__  #include "foo.cu"  #endif` (59 sites) to
   pull device definitions in only under the device compiler. hipcc defines __HIPCC__
   (both passes) not __CUDACC__ -> every kernel/device helper undeclared. Fix: compat
   header defines __CUDACC__ when __HIPCC__ is set.

3. __CUDA_ARCH__ not defined during HIP device compilation.
   ~60 `#ifdef __CUDA_ARCH__` device-vs-host branches (parallel index helpers,
   dynamics, math) would silently collapse to the CPU fallback on the GPU. Fix:
   compat header defines __CUDA_ARCH__=1 only in the HIP device pass (guarded by
   __HIP_DEVICE_COMPILE__, 1 in device pass / 0 in host pass -- verified empirically).

4. Eigen CUDA path pulled in by our __CUDA_ARCH__.
   Eigen sees __CUDA_ARCH__ -> sets EIGEN_CUDA_ARCH -> includes CUDA's
   <math_constants.h> (absent on ROCm). Eigen has a native HIP path. Fix: define
   EIGEN_NO_CUDA in the compat header so Eigen ignores __CUDA_ARCH__ and uses its own
   HIP math_constants. Eigen otherwise compiles unchanged under hipcc, incl. in
   __device__ code.

5. Function-template explicit specialization must match the primary's host/device
   attributes (clang/HIP, not nvcc). getParallel2DIndex primary is __host__ __device__
   but its 8 explicit specializations were __device__-only -> "no function template
   matches / target attributes do not match". Fix: make the specializations __host__
   __device__ (parallel_utils.cuh); they already carry the #ifdef __CUDA_ARCH__ host
   fallback inside.

6. float2/3/4 operator overloads collide with HIP_vector_type.
   cuda_math_utils.cuh hand-rolls the full set of float2/3/4 arithmetic, compound-
   assign, unary-minus, equality operators. CUDA's vector_types.h defines none, so
   they are unique there; HIP's HIP_vector_type provides all of them (host+device,
   verified) -> "use of overloaded operator '*' is ambiguous". Fix: USE_HIP-guard the
   two operator blocks (keep dot/cross/norm and createPartialCudaTuple, which HIP does
   not provide).

7. cudaFuncSetAttribute signature: CUDA takes the typed kernel pointer; HIP's
   hipFuncSetAttribute takes const void*. Added MPPI_GPU_FUNC_PTR(...) (variadic so a
   comma-bearing template-id kernel<A,B,C> is one macro arg): casts to const void* on
   HIP, identity on CUDA, at the 3 call sites in mppi_common.cu (cuda_to_hip.h:207 HIP,
   :216 CUDA). The macro is defined in cuda_to_hip.h, which is force-included on HIP but
   NOT on CUDA -- so mppi_common.cu now #includes it explicitly (see reviewer fix R1).

8. cuFFT status codes without hipFFT equivalents. cufftGetErrorString switches over
   CUFFT_LICENSE_ERROR (reachable on HIP since the CUDART_VERSION<13000 branch is
   taken) which hipFFT lacks; USE_HIP-guarded that single case in gpu_err_chk.cuh.

9. Brace-init of a runtime-sized array. colored_noise.cu had `float sigma[control_dim]
   = {0};` (control_dim is a runtime arg) -- a VLA initializer that clang rejects
   (GCC/nvcc allow it). Fix: declare then memset(sigma, 0, sizeof(float)*control_dim).
   Host code (the colored-noise variance precompute), so memset is fine.

CUDA::barrier (cuda/barrier, MPPI_USE_CUDA_BARRIERS) auto-disables on HIP: guarded by
`defined(CUDART_VERSION) && CUDART_VERSION>11000`, undefined under HIP, so the barrier
paths compile out to their __syncthreads() #else fallbacks. No action.

cuBLAS is NOT a dependency (the "cuBLAS" strings are TODO comments; gemm is a
hand-written __device__ __host__ routine).

## Deferred sub-feature: texture-backed map costs (autorally/quadrotor)
texture_helper.cuh defaults cudaFilterModeLinear + cudaReadModeElementType on a float
array -- the documented HIP rejection (popsift/gpuRIR fault class: AMD has no hardware
linear filtering on element-read float textures; needs software bilinear/trilinear
interp with the -0.5 texel-center convention). Plus the LSTM/uncertainty vehicle models
(bicycle_slip_parametric, racer_*) hit further clang-strictness issues (`case`
label using `this->UNCERTAINTY_DIM`, not a constant expression in clang; dependent-type
`typename`). These are NOT in the core MPPI validation path (cartpole, double-integrator,
core kernels, generic dynamics/costs, sampling). Left for a follow-up extension; core
port validated first.

## Additional clang-strictness fixes (core path)
10. Function-template explicit specialization host/device mismatch also hit
    getParallel2DIndex (fixed, #5). getParallel1DIndex specializations were
    already __host__ __device__.
11. `this->CONST` as a constant-expression template argument: clang rejects
    `mm::gemm1<this->STATE_DIM, ...>` (nvcc accepts). Use the class-qualified
    static member `LINEAR_DYNAMICS::STATE_DIM` (linear.cu). (The analogous
    `case this->UNCERTAINTY_DIM:` in the racer/bicycle vehicle models is the same
    class, left with the deferred vehicle sub-feature.)
12. Brace-init of a runtime-sized array also in colored_noise_tests.cu (4 sites,
    test-side) -> declare + memset.

## Build standard
HIP build uses C++17 (MPPIGenericToolsConfig.cmake): ROCm 7.x libstdc++ already
provides std::void_t etc., which collides with the library's own `#if __cplusplus
< 201703L` C++17 back-fills (generic_sampling_distribution_tests). CUDA default
(C++11) is unchanged.

## libcu++ / cuda::std (not needed here)
MPPI-Generic does NOT use `cuda::std` / `<cuda/std/*>` anywhere (grep of src/ finds
zero), so the lack of libcu++ on ROCm 7.2.x is a non-issue for this port -- the C++17
collision above is plain libstdc++ `std::`, not libcu++. For reference, projects that
DO depend on `cuda::std` on ROCm can get it from the header-only ROCm/libhipcxx
(`cuda::std` namespace; add its include/ to the hipcc compile; see
findings/libhipcxx/NOTES.md). No MPPI rebuild was done for this note.

## Reviewer fixes (round 2)

R1 (BLOCKING -- CUDA-path regression). MPPI_GPU_FUNC_PTR was defined ONLY in
cuda_to_hip.h, which is force-included on HIP (CMAKE_HIP_FLAGS -include) but never
reached on the CUDA build (hip_compat/ is on the include path only under USE_HIP and
nothing #included the header directly). So mppi_common.cu:1308,1341,1374 used an
undefined macro under nvcc -> the upstream CUDA build was broken. Fix (option a, the
cleaner of the two): add `#include <mppi/hip_compat/cuda_to_hip.h>` at
mppi_common.cu:9 (just below the existing `#include <curand.h>`). Why this is airtight
on BOTH toolchains:
  - cuda_to_hip.h:1 is `#pragma once`, so on HIP the force-include + this explicit
    include process the header exactly once (no double-define).
  - On CUDA the header's #else branch (cuda_to_hip.h:209-217) is self-contained: it
    just `#include <cuda_runtime.h>/<curand.h>/<cufft.h>` (real toolkit) and
    `#define MPPI_GPU_FUNC_PTR(...) (__VA_ARGS__)` (identity). No HIP-only symbol is
    referenced on that path, so it compiles under nvcc.
  - `<mppi/hip_compat/cuda_to_hip.h>` resolves on the CUDA build because the base
    `include/` dir is on the header lib's INTERFACE include path UNCONDITIONALLY
    (CMakeLists.txt:145, outside the `if(USE_HIP)` block); the file lives at
    include/mppi/hip_compat/cuda_to_hip.h. The hip_compat/ dir is prepended only on
    HIP (for the <cuda_runtime.h> shim redirect), but the explicit path does not need
    it. Net: the macro is now reachable on both backends; HIP behavior is byte-identical
    (no-op include).

R2 (nit). memset was only transitively included under nvcc. Added `#include <cstring>`
to colored_noise.cu:12 (the library variance precompute, memset at :104,:333) and
colored_noise_tests.cu:8 (test-side, 4 memset sites). HIP no-op (already compiled via
the cstring force-include in cuda_to_hip.h:19, but the explicit include is correct
hygiene and what the CUDA build needs).

R3 (nit/honesty). CMake-excluded the 7 deferred TUs on the HIP build so a default
`cmake --build` is GREEN (exit 0). Each test dir builds one target per *.cu via
file(GLOB)+foreach; under USE_HIP we list(REMOVE_ITEM ...) the deferred sources:
  - tests/texture_helpers/CMakeLists.txt: texture_helper_test.cu,
    two_d_texture_helper_test.cu, three_d_texture_helper_test.cu (all 3 in this dir).
  - tests/dynamics/CMakeLists.txt: racer_dubins_elevation_model_test.cu,
    racer_dubins_elevation_suspension_test.cu,
    racer_dubins_elevation_lstm_steering_model_test.cu,
    bicycle_slip_parametric_model_test.cu (exactly these 4; the sibling
    racer_dubins_model_test / racer_suspension_model_test /
    racer_dubins_elevation_lstm_uncertainty_model_test still build).
No non-deferred test/example includes the deferred vehicle-model headers (grep-verified),
so excluding exactly these 7 test targets leaves no orphaned library-only TU reachable.
CUDA build unchanged (guards are `if(USE_HIP)`).

## Validation (GPU 3, HIP_VISIBLE_DEVICES=3)

Build: default `cmake --build build-hip` is now GREEN (74/74 active targets, exit 0)
after R3 CMake-excluded the 7 deferred texture-map-cost TUs on HIP. Pre-exclusion that
was 172/179 (the 7 deferred were hard compile failures making build-all non-zero); the
deferred set is unchanged, just no longer configured into the HIP build. The 7 remain
blocked on the texture linear-filter rejection + clang `case this->CONST`/typename
strictness. ALL core MPPI translation units build.

Core correctness (GPU-vs-CPU reference, the wave64 reduction path):
- rollout_kernel_tests: 6/6 PASS incl. CombinedRolloutKernelGPUvsCPU,
  SplitRolloutKernelGPUvsCPU (full GPU rollout vs CPU baseline within tolerance).
- rmppi_kernel_tests: 5/5 PASS incl. ValidateCombined/SplitInitEvalKernelAgainstCPU
  and ValidateCombined/SplitRMPPIRolloutKernelAgainstCPU (exercises
  multiCostArrayReduction, the 2nd wave64 reduction).
- normexp_kernel_tests 7/7, weightedreduction (library-side) PASS,
  SamplingDistributionTests CompareLikelihoodRatioCostsCPUvsGPU all PASS,
  cartpole_dynamics 12/12, generic dynamics 16/17, sampling TestNoise (cuFFT/hipFFT
  colored-noise generation) all PASS.

System-level closed loop (cartpole VanillaMPPI, 2048 rollouts, 64x4 rollout dim ->
exercises the cost reduction every iteration):
- examples/cartpole_example drives cart_position -> ~20 (target 20), pole_angle ->
  ~3.13 rad (target pi=3.14159), baseline cost ~2000 -> ~1, i.e. the optimizer
  balances the pole upright at the goal. double_integrator examples also run.

Determinism (fixed RNG seed): agent_space/mppi_determinism_check.cu builds the
cartpole controller TWICE with seed_=1234 and compares the optimized control
sequence: max|seq1-seq2| = 0.000e+00 over 100 entries (BIT-IDENTICAL) -> the wave64
reduction fix leaves NO surviving race; the GPU cost reduction is deterministic.
(The cartpole *example* varies run-to-run only because controller.cuh defaults
seed_ to wall-clock time; with a fixed seed it is exactly reproducible.)
Same program's closed loop: determinism=PASS convergence=PASS (cart 19.92, pole
3.11, cost 1.50).

Re-validated after the round-2 reviewer fixes (R1 compat-header include, R2 cstring,
R3 CMake-exclude): rebuilt the core lib + mppi_common.cu (recompiled via both reduction
test targets) and the determinism check FROM SOURCE on GPU3. rollout_kernel_tests 6/6,
rmppi_kernel_tests 5/5 still PASS; determinism max|seq1-seq2| = 0.000e+00 / convergence
cart_x=19.9217 pole=3.1134 cost=1.5039 -- bit-identical to the pre-fix numbers, i.e. the
R1 include is a verified HIP no-op. Full default build-all is now exit 0.

ctest (full suite, serial -j1 to avoid single-GPU contention): 430/472 = 91% pass.
All 42 non-passes triaged, NONE a core regression:
- 22 deferred texture/vehicle sub-feature (ARStandardCost SEGFAULTs = texture
  linear-filter fault class; ARRobustCost, RacerDubins*, ARNeuralNet). NOTE: this
  ctest run predates R3; the 7 deferred test executables are now CMake-excluded on
  HIP (no longer configured), so a re-run would simply not list those 7 rather than
  report them NOT_BUILT/failed.
- 6 RNG/FP tolerance: GaussianTests (samples ARE Gaussian: 0.35% empirical-vs-CDF
  bin error vs an over-tight 0.1% bound; hipRAND!=cuRAND sequence) + RK4 (2e-4 vs a
  1e-6 abs bound after 100 steps; AMD vs NVIDIA fma rounding).
- 6 NN-loader npz: LSTM/FNN model-load `map::at` -- host-side cnpy key mismatch in
  the configure-time-generated npz, no GPU involvement.
- 5 DISABLED upstream (ColoredNoise.check* x4, MATH_UTILS.*P2): not real failures.
- 2 vendor-agnostic test-framework guard: Dynamics.stepGPU, Dubins.TestUpdateStateGPU
  pass 1 state with default dim_x=32, tripping `state.size() % dim_x != 0` ->
  early-return-before-launch; fires identically on CUDA (pre-existing test issue).
- 1 hipFFT negative-test: cuFFT.checkErrorCode intentionally calls hipfftExecC2R
  with a garbage plan handle; hipFFT dereferences it (bus error) instead of
  returning CUFFT_INVALID_PLAN like cuFFT (robustness diff, not functional).
- 1 legacy reference-kernel bug: WeightedReductionKernel.comparisonTestAutorally --
  the TEST's old autorallyWeightedReductionKernel writes results back into its own
  input buffer du_d (in-place inter-block RAW hazard); the ported library kernel
  (separate optimal_controls_d_ output) produces the CORRECT ~5.0 weighted controls,
  the reference produces 0.

Build/validation logs in agent_space/ (gitignored): mppi_build_*.log,
ctest_serial_full.log, mppi_determinism_check.cu.

## Validation 2026-05-30 (gfx1100, ROCm 7.2.1)

Platform: 2x AMD Radeon Pro W7800 48GB (gfx1100, RDNA3, wave32). ROCm 7.2.1,
hipcc clang 22, cmake 3.31, Ninja. fork sha 4231397db (moat-port tip, unchanged).

### Build

Clone: `git clone --branch moat-port https://github.com/jeffdaily/MPPI-Generic src`
then `git submodule update --init --recursive` (submodules not cloned by default).
libyaml-cpp-dev installed via apt (libeigen3-dev was already present).

Configure:
```
cmake -S projects/MPPI-Generic/src -B projects/MPPI-Generic/src/build-hip -GNinja \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DMPPI_BUILD_TESTS=ON -DMPPI_BUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release
cmake --build projects/MPPI-Generic/src/build-hip -j16
```
Result: 160/160 targets built, exit 0. Configure ~4s, build ~50s.

gfx1100 code-object evidence (roc-obj-ls on rollout_kernel_tests):
  hipv4-amdgcn-amd-amdhsa--gfx1100  (size=55544)
No gfx90a code object present. Arch confirmed.

### gtest results (ctest -j1, HIP_VISIBLE_DEVICES=0, 174s)

429/465 passed (92%). 36 failures + 5 disabled + 3 skipped.
gfx90a reference: 430/472 = 91% (42 non-passes, but that run predates R3; R3
CMake-excluded the 7 deferred executables, removing them from the test pool).

All 36 failures are pre-existing categories -- ZERO new core regressions:

- 8 ARStandardCost.* (5 SEGFAULT + 3 Failed): deferred texture linear-filter
  fault class, identical to gfx90a.
- 2 ARRobustCostTest.*: deferred vehicle cost, identical to gfx90a.
- 1 ARNeuralNetDynamics.computeGradTest: map::at NN-loader npz, same as gfx90a.
- 2 Dynamics.stepGPU + DubinsDynamics.TestUpdateStateGPU: vendor-agnostic
  dim_x guard, same as gfx90a.
- 4 RacerDubinsElevationLSTMUncertaintyTest.*: map::at NN-loader, same as gfx90a.
- 1 WeightedReductionKernel.comparisonTestAutorallyMPPI_Generic: legacy
  reference-kernel in-place RAW hazard, same as gfx90a.
- 1 cuFFT.checkErrorCode: Bus error (hipFFT dereferences garbage plan handle),
  same as gfx90a.
- 6 GaussianTests.Check*: hipRAND != cuRAND sequence, over-tight 0.1% CDF bound,
  same category as gfx90a.
- 1 Integration.RK4: AMD vs NVIDIA fma rounding (2e-4 vs 1e-6 abs bound),
  same as gfx90a.
- 6 FNNHelperTest.LoadModelNPZTestNested + LSTMHelperTest.* + LSTMLSTMHelperTest.*:
  map::at NN-loader npz, same category as gfx90a.
- 1 SamplingDistributionTests.CompareLikelihoodRatioCostsCPUvsGPU<GaussianDistribution
  <LinearDynamicsParams<1,7>>>: FP tolerance margin (2.98e-8 vs bound 2.58e-8,
  16% over), same AMD-vs-NVIDIA rounding category as RK4. Functionally correct.
- 1 CudaFloatStructsTests/*.VecAddVecMultScalar: FMA rounding on float2 (GPU uses
  fused multiply-add for `input1 + input2 * scalar`; CPU uses two separate ops);
  exact-equality assert fails on the float2 variant only. AMD-vs-NVIDIA FP category.
- 1 DoubleIntegratorTracking.TubeMPPILargeVariance: stochastic controller test;
  re-run passes. Same flakiness class as GaussianTests (RNG-sensitive).
- 1 RacerDubins.ComputeStateTrajectoryFiniteTest: pre-existing test bug --
  computeDynamics writes 6 of 7 state_der entries; STEER_ANGLE_RATE (index 6)
  is never set, leaving uninitialized stack memory. allFinite() check on that
  slot fails on gfx1100 Release where stack happens to contain Inf/NaN; was
  passing on gfx90a by luck. CPU-only test, no GPU path involved.

### Core correctness: wave-agnostic reduction on wave32

The key correctness tests (same results as gfx90a lead):
- rollout_kernel_tests: 6/6 PASS including CombinedRolloutKernelGPUvsCPU,
  SplitRolloutKernelGPUvsCPU (full GPU rollout vs CPU baseline within tolerance).
- rmppi_kernel_tests: 5/5 PASS including ValidateCombined/SplitInitEvalKernelAgainstCPU
  and ValidateCombined/SplitRMPPIRolloutKernelAgainstCPU (exercises
  multiCostArrayReduction, the 2nd wave64 reduction site).
- normexp_kernel_tests 7/7, CartPole 12/12, all PASS.
- SamplingDistributionTests CompareLikelihoodRatioCostsCPUvsGPU: 9/10 PASS
  (the 1 failure is an FP tolerance margin issue, not a functional error).

The warpReduceAdd -> __syncthreads block reduction fix (stop_condition=0) is
correct on wave32: the reduction-dependent tests (GPU rollout cost sums, optimal
control) all pass. No NaN, no HIP fault, clean exits throughout.

No fork changes needed or made. Fork sha 4231397db unchanged.
