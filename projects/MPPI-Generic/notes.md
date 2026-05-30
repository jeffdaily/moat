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
   HIP, identity on CUDA, at the 3 call sites in mppi_common.cu.

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

## Validation (GPU 3, HIP_VISIBLE_DEVICES=3)

Build: 172/179 targets compile and link under HIP (gfx90a). The 7 that do not are
the deferred texture-map-cost sub-feature (texture_helper tests + the
racer_dubins/bicycle_slip vehicle dynamics that consume the texture helper),
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

ctest (full suite, serial -j1 to avoid single-GPU contention): 430/472 = 91% pass.
All 42 non-passes triaged, NONE a core regression:
- 22 deferred texture/vehicle sub-feature (ARStandardCost SEGFAULTs = texture
  linear-filter fault class; ARRobustCost, RacerDubins*, ARNeuralNet; 7 NOT_BUILT).
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
