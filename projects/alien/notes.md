# alien notes

ROCm/HIP port of chrxh/alien (CUDA artificial-life / particle simulator). Strategy A (compat header + LANGUAGE HIP), pure CMake, NVIDIA path byte-for-byte unchanged.

## Fork / branch
- Fork: https://github.com/jeffdaily/alien (user account, not an org; `gh repo fork chrxh/alien --clone=false`). Actions disabled.
- Port branch: `moat-port` off upstream `develop` (base efbac724c171a73fb20fe80a91cee5dc93d5411a). Push with `git push --force-with-lease`.

## Build (lead arch gfx90a, ROCm 7.2.1)
Repeatable script: `projects/alien/src/build_hip.sh` (ARCH=, JOBS= overridable). Manually:

```
# one-time: vcpkg submodule must be a FULL clone (versioned ports need old git trees)
git -C projects/alien/src submodule update --init external/vcpkg
git -C projects/alien/src/external/vcpkg fetch --unshallow
bash projects/alien/src/external/vcpkg/bootstrap-vcpkg.sh -disableMetrics
# OS deps for the host GL libs vcpkg + the GUI need:
sudo apt-get install -y libgl1-mesa-dev libglu1-mesa-dev mesa-common-dev \
  libx11-dev libxcursor-dev libxrandr-dev libxinerama-dev libxi-dev libxext-dev libxfixes-dev

SRC=projects/alien/src
cmake -S $SRC -B $SRC/build -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=$SRC/external/vcpkg/scripts/buildsystems/vcpkg.cmake \
  -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
cmake --build $SRC/build -j 16
```

The same command builds gfx1100 / gfx1151 with only `-DCMAKE_HIP_ARCHITECTURES=<arch>` (no source change; the arch is defaulted only when unset and read from `${CMAKE_HIP_ARCHITECTURES}` everywhere).

Use ABSOLUTE paths for `-S/-B/-DCMAKE_TOOLCHAIN_FILE`; a relative toolchain path is resolved against the build dir and breaks the vcpkg bootstrap inside `project()`.

First configure builds ~86 host packages via vcpkg (boost, openssl, glew, glfw, imgui, ...) -- 15-25 min cold, then cached.

## Port shape (Strategy A)
- `source/EngineKernels/cuda_to_hip.h` -- the single compat header. Force-included on every HIP and host C++ TU (`-include`, set in the top-level CMakeLists under USE_HIP). Aliases the cuda* runtime/graph/GL-interop/stream symbols to hip*, plus `atomicAdd_block/atomicExch_block -> atomicAdd/atomicExch` and `cudaGraphicsMapFlagsWriteDiscard -> hipGraphicsRegisterFlagsWriteDiscard`. Defines `__HIP_PLATFORM_AMD__` for host C++ TUs (the HIP language compiler defines it automatically; plain host CXX does not, and a host .cpp pulls `<cuda_fp16.h>`). Project identifiers that merely look like CUDA names (cudaSimulationParameters, cudaSettings, cudaTO*, the cudaNextTimestep_* kernels) are deliberately NOT aliased.
- `source/hip_compat/` -- forwarding shims for the toolkit angle-bracket includes the sources use (`cuda_runtime.h`, `cuda_runtime_api.h`, `device_launch_parameters.h`, `sm_60_atomic_functions.h`, `vector_types.h`, `cuda_fp16.h`, `cuda_gl_interop.h`, `cooperative_groups.h`, `cooperative_groups/reduce.h`, `cuda/helper_cuda.h`, `cuda/helper_string.h`). On the HIP include path ONLY, prepended with `include_directories(BEFORE ...)` so `cuda/helper_cuda.h` shadows the project's vendored `external/cuda/helper_cuda.h` (which does not parse under hipcc). On CUDA the dir is absent so the real toolkit headers win.
- CMake: `option(USE_HIP)`; `enable_language(HIP)` vs CUDA; NVCC-only flags (`-lineinfo --use-local-env -use_fast_math --threads 0 --Werror=all-warnings` and `-Werror` on CXX) gated to the CUDA branch; `-ffast-math` for HIP. `alien_mark_hip_target()` retags `.cu` LANGUAGE HIP and sets HIP separable compilation on EngineKernels/EngineImpl; `alien_link_hip_rdc()` adds the device-link flags on the consuming executables (see rdc note). A stub `cudart` INTERFACE target -> `hip::host` so the subdir `target_link_libraries(... cudart)` need no edit.

## Fault fixes (the real semantic ones)
1. compute-capability gate (SimulationCudaFacade.cu:717). The `major*100+minor >= 600` check rejects AMD GPUs; guarded `#if !defined(USE_HIP)`. Device selection (highest CC) still picks a device; the throw is skipped. Verified the CLI logs "AMD Instinct MI250X / MI250 with compute capability 9.0 ... device 0 selected".
2. cg::reduce in the SPH fluid kernel (ObjectProcessor.cuh, 9 calls across 2 functions). HIP CG (7.2) has no cg::reduce / cg::plus. Implemented both in the `cooperative_groups/reduce.h` shim: `reduce(tile, v, op)` is a shfl_xor butterfly bounded by `tile.size()`, `plus/greater/less` functors. `tiled_partition<32>` is tile-relative-correct on wave64 (the shfl width is the tile size, stays within the tile's 32-lane span), and here the fluid block size is `(ceil(smoothingLength*2)*2+1)^2` (can exceed 32 -> multiple tiles/block, each tile reduces then atomicAdd_block accumulates across tiles). Correct on wave32 AND wave64; call sites untouched; CUDA keeps real cg::reduce.
3. CUDA Graphs (SimulationKernelsService.cu) -- symbol aliases only (hipStreamBeginCapture / hipGraph_t / hipGraphInstantiate(exec,graph,nullptr,nullptr,0) / hipGraphLaunch). No logic change; the graph capture+replay path works on ROCm 7.2.1 (validated by the CLI smoke: 1000 timesteps, 466 TPS, fault-free, energy conserved).
4. float2/float3/int2 operators (Math.cuh). HIP's HIP_vector_type defines component-wise +,-,*scalar,/scalar,+=,-=,*=,/=,== with the same semantics, so the project's same-signature helpers are ambiguous on HIP. Guarded the same-signature ones out with `#if !ALIEN_NATIVE_VECTOR_OPS` (= USE_HIP); kept the mixed-type `operator-(float2, int2)` (HIP has no exact overload for it -- it disambiguates the otherwise-ambiguous HIP conversion path). CUDA keeps all of them (CUDA vector types carry no operators).
5. `max(int, uint32_t)` (NeuronProcessor.cuh:95). HIP's mixed-arithmetic max overload returns `double` -> `uint64_t % double` is invalid. Cast both operands to uint64_t. Arch-neutral.
6. lambda return-type deduction (MuscleProcessor.cuh:180). clang rejects a lambda whose branches return literal `0.0f`/`1.0f` and a `min/max(...)` expression as inconsistent; added explicit `-> float`. Arch-neutral.
7. `HOST_DEVICE` macro (EngineInterface/Definitions.h:58). Was gated `#if defined(__CUDACC__)`; hipcc does not define `__CUDACC__` (it defines `__HIPCC__`), so device functions would lose `__host__ __device__`. Changed to `#if defined(__CUDACC__) || defined(__HIPCC__)`. Arch-neutral.
8. `struct cudaGraphicsResource;` forward decl (SimulationCudaFacade.cuh:28). The macro turns it into `struct hipGraphicsResource;`, which clashes with the typedef `hipGraphicsResource`. Guarded out on HIP (the HIP runtime already declares it via the force-included compat header).

## Relocatable device code (rdc) -- the load-bearing build detail
The CUDA path uses CUDA_SEPARABLE_COMPILATION + CUDA_RESOLVE_DEVICE_SYMBOLS because `__constant__ cudaSimulationParameters` (ConstantMemory.cu) is referenced by device code in sibling EngineKernels TUs. On HIP:
- CMake 3.31 does NOT reliably inject `-fgpu-rdc` from the `HIP_SEPARABLE_COMPILATION` property, so it is added explicitly to the `.cu` HIP compiles (`alien_mark_hip_target`). Without it: `lld: error: undefined protected symbol: cudaSimulationParameters` at the per-object device link.
- CMake also does NOT generate a separate HIP device-link step (unlike CUDA). With `-fgpu-rdc` the device code is relocatable in the static `.a` and the device link must happen at the EXECUTABLE link via `clang++ -fgpu-rdc --hip-link`. Without it: `ld.lld: undefined hidden symbol __hip_gpubin_handle_* / __hip_fatbin_*`. `alien_link_hip_rdc()` adds those flags to the 4 executables that link the engine libs (alien, cli, EngineTests, PersisterTests). EngineImpl's `.cu` are host-side kernel launchers (no `__global__`/`__device__` defs), so all device code lives in EngineKernels.

## GPU validation (gfx90a / MI250X, ROCm 7.2.1)
Box has 4 GCDs (0-3); always check `rocm-smi` and pin a free one with `HIP_VISIBLE_DEVICES=<n>` (do NOT trust a hardcoded device -- the app selects by highest CC and all GCDs report the same on AMD; pinning makes the chosen GCD appear as device 0).

- EngineTests: run `-d` FIRST (debug mode = direct kernel launch, bypasses graphs) then without `-d` (graph path). Run serially on one GPU (no parallel gtest). Filter out `GeometryTests.*` (needs an on-screen GL context; see below) and the `NeuronPerformanceTests.*` micro-bench.
- CLI smoke (graph path, headless): generate a sim with the throwaway generator in `agent_space/alien_gen/gen_sim.cpp` (energy particles + cells-as-creatures), then `HIP_VISIBLE_DEVICES=<n> build/cli -i smoke.sim -o out.sim -t 1000`. PASSED: 1000 steps, 466 TPS, fault-free, total energy conserved at 82500 across the run, cells 25->10 (plausible decay). NOTE: free `ObjectDesc()` cells with no creature trip `unordered_map::at` in DescConverter (creatureIdByTOIndex) -- a sim-input requirement, not a port bug; use `addCreature(...)`.

### Known platform-ordering test difference (NOT a port bug)
`CommunicatorTests.sender_signalPriority_lowerNumTimesSentWins` FAILS on gfx90a (deterministically: receiver gets numTimesSent=4 / channel=+1 instead of the expected 2 / -1). Root cause: `CommunicatorProcessor::tryTransmitSignal` does NOT compare priorities -- it locks the receiver and unconditionally overwrites (last-writer-wins). When two senders transmit in one timestep the winner depends purely on block execution order, which differs AMD-vs-NVIDIA. This is an unspecified race in the upstream algorithm; the test encodes the NVIDIA scheduler's incidental ordering. Per the chaotic-sim determinism bar (atomic-ordered accumulation, run-to-run identity not guaranteed) this is expected and acceptable; the kernels are correct (no fault, energy/physics invariants hold).

### GeometryTests / GUI interop -- scoped out of headless validation
`GeometryTests` calls `glfwInit()/glfwCreateWindow/gladLoadGLLoader` and `tryCopyBuffersFromCudaToOpenGL` (the CUDA-GL interop, render path). On a headless validator (no DISPLAY) it core-dumps at GL context creation. The interop is ported mechanically (hipGraphics* 1:1) but is render-path only -- NOT hit by the headless compute path (calcTimesteps / EngineTests / cli). Re-run under xvfb (`xvfb-run`) if interop runtime proof is wanted; it does not gate sim correctness.

## Inter-project deps
None. All third-party deps are host-only via vcpkg.
