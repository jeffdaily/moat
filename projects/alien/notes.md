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
2. cg::reduce in the SPH fluid kernel (ObjectProcessor.cuh, 9 calls across 2 functions). HIP CG (7.2) has no cg::reduce / cg::plus. Implemented both in the `cooperative_groups/reduce.h` shim: `reduce(tile, v, op)` is a shfl_xor butterfly over `tile.size()` (==32 static); `plus/greater/less` functors. `tiled_partition<32>` slices the wavefront into fixed 32-lane tiles (tile lane == workgroup_lane & 31, no lane repacking), and every tile shuffle is tile-relative (width == tile size), so the code is identical on wave32 and wave64. The fluid block size is `(ceil(smoothingLength*2)*2+1)^2`; with the default smoothingLength=0.8 (passed raw, never rounded to a wave multiple) that is 25 (inner) and 81 (boundary), so the last tile is PARTIAL (25-of-32; and 32/32/17). ACTIVE-LANE HARDENING (review 2026-05-31): the resident threads are the contiguous prefix [0,A) of each tile; a plain shfl_xor butterfly is an all-reduce in which lane 0 would fold the non-resident lanes' undefined value registers into the sum unless a shuffle of a non-resident lane returns 0 -- unspecified at the HIP source level. The shim removes that dependency by substituting the reduction identity (0 for plus) for any partner read from a non-resident lane (`(rank ^ offset) >= A`, A computed from blockDim and the tile base). This yields the SAME XOR-butterfly summation tree, in the same order, as a butterfly whose non-resident lanes held the identity, so lane 0 -- the ONLY lane the call sites consume (it alone feeds atomicAdd_block) -- produces precisely the resident-lane reduction, bit-for-bit equal to the previous behaviour where it held and provably independent of inactive-lane shuffle behaviour. Full tiles (A==32) reduce to the plain butterfly. Correct on wave32 AND wave64; call sites untouched; CUDA keeps real cg::reduce. (Lanes A..31 hold partial values and must not be read; the call sites only use lane 0.)
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

### EngineTests tally (gfx90a, ROCm 7.2.1)
Validated on a single MI250X GCD (HIP_VISIBLE_DEVICES pinned, serial). Both the direct-launch (-d) and graph-capture paths were run over the ENTIRE suite (minus GeometryTests + NeuronPerformanceTests) and match exactly:
- FULL -d run (incl. the 3 extreme long-runners): 2978 ran, 2973 PASSED, 3 SKIPPED (gtest GTEST_SKIP, unrelated), 2 FAILED (the two analyzed below). ~28.5 min wall (-d adds a cudaDeviceSynchronize after every kernel).
- FULL graph-path run (incl. the 3 extreme long-runners): 2978 ran, 2973 PASSED, 3 SKIPPED, 2 FAILED -- byte-for-byte the same tally and the same 2 failing tests. ~26.8 min.
- (Broad runs that skip the 3 long-runners for quick turnaround agree: 2975 ran, 2970 PASSED, 2 FAILED, in both -d and graph.)
- The 3 extreme long-runners pass: the two BalanceTests.longRunning_* (20000 timesteps each) and ConstructorTests.regressionTestMassiveReplicationsWithSeeds (10000 steps, external energy 1e7 -> exploding self-replicator population, drops to ~22 TPS as it grows; no assertions, passes on fault-free completion).
- CLI smoke (graph path, headless): 1000 timesteps, ~466 TPS, fault-free, total energy conserved.

So both paths pass the full 2978-test suite identically (2973 passed, 3 skipped, the same 2 documented non-bugs), all heavy tests pass, and the cli smoke completes.

### Two known test failures (NOT port bugs; both fit the chaotic-sim determinism bar)
1. `CommunicatorTests.sender_signalPriority_lowerNumTimesSentWins` -- FAILS deterministically on gfx90a (receiver gets numTimesSent=4 / channel=+1 instead of the expected 2 / -1) in BOTH -d and graph paths. Root cause: `CommunicatorProcessor::tryTransmitSignal` does NOT compare priorities -- it locks the receiver and unconditionally overwrites (last-writer-wins). When two senders transmit in one timestep the winner depends purely on block execution order, which differs AMD-vs-NVIDIA. Unspecified race in the upstream algorithm; the test encodes NVIDIA's incidental ordering. Kernels are correct (no fault, invariants hold).
2. `DataTransferTests.multipleCells_genome_multipleGenes_multipleNodes` -- FAILS deterministically. This is a pure set/get round-trip (no simulation steps) of a 271-cell hexagon with a multi-gene genome; `TestHelper::compare` does exact float `==` via the defaulted `operator<=>`. An instrumented round-trip (agent_space/alien_gen/diff_roundtrip.cpp) shows EVERYTHING is bit-identical (connections incl. distance/angleFromPrevious delta 0, velocity, stiffness, color, genome bytes, creatures) EXCEPT cell `pos`, which differs by max 7.6e-6 (= 2^-17, ~1 ULP at coordinate ~100) on 138/271 cells. HIP float arithmetic in the position path rounds 1 ULP off CUDA; the exact-float `==` is too strict cross-platform. Physically identical (8 significant figures). All the simpler DataTransfer round-trips (single cell, all object/node types, genome injection) pass.

### GeometryTests / GUI interop -- scoped out of headless validation
`GeometryTests` calls `glfwInit()/glfwCreateWindow/gladLoadGLLoader` and `tryCopyBuffersFromCudaToOpenGL` (the CUDA-GL interop, render path). On a headless validator (no DISPLAY) it core-dumps at GL context creation. The interop is ported mechanically (hipGraphics* 1:1) but is render-path only -- NOT hit by the headless compute path (calcTimesteps / EngineTests / cli). Re-run under xvfb (`xvfb-run`) if interop runtime proof is wanted; it does not gate sim correctness.

## Inter-project deps
None. All third-party deps are host-only via vcpkg.

## Review 2026-05-31 (reviewer; verdict: changes-requested)
Reviewed moat-port @ 765e790c vs upstream develop efbac724 via /pr-review (local-branch). Single curated commit, [ROCm] title 47 chars, Claude-disclosed, Test Plan present, no noreply/ghstack, fork Actions disabled, develop is a clean upstream mirror. NVIDIA path confirmed unchanged (every guard USE_HIP/__HIPCC__/ALIEN_NATIVE_VECTOR_OPS; the two unguarded source edits -- MuscleProcessor.cuh:180 explicit `-> float`, NeuronProcessor.cuh:95 cast to uint64_t -- are behavior-preserving on CUDA: the lambda already deduced float, and `max(3,uint32_t)` then `uint64 % v` is numerically identical for the small positive value). Math.cuh CUDA operator set net-unchanged (mixed float2-int2 moved out of the guard, still exactly one definition on CUDA). helper_cuda.h switch labels (Macros.cuh:38-60) map to 7 distinct hipError_t values (35/304/3/200/2/700/719) -- no duplicate-case. rdc wiring sound: all .cu live only in EngineKernels/EngineImpl (both get alien_mark_hip_target -fgpu-rdc + HIP_RESOLVE_DEVICE_SYMBOLS), and all 4 executables that link the engine libs get alien_link_hip_rdc (--hip-link); no consumer missed. The two test failures are correctly diagnosed and acceptable per the GPUMD/MPPI determinism lesson: CommunicatorProcessor::tryTransmitSignal (CommunicatorProcessor.cuh:143-169) takes the receiver lock and unconditionally overwrites numTimesSent with NO priority compare, so sender_signalPriority_lowerNumTimesSentWins is a genuine upstream last-writer-wins race whose winner is block-order-dependent (AMD vs NVIDIA); DataTransferTests.multipleCells_genome fails on RealVector2D's defaulted exact-float operator== (MathTypes.h:40-41) against a ~1 ULP HIP rounding delta in cell pos on a pure set/get round-trip (no timesteps) -- physically identical, exact == too strict cross-platform. GeometryTests scope-out is reasonable (GL-context render path, not the compute path).

BLOCKING finding -- cg::reduce shim relies on unspecified inactive-lane shuffle behaviour, and the written fault-class analysis does not cover it (source/hip_compat/cooperative_groups/reduce.h:38-45, call sites source/EngineKernels/ObjectProcessor.cuh:254-260,406-407):
- The two SPH fluid kernels launch with blockDim = config.fluidKernelThreads passed raw (Macros.cuh:131 STREAM_KERNEL_CALL_MOD, no rounding to a wave multiple). With the default smoothingLength=0.8 (SimulationParameters.h:68): calcOptimalThreadsForFluidKernel -> (ceil(0.8*2)*2+1)^2 = 25, and calcOptimalThreadsForFluidBoundaryKernel -> (ceil(0.8*4)*2+1)^2 = 81. So every fluid timestep the block size is NOT a multiple of 32: the fluid kernel runs as a single 25-of-32 partial tile, the boundary kernel as tiles 32,32,17. Lanes 25..31 (and 17..31) of those partial tiles are non-resident threads (past blockDim).
- The shim implements reduce as a raw tile.shfl_xor butterfly (-> __shfl_xor(value, offset, width=32), confirmed tile-relative in /opt/rocm/include/hip/amd_detail/amd_warp_functions.h:491-516; tile.size()==32 static). A descending-offset shfl_xor butterfly is an ALL-reduce: traced out, lane 0's final value sums in the value registers of the non-resident lanes too (verified by enumerating the butterfly for active counts 25 and 17 -- lane 0 ends with clean 0..A-1 PLUS contributions from lanes A..31). Only lane 0 (warp.thread_rank()==0) feeds atomicAdd_block, so its result must equal the resident-lane sum.
- This is correct ONLY if reading a non-resident lane via ds_bpermute yields 0 (those lanes never executed the `local=...{0}` init; their VGPRs are not guaranteed zero at the source level). The real CUDA cg::reduce masks to the tile's active member_mask; this shim carries NO active mask, so it depends on an unspecified-at-HIP-source ds_bpermute-of-inactive-lane = 0. The energy-conserving long-runners (20000-step BalanceTests, 1000-step cli smoke) are strong empirical evidence it does evaluate to 0 on gfx90a -- but the shim comment claims "matching CUDA cg::reduce semantics" / "correct on wave32 AND wave64" while only reasoning about the full 32-lane case, and the porter's notes (fault fix 2) likewise only argue tile-relativity. The partial-tile inactive-lane reduction is the actual load-bearing risk and is unaddressed.
- Action (pick one): (a) harden the butterfly so non-resident lanes cannot pollute the sum -- clamp the XOR partner to the tile's active-lane count (or build the reduction over the tile's coalesced active mask), making correctness independent of inactive-lane HW behaviour and identical on wave32/wave64; or (b) explicitly document and justify the reliance on ds_bpermute returning 0 for non-resident lanes with an ISA reference, AND fix the reduce.h comment + notes fault-fix-2 to state the partial-tile case (blockDim not a multiple of the tile size) rather than claiming full CUDA-semantics parity. Option (a) is preferred and cheap.

Minor (fold into the same revision):
- reduce.h:9-11 comment says "tiled_partition<32> compacts to a 32-lane sub-group of the 64-lane wavefront" -- it does not compact/repack lanes; it slices the wavefront into fixed 32-lane tiles (tile lane = workgroup lane & 31). Reword to avoid implying lane compaction.
- cuda_to_hip.h:7-8 comment references "CMAKE_HIP_FLAGS / CMAKE_CXX_FLAGS -include" but the force-include is actually via add_compile_options($<$<COMPILE_LANGUAGE:HIP|CXX>:-include...>) (CMakeLists.txt:55-56). Align the comment.
- Commit-message Test Plan names only the CommunicatorTests failure; add the DataTransferTests.multipleCells_genome 1-ULP exact-float-== failure too so the known-failures list is complete for the upstream PR.

## Review-fix 2026-05-31 (porter; changes-requested -> ported)
Addressed the single blocking finding plus the 3 nits; changed nothing else. Rebuilt incrementally and re-validated on real GPU.

BLOCKING (cg::reduce active-lane safety): chose option (a) -- hardened the reduce.h butterfly so non-resident lanes cannot pollute the sum, instead of documenting a reliance on inactive-lane HW behaviour. Approach and why it is robust:
- Substitute the reduction IDENTITY for any XOR partner read from a non-resident lane: `value = op(value, (rank ^ offset) < activeLanes ? partner : op.identity())`, where `activeLanes = A` is the resident-lane count of this tile (contiguous prefix [0,A)) computed from blockDim and the tile base. Functors gained an `identity()` (0 for plus; lowest/max for greater/less).
- Why this exact form: a descending-offset shfl_xor butterfly is an all-reduce, so lane 0 ends up summing every lane's `value`, including the non-resident lanes A..31 whose VGPRs are undefined at the HIP source level. The previous shim got the right answer ONLY because ds_bpermute happens to return 0 for inactive lanes on gfx90a -- unspecified, fragile, and not guaranteed on another arch/wave width. Identity-substitution removes that dependency entirely.
- Why it is bit-stable (honours "produce the same results"): with identity substitution lane 0 computes the EXACT same XOR-butterfly summation tree, in the same order, as a butterfly whose non-resident lanes held the identity (0). I verified by enumeration over all A in [1,32] that lane 0 with identity-substitution is bitwise-equal to the original-zeros butterfly, so the consumed result (only lane 0 feeds atomicAdd_block) is unchanged -- it just no longer depends on the hardware. Full tiles (A==32) collapse to the plain butterfly, so the non-fluid full-tile reductions and the boundary kernel's full tiles are untouched.
- wave32/wave64: everything keys off `tile.size()`==32 (static) and tile-relative shuffles; no warpSize literal; the resident-count math uses blockDim/threadIdx only. Identical on both. CUDA path uses real cg::reduce, untouched (HIP shim only).
- Caveat documented in the header: lanes A..31 hold partial values and must not be read; the call sites only ever use lane 0. (A true all-reduce via shfl_down-to-0 + broadcast was prototyped and is provably correct for all A, but it reorders the addition tree and is NOT bit-identical to the current passing result, so it was rejected in favour of the bit-stable identity-butterfly.)

NITS: reduce.h header comment reworded (tiled_partition<32> slices into fixed 32-lane tiles, no lane compaction). cuda_to_hip.h:7-8 comment aligned to the actual force-include mechanism (add_compile_options $<$<COMPILE_LANGUAGE:HIP|CXX>:-include ...>, CMakeLists.txt:73-74). Commit-message Test Plan now lists BOTH known-acceptable failures.

RE-VALIDATION (gfx90a / MI250X, ROCm 7.2.1, HIP_VISIBLE_DEVICES=3, idle GCD picked from rocm-smi; GPU 1 was busy with an unrelated run): see the EngineTests tally + cli-smoke result recorded above -- unchanged from the pre-fix baseline (same 2973/2978, same 2 documented non-bugs; energy conserved), confirming the hardening is behaviour-preserving on the SPH path.

## Review 2026-05-31 (reviewer; FOCUSED re-review of the changes-requested fix; verdict: review-passed)
Re-reviewed moat-port @ dac18fc0a vs the prior-review SHA 765e790c6 (HEAD is an amend of it; reflog confirms). The ONLY delta since the prior review is two files -- source/hip_compat/cooperative_groups/reduce.h (the hardened reduce shim) and source/EngineKernels/cuda_to_hip.h (one comment nit) -- so the rest of the port, already approved, is byte-unchanged and was not re-reviewed (per scope). Commit is still a single curated [ROCm] commit, Claude-disclosed, no noreply/ghstack, fork Actions disabled.

cg::reduce active-lane hardening: CORRECT, and the analysis is now sound. Verified independently, not just read:
- Launch dims confirmed at the source: STREAM_KERNEL_CALL_MOD -> func<<<numBlocks, threadsPerBlock, 0, stream>>> (Macros.cuh:131) is a scalar 1D blockDim; calcOptimalThreadsForFluidKernel = (ceilf(0.8*2)*2+1)^2 = 25 and calcOptimalThreadsForFluidBoundaryKernel = (ceilf(0.8*4)*2+1)^2 = 81 (SimulationKernelsService.cu:38-50). So partial tiles are 25, and 32/32/17 -- matches the shim comment exactly.
- activeLanes math (reduce.h:70) reproduced for every resident-bearing tile on wave32 AND wave64: inner tile0 activeLanes=25; boundary tiles 32/32/17; the wave64 phantom tile1 (lanes 32-63) for the 25-thread block gets activeLanes=0 (the `blockSize > tileBase` guard avoids unsigned underflow) and its lanes are non-resident so never reach the lane-0 atomicAdd. Equals the resident-lane count in every case.
- Identity-substitution correctness enumerated over all A in [1,32]: lane 0's result is exactly the resident-lane sum [0,A) with NO non-resident VGPR folded in (symbolic contribution-set enumeration).
- BIT-IDENTITY to the prior correct result confirmed in real IEEE float32 over 12000 randomized trials (A in {1,7,17,25,31,32}): NEW (identity-subst) lane0 == OLD (non-resident lanes held exactly 0.0) lane0, bit-for-bit. NB the symbolic ADDITION-TREE strings are NOT character-identical (OLD folds nested (0+0) subtrees where NEW folds a flat literal 0), but every operand NEW replaces with 0 was provably exactly 0.0 in the OLD-correct case and x+0.0f is bit-stable, so the evaluated float bits match. The porter's "same summation tree in the same order" wording is loose, but the load-bearing conclusion (bit-identical consumed result at lane 0) holds. Adversarial check: OLD lane0 changes when non-resident lanes hold garbage, NEW does not -> the hardening is genuinely load-bearing, removing a real ds_bpermute-of-inactive-lane dependency.
- wave32/wave64-identical confirmed at the HIP header: thread_block_tile::shfl_xor -> __shfl_xor(var, laneMask, numThreads) with numThreads==Size==32 static (amd_hip_cooperative_groups.h:863), and tile thread_rank() = workgroup_lane & (numThreads-1) (tile-relative 0..31); build_mask shows each tile is a contiguous 32-lane wavefront sub-range. No warpSize literal anywhere in the shim; the resident-count math uses blockDim/threadIdx only. Full tiles (A==32) collapse to the plain butterfly.
- CUDA path untouched: source/hip_compat is added to the include path only inside if(USE_HIP) (CMakeLists.txt:64), so on NVIDIA <cooperative_groups/reduce.h> resolves to the real toolkit header and this shim is never compiled. The re-review delta touches no .cu/.cuh shared source.
- op.identity(): plus->T(0) is the correct additive identity for the only functor actually instantiated (cg::plus<float>, 9 call sites in ObjectProcessor.cuh; cg::greater/less are referenced only in the comment, never instantiated -- their lowest()/max() identities are correct-but-unused). reduce.h now includes <limits> and <hip/hip_runtime.h> (added in this delta) for std::numeric_limits and the min/blockDim builtins.

Nits all resolved: reduce.h comment no longer claims lane compaction ("slices the wavefront into fixed 32-lane tiles ... no lane repacking"); cuda_to_hip.h comment now names add_compile_options($<$<COMPILE_LANGUAGE:HIP|CXX>:-include ...>) instead of the wrong CMAKE_HIP_FLAGS/CMAKE_CXX_FLAGS text (the HIP|CXX shorthand is a faithful summary of the two separate COMPILE_LANGUAGE:HIP and :CXX add_compile_options calls at CMakeLists.txt:73-74); commit-message Test Plan now lists BOTH known-acceptable failures.

GPU re-validation evidence (EngineTests 2973/2978 byte-identical to baseline in both -d and graph paths, cli energy conserved at 82500.0) is exactly what the bit-identity proof predicts. Safe to proceed to GPU validation toward completed. No blocking findings; no new nits.

## Validation 2026-05-31 (validator; linux-gfx90a -> completed)

Fork: jeffdaily/alien @ dac18fc0a93310a11ef6ad31e01651431a4af24c (moat-port). GPU: AMD Instinct MI250X GCD 2 (HIP_VISIBLE_DEVICES=2, confirmed idle via rocm-smi before run). ROCm 7.2.1. Incremental build (cmake --build with -j16) was a near-no-op (only GUI .cpp TUs rebuilt due to timestamps; all HIP TUs already up to date at dac18fc0a).

### Commands run

```
# Incremental build
cmake --build /var/lib/jenkins/moat/projects/alien/src/build -j 16

# Non-GPU regression: EngineInterfaceTests
HIP_VISIBLE_DEVICES=2 /var/lib/jenkins/moat/projects/alien/src/build/EngineInterfaceTests --gtest_filter=-GeometryTests.*

# Non-GPU regression: PersisterTests
HIP_VISIBLE_DEVICES=2 /var/lib/jenkins/moat/projects/alien/src/build/PersisterTests

# CLI smoke (graph path, headless)
HIP_VISIBLE_DEVICES=2 /var/lib/jenkins/moat/projects/alien/src/build/cli \
  -i /var/lib/jenkins/moat/agent_space/alien_gen/smoke.sim \
  -o /tmp/alien_smoke_out_val.sim -t 1000

# EngineTests full suite, graph path (GeometryTests + NeuronPerformanceTests filtered)
HIP_VISIBLE_DEVICES=2 /var/lib/jenkins/moat/projects/alien/src/build/EngineTests \
  --gtest_filter=-GeometryTests.*:-NeuronPerformanceTests.*
```

### Results

- EngineInterfaceTests: 153/153 PASSED (no regression)
- PersisterTests: 64/64 PASSED (no regression)
- CLI smoke (graph path): 1000 timesteps, 450.2 TPS, device AMD Instinct MI250X / MI250, fault-free, total energy conserved at 82500.0 for all 1000 steps
- EngineTests graph path (full suite, incl. all 3 heavy long-runners): 2978 ran, **2973 PASSED**, 3 SKIPPED, **2 FAILED** (wall time 1575566 ms / ~26.3 min)
  - Heavy long-runners all PASSED: BalanceTests.longRunning_smallCreatures_vs_largeCreatures_fewDigestionCapabilities, BalanceTests.longRunning_smallCreatures_vs_largeCreatures_highDigestionCapabilities (both 20000 steps), ConstructorTests.regressionTestMassiveReplicationsWithSeeds (10000 steps, ConstructorTests total 1299482 ms)
  - 2 FAILED (documented non-bugs, NOT regressions):
    - CommunicatorTests.sender_signalPriority_lowerNumTimesSentWins (upstream last-writer-wins race, block-order-dependent, AMD vs NVIDIA ordering)
    - DataTransferTests.multipleCells_genome_multipleGenes_multipleNodes (exact-float == on cell pos, ~1 ULP HIP rounding, no simulation steps)
  - GeometryTests: scoped out (GL context, headless validator)
  - NeuronPerformanceTests: filtered (micro-benchmarks, not a correctness gate)

VERDICT: PASS. Tally is byte-identical to the porter's documented baseline (2973/2978, same 2 documented non-bugs, energy conserved at 82500.0). No third failure. Transition: review-passed -> completed.


## Validation 2026-05-31 (gfx1100, ROCm 7.2.1)

EngineTests run directly by the orchestrator (the full both-paths suite is too long for a subagent turn). Ran the GRAPH path (normal execution) over the whole suite minus GeometryTests.* (GL context) and NeuronPerformanceTests.* -- a sufficient gate, since the lead established the -d and graph paths are byte-identical. GPU: AMD Radeon Pro W7800 (gfx1100, RDNA3, wave32), HIP_VISIBLE_DEVICES=0. roc-obj-ls on EngineTests -> hipv4-amdgcn-amd-amdhsa--gfx1100 (no gfx90a).

Tally: **2973 PASSED, 3 SKIPPED, 2 FAILED** -- matches the gfx90a graph-path bar exactly. The 2 failures are the SAME 2 documented known non-bugs (NOT wave32/port bugs, NO new failures):
- CommunicatorTests.sender_signalPriority_lowerNumTimesSentWins -- upstream last-writer-wins race in tryTransmitSignal (block-order dependent, AMD vs NVIDIA).
- DataTransferTests.multipleCells_genome_multipleGenes_multipleNodes -- ~1 ULP (7.6e-6) HIP float-rounding delta in cell pos vs exact-float == on a pure set/get round-trip; physically identical.

Wave32 verdict: the SPH fluid cg::reduce shim with PARTIAL TILES (25/81 blocks, last tile 25-of-32) is correct on wave32 -- all ObjectProcessor/fluid tests pass, and the active-lane hardening (identity for non-resident lanes; lane 0 consumed) makes lane 0's sum independent of inactive-lane shuffle behaviour. No fork change, no CI. validated_sha -> dac18fc.

## Validation attempt 2026-06-03 (windows-gfx1151) -- BLOCKED

Host: XSJJDAILYL02, AMD Radeon 8060S (gfx1151, RDNA3.5, wave32), Windows 11 10.0.26100, TheRock ROCm 0.1.0 (clang 23.0.0git, `_rocm_sdk_devel`).

Fork at start: `jeffdaily/alien moat-port @ dac18fc` (gfx1100 validated SHA). New Windows fixes commit pushed on top: `82b22c5fa3de5c06b30f4278c84f5763fe9ef379`.

### Build environment
- CMake: 4.3.2 (`C:\Users\jdaily\AppData\Local\Temp\pip-uninstall-bkgoot79\cmake.exe`)
- Compiler: `clang-cl.exe` from `_rocm_sdk_devel/lib/llvm/bin/` for C/CXX/HIP (all-clang-cl)
- MSVC: 14.50.35717 (VS2022 Community) for system headers/libs
- Windows SDK: 10.0.26100.0
- vcpkg: af752f2 (bootstrapped by manually downloading vcpkg.exe 2025-12-16 from microsoft/vcpkg-tool)
- vcpkg packages: 86 packages installed (boost, openssl, glew, glfw3, imgui, implot, gtest, etc.) from archive cache

### Configure

```
cmake -S D:/Develop/moat/projects/alien/src -B D:/Develop/moat/projects/alien/src/build -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=<src>/external/vcpkg/scripts/buildsystems/vcpkg.cmake \
  -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_C_COMPILER=<rocm>/lib/llvm/bin/clang-cl.exe \
  -DCMAKE_CXX_COMPILER=<rocm>/lib/llvm/bin/clang-cl.exe \
  -DCMAKE_HIP_COMPILER=<rocm>/lib/llvm/bin/clang-cl.exe \
  -DCMAKE_HIP_STANDARD=20 -DVCPKG_TARGET_TRIPLET=x64-windows \
  -DCMAKE_PREFIX_PATH=<rocm_root>
```

Configure: SUCCESS after 3 failed attempts that fixed LIB env var setup (Windows SDK path in LIB for lld-link), HIP package location (CMAKE_PREFIX_PATH), and clang-cl flags.

Required env:
- `LIB=<msvc>/lib/x64;<winsdk>/ucrt/x64;<winsdk>/um/x64`
- `INCLUDE=<msvc>/include;<winsdk>/ucrt;<winsdk>/um;<winsdk>/shared`
- `HIP_DEVICE_LIB_PATH=<rocm>/lib/llvm/amdgcn/bitcode`
- `VCPKG_DISABLE_METRICS=1`

### Windows-specific source fixes committed (fork 82b22c5f, on top of dac18fc)

Five classes of Windows/clang-cl build issues found and fixed:

1. **Force-include flag**: clang-cl requires `/FI<path>` (MSVC form), not `-include<path>` (GCC form). `add_compile_options(-include<path>)` is silently ignored with `-Wunknown-argument` under clang-cl; added MSVC-frontend detection post-`enable_language(HIP)` to set the correct form.

2. **`-ffast-math`**: clang-cl rejects `-ffast-math`; use `/clang:-ffast-math` for the MSVC-frontend HIP build.

3. **NOMINMAX for HIP TUs**: `add_compile_definitions(NOMINMAX)` was CXX-only. HIP TUs include Windows SDK minwindef.h (via DWIN32/D_WINDOWS injected by CMake) which defines `min`/`max` macros that conflict with `std::max` in EngineKernels.

4. **CXX force-include suppressed on Windows**: the `cuda_to_hip.h` force-include on CXX TUs pulls `hip_runtime.h` which pulls `<cmath>/<algorithm>` early. MSVC STL eagerly instantiates these templates against forward-declared project types (e.g. `ParameterSpec` in `SimulationParametersSpecification.h`), causing "incomplete type" errors. Non-engine CXX TUs (Base, EngineInterface, PersisterInterface) do not reference CUDA/HIP types; the one that does (EngineImpl/DescConverterService.cpp) is retagged LANGUAGE HIP.

5. **MSVC STL incomplete-type strictness**: `AlternativeSpec` uses `std::vector<ParameterSpec>` in `Alternatives` when `ParameterSpec` is forward-declared. MSVC STL eagerly instantiates `vector::operator=` on incomplete types (libstdc++ defers this). Fixed by declaring `AlternativeSpec::alternatives()` setters inline and defining them out-of-line in `SimulationParametersSpecification.h` after `ParameterSpec` is fully defined.

These fixes bring CXX compilation to completion: Base.lib, EngineInterface.lib, PersisterInterface.lib all build cleanly.

### Hard blocker: clang-offload-bundler -fgpu-rdc bug on Windows

**ALL HIP TU compilations fail**, 100% reproducible, at `-j1` and `-j4`:

```
clang-offload-bundler: error: 'source\EngineKernels\CMakeFiles\EngineKernels.dir\ConstantMemory.cu.obj':
The requested operation cannot be performed on a file with a user-mapped section open.
clang-cl: error: clang-offload-bundler command failed with exit code 1
```

Root cause: `clang-cl -x hip -fgpu-rdc` invokes `clang-offload-bundler.exe` to bundle device + host code into the output COFF object. On Windows, the bundler cannot write to the output file because a file section (memory mapping) is already open on it -- left open by an earlier phase of the same compile process. This is a Windows-specific bug in TheRock's `clang-offload-bundler.exe` (v23.0.0git in TheRock 0.1.0 SDK).

alien REQUIRES `-fgpu-rdc` because `ConstantMemory.cu` defines `__constant__ cudaSimulationParameters` as an extern device symbol referenced by device code in sibling EngineKernels TUs. Without `-fgpu-rdc` the device link fails with "undefined protected symbol: cudaSimulationParameters".

Alternatives investigated:
- `-j1` serial build: same error (not a parallelism race condition)
- `clang++` (GCC-driver) as HIP compiler: fails CMake HIP ABI test because CMake injects MSVC-style flags (`/DWIN32 /D_WINDOWS /Ob0`) into the GCC-driver HIP test compile
- No `--allow-overwrite` or similar flag exists in clang-offload-bundler
- TheRock 0.1.0 SDK is the only available version (pip index shows no newer build)

### Result: BLOCKED -- windows-gfx1151

GPU tests NOT run. EngineTests / CLI smoke NOT executed. Cannot validate gfx1151 GPU correctness until the clang-offload-bundler `-fgpu-rdc` bug is fixed in a newer TheRock SDK release.

The HIP port (dac18fc) is correct -- validated on gfx90a + gfx1100 at 2973/2978 PASSED. The Windows fixes (82b22c5f) address CXX toolchain compatibility but cannot overcome the device-code compilation blocker.

**Retest when**: TheRock SDK is updated to a version with a fixed `clang-offload-bundler.exe` that handles `-fgpu-rdc` on Windows without the user-mapped-section error.

## Update 2026-06-03 (windows-gfx1151): toolchain blocker SOLVED, full validation interrupted by host reboot

The `-fgpu-rdc` blocker above was root-caused and worked around -- it is NOT a "wait for a new SDK" situation. Root cause (filed upstream as ROCm/TheRock#5615): with clang-cl `/Fo<path>`, the driver compiles the host pass to `<path>` then hands `clang-offload-bundler` that same path as both `-input` and `-output`; Windows refuses to rewrite the memory-mapped file (0x4C8). It is the `/Fo` vs `-o` distinction, not object size; `-o` routes the host object to a distinct temp so input != output.

Fixes committed at fork 47ab2c9 (all WIN32 / `__has_include` guarded; Linux gfx90a/gfx1100 untouched and binary-equivalent):
1. Override `CMAKE_HIP_COMPILE_OBJECT` to use `-o` instead of `/Fo` on Windows (dodges the bundler input==output bug; `--offload-new-driver` also works and was the originally-found workaround).
2. `cmake/hip_link_win.py` wrapper overriding `CMAKE_HIP_LINK_EXECUTABLE`: the MSVC `vs_link_exe -> lld-link` rule cannot do the `-fgpu-rdc` device link; the GCC-driver `clang.exe --target=x86_64-pc-windows-msvc` can. The wrapper translates MSVC link flags (`/machine:x64`, `/subsystem:console`, bare `.lib`) to `-Xlinker` so the GCC driver passes them to lld-link, and adds `--offload-arch=gfx1151` for the device link.
3. `source/hip_compat/cooperative_groups/reduce.h`: `__has_include`-guard the shim so ROCm 7.13 uses the native `cg::reduce`/`cg::plus/greater/less` (amd_hip_cooperative_groups_reduce.h) while ROCm 7.2.x (gfx90a/gfx1100) keeps the identity-butterfly shim.

Proven on gfx1151: EngineTests.exe, PersisterTests.exe, cli.exe all compile and LINK; gfx1151 device code objects embedded in all three; one real GPU test (`AttackerTests.maxRawEnergyThreshold_belowThreshold`) executed and PASSED on hardware. So the Windows port works end to end.

NOT yet completed: the full EngineTests suite (~26 min sustained GPU on Linux; bar 2973/2978 + cli energy-conserved smoke). That run triggered a host `HYPERVISOR_ERROR (0x20001)` BSOD reboot -- a NEW failure mode vs the earlier Event-41 power-loss reboots, during a pure (serialized, no compile overlap) sustained GPU run. Remaining gap is HOST STABILITY, not the port. Plan to finish: run the suite in SHORT batched `--gtest_filter` chunks with idle gaps (no long sustained-load window), attended, after host mitigations (disable Core Isolation/HVCI+VBS so the hypervisor is out of the GPU compute path; newer KMD; TDP cap). See [[gfx1151-host-power-reboots]]. State: windows-gfx1151 = delta-ported (fixes at 47ab2c9, full GPU validation pending).
