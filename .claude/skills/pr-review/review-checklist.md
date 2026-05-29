# Port Review Checklist

Use with SKILL.md. Spawn sub-agents to verify items against the actual code. Report only violations, each with a file:line.

## 1. Port strategy
- [ ] Strategy matches the build type (PORTING_GUIDE Build classification): pure CMake -> Strategy A (compat header + LANGUAGE HIP); pytorch extension -> Strategy B (torch hipify). A correct implementation of the wrong strategy is Request Changes.
- [ ] Strategy A: a single cuda_to_hip.h compat header carries all aliases; it is a no-op on NVIDIA; there is no second HIP-aware header.
- [ ] Strategy A: .cu files are marked LANGUAGE HIP, not renamed to .hip (unless renaming is genuinely required).
- [ ] Strategy B: no hand-renamed symbols and no compat header; only fixes hipify cannot do (e.g. warp size) are in source, guarded by USE_ROCM.

## 2. Fault classes (AMD strict where CUDA lenient)
- [ ] No hardcoded warp/lane count of 32. Device code uses a per-arch constant (64 on __GFX9__, else 32); host code queries warpSize at runtime. Static shared-memory arrays use a compile-time upper bound or 64.
- [ ] Lane masks for __shfl*/__ballot/__activemask are 64-bit where the API takes a mask; no uint32 mask assumes wave32.
- [ ] Texture/stream/event RAII wrappers have rule-of-five: default-init handle to 0, move-only, guarded destructor. No double-free of a default handle.
- [ ] Kernels reading neighbor indices (+/-1, +/-width, stencils) clamp at edges; no read one-past-end relied upon.
- [ ] Pitched 2D texture binds respect 256-byte row pitch on AMD; point-sampled fetches prefer a linear bind to avoid pitch.
- [ ] Library calls swapped correctly (cuBLAS->hipBLAS, cuFFT->hipFFT, cuRAND->hipRAND, cuSPARSE->hipSPARSE, cuDNN->MIOpen, Thrust/CUB->rocThrust/hipCUB); handle types and v2-enum signature differences handled.

## 3. Minimal footprint
- [ ] Host (non-.cu) C++ is untouched except where genuinely required; the diff is small.
- [ ] `#if defined(USE_HIP)` guards are rare and only where behavior truly diverges.
- [ ] Plain CUDA spelling preserved outside the compat header (Strategy A).
- [ ] The NVIDIA/CUDA build still works; the change is additive and the CUDA path is unchanged (see bc-guidelines.md).

## 4. Arch-unified (shared branch)
- [ ] Fixes to shared (non-arch-guarded) code are correct on both wave32 and wave64; no per-arch hack that would regress the other AMD target.
- [ ] No change silently alters the CUDA-path numerics.

## 5. Build system
- [ ] enable_language(HIP) used (Strategy A); HIP gated behind a USE_HIP option (default OFF).
- [ ] HIP_ARCHITECTURES / arch flags set; the target builds for the intended gfx arch.
- [ ] ROCm dependencies are gated behind the HIP build; no new hard dependency on the CUDA/CPU build.
- [ ] CMake finds the HIP compiler (or documents CMAKE_HIP_COMPILER).

## 6. Testing
- [ ] The plan's real GPU test suite is actually run on the target arch. A port without a real-GPU run is Request Changes.
- [ ] Non-GPU tests are not regressed versus upstream.
- [ ] Any CPU-only docker smoketest is treated as a compile-only tripwire, never the validation gate.

## 7. Commit hygiene
- [ ] Title prefixed [ROCm], <= 72 chars.
- [ ] Body explains the change and mentions Claude by name; NO Co-Authored-By noreply trailer; has a Test Plan section.
- [ ] Curated history pushed with --force-with-lease to the fork; no bare --force; no ghstack.
- [ ] No references to an AMD-internal account; all under jeffdaily.
- [ ] No em-dash; ASCII only; "ROCm" casing in prose.

## 8. General code quality
- [ ] Matches the project's existing style and patterns (read the surrounding code).
- [ ] No dead code, debug prints, or commented-out blocks left behind.
- [ ] Comments carry non-obvious context, not restatement; ASCII only in new comments.
- [ ] Error handling on HIP API calls is not silently dropped.
