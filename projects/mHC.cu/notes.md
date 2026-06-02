# mHC.cu notes

planner: perf-critical kernels -- assess a mechanical HIP port vs an AMD-native (rocWMMA/CK/MFMA) rewrite of the hot kernels; a correctness-first port is a valid first step.

## Port status (linux-gfx90a)
- Fork: https://github.com/jeffdaily/mHC.cu, branch moat-port, HEAD 29a9acf. Actions disabled.
- Validated deliverable: the standalone CMake test/bench harness (src/csrc). Strategy A (USE_HIP option + enable_language(HIP) + LANGUAGE HIP on the .cu, one cuda_to_hip.h shim reached via cuda_stubs/ redirect). The CUDA path is byte-identical (all changes behind USE_HIP / __HIP_PLATFORM_AMD__).
- Validation gate met on real gfx90a (MI250X, ROCm 7.2, GCD 1): multi-arch build -DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100" is clean and embeds BOTH code objects (roc-obj-ls); all 8 test_*.cu PASS vs their CPU fp32 references; sinkhorn asserts doubly-stochastic (row/col err ~1e-7); hipBLASLt loads the gfx90a Tensile library (native dispatch, AMD_LOG_LEVEL=3), not a fallback.

## Build / test (lead)
```
export HIP_VISIBLE_DEVICES=1
cmake -B build -S projects/mHC.cu/src/src/csrc -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100" \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
cmake --build build -j
for t in test_rmsnorm test_rmsnorm_backward test_sinkhorn_knopp test_mhc_layer \
         test_stream_ops test_stream_ops_backward test_fused_rmsnorm_matmul \
         test_fused_rmsnorm_matmul_backward; do build/$t; done
```

## Gotchas / lessons
- NVPTX timer: s_memrealtime exists on CDNA (gfx90a) but NOT RDNA (gfx1100) -- __builtin_amdgcn_s_memrealtime fails the gfx1100 leg of a fat build with "needs target feature s-memrealtime". Use __builtin_readcyclecounter() (portable clang builtin, lowers per-arch). get_smid -> __builtin_amdgcn_s_getreg(HW_ID) is fine on both. The profiler is if-constexpr-disabled by default; these only need to compile.
- HIP cooperative_groups has tiled_partition/shfl_xor but NEITHER cg::reduce NOR cg::plus. The shim provides both as a tile.size() shfl_xor butterfly (all-reduce semantics, wave32/wave64-correct). <cooperative_groups/reduce.h> does not exist on HIP either; the cuda_stubs redirect routes it to the shim.
- hipBLASLt does NOT honor HIPBLASLT_ORDER_ROW like cuBLASLt. The shim runs hipBLASLt native column-major: swap rows<->cols at layout create, drop the ORDER_ROW set-attribute, and swap A/B operands + TRANSA/TRANSB in BOTH cublasLtMatmul and cublasLtMatmulAlgoGetHeuristic so the heuristic and the matmul agree and C lands row-major. cublasGemmEx -> hipblasGemmEx is the fallback when the Lt heuristic returns 0.
- The cuda_to_hip.h shim file name ends in "_hip.h" -- a `rm src/**/*_hip.h` cleanup glob will delete it. It is an untracked-then-tracked porter artifact; keep it committed.

## Torch extension (Strategy B) -- NOT completed, future work
- A ROCm torch IS importable here (torch 2.13 hip 7.2). setup.py builds bindings.cu as a CUDAExtension; torch.utils.hipify renames cuda*/cublas* in the .cu/.cuh at build time.
- hipify defeats the cuda_to_hip.h shim: it RENAMES the shim's own cublasLt* wrapper functions to hipblasLt* (clashing with the real ones -> ambiguous/redeclaration) and copies cuda_to_hip.h to a hip_to_hip.h sidecar. So the standalone shim cannot be shared with the torch path.
- With a separate hipify-safe supplement (no cuda/cublas tokens: nv_bfloat16 aliases + cg::reduce/plus + a cooperative_groups/reduce.h stub) the bf16 and cg gaps close, but hipify still leaves several cuBLASLt symbols un-renamed (cublasLtOrder_t, CUBLASLT_ORDER_ROW, CUBLASLT_MATRIX_LAYOUT_ORDER) and hipFuncSetAttribute needs the const-void* cast. Finishing the extension needs per-symbol handling of those hipify coverage gaps plus replicating the row-major->col-major Lt swap on the torch path (or forcing use_lt=false to take the hipblasGemmEx fallback). Deferred; the standalone harness is the sufficient GPU correctness gate per the plan.

## Follower delta (gfx1100/gfx1151)
- Multi-arch build already proves gfx1100 codegen is clean (s-memrealtime trap fixed). Warp work is width-32 logical with __syncthreads-bounded cross-subgroup reductions, so expected wave32-safe; revalidate by running the same 8 tests on a gfx1100 host with no source change.

## Review 2026-06-02 (reviewer, linux-gfx90a, moat-port @ 29a9acf)
Verdict: review-passed. Reviewed git diff a426939...HEAD (10 files, +267/-6: CMakeLists USE_HIP path, cuda_stubs/ redirect headers, cuda_to_hip.h shim, utils.cuh timer/smid HIP branch). The CUDA path is byte-identical and every ROCm change is guarded by USE_HIP / __HIP_PLATFORM_AMD__; bindings.cu, setup.py, and all kernels/*.cuh are unchanged vs base, so the deferred torch extension is cleanly reverted (not half-broken), with the hipify cuBLASLt coverage gaps documented as future work.

Re-verified on this host (GCD 1, ROCm 7.2.1):
- Multi-arch build -DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100" configures and builds clean (rc=0, no errors); roc-obj-ls shows BOTH gfx90a and gfx1100 code objects; llvm-objdump shows zero s_memrealtime (the RDNA trap the timer fix avoids).
- globaltimer() -> __builtin_readcyclecounter() compiles on both arches (load-bearing); get_smid() -> __builtin_amdgcn_s_getreg compiles on both.
- Spot-run on real gfx90a: test_rmsnorm, test_sinkhorn_knopp (doubly-stochastic, row/col err ~1e-7), test_fused_rmsnorm_matmul (hipBLASLt row->col swap shim correct, fwd max diff 5.6e-2 within 6e-2 tol), test_stream_ops_backward all PASS rc=0.
- Fault classes clean: no hardcoded physical-warp 32 (all /32, %32, tiled_partition<32> are logical-subgroup partitioning, wave-agnostic); cross-subgroup reductions are __syncthreads/block.sync-bounded; s_warp_sums[8] reduce-partials kernels launch at 128 threads with __launch_bounds__(256) -> max 8 logical subgroups, no overflow; cg::reduce/cg::plus shim is a tile.size() shfl_xor butterfly (all-reduce, correct on wave32/wave64); hipBLASLt operand+TRANSA/TRANSB swap is applied identically in both matmul and heuristic so they agree; cublasGemmEx->hipblasGemmEx fallback present.
- Hygiene: commit title "[ROCm] Port mHC.cu standalone CMake harness to HIP" 50 chars, mentions Claude, no noreply/Co-Authored-By trailer, no ghstack, no em-dash; fork main == upstream base a426939 (clean mirror); Actions disabled on the fork; no AMD-internal account references.

Non-blocking (porter may fix opportunistically, not gating):
- utils.cuh:489 -- the get_smid() HIP branch passes literal 0x1e to __builtin_amdgcn_s_getreg with comment "HW_REG_HW_ID, offset 0, size 32", but 0x1e decodes to hwRegId=30/offset=0/size=1, not HW_ID (hwRegId=4) nor a 32-bit field; the returned value is not the CU/SM id. It compiles and is dead code (get_smid is only called under if constexpr(DO_PROFILE), default false, and the profiler output is explicitly not validated per plan risk 2), so it does not affect the deliverable; fix the comment/encoding only if the profiler is ever enabled.

## Validation 2026-06-02 (validator, linux-gfx90a, moat-port @ 29a9acf)

Platform: MI250X (AMD Instinct), ROCm 7.2.1, GCD 1 (HIP_VISIBLE_DEVICES=1).
GPU arch: gfx90a (amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-).
Compiler: clang++ 22.0.0 (/opt/rocm/llvm/bin/clang++).

Build command:
```
export HIP_VISIBLE_DEVICES=1
cmake -B build -S projects/mHC.cu/src/src/csrc -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100" \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
cmake --build build -j$(nproc)
```

Build result: rc=0, no errors. All 8 test_*.cu and 9 bench_*.cu link clean.

Multi-arch code object check (roc-obj-ls on test_rmsnorm):
- hipv4-amdgcn-amd-amdhsa--gfx1100 (offset=8192, size=364200)
- hipv4-amdgcn-amd-amdhsa--gfx90a  (offset=372736, size=395464)
Both code objects present in every test binary checked.

s_memrealtime check: llvm-objdump over test_rmsnorm and test_fused_rmsnorm_matmul: 0 occurrences. RDNA trap confirmed absent.

Test run (8 tests, gfx90a GCD1, AMD_LOG_LEVEL=3):
- test_rmsnorm:                    max diff 1.22e-05, PASS (tol 1e-2)
- test_rmsnorm_backward:           d_inp 1.45e-02, d_weight 9.54e-07, PASS (tol 2e-2)
- test_sinkhorn_knopp:             row err ~2.4e-7, col err ~4.8e-7 (<=1e-7 scale varies by size), max diff 0, PASS; doubly-stochastic assertion holds
- test_mhc_layer:                  static 1.19e-06, dynamic 7.15e-07, PASS (tol 1e-1)
- test_stream_ops:                 aggregate 1.19e-07, distribute 2.38e-07, PASS
- test_stream_ops_backward:        all 6 gradients PASS
- test_fused_rmsnorm_matmul:       fwd 5.65e-02, rms 2.65e-04, PASS (tol 6e-2 / 1e-3)
- test_fused_rmsnorm_matmul_backward: dW 2.06e-02, dx 3.05e-02, PASS (tol 5e-2 / 6e-2)

Result: 8 PASS, 0 FAIL.

hipBLASLt native dispatch confirmed (AMD_LOG_LEVEL=3):
- Kernels.so-000-gfx90a.hsaco loaded
- TensileLibrary_BB_SB_UA_..._gfx90a.co loaded
- "Using native code object for device: amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-" (not a fallback)

Deferred: torch extension (Strategy B) -- see "Torch extension" section above.

Verdict: COMPLETED. linux-gfx90a validated_sha=29a9acf.
