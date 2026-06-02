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
