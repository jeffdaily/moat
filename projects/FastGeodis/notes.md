# FastGeodis notes

## Build (linux-gfx90a)

Strategy B PyTorch extension. The only required change was fixing setup.py to detect
`ROCM_HOME` when `CUDA_HOME` is not set, enabling the GPU extension build on ROCm PyTorch.

PyTorch's hipify automatically translates `fastgeodis_cuda.cu` at build time; no kernel-level
changes are needed. All 4 kernels use only standard CUDA runtime APIs and `__syncthreads()`
that map 1:1 to HIP.

Build command:
```bash
HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx90a pip install -e . --no-build-isolation
```

## Test (linux-gfx90a)

300 tests pass in ~72s:
```bash
HIP_VISIBLE_DEVICES=0 python -m pytest tests/ -v
```

Test coverage: 2D/3D geodesic distance transforms (GPU + CPU), signed/unsigned variants,
plus CPU-only algorithms (Toivanen, pixelqueue, fastmarch).

## Gotchas

- ROCm detection: Original setup.py only checks `CUDA_HOME`, which is None on pure ROCm
  systems. Fixed by also checking `ROCM_HOME` from `torch.utils.cpp_extension`.

## Review 2026-06-05

Reviewed by: reviewer agent

**Verdict: Approve**

No problems found. This is an exemplary minimal Strategy B port:
- setup.py change correctly detects ROCM_HOME as alternative to CUDA_HOME
- No kernel changes needed -- hipify handles cudaMemcpyToSymbol and __syncthreads() 1:1
- No warp intrinsics, textures, CUDA libraries, or RAII handles -- no fault classes apply
- CUDA build preserved (additive change)
- Commit message correct: [ROCm] prefix, Claude mention, Test Plan, no noreply trailer
- 300/300 tests passed on gfx90a

Ready for validation.
