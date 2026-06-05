# TTT3R notes

## Build (linux-gfx90a)

```bash
cd src/croco/models/curope
rm -rf build *.so *.hip
PYTORCH_ROCM_ARCH=gfx90a python setup.py build_ext --inplace
```

## Validation

```bash
cd src/croco/models/curope
python -c "import torch; import curope; print('curope imported successfully')"
```

GPU vs CPU correctness test:
```bash
python -c "
import torch, curope
tokens = torch.randn(2, 16, 8, 64, device='cuda')
pos = torch.randint(0, 10, (2, 16, 2), device='cuda', dtype=torch.int64)
ref = tokens.clone().cpu()
pos_cpu = pos.cpu()
curope.rope_2d(tokens, pos, 100.0, 1.0)
curope.rope_2d(ref, pos_cpu, 100.0, 1.0)
diff = (tokens.cpu() - ref).abs().max().item()
assert diff < 1e-5, f'diff={diff}'
print(f'PASS (diff={diff:.2e})')
"
```

## Port details

Changes:
1. setup.py: Detect ROCm via torch.version.hip, skip cuda.get_gencode_flags() on ROCm (fails on AMD), use HIP-compatible flags
2. kernels.cu: Fix deprecated tokens.type() -> tokens.scalar_type() (required for modern PyTorch)

The kernel is portable (no warp intrinsics, no CUDA libraries, only __syncthreads__ and standard math) so PyTorch's automatic hipify handles the CUDA->HIP translation at build time.

Gotchas:
- Cannot use -ffast-math with hipcc on the kernel file because PyTorch headers use std::isinf() which conflicts with -ffinite-math-only (part of -ffast-math)
- The cuRoPE2D module expects input in (B, N, H, D) layout or (B, H, N, D) created from transpose of (B, N, H, D) to maintain correct strides

## Review 2026-06-05

Reviewed branch `moat-port` against upstream `main`.

**Verdict**: Approve

**Summary**: Clean Strategy B (PyTorch extension) port. The setup.py detects ROCm via `torch.version.hip` and uses HIP-compatible compile flags; the CUDA path is preserved unchanged. The only kernel change (`tokens.type()` -> `tokens.scalar_type()`) is a legitimate PyTorch modernization that applies to both platforms.

The kernel uses only portable constructs (`__global__`, `__syncthreads__`, `extern __shared__`, standard math functions). No warp intrinsics, no CUDA libraries, no textures. PyTorch's automatic hipification handles the CUDA->HIP runtime symbol translation at build time.

GPU vs CPU validation on gfx90a: max diff 2.26e-06 (within 1e-5 tolerance).

No findings.

## Validation 2026-06-05

Platform: linux-gfx90a (MI250X, HIP_VISIBLE_DEVICES=1)
Commit: d162988b5af74e7939faaea422c004ddc1511503

Build:
```bash
cd /var/lib/jenkins/moat/projects/TTT3R/src/src/croco/models/curope
HIP_VISIBLE_DEVICES=1 PYTORCH_ROCM_ARCH=gfx90a python setup.py build_ext --inplace
```

Test results:
- Import test: PASS
- GPU vs CPU correctness: PASS (max diff=2.38e-06, tolerance 1e-5)

Validation commands:
```bash
HIP_VISIBLE_DEVICES=1 python -c "import torch; import curope; print('curope imported successfully')"

HIP_VISIBLE_DEVICES=1 python -c "
import torch, curope
torch.manual_seed(42)
tokens = torch.randn(2, 16, 8, 64, device='cuda')
pos = torch.randint(0, 10, (2, 16, 2), device='cuda', dtype=torch.int64)
ref = tokens.clone().cpu()
pos_cpu = pos.cpu()
curope.rope_2d(tokens, pos, 100.0, 1.0)
curope.rope_2d(ref, pos_cpu, 100.0, 1.0)
diff = (tokens.cpu() - ref).abs().max().item()
assert diff < 1e-5, f'diff={diff}'
print(f'PASS: GPU vs CPU match (max diff={diff:.2e})')
"
```

Real GPU validation passed.
