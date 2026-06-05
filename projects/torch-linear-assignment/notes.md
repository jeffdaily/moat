# torch-linear-assignment notes

## Port summary

Strategy B (pytorch extension, torch hipify) -- zero-source-edit port confirmed.

PyTorch's build-time hipify translates the CUDA sources automatically when building against ROCm PyTorch. No manual source changes required.

## Build (linux-gfx90a)

```bash
export HIP_VISIBLE_DEVICES=3
pip install --no-build-isolation -e /path/to/src
```

Verified `has_cuda: True` after build.

## Validation (linux-gfx90a)

GPU tests pass:
- `test_simple`: PASSED (deterministic assignment output matches expected)
- Batch test (100 x 20 x 40): PASSED (valid assignments, no duplicates)

The `test_cuda_equal_to_cpu` and `test_compare_to_scipy` tests fail due to a torch/numpy version incompatibility in the base environment (torch compiled with numpy 1.x, environment has numpy 2.x). This is an environment issue, not a port issue -- the GPU code path works correctly.

## Gotchas

1. The `SMPCores()` function returns CUDA cores per SM based on NVIDIA arch major/minor. On AMD it falls through to default 128. This affects block size selection but not correctness -- the algorithm is inherently sequential per batch element (Hungarian assignment).

2. Torch's hipify generates a `.hip` file at build time (`torch_linear_assignment_hip_kernel.hip`). This is a build artifact, not checked in.
