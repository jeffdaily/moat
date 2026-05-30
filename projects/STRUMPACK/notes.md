# STRUMPACK notes

## Target type: validate existing ROCm support (NVIDIA cuDSS substitute)

STRUMPACK (LBNL, BSD) is a GPU sparse DIRECT solver (multifrontal sparse LU + HSS/BLR low-rank) that ALREADY supports HIP/ROCm and runs on AMD GPUs (Frontier at OLCF); a 2025 paper reports its single-GPU factorization is ~1.9x faster than NVIDIA cuDSS. So this is a VALIDATE-AND-DOCUMENT target (like fused-ssim), NOT a fresh CUDA->HIP port: build for gfx90a, confirm a sparse factorize+solve runs correctly and deterministically vs a CPU reference, and document it as the ROCm substitute for the proprietary/closed-source NVIDIA cuDSS (which cannot be ported). It is the (pending) dependency for RXMesh's solver/autodiff path.
