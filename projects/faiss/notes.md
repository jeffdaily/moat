# faiss notes

## Provenance / why adopted (2026-05-31)
Adopted at Jeff's suggestion after finding FAISS GPU code vendored inside Open3D.
Open3D does NOT use upstream facebookresearch/faiss as a submodule or fetched dep --
it VENDORS a SUBSET of FAISS's GPU warp-select kNN kernels directly as source under
cpp/open3d/core/nns/kernel/ (BlockSelect*, WarpShuffle.cuh, MergeNetwork, L2Select,
PtxUtils.cuh; MIT, "Copyright (c) Facebook, Inc."). The Open3D ROCm port ported that
subset in-place (PtxUtils inline-PTX -> HIP intrinsics; __shfl_*_sync masks ->
OPEN3D_FULL_WARP_MASK; unified kWarpSize=32 two-32-lane-halves model for wave64).

So Open3D's port covers only the kNN-selection subset. Upstream FAISS proper (the IVF/
IVFPQ/IVFFlat GPU indexes, StandardGpuResources, the full GpuIndex* hierarchy, cuVS
integration) is a MUCH larger standalone CUDA library and is NOT redundant with Open3D.
It is a strong MOAT target in its own right (popular GPU similarity search). The Open3D
nns/kernel port is a useful reference for the warp-select/selection-network HIP translation.
