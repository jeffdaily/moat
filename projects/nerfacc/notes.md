# nerfacc notes

## Why this project is in MOAT

nerfstudio-project/nerfacc (1457 stars) is a PyTorch CUDA extension for efficient
volumetric ray-marching / sampling in NeRFs. It was a discovery blind spot (a
.cu-based CUDA library in a Python-dominant repo, no "cuda" in name/topics).

Added because gsplat's test suite uses nerfacc as an optional pure-torch
reference (the eval3d accumulator and several rasterize_to_pixels comparisons).
nerfacc ships a CUDA-only wheel whose `_C` extension is unavailable on ROCm, so
`import nerfacc` fails and 38 gsplat reference tests are a documented known-fail
on every platform. Porting nerfacc to ROCm closes that gap and stands on its own
as a widely used library.

## Classification (preliminary, for the planner)

- ext_type: torch-extension (Strategy B -- torch hipifies the .cu/.cuh at build
  time), same family as gsplat and FastGeodis.
- Sibling of gsplat in the nerfstudio ecosystem; no MOAT-internal deps.
- Upstream default branch: master.
