# FaithC notes

## Port summary (linux-gfx90a, lead)
Strategy B (torch hipify). The whole port is: restore a `setup.py` with a
`CUDAExtension`/`BuildExtension` over `_C/{bindings.cpp,kernels.cu}` (the v1.5
pyproject dropped it, so a source install had no compiled `_C` while ops.py
hard-imports it), plus one mechanical source fix for a hipify parser limit.
Fork: https://github.com/jeffdaily/FaithC @ moat-port (head ec2fae2).

## Gotchas

### hipify cannot rewrite a parenthesized ternary kernel launch
`aabb_tri_sat_clip_select_cuda` originally launched a template kernel chosen by
a ternary directly in the launch:
`(max_vert==8 ? sat_clip_kernel<scalar_t,8> : sat_clip_kernel<scalar_t,7>)<<<...>>>`.
hipify's regex `<<<...>>>` -> `hipLaunchKernelGGL` rewrite mis-parses the
parenthesized expression and emits a mangled `...<scalar_t,7hipLaunchKernelGGL((>))`
token, giving `invalid suffix 'hipLaunchKernelGGL' on integer constant`. Fix:
hoist the selected kernel into a local function pointer
(`auto kernel = cond ? k<...,8> : k<...,7>;`) then launch `kernel<<<...>>>`.
Parses cleanly under both hipify and nvcc; identical semantics. Applies to both
the mode-1 (sat_centroid) and mode-2 (sat_clip) launch sites.

### Build / incremental
- `cd src && PYTORCH_ROCM_ARCH=gfx90a python setup.py build_ext --inplace`.
- Multi-arch fat binary: `PYTORCH_ROCM_ARCH="gfx90a;gfx1100" ... build_ext --inplace`,
  verify `llvm-objdump --offloading _C*.so | grep -E "gfx90a|gfx1100"`.
- After editing a `.cu`, delete the stale `src/faithcontour/_C/kernels.hip` and
  the `build/` dir before rebuilding so hipify regenerates (Strategy B incremental
  trap). `.gitignore` now excludes `*.hip`, `*.prehip`, `*.so.*`.
- Pybind module is `TORCH_EXTENSION_NAME` = `_C`; to load the `.so` standalone
  (without importing the `faithcontour` package, which pulls scipy/utils.grid),
  load it under spec name `_C` (PyInit symbol is `PyInit__C`).

### Numerics
- No `-ffp-contract=on` pin needed. The Moller-Trumbore dot drift vs a torch CPU
  reference is 3.5e-7, well inside the kernels' eps guards.
- wave-size-agnostic: zero warp intrinsics, no shfl/ballot/cub; `extern __shared__`
  is sized by `blockDim.x` and fully `__syncthreads`-fenced. `dim3(32,32)` blocks
  are 2D tile dims, not warp assumptions. No wave64 work; gfx1100 expected to pass
  by rebuild with no delta.

## Validation (real gfx90a, MI250X, GCD 1)
Harness `agent_space/faithc_harness.py` drives all four bindings on GPU vs a
pure-torch CPU reference. Atomic kernels (segment_tri_intersection_fused,
gen_candidates_overlap) compared as ORDER-INDEPENDENT sorted (a,t) pair sets
(atomicAdd slotting is nondeterministic); non-atomic kernels (voxelize_mark
use_sat F/T, aabb_tri_sat_clip_select modes 0/1/2) compared exactly + rerun
determinism; overflow path exercised with a small cap. 16/16 PASS.
`AMD_LOG_LEVEL=3` confirms native gfx90a code-object dispatch.

## Deferred: end-to-end demo dependencies
`demo.py` / the encoder/decoder need two GPU deps NOT in this repo, so the full
pipeline is a follow-up (the four `_C` kernels are the validated lead gate):
- `atom3d` (Luo-Yihao/Atom3d, ~25% CUDA: MeshBVH + octree). Its own build is
  CUDA-only; would need a separate MOAT port. Recommend scaffolding it and
  recording FaithC `depends_on` it for the e2e story.
- `torch_scatter` (rusty1s/pytorch_scatter): builds on ROCm torch via auto-hipify
  but must be compiled for this ROCm; external pip dep, not a MOAT project.

## Review 2026-06-02 (reviewer, gfx90a)
Verdict: review-passed. Independently reproduced on real gfx90a (MI250X, GCD 3, ROCm torch hip 7.2.53211).

Verified clean:
- setup.py CUDAExtension + BuildExtension drives a clean-from-scratch HIP build (auto-hipify generates kernels.hip, compiles, links amdhip64/c10_hip/torch_hip); sources stay CUDA-native (.cu/.cpp/.h tracked, no .hip/.so committed; .gitignore covers *.hip/*.prehip/*.so.*).
- Multi-arch: PYTORCH_ROCM_ARCH="gfx90a;gfx1100" fresh build -> both code objects in _C*.so (llvm-objdump --offloading).
- Ternary-launch hoist (kernels.cu:730 mode-1 sat_centroid, kernels.cu:745 mode-2 sat_clip): `auto kernel = max_vert==8 ? K<scalar_t,8> : K<scalar_t,7>; kernel<<<...>>>` is semantically identical to the original `(cond ? A : B)<<<...>>>`; both template branches differ only in the non-type MAXV arg so they share one function-pointer type that `auto` deduces cleanly. Parses under hipify and nvcc.
- Wave-agnostic confirmed: zero warp intrinsics (no shfl/ballot/activemask/any/all), no cooperative groups, no threadfence, no textures/surfaces, no cub/thrust/cublas/curand/cufft. Only atomicAdd + __syncthreads + dynamic extern __shared__ (sized by blockDim.x, fully fenced) + float math. dim3(32,32) are 2D tile dims, not warp assumptions.
- Correctness harness reproduced: 16/16 PASS. atomicAdd kernels (seg_tri, gen_candidates_overlap) order-independent sorted-pair sets; non-atomic kernels exact + rerun-deterministic; overflow path exercised; Moller-Trumbore dot drift 3.5e-7, eps-absorbed (no -ffp-contract pin needed).
- Commit hygiene: [ROCm] title 54 chars, mentions Claude, no noreply/ghstack/em-dash. Fork main == upstream main (1580e2e) clean mirror; moat-port HEAD == ec2fae28; fork Actions disabled. No AMD-internal account reference.

Minor (non-blocking, recorded for the e2e follow-up): the harness cross-checks mode-0 hit_mask against a CPU SAT reference, but mode-1/mode-2 centroid/poly-vert VALUES are validated only for determinism + index alignment + range, not against an independent CPU Sutherland-Hodgman reference (plan called for "centroids/areas within tol"). The clip geometry is unchanged from upstream CUDA and the shared SAT plane logic is cross-checked via mode-0, so this is a coverage gap, not a defect; close it when atom3d enables the end-to-end demo run.
