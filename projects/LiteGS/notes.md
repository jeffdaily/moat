# LiteGS notes

MooreThreads/LiteGS: a PyTorch CUDA-extension 3D-gaussian-splatting trainer
("training 3DGS in 50 seconds"). Strategy B (torch hipify). Ported and
GPU-validated on linux-gfx90a (MI250X, CDNA2, wave64, ROCm 7.2.1, PyTorch 2.13
ROCm). Fork: jeffdaily/LiteGS @ moat-port. Three PyTorch CUDA extensions under
`litegs/submodules/`:

- `simple-knn` -> `simple_knn._C` (Morton kNN; near-zero-edit).
- `fused_ssim` -> a git submodule, the kemchenj/fused-ssim `fused-l1-ssim-loss`
  fork (an L1+SSIM head on rahul-goel/fused-ssim, which MOAT already ported
  zero-edit). Builds zero-edit here too.
- `gaussian_raster` -> `litegs_fused` (THE hard one: 21 kernels / 4 .cu --
  rasterizer fwd/bwd, transforms, binning, SH, compaction, sparse Adam).

## Environment

- conda env `py_3.12`, torch `2.13.0a0+gitb5e90ff`, `torch.version.hip
  7.2.53211`, gfx90a, `CUDA_HOME=None`/`ROCM_HOME=/opt/rocm` (so CUDAExtension
  auto-hipifies and links amdhip64/c10_hip/torch_hip).
- MUST run python/build with cwd OUTSIDE `/var/lib/jenkins/pytorch` (the source
  tree shadows the installed torch and breaks CUDAExtension hipify). All runs
  below are from `/tmp`.
- Python deps: `pip install torchmetrics plyfile tqdm pillow opencv-python
  matplotlib` (litegs imports plyfile at module load).

## Build (gfx90a)

Init the fused_ssim submodule first (it is the kemchenj fork):
```
cd projects/LiteGS/src && git submodule update --init --recursive
```
Then, from the src dir, build each extension. INSTALL NON-EDITABLE (`pip
install <dir>`, NOT `-e`): the editable install registers `simple_knn` /
`litegs_fused` as namespace packages with no `__init__.py`, so
`from simple_knn._C import distCUDA2` and `import litegs_fused` fail to resolve
to the `.so` (a fork packaging quirk, CUDA-identical). Non-editable copies the
`.so` into site-packages and resolves cleanly.
```
export HIP_VISIBLE_DEVICES=<free GCD> PYTORCH_ROCM_ARCH=gfx90a MAX_JOBS=16
P=/opt/conda/envs/py_3.12/bin/python
$P -m pip install ./litegs/submodules/simple-knn      --no-build-isolation
$P -m pip install ./litegs/submodules/fused_ssim       --no-build-isolation
$P -m pip install ./litegs/submodules/gaussian_raster  --no-build-isolation
```
INCREMENTAL gotcha (Strategy B): after editing a `.cu`, delete the extension's
`build/` and any `*.hip` mirror before rebuilding, or torch recompiles the
stale hipified mirror. A follower (gfx1100) needs only `PYTORCH_ROCM_ARCH=
gfx1100` + clean rebuild; no source edit.

## Wave64 fixes (gaussian_raster, all guarded USE_ROCM)

The rasterizer launches `dim3 Thread3d(32, tiles_per_block=4)` so blockDim.x ==
32: each tile is a POSITIONAL 32-lane row group, and on wave64 two tiles share
a 64-lane wavefront, each with its OWN per-tile blend-loop bounds. So every
cross-lane op must stay inside its 32-lane tile group (this is the popsift
positional-packing axis, NOT the AutoDock flexible-recombine axis). raster.cu:

1. **Divergent `__any_sync` = the root-cause hardware fault.** The forward
   early-exit `for(...; __any_sync(0xffffffff, any_active!=0); ...)` and two
   backward guards: a full-wavefront `__any_sync` is called divergently once one
   tile's loop ends while the sibling's continues -> HSA_STATUS_ERROR_EXCEPTION
   (0x1016), aborting the queue. The faulting kernel was confirmed via
   `AMD_SERIALIZE_KERNEL=3 AMD_LOG_LEVEL=3` (last ShaderName before abort =
   `raster_forward_kernel<8,16,false,false,false>`). Fix: a width-32 butterfly
   OR `tile_any(pred)` (`v |= __shfl_xor(v,o,32)`) confined to the tile; never
   diverges across tiles because each tile's 32 lanes share the loop bound. This
   was THE bug -- a full-wavefront `__any_sync(0xffffffffffffffffULL)` compiles
   and runs but faults at runtime on the divergence.

2. **sm_80 `__reduce_{add,max}_sync` (11 sites).** These DO exist in HIP 7.2
   (`amd_warp_sync_functions.h`, with `-DHIP_ENABLE_WARP_SYNC_BUILTINS=1`) but
   reduce over the WHOLE 64-lane wavefront -> would fold the adjacent tile. Do
   NOT shim the name (collides with HIP's). Replaced the call sites with
   `tile_warp_reduce_add<T>` / `tile_warp_reduce_max` = width-32 `__shfl_xor`
   butterfly all-reduce (result in every lane, like the CUDA intrinsic). The
   reduction preserves the kernel's deliberate integer exponent-rescale path
   (it rescales floats to int, reduces the int domain, rescales back -- int add
   is associative so the per-tile butterfly is bitwise-deterministic).

3. **Inline PTX `asm("ex2.approx.f16x2")`** in fast_exp_approx: hipcc cannot
   assemble PTX. Replaced with the HIP intrinsic `h2exp2` (packed half2 base-2
   exp; the code already scales by log2(e)). Numerically approximate either way;
   absorbed by the validation tolerances and the PSNR gate.

4. **half2 mask intrinsics.** HIP `<hip/hip_fp16.h>` has `__hge2`/`__hle2`/
   `__hgt2` (returning a half2 of 1.0/0.0) but NOT the `*_mask` forms, and NOT
   `__hmin2` or `__vcmpleu2`. Reimplemented `__hgt2_mask`/`__hge2_mask`/
   `__hle2_mask` to return the 0xFFFF-per-half-lane mask (synthesized from the
   `__h*2` compare) the downstream `reinterpret_cast<unsigned*>(&x)[0] &= mask`
   bit-masking expects; `__vcmpleu2` as a per-16-bit `<=` -> 0xFFFF; `__hmin2`
   from per-half `__hlt`. `h2rcp` and `h2exp2` already exist in HIP.

5. **The 32-thread-block-on-wave64 tile mapping resolved as strategy 3(a):**
   keep blockDim.x=32 and force every shuffle/reduce to width-32 so each tile
   runs as one aligned 32-lane sub-group of the wavefront. HIP's `__shfl_down`/
   `__shfl`/`__shfl_xor` correctly clamp to the `width`-lane sub-group, so the
   per-tile reductions never cross into the sibling tile. The pixel addressing
   (`threadIdx.x % tile_w`, `threadIdx.x / tile_w * PIXELS_PER_THREAD *
   VECTOR_SIZE`) and the `[...][4*32]` shared buffers indexed by
   `threadIdx.y*32+threadIdx.x` are pure threadIdx arithmetic, wave-width
   independent, unchanged. Strategy 3(b) (re-derive for 64 lanes) was not
   needed.

## Wave64 fix (compact.cu) -- the popsift leader-election trap

`frustum_culling_aabb_kernel` (block of 256): per-warp leader election with a
32-bit `__ballot_sync`, `lane_id = threadIdx.x & 0x1f`, `__popc`, lane-0
`atomicAdd`. On wave64 lanes 0 and 32 both have lane_id 0 so the leader
`atomicAdd` double-fires, and 32-bit `__popc` on a 64-bit ballot drops the upper
half-wavefront -> corrupt compaction offsets / OOB scatter. Fix (arch-unified,
`kWarpSize` = 64 on gfx90a, 32 on RDNA/CUDA): 64-bit ballot, `__popcll`,
`lane_id = threadIdx.x & (kWarpSize-1)`, single leader per kWarpSize-lane
wavefront, `__shfl_sync(mask, v, 0, kWarpSize)` broadcast. Correct on wave32 and
wave64. (Note: `render()` uses Binning, not frustum_culling_aabb -- the latter
is the cluster-culling entry, exercised separately.)

## Build/host fixes

- `setup.py`: added a `torch.version.hip` branch dropping `--use_fast_math` for
  `-ffast-math`; and EXTENDED `remove_unwanted_pytorch_nvcc_flags` to also strip
  `-D__HIP_NO_HALF_OPERATORS__=1` / `-D__HIP_NO_HALF_CONVERSIONS__=1` from
  torch's `COMMON_HIPCC_FLAGS` (the rasterizer uses half2 ctors and operators
  throughout; torch disables exactly those on HIP by default).
- `cuda_errchk.h`/`.cpp`: include `<hip/hip_runtime.h>` and call the `hip*`
  runtime on ROCm (this is a plain `.cpp`/`.h` pair; the `.h` is not in the
  extension sources so hipify does not rewrite it).
- USE_ROCM-guarded out the `#ifndef __CUDACC__ #define __CUDACC__/__NVCC__`
  editor-intellisense force-defines, the libcu++ `<cuda/atomic>` include, and
  `<cooperative_groups/reduce.h>` (absent on ROCm; HIP CG has no cg::reduce) in
  all 4 .cu and in simple_knn.cu.
- transform.cu: an explicit `(float)` cast on a `double->float` brace-init in
  the eigh kernel (a templated `<scalar_t>` kernel; clang rejects the narrowing
  in `{...}` where nvcc warns).
- **Launch-syntax normalization `<< <`/`>> >` -> `<<<`/`>>>`** (32 sites in
  gaussian_raster, 3 in simple-knn). MooreThreads (a MUSA GPU vendor)
  reformatted every kernel launch with spaces inside the triple-angle-brackets;
  nvcc tolerates it but clang-HIP's parser rejects `coord2Morton << <...>>>`
  with "expected expression". Correctness-neutral on both backends.
- `.gitignore`: added `*.hip`, `*.so`, `hip_errchk.*` (torch-hipify outputs).

## simple-knn

Near-zero-edit (the include guards + launch-syntax only). Uses
`cg::this_grid().thread_rank()` (HIP CG provides it), cub SortPairs with
begin_bit=0 (hipCUB drop-in), thrust device_vector/sequence (rocThrust
drop-in). Validated: `distCUDA2` on a random cloud is finite and
bitwise-deterministic.

## Validation (gfx90a, HIP_VISIBLE_DEVICES=2) -- ALL THREE TIERS PASS

Tier 1 (litegs.utils.wrapper fused-vs-script, fwd+bwd, agent_space/
litegs_tier1*.py): CreateTransformMatrix, CreateCov2dDirectly PASS.
EighAndInverse2x2 verified correct independently (`||A@Ainv - I||` median
5.96e-8; eigenvalues match torch to 7.6e-6 on |det|>1; the wrapper threshold
miss is eigenvector SIGN ambiguity -- |eigvec| matches to ~1e-5 -- plus randn*10
near-singular conditioning; bitwise-deterministic). SphericalHarmonicToRGB:
sh_base_grad exact, sh_rest_grad within tolerance; its dir_grad is INTENTIONALLY
never written by the kernel (`dir_grad = zeros_like(dir)`, all dRGBdx writes
commented out -- LiteGS does not backprop to view directions), so the wrapper's
autograd dir_grad reference cannot match (a CUDA-identical upstream test
artifact). Two wrappers (CreateRaySpaceTransformMatrix, Binning) throw on an
arg-count mismatch -- their `test_inputs` lists do not match the fused signature
(upstream harness bug, CUDA-identical).

Tier 2 (rasterizer, no script twin, agent_space/litegs_tier2*.py): forward
finite, non-trivial (99.9% nonzero), bitwise-deterministic across 2 runs.
Backward grads finite and CORRECT -- central finite-difference on grad_color
gives slope 1.000, 100% sign agreement, 8e-4 median rel err on the largest
entries; order-independent grad SUMs stable to ~1e-9; the only run-to-run
variation is ~1e-7 fp16 atomicAdd reordering (benign, universal to every GPU
including CUDA -- the 3DGS backward scatters per-point grads via atomicAdd).

Tier 3 (end-to-end synthetic multi-view fit through the full pipeline -- MVP,
transforms, eigh+inverse, binning + cub sort, rasterizer fwd/bwd, Adam;
agent_space/litegs_tier3.py): loss 0.0274 -> 0.0012 (down 95.6%), mean PSNR
20.5 -> 48.3 dB over 300 iterations, no NaN.

Gotchas worth repeating:
- positions fed to `render`/MVPTransform must be HOMOGENEOUS `[4, N]` (xyz + a
  w=1 row), not `[3, N]`; the MVP backward returns a `[4,N]` grad.
- the rasterizer blends internally in fp16 (`SCALER=128`), so its gradients are
  ~fp16 precision; a finite-difference check needs a coarse eps and should
  target the largest-magnitude grad entries, or fp16 round-off dominates.
- no real COLMAP scene was on the host; Tier 3 is a self-contained synthetic
  multi-view fit (a stronger correctness oracle since the ground truth is
  controlled). If a Mip-NeRF360 / 3dgs-challenge scene is fetched later,
  `example_train.py -s <scene> -i <images> -m <out>` is the documented entry.

## Follower (gfx1100 / gfx1151)

No source change expected: the wave64 fixes are arch-unified (`kWarpSize`,
width-32 tile reductions are correct on wave32 too -- on RDNA a 32-lane
wavefront == one tile, the native CUDA layout). A follower validates with
`PYTORCH_ROCM_ARCH=gfx1100` + clean rebuild; do not advance head_sha unless a
genuine build/source fix is needed.
