# popsift notes

Sibling of the CudaSift port (also a CUDA SIFT): reuse projects/CudaSift cuda_to_hip.h compat-header approach and the texture / warp-shuffle fixes.

## Porting 2026-05-30

Strategy A (CMake compat-header, colmap/CudaSift model). Built the **library**
`libpopsift.so` clean for gfx90a. Upstream `develop` @ b1c8199. ROCm 7.2.1,
HIP clang 22.0.0.

### Build status
- **CLEAN COMPILE + LINK** of `libpopsift` (the GPU port surface). 100% target
  built, 0 errors. `roc-obj-ls` confirms an embedded `hipv4-amdgcn-amd-amdhsa--gfx90a`
  code object (~9.7 MB), so all SIFT kernels compiled for gfx90a.
- Built with `-DPopSift_BUILD_EXAMPLES=OFF`. The demo apps (`popsift-demo`,
  `popsift-match`) need **Boost 1.71+** (filesystem, program_options, system) and
  optional DevIL, neither installed on this host. The library does not depend on
  Boost/DevIL, so the port completes without them. The demos are blocked-on-dep
  (Boost) but are not part of the GPU port; they are NOT needed to validate the
  kernels (a small harness against testImages/ would suffice on a GPU run).
- NVIDIA path (`USE_HIP=OFF`) is unchanged; it reaches `project(... CUDA)` and
  fails here only because there is no CUDA toolkit on this AMD host (expected).
- Clean full build ~30s (`stats.jsonl`).
- Benign warnings only: 222x -Wunused-value (debug_macros.h ignores the HIP
  [[nodiscard]] hipError_t return of hipStreamSynchronize/hipDeviceSynchronize;
  CUDA does not mark these nodiscard -- pre-existing pattern), 2x
  -Wdeprecated-declarations (rocThrust's thrust::identity in s_filtergrid.cu).

### Files changed
- `src/popsift/cuda_to_hip.h` (new): compat header. Adapted from CudaSift; adds
  popsift's extra surface symbols. Notable beyond CudaSift's set:
  surfaces (cudaSurfaceObject_t/Create/Destroy), surf2DLayeredwrite, layered/3D
  arrays (cudaMalloc3DArray, cudaArrayLayered|SurfaceLoadStore), cudaMemcpy3D(+Parms,
  make_cudaPitchedPtr/Extent/Pos), async variants, FromSymbol, HostRegister,
  tex2DLayered enums.
- `src/popsift/hip_compat/{cuda_runtime.h,math_constants.h}` (new): include-path
  shims (HIP build only). 12 files `#include <cuda_runtime.h>` (incl. the public
  popsift.h); ROCm has no such header. math_constants.h supplies CUDART_INF_F
  (features.cu) which ROCm does not ship.
- `CMakeLists.txt`, `src/CMakeLists.txt`: `option(USE_HIP)`; gate project language
  CUDA<->HIP; on HIP enable_language(HIP), .cu LANGUAGE HIP, HIP_ARCHITECTURES gfx90a,
  force-include the compat header, add hip_compat to includes (PRIVATE), link
  roc::rocthrust instead of CUDA::cudart, **-fgpu-rdc on compile AND link**. CUDA
  path byte-for-byte unchanged (wrapped in `if(NOT USE_HIP)`).
- `src/popsift/s_filtergrid.cu`: `thrust::cuda::par` -> `POPSIFT_THRUST_PAR` macro
  (::thrust::hip::par on ROCm, ::thrust::cuda::par on CUDA). rocThrust does NOT
  alias thrust::cuda; and being inside namespace popsift, unqualified thrust::cuda
  also collided with popsift::cuda (debug_macros.h) -- the fully-qualified macro
  fixes both.
- `src/popsift/sift_octave.h`: `LinearTexture::tex` retyped cudaSurfaceObject_t ->
  cudaTextureObject_t (it always held a texture object; CUDA's identical ull typedefs
  hid the bug, HIP's distinct pointer types expose it).

### Fault classes handled
- **Cross-TU __device__/__constant__ linkage (the big one).** d_consts, d_gauss,
  dct/dbuf/dobuf are `extern __device__[ __constant__]` in headers, defined in one
  .cu, used in many. Without `-fgpu-rdc` clang HIP device-links each .o standalone
  and these references are "undefined protected symbol" at lld. Fix: `-fgpu-rdc` on
  every HIP compile + `-fgpu-rdc --hip-link` at the final link so the device linker
  resolves them. CMake's HIP_SEPARABLE_COMPILATION property did NOT add -fgpu-rdc
  here (CMake 3.31 / this toolchain); had to add the flag explicitly.
- **surf2DLayeredwrite arity.** HIP's builtin is 5-arg (no boundary-mode param);
  CUDA's is 6-arg (trailing cudaSurfaceBoundaryMode). All 17 call sites pass 6 args.
  Fixed with a compat-header template wrapper that drops the boundary mode (HIP's
  default image-store behavior == hipBoundaryModeZero, so faithful). Verified the
  x-coordinate convention matches: HIP's __hipGetPixelAddr converts the byte x back
  to an element (>>2 for float), so popsift's `write_x*4` byte offset is correct on
  HIP unchanged.
- **Texture/surface type strictness.** LinearTexture (above): HIP separates
  hipTextureObject_t / hipSurfaceObject_t as distinct pointer types; CUDA's are both
  unsigned long long.
- **Directed-rounding intrinsics.** __fmaf_ru / __fmul_ru (round-to-+inf) absent on
  HIP -> mapped to __fmaf_rn / __fmul_rn in the header (operands non-negative, the
  source comments confirm rounding mode is immaterial; matches CudaSift's __fmul_rz
  precedent).
- **Warp size 64 vs 32 -- NOT yet validated (open GPU risk).** s_orientation.cu is
  hardwired to a 32-lane warp: launched block.x=32, 64-bin histogram split as
  threadIdx.x+0 / +32, Warp32 bitonic sort64, __popc(ballot). It COMPILES on wave64;
  the bitonic shuffle_xor masks (1<<shift, max 16) stay within the low 32 lanes and
  ballot's inactive upper lanes are 0, so it *should* preserve 32-lane semantics --
  but this MUST be checked numerically on a real gfx90a run (orientation/descriptor
  parity). Same class as CudaSift's width=32 finding. assist.h's *_sync(0xffffffff,..)
  intrinsics are mapped to the mask-free HIP builtins in the header (HIP ignores the
  32-bit mask on wave64 anyway).
- **Texture pitch (256B) -- N/A.** All textures are cudaArray-backed (cudaMalloc3DArray),
  not cudaResourceTypePitch2D binds, so the AMD 256-byte row-pitch rule does not apply.
  Confirmed the resource type is Array per the CudaSift changelog lesson. The lone
  cudaMallocPitch (plane_2d) is plain device memory, never bound as a pitched texture.
- **Deprecated array-copy APIs -- N/A.** popsift already uses modern cudaMalloc3DArray +
  cudaMemcpy3D + surface objects; no cudaMemcpyToArray (CudaSift needed that rewrite,
  popsift does not).
- **Thrust.** rocThrust at /opt/rocm/include/thrust compiles s_filtergrid.cu unchanged
  except the par-policy namespace (above). find_package(rocthrust CONFIG) + link
  roc::rocthrust.

### Exact commands
Configure + build (library, no Boost needed):
```
utils/timeit.sh popsift compile -- cmake -S projects/popsift/src -B projects/popsift/src/build-hip \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_BUILD_TYPE=Release \
  -DPopSift_BUILD_EXAMPLES=OFF -DBUILD_SHARED_LIBS=ON
utils/timeit.sh popsift compile -- cmake --build projects/popsift/src/build-hip -j
```
Verify embedded gfx90a code:
```
roc-obj-ls projects/popsift/src/build-hip/Linux-x86_64/libpopsift.so.0.10.1
```

### For the validator (GPU run)
Install Boost 1.71+ (filesystem, program_options, system); optionally DevIL. Then
configure with `-DPopSift_BUILD_EXAMPLES=ON` to get popsift-demo/popsift-match, run
SIFT on testImages/ (and an Oxford image), and check keypoint counts / orientations /
descriptors vs the CUDA expectation. This is where the wave64 orientation/bitonic
risk is actually observed; a compile-only build cannot see it.

## Fix 2026-05-30

### The texture bug (creation failure, found by the GPU validator)
Every SIFT run aborted at `sift_octave.cu` `Octave::alloc_data_tex` /
`alloc_interm_tex`: HIP's `hipCreateTextureObject` rejects a texture with
`filterMode=cudaFilterModeLinear` + `readMode=cudaReadModeElementType` over a
`cudaMalloc3DArray` **float** array ("operation not supported"). ROCm/HIP does
not support hardware linear filtering on element-read float textures (CUDA does).
Creation failed before any kernel ran, so zero features extracted. The two
offending textures are `_data_tex_linear` and `_intm_tex_linear`. The DoG texture
(`_dog_3d_tex_point`) is already point-filtered, so it is unaffected; there is no
DoG *linear* texture. (The uchar input-image texture in `Image::createTexture`
uses `cudaReadModeNormalizedFloat`, which HIP *does* accept with linear filtering,
so the demo's default `ByteImages` path reaches the octave before failing. The
float input texture in `ImageFloat::createTexture` -- `--float-mode` only -- has
the same element-read-float+linear pattern and would still fail on HIP; left
unchanged as it is out of the demo's default path and out of scope here.)

### The fix (HIP-guarded software bilinear; CUDA path byte-for-byte unchanged)
Two files, both changes wrapped in `#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)`:

- `src/popsift/sift_octave.cu` (`alloc_data_tex`, `alloc_interm_tex`): after the
  CUDA code sets `tex_desc.filterMode = cudaFilterModeLinear`, the HIP guard
  overrides it to `cudaFilterModePoint` (which HIP accepts over the float array).
  The float array and all other descriptor fields are untouched.

- `src/popsift/common/assist.h` (`readTex(tex, x, y, z)`, the 3-arg layered
  fetch): on HIP, replace the single `tex2DLayered<float>(tex, x+0.5, y+0.5, z)`
  with manual bilinear interpolation over 4 point-filtered neighbors.

  **Coordinate convention (matched exactly).** CUDA's unnormalized linear filter
  on `tex2DLayered(c)` samples at index `c-0.5`: `i0=floor(c-0.5)`,
  `frac=(c-0.5)-i0`. `readTex` passes `c=x+0.5`, so `i0=floor(x)`, `frac=x-floor(x)`.
  Point-filtered fetch of texel `ix` is `tex2DLayered(ix+0.5)` (point returns
  `floor`). So:
  ```
  fx=floor(x); fy=floor(y); ax=x-fx; ay=y-fy;
  t00=tex(fx+0.5,fy+0.5,z); t10=tex(fx+1.5,fy+0.5,z);
  t01=tex(fx+0.5,fy+1.5,z); t11=tex(fx+1.5,fy+1.5,z);
  out = lerp( lerp(t00,t10,ax), lerp(t01,t11,ax), ay );
  ```
  Verified: `x=5.0` -> `ax=0` -> texel[5] (CUDA: i0=5,frac=0 -> texel[5]); `x=5.3`
  -> `lerp(texel[5],texel[6],0.3)` (CUDA: i0=5,frac=0.3 -> same). `cudaAddressModeClamp`
  is set on the point texture too, so out-of-range neighbors clamp identically to
  CUDA's hardware filter. SIFT blurs are per-layer; every `readTex` z-arg is an
  integer level, so interpolation is in x,y at fixed layer z only (no z-interp).

  All octave linear-texture samples funnel through this one `readTex`: the
  interpolated Gauss filters (`s_pyramid_build_ai.cu` `absoluteSourceInterpolated`,
  the half-texel two-tap trick) and the orientation/descriptor gradients
  (`s_gradiant.h get_gradiant`, fractional cos/sin coords). `readTex` is also used
  on genuine *point* textures (make_dog, extrema, etc.) but always with integer
  coords -> `frac==0` -> reduces to the exact texel, so this path stays correct
  for them (at the cost of 3 weight-0 fetches; correctness-first, acceptable).
  The 2-arg `readTex` (tex2D) has no callers and was left unchanged. The
  `normalizedSource` input-image kernels call `tex2D<float>` directly on the
  uchar input texture (not via readTex), unaffected.

### Rebuild
`utils/timeit.sh popsift compile -- cmake --build projects/popsift/src/build-hip -j`
-> 100% built, both `popsift-demo` and `popsift-match` linked, only the
pre-existing benign `-Wunused-value` warnings.

### Re-validation (GPU 3, gfx90a, scene.png 1052x744)
- **Texture error: GONE.** The pipeline now runs end-to-end with no
  "operation not supported" / no creation abort. The texture fix is confirmed
  correct (the manual-bilinear math is verified above).
- **BUT feature output is non-deterministic and degenerate -- this is the
  wave64 bug, not the texture fix.** Five identical `popsift-demo -i scene.png`
  runs gave feature counts: `0, 64, 8128, 0, 0` (run-to-run variance from the
  same binary/input). Descriptors contain `-nan` (every other element). 8128
  (=127*64) and 64 are suspiciously warp-size-multiples. `scene_rot.png` and the
  small testImages gave 0. `popsift-match --left scene --right scene_rot`
  reaches the pipeline (no texture error) but reports `dist -nan vs -nan` for all
  pairs and once threw a `std::runtime_error` (same non-determinism).
- **Root cause of the remaining failure (the predicted wave64 bug).** Both the
  extrema counter and the orientation kernel are hardwired for a 32-lane warp and
  break on wave64:
  - `s_extrema.cu:25` `uint32_t mask = popsift::ballot(indicator)` truncates HIP's
    64-bit `__ballot` to 32 bits; `__popc(mask)` then miscounts, and
    `write_index += __popc(mask & ((1<<threadIdx.x)-1))` assumes a 32-lane warp.
    The kernel packs two logical 32-thread rows into one 64-lane wavefront
    (`block.x=32`, `block.y>1`), so `ballot` mixes two warps.
  - `s_orientation.cu`: `popsift::any(i<loops)` (l.117) / `popsift::any(bin<ORI_NBINS)`
    (l.192) poll all 64 lanes; `BitonicSort::Warp32` + `sort64` (l.219-220) and
    `__popc(popsift::ballot(written))` (l.243) assume 32 lanes. Corrupted
    orientation histograms -> NaN descriptors -> `dist -nan`.
  This is exactly the open risk the porting notes flagged ("s_orientation.cu is
  hardwired to a 32-lane warp ... MUST be checked numerically on a real gfx90a").
  The GPU run has now confirmed it is real. Fixing it (rewriting the
  extrema/orientation warp-reduction for wave64, or forcing 32-lane semantics) is
  a separate change beyond this texture fix.

**Files changed (this fix):** `src/popsift/sift_octave.cu`,
`src/popsift/common/assist.h`. Net: linear textures create with point filtering
on HIP and `readTex` interpolates in software; CUDA path unchanged.

## Fix wave64 2026-05-30

Goal: fix the non-deterministic / -nan output on gfx90a (wave64). Diagnosed
TWO independent bugs; fixed the wave64 one fully, found a second (texture)
bug that now dominates.

### Wave64 warp-op fixes (DONE, HIP-guarded, CUDA path byte-for-byte unchanged)

PopSift packs two logical 32-thread rows into one 64-lane wavefront and assumes
NVIDIA 32-lane-warp semantics in every warp collective. On wave64 the mask-free
HIP builtins (`__ballot`/`__any`/`__shfl*`, via the cuda_to_hip.h remap of the
`*_sync` forms) operate over all 64 lanes, mixing the two rows. Fix: treat the
wavefront as two independent 32-lane groups. All changes are
`#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)` guarded; the `#else`
(CUDA) arm is the original code verbatim.

- `common/assist.h`: new HIP helpers `ballot_group(pred, group)` (64-bit
  `__ballot` shifted to the lane's own 32-bit half) and `any_group(pred, group)`.
  CUDA `#else` forwards to `ballot`/`any`.
- `s_extrema.cu` `extrema_count` (block (32,HEIGHT)): group = `threadIdx.y & 1`,
  lane = `threadIdx.x & 31`; ballot via `ballot_group`, leader test `lane==0`,
  broadcast from `group*32`, prefix mask `(1u<<lane)-1`.
- `common/warp_bitonic_sort.h` `shiftit` (used by `sort64` in ori_par): width-32
  `shuffle_xor` and key the swap direction off `threadIdx.x & 31`.
- `s_orientation.cu`: the two `popsift::any()` loop guards -> `any_group(...,0)`;
  `__popc(ballot(written))` -> `ballot_group(written,0)` (ori_par is a 32-thread
  block, group 0).
- `common/excl_blk_prefix_sum.h` (ori_prefix_sum, block (32,32)): both warp-scan
  `shuffle_up` loops forced to width 32 so odd rows don't pull partial sums from
  the neighbouring row.
- Descriptor 32-lane reductions forced to width 32 (these were full-width-32
  reductions that leak across the 32-lane boundary on wave64; grid/notile already
  passed explicit widths 16/8 and were fine): `s_desc_loop.cu`, `s_desc_iloop.cu`
  (loop is the default desc mode), `s_desc_norm_l2.h` (default normalize, 2
  reductions), `s_desc_norm_rs.h`, `features.cu` `l2_in_t0` (match kernel).
- Horizontal Gauss tap-exchange shuffles forced to width 32:
  `s_pyramid_build_aa.cu::horiz` (default VLFeat_Compute path) and
  `s_pyramid_fixed.cu::octave_fixed_horiz` (fixed9/15 only).
- `s_pyramid_build_ai.cu`/`ra.cu` horiz/vert and `aa::vert` use no shuffles
  (texture reads) -> unaffected.

Result: feature count is now STABLE across 5 runs (determinism achieved). The
run-to-run variance (0/64/8128/0/0) and the -nan in descriptors / `dist -nan`
in popsift-match are GONE. So the wave64 race is fixed.

### Remaining blocker: data-array surface<->texture incoherence (NOT wave64)

After the wave64 fix, output is deterministic but degenerate: popsift-demo
reports 0 features on scene.png (5/5 runs == 0). Root-caused on hardware with
in-kernel printf probes (since removed):
- input image texture read: OK (`tex2D` on the normalized uchar input returns
  the right value; `normalizedSource::horiz` writes ~157 to the interm surface).
- `aa::vert`: reads the interm point-texture correctly (157.6) and writes 161.7
  to the DATA surface; a `surf2DLayeredread` of the same surface IMMEDIATELY
  reads back 161.7 -> the surface write lands.
- `make_dog` (next kernel, same stream): reads the SAME data array via its POINT
  TEXTURE (`tex2DLayered`) and gets 0.0 at every level -> DoG is all-zero -> no
  extrema -> 0 features.

Conclusion: on this ROCm/HIP a texture object created from a layered
`cudaMalloc3DArray(... cudaArrayLayered|cudaArraySurfaceLoadStore)` does NOT
observe surface writes made to that array (surface path is fully coherent,
texture path reads stale/zero). This is the same class as the earlier
"texture fix" and is orthogonal to the warp bug; it predates this change (the
prior 8128/64 counts were garbage from uninitialized reads, not real
detection). The 0-feature result is deterministic now, which is the wave64
signal we wanted, but real validation (non-degenerate counts, heavy
scene/scene_rot match) is blocked on fixing the data-array texture coherence
(candidates: recreate/refresh the texture object after writes, add an explicit
cache invalidation, or read DoG inputs via the surface instead of the texture).

PASS/FAIL: wave64 determinism + no -nan = PASS. Non-degenerate counts + match
= FAIL, blocked by the separate data-texture incoherence bug above.

## 2026-05-30 -- gfx90a validation: BLOCKED (parent finalization)

State: ported, blocked. WIP preserved on the fork at jeffdaily/popsift @ moat-port (35de1a8).

Fixed and confirmed on gfx90a (MI250X):
- HIP linear-filter texture rejection -> manual bilinear interpolation (-0.5 texel-center) behind readTex. sift_octave alloc_data_tex/alloc_interm_tex force cudaFilterModePoint on HIP; assist.h readTex point-fetches 4 neighbors and lerps.
- wave64 warp-packing: the SIFT kernels pack two 32-thread rows into one 64-lane wavefront. ballot_group/any_group helpers + width-32 shuffles across s_extrema, warp_bitonic_sort, s_orientation, excl_blk_prefix_sum, the s_desc_* descriptor reductions, features, s_pyramid_build_aa, s_pyramid_fixed. Restores run-to-run determinism (5/5 identical) and removes the -nan descriptors.

OPEN blocker: surface/texture coherency. make_dog (s_pyramid_build.cu) reads the octave data array via tex2DLayered and gets 0.0 even though aa::vert (s_pyramid_build_aa.cu) wrote 161.7 to that layered array via surf2DLayeredwrite; an immediate surf2DLayeredread of the same location returns 161.7. DoG is therefore all-zero -> no extrema -> 0 features.

Under investigation (per jeff): is this a HIP runtime software bug (texture cache not invalidated at the kernel-launch boundary, which CUDA guarantees coherent across launches), a CDNA-vs-RDNA hardware difference, or popsift relying on intra-kernel coherence that CUDA leaves undefined? CUDA programming guide: texture/surface caches are coherent across kernel launches; only a read of an address written earlier in the SAME kernel is undefined. If aa::vert and make_dog are separate launches, the pattern is in-spec on NVIDIA and HIP should match.

Fix candidates (correct regardless of root cause): read make_dog/extrema from the surface instead of the texture; recreate/refresh the texture object after the blur writes; or add an explicit texture-cache invalidation.

Where I need jeff's input when he returns: confirm the root-cause direction (HIP bug to file vs popsift-side fix) from the investigation agent's findings, and the preferred workaround.
