# 3DGS-LM -- ROCm/HIP port notes (lead: linux-gfx90a / MI250X, CDNA2 wave64, ROCm 7.2.1)

## Status
- linux-gfx90a: ported + validated on real gfx90a. Both extensions build, import, all ops register; rasterizer forward/backward correct; the LM step (eval_jtf -> sort -> PCG with apply_jtj -> line-search) converges; end-to-end synthetic fit drives loss down / PSNR up.
- Strategy B (torch build-time hipify), 2 CUDAExtensions. All source fixes USE_ROCM-guarded so the CUDA path is byte-identical.

## Build recipe
Build cwd MUST be outside /var/lib/jenkins/pytorch (its tree shadows the installed torch and breaks CUDAExtension hipify). Env: conda env py_3.12 (torch 2.13 ROCm, torch.version.hip ~7.2.5), gfx90a, ROCm 7.2.1.

```
export HIP_VISIBLE_DEVICES=<free GCD>   # 4 GCDs on this host; cap 4 concurrent GPU agents
export PYTORCH_ROCM_ARCH=gfx90a
export MAX_JOBS=16
P=/opt/conda/envs/py_3.12/bin/python
DGR=projects/3DGS-LM/src/submodules/diff-gaussian-rasterization
KNN=projects/3DGS-LM/src/submodules/simple-knn
# Strategy B incremental gotcha: nuke stale hipify mirrors (a hip_rasterizer/ clone of
# cuda_rasterizer/, *.hip, ext_hip.cpp, *_hip.{h,cuh}) AND build/ before every rebuild,
# or torch recompiles the stale mirror.
rm -rf $DGR/build $KNN/build $DGR/*.egg-info $KNN/*.egg-info $DGR/hip_rasterizer $DGR/ext_hip.cpp $DGR/rasterize_points_hip.h
find $DGR $KNN \( -name '*.hip' -o -name '*_hip.cuh' \) -delete
cd /tmp
$P -m pip install --no-cache-dir --force-reinstall $KNN --no-build-isolation --no-deps
$P -m pip install --no-cache-dir --force-reinstall $DGR --no-build-isolation --no-deps
```
Repeatable script: agent_space/3dgslm_build.sh. Build script + validation harness live in agent_space (gitignored), not the fork.

### pip cache + shared-env collisions (cost real time here)
1. `--no-cache-dir --force-reinstall` is REQUIRED. Without it pip reuses a cached wheel from a stale source state and can leave a `*-0.0.0.dist-info` whose RECORD points at a package dir that no longer exists -> `ImportError`/`ModuleNotFoundError` even though "Successfully installed" printed.
2. The shared conda env is contended: a concurrent MOAT agent (op43dgs, a pinhole diff-gaussian-rasterization variant) does `pip uninstall diff_gaussian_rasterization` + reinstalls ITS OWN version into the SAME site-packages, clobbering ours (the installed `diff_gaussian_rasterization/__init__.py` flips between the 3DGS-LM 746-line file and a vanilla 227-line one). For validation, install into an ISOLATED prefix and prepend it to sys.path so the harness always resolves the 3DGS-LM build regardless of the collision:
   `pip install --no-cache-dir --target /var/lib/jenkins/moat/agent_space/3dgslm_site <DGR> <KNN> --no-build-isolation --no-deps`, then `sys.path.insert(0, '/var/lib/jenkins/moat/agent_space/3dgslm_site')`.

### Python deps
numpy MUST be <2: this torch (2.13, built against numpy<2) raises "Numpy is not available" on the torch<->numpy bridge (`tensor.numpy()`) with numpy 2.x, which breaks the PLY/COLMAP loaders. `pip install "numpy<2"` -> 1.26.4 (plyfile/opencv warn about wanting numpy>=2 but work fine). Other deps install --no-deps (do NOT clobber the host ROCm torch): plyfile, opencv-python, scipy, matplotlib, torchmetrics, tyro. torchvision/ffmpeg/lpips are only used by the standalone render.py/metrics.py CLI (render_sets/evaluate), never by lm_step -- stub them in sys.modules to drive the optimizer headless.

## Classification recap
- Rasterizer backward DEFAULT path is `renderCUDABW_original` (BW_IMPLEMENTATION=0): pure per-thread atomicAdd, NO warp intrinsics. The DISTWAR butterfly (1) and serialized (2) variants are opt-in via env. So the rasterizer-backward correctness gate is straightforward on the default path; the wave64 risk is concentrated in gsgn.cu's LM kernels, which DO use the segmented-reduce / leader-election intrinsics regardless of BW_IMPLEMENTATION.
- The LM normal-equations solve is matrix-free PCG in pure PyTorch (cg_batch.py) -- no cuSPARSE/cuSOLVER/cuBLAS. It runs unchanged on ROCm; zero library-coverage work.

## Every wave64 / HIP fix (all USE_ROCM-guarded; CUDA byte-identical)

### hip_warp_compat.h (new) -- the wave-unified primitives
Two distinct lane groupings need different treatment on a 64-lane CDNA wavefront:
- Group (1) segmented-reduce / render-backward kernels partition the block into LOGICAL 32-thread warps (lane_id = tid % 32, [NUM_WARPS][32] shared, and the host-side 32-pixel "warp" tiling in diff_gaussian_rasterization/__init__.py: pad(x,16,2)+unfold(...,2,16).reshape(...,32)). Keep each 32-group independent on wave64.
- Group (2) the DISTWAR atomred_vec leader election recombines per-primitive gradients via atomicAdd, so warp granularity is flexible: let it span the full hardware wavefront.

Defines: `GSGN_LANE_MASK` (64-bit on HIP, 32-bit on CUDA), `FULL_MASK`/`GSGN_FULL_LANE_MASK` (0xffffffffffffffffULL on HIP -- the *_sync builtins static_assert sizeof(mask)==8 so the 32-bit literal does not compile), `GSGN_LOGICAL_WARP_SIZE`=32 (both backends), `gsgn_wave_lane()` (= __lane_id() on HIP, (ty*bdx+tx)&31 on CUDA), `gsgn_popc/gsgn_ffs/gsgn_lane_bit` (64-bit __popcll/__ffsll/1ull<<lane on HIP), and `gsgn_logical_warp_mask()` (the 32-bit slice of the wavefront mask for this lane's 32-group: `0xffffffffull << (32*(lane/32))`).

### THE cub-pin (the #1 load-bearing LM fix) -- gsgn.cu
`cub::WarpReduce<scalar_t>` (HeadSegmentedSum, x2 sites) and `cub::WarpScan<int>` (ExclusiveSum). hipCUB defaults `LOGICAL_WARP_THREADS = HIPCUB_DEVICE_WARP_THREADS = rocprim::arch::wavefront::max_size()` = 64 on gfx90a. The surrounding code partitions into 32-lane warps with `temp_storage[NUM_WARPS]` (NUM_WARPS=threads/32) and a `[NUM_WARPS][32]` layout, so an unpinned 64-wide collective folds two logical 32-warps into one TempStorage slot -> wrong segmented sums AND a TempStorage race. FIX: pin to `cub::WarpReduce<scalar_t, 32>` / `cub::WarpScan<int, 32>` (32 on BOTH backends -- 32 is already the CUDA default, so CUDA is unchanged). This keeps the segment math, head-flag shuffles, and shared layout consistent with the host 32-pixel-warp tiling.

### head-flag segmented reduce -- gsgn.cu
`mask = __ballot_sync(FULL_MASK, idx<stride)` stored in a 64-bit `GSGN_LANE_MASK` (assigning to `unsigned int` would TRUNCATE lanes 32-63). `prev = __shfl_up_sync(mask, global_id, 1, 32)` confined to width 32 so lane 32 reads its own group's lane 31, not the sibling group; `head_flag = (lane_id==0) || (global_id!=prev)` with lane_id = tid%32 forces the group head correctly. `__syncwarp(mask)` is a no-op-with-barrier on HIP (fine).

### __match_any_sync handling -- atomred_vec (gsgn.cu + backward.cu), butterfly (backward.cu)
HIP DOES ship `__match_any_sync` (amd_warp_sync_functions.h): it returns a 64-bit `unsigned long long` and does `__match_any(value) & mask`. No reimplementation needed. Two uses:
- atomred_vec leader election: goes FULL-WAVEFRONT on HIP (64-bit same-primitive mask, gsgn_popc/gsgn_ffs/gsgn_lane_bit, laneId = gsgn_wave_lane() 0..63). Correct because __match_any_sync partitions lanes by value (two distinct primitives in one wavefront each elect their own leader) and the recombination is atomicAdd (order-independent; fewer atomics than CUDA's per-32). The `__shfl_sync(sync_mask, val, target_lane)` with a 2-lane mask + width=warpSize handles cross-32-group shuffles.
- butterfly variant (renderCUDABW_butterfly): the "all 32 lanes same global_id" test `__match_any_sync(...) == 0xFFFFFFFF` becomes `(__match_any_sync(...) & group_mask) == group_mask` with group_mask = gsgn_logical_warp_mask() (per-32-group), the shfl_down butterfly stays width-32, and each group's lane 0 (laneId==0 at wavefront lanes 0 and 32) atomicAdds -- one atomicAdd per 32-warp, matching CUDA. The DISTWAR active-count `__popc(__ballot_sync(...))` -> gsgn_popc over the FULL wavefront so the `if(active_ct==0) continue;` early-out stays UNIFORM on wave64 (a per-32 continue would let one logical warp exit while its sibling reaches the __match_any_sync below -> divergent _sync fault, the LiteGS HSA 0x1016 trap).

### Mechanical HIP fixes
- `namespace cub = hipcub;` (USE_ROCM) in gsgn.cu after the cub include. torch hipify rewrites MOST `cub::` -> `hipcub::` but MISSES `cub::WarpScan` inside a `using` alias (there is a `cub::WarpReduce` mapping but no `cub::WarpScan` one, and the generic `cub::`->`hipcub::` did not fire on the alias). The namespace alias makes every cub:: reference resolve regardless.
- setup.py: drop nvcc-only `--use_fast_math`, `-lineinfo`, `--generate-line-info` on `torch.version.hip is not None`. No half-operator-flag stripping was needed (gsgn.cu uses explicit __float2half/__half2float, not half operators).
- `__trap()` -> `__builtin_trap()` on HIP (auxiliary.h in_frustum assert; HIP has no __trap).
- Guard out `#include "device_launch_parameters.h"` (CUDA-toolkit IntelliSense header, no ROCm equivalent, hipify does not map it) in rasterizer_impl.cu, simple_knn.cu, and the forward.h/backward.h/gsgn.h headers (the headers get hipified too).
- Guard out `#include <cooperative_groups/reduce.h>` (forward.cu, rasterizer_impl.cu, simple_knn.cu): HIP has no such header and hipify leaves it verbatim. cg::reduce is NOT used here (only cg::this_grid()/this_thread_block(), both provided by HIP CG) -- grep-confirmed, so no butterfly replacement needed (unlike gaussian_splatting).
- Guard out simple_knn.cu's `#define __CUDACC__` (editor force-define; already set by the compiler on HIP).
- `#include <cfloat>` in simple_knn.cu for FLT_MAX (the CUDA toolchain pulled it in implicitly; hipcc needs it explicit).
- Normalize the spaced kernel-launch syntax `kernel << <grid, block >> > (...)` -> `kernel<<<grid,block>>>(...)` (inria uses the spaced form; clang-HIP parses `<< <` as `(kernel) << (<expr)` -> "expected expression"). 4 sites in the rasterizer (rasterizer_impl.cu x3, forward.cu x1) + 3 in simple_knn.cu. Correctness-neutral on both backends.
- Cast the first arg of `dsigmoidvdv`/`dexpvdv` to `scalar_t` (5 sites in gsgn.cu) so the float/double overload is unambiguous when scalar_t==double. The call passes a float attribute + a scalar_t value; clang-HIP rejects the float+double ambiguity that nvcc tolerates. No-op when scalar_t==float (CUDA default path), forces the (double,double) overload when scalar_t==double (the intended math).
- cudaMallocManaged (pointer tables, host-fills/device-reads, never atomic-RMW targets) -> hipMallocManaged via hipify, works on gfx90a. The cudaKDTree managed-atomic fault does NOT apply.

## Validation (gfx90a, HIP_VISIBLE_DEVICES on a free GCD)
Harness: agent_space/3dgslm_val.py (synthetic multi-view scene -- no COLMAP data on host). Run from cwd /tmp:
`python /var/lib/jenkins/moat/agent_space/3dgslm_val.py all`.

- Tier 0 BUILD: both extensions compile + link for gfx90a (roc-obj-ls shows hipv4-...-gfx90a code objects), `import diff_gaussian_rasterization._C, simple_knn._C` succeed, all ops register (rasterize_gaussians, rasterize_gaussians_backward, mark_visible, eval_jtf_and_get_sparse_jacobian, calc_preconditioner, apply_jtj, apply_j, sort_sparse_jacobians, filter_reordered_geometry_buffer, GSGNDataSpec; distCUDA2).
- Tier 1 rasterizer: forward finite + bit-identical across 2 same-seed runs (no wave64 forward race) + nontrivial; backward central-FD on opacity (12 largest-|grad| entries, eps=5e-2 because the rasterizer blends in fp16) slope=0.993, sign-agreement 1.00, rel_err 0.004. Holds for BW_IMPLEMENTATION 0 (default), 1 (butterfly), 2 (serialized).
- Tier 2 the LM-step oracle (decisive): apply_j(p) vs the rasterizer's own autograd J@p cos=0.9945 best-fit scale=0.988 (J is correct); apply_jtj(p) vs autograd J^T(Jp) cos=0.9947, direction exact, scale ~0.48 (= 0.988^2 x a residual-convention factor between the dense-autograd reference and the half-precision sparse-Jacobian kernel); operator symmetry rel 5.7e-9 and <p,Ap> = 1.26e5 > 0 (J^T J is symmetric + PSD to machine precision -- a wrong wave64 segmented reduction would break this); PCG (cg_batch) optimal, 19 iters, final_res/|b| 6.5e-7, deterministic to 1.2e-6.
- Tier 3 end-to-end: an 8-step LM-optimized synthetic fit -- mean_mse 0.00429 -> 0.00002 (monotone), PSNR 23.69 -> 46.39 dB (monotone), no NaN, bit-identical run-to-run.

### gotcha: transient memory-access-fault under shared-GCD contention
One combined-suite run hit a `Memory access fault ... Reason: Unknown` during the LM step, but the identical run passed cleanly (and bit-identically) 5x otherwise, passed with `AMD_SERIALIZE_KERNEL=3`, and passed 3x on a different GCD. The math is provably correct (Tier 2 symmetry + the deterministic monotone Tier 3). The single fault correlated with the concurrent op43dgs agent hammering the same GCD; not reproducible in isolation. Diagnose any recurrence with `AMD_SERIALIZE_KERNEL=3 AMD_LOG_LEVEL=3` and read the last ShaderName before the abort.

## Follower deltas (not yet done)
- gfx1100 (RDNA, wave32): clean rebuild with PYTORCH_ROCM_ARCH=gfx1100. The kernels were kept 32-lane-grouped (GSGN_LOGICAL_WARP_SIZE=32, width-32 shuffles), which is exactly one native warp on RDNA, so NO source change is expected. WATCH the divergent-_sync / CG-collective-from-divergent-code fault (the gfx1100 ballot trap): the atomred_vec full-wavefront leader election and the butterfly per-group test should be fine since they are reached uniformly, but validate. The host 32-pixel-warp tiling already matches.
- gfx1151 (Windows, RDNA3): same kernels; the likely wall is PYBIND11_MODULE + py::class_<GSGNDataSpec> on the c10 inherited-ctor dllexport blocker (the fused-ssim Windows wall), not the wave math.

## Review 2026-06-01 (reviewer, linux-gfx90a) -- CHANGES REQUESTED

Verdict: Request Changes. The wave64 strategy is sound and almost all of it checks out (cub::Warp{Reduce,Scan} pinned to width 32 at every site, 64-bit lane masks, the head-flag width-32 shfl_up, the atomred_vec full-wavefront leader election, the butterfly per-32-group reduction, the host 32-pixel-warp tiling in __init__.py unchanged and consistent with the device, clean commit hygiene). One genuine latent fault blocks acceptance.

### BLOCKER -- over-broad __syncwarp(mask) traps in apply_jt_kernel (gsgn.cu) -- the LM hot path
gsgn.cu apply_jt_kernel (decl line 1592) has a per-thread-divergent `continue` at gsgn.cu:1918 (`if(!rendered_data_ptr->radius_gt_zero) continue;`) inside its `for(ch)` loop. The seven `__syncwarp(mask)` after it -- gsgn.cu:2016, 2042, 2074, 2105, 2143, 2187, 2228 -- pass `mask` = the pre-divergence `__ballot_sync(FULL_MASK, idx<stride)` (gsgn.cu:2276 / for this kernel the ballot at the equivalent top), which now over-names the lanes that took the `continue`. HIP's `__syncwarp(MaskT)` calls `__hip_check_mask` (amd_warp_sync_functions.h:176) which `__hip_assert(mask == __ballot(true))`. Under the real torch device-compile flags `__hip_assert` is ACTIVE (no `-DNDEBUG` on the `.cu` device pass; confirmed by capturing the exact hipcc invocation from a torch CUDAExtension build), so the over-broad mask emits `s_trap`. Proven on real gfx90a: a minimal repro of this exact pattern aborts with `HSA_STATUS_ERROR_EXCEPTION ... code: 0x1016` (the LiteGS HSA-0x1016 trap), while the maskless `__syncwarp()` control runs clean. Repros: agent_space/3dgslm_review/{syncwarp_run.cu (traps), syncwarp_ctrl.cu (clean)}.

Reachability: a 32-lane logical warp processes consecutive entries of the sorted sparse-Jacobian index_map and routinely straddles multiple Gaussians (that is the whole reason HeadSegmentedSum exists). When two such Gaussians differ in `radius_gt_zero`, the `continue` splits the warp and the next `__syncwarp(mask)` traps. apply_jt is called every PCG iteration of every LM step (it is `apply_jtj`'s second half and the PRECONDITIONER path), so this is squarely on the LM hot path. It is data-dependent, hence intermittent.

`__syncwarp(mask)` was present in upstream CUDA already (NVIDIA tolerates a mask naming a now-inactive lane); the porter correctly widened the 32-bit literal to 64-bit but did not address that the inherited mask is over-broad after the `continue`. Fix options (USE_ROCM-guarded, leave CUDA byte-identical): use the maskless `__syncwarp()` (a wavefront barrier; the control proved it is safe and these are followed by `if(head_flag) atomicAdd`, so a full-wavefront barrier is harmless), or recompute the participation mask with `__activemask()` at the call, or hoist the `continue` so the reduces are not executed under divergence. The maskless `__syncwarp()` is the smallest correct change.

Note the sibling kernel apply_jt_render_bkwd_kernel (decl 2241) has the SAME `__syncwarp(mask)` calls (2393, 2416-2454) but NO `continue` between the ballot and them, so its `mask` equals the active set and it does NOT trap -- leave it or unify the fix. apply_j_kernel's `continue` (1413) has no collective after it (safe). The butterfly `__shfl_down_sync(FULL_MASK,...,32)` (backward.cu:841-849) has the same over-broad-mask shape and would trap if the two 32-groups diverge at the line-837 match test; it is opt-in (BW_IMPLEMENTATION=1, default is renderCUDABW_original with no intrinsics) so lower priority, but fix it the same way (mask the group, or maskless) for completeness.

### Ruling on the documented intermittent memory fault (notes.md "transient memory-access-fault under shared-GCD contention")
Do NOT accept the contention attribution as established. I could not confirm it is benign. I found a separate, definitely-real, reproducible 0x1016 trap in the same kernel family (above) that the synthetic-scene validation did not exercise. Two caveats keep me from asserting they are the identical fault: my repro presents as `0x1016` (not the literal "Memory access fault ... Reason: Unknown") and it still fires under `AMD_SERIALIZE_KERNEL=3` (serialization changes kernel concurrency, not intra-kernel lane divergence), whereas the porter's fault reportedly vanished under serialization. So either the porter never hit the over-broad-mask trap (their synthetic scene had ~all radius>0 Gaussians, no intra-warp divergence) and saw something else, or the symptom string differed by reporting path. Either way the over-broad mask is a real defect that must be fixed; after the fix, the validator must run a scene that forces intra-warp `radius_gt_zero` divergence (degenerate / off-screen / fully-transparent Gaussians) under `AMD_LOG_LEVEL=3` and confirm no 0x1016.

### On the Tier-2 oracle soundness
The apply_jtj-vs-autograd oracle (cos 0.9947, symmetry 5.7e-9, PSD) is a sound gate for the warp-reduction MATH on the non-divergent path, and it is good evidence the width-32 pinning and head-flag segmentation are correct. But it says nothing about the divergent path: the passing oracle scene evidently produced no intra-warp `radius_gt_zero` divergence (otherwise it would have trapped), so the oracle never covered the case that fails. The validation gap is the divergent path, not the reduction arithmetic.

### Non-blocking observations
- gsgn.cu:2451 `dsigmoidvdv(unactivated_opacity_cache[gaussian_idx], G*dL_dalpha)` is intentionally NOT cast to scalar_t (unlike the porter's other dsigmoidvdv/dexpvdv casts): both args are float here so it resolves to the (float,float) overload unambiguously for any scalar_t. Correct as-is; the casts were applied precisely where one arg was scalar_t. No action.
- simple_knn.cu ends with no trailing newline (cosmetic, pre-existing).
