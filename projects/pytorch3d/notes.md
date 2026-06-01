# pytorch3d

3D deep-learning library (facebookresearch/pytorch3d). ext_type: torch-extension (Strategy B -- torch `BuildExtension` auto-hipify).

## gfx90a: COMPLETED upstream (exception to the jeffdaily-fork model)

The gfx90a ROCm port landed UPSTREAM as facebookresearch/pytorch3d PR #2039 "[ROCm] Port pytorch3d" (author jeffdaily), commit `b73d735ecf194c31de812feffef3a55cc3726128` (2026-06-01). Verified by the PR on AMD Instinct MI250X (gfx90a, warpSize=64), HIP 7.2, PyTorch 2.13, including the pulsar subrenderer and the test suite.

This is an exception to the usual MOAT model: the canonical port is upstream, not a jeffdaily moat-port fork. So `fork_url` points at upstream and `validated_sha` is the landed commit. MOAT adopts it as the gfx90a baseline. Optional: re-confirm on this MOAT gfx90a host with a validator run; not required since the PR already verified on MI250X.

### What the port did (from PR #2039)
- `setup.py`: detect ROCm via `torch.version.hip is not None`; treat `ROCM_HOME` as the GPU-toolkit-root analogue of `CUDA_HOME` (without it, `CUDA_HOME is None` silently demoted the build to a CPU-only `CppExtension`); skip `CUB_HOME`, CUDA-13 visibility flags, and `-ccbin=` on ROCm.
- torch `BuildExtension` auto-hipifies the `.cu` (`cuda_runtime.h`->`hip/hip_runtime.h`, `cub::`->`hipcub::`, `cudaStream_t`->`hipStream_t`), so most of the lift is build-system glue.
- `pytorch3d/csrc/pulsar/gpu/commands.h`: the CUDA `_rn`-suffixed FP rounding intrinsics (`__fadd_rn`/`__fdiv_rn`/`__fsqrt_rn`/`__fmaf_rn`/`__frcp_rn`) and `__saturatef` have no HIP equivalent (AMD's ISA has no instruction-level rounding-mode override), so on the `USE_ROCM` arm they expand to plain operators / `sqrtf` / `fmaf` / `1.0f/x` / `fmaxf(0,fminf(1,x))` -- rounding-mode-equivalent (both round-to-nearest-even). If FMA-fusion drift ever surfaces, add `-ffp-contract=off` to the pulsar TU. This is exactly our UPSTREAM_FINDINGS B1 (`__fsqrt_rn` 1-ULP) / B4 (`-ffp-contract`) story, now resolved in-tree upstream.

## gfx1100 + gfx1151: port-ready -- IMMEDIATE verification target

Followers reuse the upstream landed commit (validate-first; delta-port only on failure). This MOAT gfx90a host cannot validate gfx1100/gfx1151 -- a gfx1100 host's `orient.sh` selector picks up the `port-ready` follower.

### gfx1100 verification task
1. Clone upstream facebookresearch/pytorch3d at `b73d735ecf194c31de812feffef3a55cc3726128`.
2. Build the `_C` extension against a ROCm-built PyTorch for gfx1100 (RDNA3, wave32).
3. Run the pytorch3d test suite (`tests/`), including the pulsar renderer tests, on real gfx1100.

The decisive risk: the landed port was verified only on gfx90a (warpSize=64); gfx1100 is **wave32** (warpSize=32). Audit for wave64-baked assumptions the gfx90a verification could not catch:
- explicit-mask `__shfl*`/`__ballot` intrinsics (32- vs 64-bit masks) and hardcoded warpSize / `%32` / `/32` lane math, especially in the pulsar renderer's warp reductions.
- `cub`/`hipcub` `WarpReduce`/`WarpScan` defaulting to the 64-wide logical warp on hipCUB (pin `<...,32>` if any survive).
- the `_rn`->plain-op change is wave-agnostic (fine on wave32).
- the rocPRIM 4.2.0 GFX10/11 DPP bug: any `cub::DeviceSegmentedRadixSort` / `warp_exchange` path can fail to COMPILE on gfx1100 ("wavefront shifts are not supported on GFX10+") -- the raft lesson; may need a target exclusion or an upstream rocPRIM fix.

Outcome handling:
- Passes as-is -> set gfx1100 `completed`, validated_sha = `b73d735ecf...` (same commit; pure verification, no delta).
- wave32 breaks -> this would be the first time pytorch3d needs a jeffdaily fork: delta-port the wave32 fixes onto `jeffdaily/pytorch3d` @ `moat-port` (branched off the landed commit), validate on gfx1100, then the wave32 fix is a candidate to upstream as a follow-up PR (user-gated, like any upstream PR).

## Adoption note
priority 7.075 is the computed discovery score (stars 9891 -> 3.995, forks 1455 -> 1.582, pushed 2026-06-01 -> recency 1.498; weights stars 1.0 / forks 0.5 / recency 1.5 / half-life 180d). That places pytorch3d near the top of the table (above cuml 6.61 and cugraph 6.11; below only vllm 8.53 in the discovery pool), versus the lowest real table score op43dgs 3.393.
