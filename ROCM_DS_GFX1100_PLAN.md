# ROCm-DS gfx1100 (RDNA3, wave32) enablement plan

Tier 2 of the ROCm-DS contribution backlog (see ROCM_DS_GAP_ANALYSIS.md). This plan is the
cross-host handoff: it is committed to the MOAT repo so the gfx1100 host's `orient.sh`
(`git pull --rebase`) makes it available; that host has a real gfx1100 GPU, which this gfx90a
host does not, and GPU validation on gfx1100 is the whole point.

## Why this is a distinct effort from PR #9

PR #9 (ROCm-DS/hipRaft, merged-or-pending) added the **Composable Kernel MFMA** distance
backend. CK XDL requires matrix cores, which exist on CDNA (gfx90a/gfx942) but NOT on RDNA3
(gfx1100). So on gfx1100 the CK path must cleanly no-op and fall back to SIMT. gfx1100
enablement is therefore a *separate* concern: the **wave32 portability layer** that lets the
ROCm-DS RAPIDS-derived libraries build and run correctly when the wavefront is 32 lanes
instead of 64. ROCm-DS today is CDNA-only (gfx90a + gfx942) and in places explicitly disables
RDNA (rocGRAPH hard-comments-out gfx11xx; hipDF gates gfx1100 as experimental/separate-build).
MOAT already validated all of this on real gfx1100 in our own forks; this effort upstreams it.

## Source material (already gfx1100-validated in MOAT forks)

The wave32 fixes live in our MOAT forks and are the transplant inputs. For raft (the first
target), the concrete files:

- `cpp/include/raft/util/warp_primitives.cuh` -- `WarpMask`/`RAFT_LANE_MASK_ALL` 64-bit on HIP, `active_mask()` (`__activemask()`), ballot/popc/ffs sized to the mask.
- `cpp/include/raft/util/reduction.cuh` -- `active_mask()` in the logical-warp reductions; `binaryBlockReduce` 64-bit ballot + `__popcll`.
- `cpp/include/raft/util/cudart_utils.hpp` -- `warp_full_mask()` 64-bit, `warp_size()` runtime query.
- `cpp/include/raft/util/cuda_dev_essentials.cuh` -- `WarpSize` per-arch (64 on `__GFX9__`, else 32), `laneId()` -> `__lane_id()`.
- The **runtime host warp-size query** replacing the compile-time `RAFT_HOST_WARP_SIZE` constant (MOAT fork commit `ce0fa68c`), which is what lets a single fat binary serve wave64 and wave32.

NOTE: hipRaft already widened its warp mask to 64-bit and uses `__activemask()` (parity with
MOAT -- do not re-contribute those). The genuine gap is the **wave32 path**: per-arch
`WarpSize`/`warp_size()` and the host-side runtime dispatch so gfx1100 (wave32) is correct in
the same build, plus un-gating gfx1100 in the arch lists.

## Approach (Family-A contribution playbook, proven by PR #9)

The RAPIDS-derived ROCm-DS repos (hipMM, hipDF, hipRaft, hipVS) keep raft/cudf's exact layout,
so files transplant path-for-path. Per repo:

1. `gh repo fork ROCm-DS/<repo>` (if not already forked); disable Actions:
   `gh api -X PUT repos/jeffdaily/<fork>/actions/permissions -F enabled=false`.
2. Branch off the repo's `release/rocmds-25.10` (e.g. `gfx1100-wave32`).
3. Transplant the wave32 portability diff from the corresponding MOAT fork into the identical
   paths. Adapt the HIP guard `USE_HIP` -> `__HIP_PLATFORM_AMD__` (ROCm-DS convention; they do
   NOT define USE_HIP). Keep CUDA/wave64 byte-identical.
4. Un-gate gfx1100 in the arch handling (CMake arch lists / wave-size dispatch). Confirm a
   single build can carry gfx90a + gfx1100 (or document if the repo requires per-wave builds).
5. Add/point AMD tests under `cpp/tests/amd/` (or the repo's test tree).
6. Build with the repo's own `build.sh ... --gpu-arch=gfx1100` (deps auto-fetch via their
   rapids-cmake CPM). For hipRaft also confirm the **PR #9 CK code compiles for gfx1100 and
   `*_ck_can_dispatch` returns false at runtime on RDNA** (CK XDL is CDNA-only), so it falls
   back to SIMT -- this is a robustness check for PR #9 on the non-CDNA arch.
7. Run the tests on the real gfx1100 GPU. Validate correctness (wave32 ballot/reduction
   results, no HSA memory faults).
8. Jargon-scrub (PORTING_GUIDE refs are the usual leak), ASCII-only, ROCm/HIP casing.
9. Open the PR to ROCm-DS -- **gated: needs jeff's explicit approval** (upstream-visible AMD
   repo) plus AMD-internal coordination. Draft body, wait for yes.

## Ordering (start small, build confidence)

1. **hipRaft** -- first; we have the fork (`jeffdaily/hipRaft`), an open PR (#9), and the
   tightest raft wave32 diff. Smallest, highest-confidence gfx1100 win. The CK fallback check
   (step 6) also de-risks PR #9 on RDNA.
2. **hipMM** -- trivial (header-only); just enable the gfx1100 arch flag and validate the rmm
   gtest suite on gfx1100.
3. **hipVS** -- extend the wave64-only support matrix to RDNA3; distance is the validated slice
   (CAGRA is already wave-dynamic upstream, may need only test coverage).
4. **hipDF** -- promote gfx1100 from experimental/separate-build to validated; offer the
   arch-unified logical-32 path (`ballot_32`/`tile_any_32`) as an alternative to the
   compile-time wave switch.
5. **rocGRAPH** (Family B, AMD library template -- different layout: `library/`+`clients/`) --
   un-disable the commented-out gfx11xx targets; contribute `warp_size_ct.hpp` host/device
   dispatch + the frontier-prim ballot-mask fix; validate SG BFS/SSSP/PageRank on gfx1100.

## What the gfx1100 host needs

- ROCm (>= 7.0.2; 7.2.x fine) and a gfx1100 GPU (e.g. Radeon Pro W7800).
- The wave32 source diff: clone `jeffdaily/<lib>` @ `moat-port` (the MOAT fork is on GitHub;
  `projects/<lib>/src` is local to the gfx90a host and will NOT be on the gfx1100 host). The
  validated wave32 fixes are in that fork's tree at the paths listed under "Source material".
- This plan + ROCM_DS_GAP_ANALYSIS.md (already in MOAT, auto-pulled).
- gh auth as jeffdaily.

## Status

- gfx90a CK backend: done, PR #9 (https://github.com/ROCm-DS/hipRaft/pull/9).
- gfx1100 enablement: NOT STARTED -- to be executed on the gfx1100 host per this plan.
