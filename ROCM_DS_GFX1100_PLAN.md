# ROCm-DS gfx1100 (RDNA3, wave32) enablement plan

Tier 2 of the ROCm-DS contribution backlog (see ROCM_DS_GAP_ANALYSIS.md). This plan is the
cross-host handoff: it is committed to the MOAT repo so the gfx1100 host's `orient.sh`
(`git pull --rebase`) makes it available; that host has a real gfx1100 GPU, which this gfx90a
host does not, and GPU validation on gfx1100 is the whole point.

## Relationship to PR #9, and the two gfx1100 workstreams

PR #9 (ROCm-DS/hipRaft) added the Composable Kernel distance backend using CK's **XDL** instance
(`DeviceGemmMultipleD_Xdl_CShuffle`), which targets the CDNA matrix cores (MFMA, gfx90a/gfx942)
in our fp32 regime. gfx1100 (RDNA3, wave32) enablement is two workstreams:

**A. Wave32 portability layer (library-wide -- the primary, correctness-critical work).** What
lets the ROCm-DS RAPIDS-derived libraries build and run correctly when the wavefront is 32 lanes
instead of 64: runtime warp-size, per-arch `WarpSize`, the `active_mask()` partial-wavefront
handling, and ballot-mask widths. ROCm-DS today is CDNA-only and in places explicitly disables
RDNA (rocGRAPH hard-comments-out gfx11xx; hipDF gates gfx1100 experimental). MOAT already
validated all of this on real gfx1100 in our forks; this effort upstreams it.

**B. CK WMMA distance instance for RDNA3 (extends PR #9 -- secondary, perf).** CK *does* support
RDNA, via WMMA rather than XDL. The install carries `DeviceGemmMultipleD_Wmma_CShuffle` with the
SAME `CDEElementwiseOperation` epilogue our XDL instance uses, plus `ck::is_gfx11_supported()` /
`is_xdl_supported()` to arch-gate at runtime (the gfx11 list includes gfx1150/gfx1151, useful for
the later Strix Halo target). So the distance fast path can be extended to gfx1100 by adding a
WMMA `DeviceGemm` alongside the XDL one, selected by arch. **Precision caveat (verified in the CK
WMMA traits):** RDNA3 WMMA inputs are fp16 / bf16 / int8 only -- there is NO fp32-input WMMA on
gfx11 (fp8/bf8 are gfx12). So the **fp32** distance that PR #9 accelerates has no CK fast path on
gfx1100 and stays on SIMT there; the WMMA opportunity is the **half / bf16** distance dtypes.
Net: correctness on gfx1100 comes entirely from workstream A; B is added matrix-core acceleration
for the reduced-precision distances, and is optional/follow-on.

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

raft is the worked example above; each other target repo has its own gfx1100-validated wave32
diff in the corresponding MOAT fork (`jeffdaily/<lib>` @ `moat-port`). Per-repo inputs (see
ROCM_DS_GAP_ANALYSIS.md for the exact file cites already gathered):

| Target ROCm-DS repo | Source MOAT fork | Key wave32 portability code |
|---|---|---|
| hipRaft | jeffdaily/raft | util/{warp_primitives,reduction,cudart_utils,cuda_dev_essentials}; runtime host warp-size query |
| hipMM | jeffdaily/rmm | header-only -- arch flag only, no wave32 code |
| hipVS | jeffdaily/cuvs | distance SIMT is wave-agnostic via raft::WarpSize; relies on hipRaft's wave32 layer; CAGRA already wave-dynamic upstream |
| hipDF | jeffdaily/cudf | detail/utilities/hip/warp_primitives.cuh (`ballot_32`/`tile_any_32`), valid_if.cuh, merge.cu logical-32 path |
| rocGRAPH | jeffdaily/cugraph | utilities/warp_size_ct.hpp (host/device dispatch), the frontier-prim ballot-mask fix |

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
   tightest raft wave32 diff. Smallest, highest-confidence gfx1100 win (workstream A). The CK
   fp32 path correctly stays SIMT on gfx1100 -- confirm it compiles and `*_ck_can_dispatch`
   returns false at runtime on RDNA (step 6), de-risking PR #9 on the non-CDNA arch. Workstream
   B (CK WMMA half/bf16 distance) is best evaluated here too, since the CK plumbing from PR #9
   is already in place -- add a `DeviceGemmMultipleD_Wmma_CShuffle` instance arch-gated by
   `ck::is_gfx11_supported()`.
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
