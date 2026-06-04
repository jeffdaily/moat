# ROCm-DS vs MOAT RAPIDS: gap analysis and contribution plan

Assessment of AMD's official RAPIDS-on-ROCm product (the **ROCm-DS** GitHub org,
https://github.com/ROCm-DS) against MOAT's independent RAPIDS ports, to direct future
effort only at genuine gaps. Method: three-way diff per library (MOAT fork vs ROCm-DS repo
vs the rapidsai upstream tag ROCm-DS derives from), classifying every delta as a real gap, a
ROCm-DS-only advantage, or mere version skew. MOAT forks are local under `projects/<lib>/src`;
ROCm-DS repos and rapidsai reference tags were cloned under `agent_space/rocm-ds/`.

## TL;DR

ROCm-DS is real, GA (release 25.10, Nov 2025), and on the four core libraries it is
**equal-or-ahead of MOAT** in functionality. Its repos are NOT forks of rapidsai -- they are
renamed, restructured derivations pinned to **older RAPIDS bases (25.02; cuGraph 24.06)**,
and they target **only CDNA (gfx90a + gfx942), Linux**. MOAT forks are thin HIP-shim forks of
**newer rapidsai (25.08; cugraph/cucim 26.08)** validated on **gfx90a AND gfx1100 (RDNA3)**.

Two findings drive everything:

1. **The one consistent, defensible MOAT contribution across every library is RDNA3 (gfx1100,
   wave32) support.** ROCm-DS is uniformly CDNA-only and in several repos explicitly disables
   RDNA (rocGRAPH hard-errors on gfx11xx; hipDF gates gfx1100 as experimental/separate-build;
   hipVS lists wave64-only). MOAT validated gfx1100 on real hardware across all seven ports.
   This is the headline deliverable to push into ROCm-DS.
2. **cuML and cuCIM have NO ROCm-DS counterpart at all.** MOAT has working, GPU-validated
   ports of both. These are greenfield: propose new ROCm-DS projects (hipML, hipCIM).

The version skew (MOAT newer) is mostly inert -- it is a rebase ROCm-DS will do anyway, not a
MOAT achievement. Most "MOAT-only headers" are skew, not gaps. Honest net: MOAT does not
out-feature ROCm-DS on the existing five libraries except on architecture reach (gfx1100) and
two genuinely-missing libraries (cuML, cuCIM), plus one real functional gap in hipRaft (CK
distance backend).

> **CORRECTION (2026-06-04, from a ROCm-DS developer).** This analysis was built on the PUBLIC
> ROCm-DS forks, which are pinned to an OLDER RAPIDS version; AMD's latest ports are unreleased
> and partly in other orgs. Three corrections: (1) the AMD RAPIDS-on-ROCm ecosystem spans
> multiple orgs -- **ROCm-DS** + **ROCm-LS** (Life Sciences) + **AMD-AIOSS** (private). (2) cuCIM
> is NOT a gap: **ROCm-LS/hipCIM** is an official, active port (CDNA gfx942/gfx950 only). (3) the
> hipRaft CUTLASS-distance gap is a STALE-PUBLIC artifact: RAPIDS moved distance/NN from raft to
> cuvs, and AMD already replaced CUTLASS with **CK-tile** kernels in the private AMD-AIOSS/hipVS
> (amd-integration). So PR #9 (CK in hipRaft) is misdirected+duplicative. The one durable MOAT
> value-add across all of these remains **RDNA3/gfx1100 (consumer) coverage**, which every AMD
> port omits. cuML/hipML IS genuinely greenfield and wanted by the ROCm-DS team.

## Library-to-library map

| MOAT port (base) | ROCm-DS repo (base) | ROCm-DS status | Net standing |
|---|---|---|---|
| rmm (25.08) | hipMM (25.02) | Production v3.0.0 | ROCm-DS ahead (Python); MOAT adds gfx1100/Windows |
| raft (25.08) | hipRaft (25.02) \* | Production v0.1.0 | mixed; MOAT has CK distance backend hipRaft lacks; ROCm-DS builds more instantiations |
| cudf (25.08) | hipDF (25.02) | Production v2.0.0 | ROCm-DS far ahead (full libcudf + cudf.pandas); MOAT adds validated gfx1100 |
| cuvs (25.08) | hipVS (25.02) | Production v0.1.0 | ROCm-DS far ahead (full ANN + C/Py/Rust); MOAT = distance-only + gfx1100 |
| cugraph (26.08) | hipGRAPH + rocGRAPH (24.06) | Early-access beta | ROCm-DS broader (MG/MTMG, C-API, Python); MOAT adds RDNA3, which ROCm-DS disables |
| **cuml (25.08)** | **none** | **MISSING** | greenfield -> propose hipML |
| **cucim (26.08)** | **none** | **MISSING** | greenfield -> propose hipCIM |
| cuCollections | none (NVIDIA, not RAPIDS) | - | dependency only; out of scope |

\* PR #9 (CK distance/fused-NN into ROCm-DS/hipRaft, gfx90a-validated DISTANCE 11/11 + FUSED_NN 12/12) was **CLOSED 2026-06-04** as misdirected: per a ROCm-DS developer, RAPIDS moved distance/NN from raft to cuvs, and AMD already replaced CUTLASS with CK-tile kernels in the private AMD-AIOSS/hipVS. The "CUTLASS commented out" gap was a stale-public-fork artifact (see the CORRECTION note above). https://github.com/ROCm-DS/hipRaft/pull/9

ROCm-DS org full contents (verified via API): hipMM, hipDF, hipRaft, hipVS, hipGRAPH,
rocGRAPH, plus infra (ROCm-DS docs, ROCmDS-cmake, rocmds-logger). No hipML, no hipCIM, no
hipSpatial/hipProj.

---

## rmm -> hipMM

- **Bases:** MOAT 25.08, hipMM 25.02 (its `VERSION` 3.0.0 is the product version; README states
  "derived from RAPIDS RMM version 25.02"). Skew = 6 minor releases.
- **Public API:** zero real gaps. The only headers MOAT has that hipMM lacks (`rmm/logger.hpp`,
  `detail/cuda_memory_resource.hpp`) are 25.02->25.08 upstream additions (SKEW). The memory-resource
  framework, device_buffer/uvector/scalar, stream/event, prefetch/advise, and the stream-ordered
  async pool resources are byte-identical across all three trees.
- **ROCm-DS ahead:** hipMM ships a full working **Python/Cython package** (`python/rmm` pyx
  bindings); MOAT scoped Python out entirely.
- **Architecture:** hipMM = gfx942 + gfx90a, Linux only (`docs/.../install/hipMM-support.rst`).
  MOAT = gfx90a + gfx1100 + gfx1151 (Windows), all validated. RMM is header-only, so arch support
  is just the `-DCMAKE_HIP_ARCHITECTURES` flag.
- **Bottom line:** hipMM is missing nothing real in RMM's C++ API and is ahead on Python. MOAT's
  sole value-add is broadened hardware/OS reach (gfx1100 + gfx1151 + Windows).

## raft -> hipRaft

- **Bases:** MOAT 25.08, hipRaft 25.02 (branch `release/rocmds-25.10`, product 0.1.0). Skew = one
  full RAPIDS cycle; dominates the header deltas.
- **Documented-unsupported cross-check (the contribution-candidate question):** hipRaft documents
  `randomized_svd`, `interruptible.hpp`, `masked_matmul`-half, `make_regression`-double as
  unsupported. MOAT does **not** genuinely cover the first three either (randomized_svd would not
  even link -- no rocSOLVER `gesvdr`; interruptible carried but validation-deferred; masked_matmul
  in the SPARSE group which defaults OFF). Only `make_regression`-double is a plausible (low-
  confidence) MOAT win. So the headline "MOAT supports what hipRaft doesn't" mostly does NOT hold.
- **REAL GAP (MOAT ahead, high confidence):** MOAT reimplemented the **CUTLASS expanded-distance +
  fused-NN path against Composable Kernel** (`distance/detail/pairwise_matrix/dispatch_ck.cuh`,
  `fused_distance_nn/dispatch_fused_nn_ck.cuh`; GPU-validated DISTANCE 11/11, FUSED_NN 12/12).
  hipRaft has CUTLASS **commented out** (`cpp/CMakeLists.txt:240,262`, "re-add cutlass when it is
  available on AMD") with **no CK substitute** -- that path is dead source there. This gates
  distance performance for cuvs/cuml and is the strongest functional contribution candidate.
- **ROCm-DS ahead:** hipRaft compiles MORE instantiations -- the 4 lanczos solver TUs and
  `coalesced_reduction.cu` -- which MOAT defers (lanczos blocked by a documented upstream thick-
  restart bug, not a missing capability). hipRaft's lanczos compiles but shows no GPU validation.
- **Skew, not gaps:** all other MOAT-only headers (matrix/shift, sparse laplacian/diagonal,
  core multi_gpu/SNMG/nccl) are 25.08 additions hipRaft predates.
- **Architecture:** hipRaft = gfx90a + gfx942, Linux. MOAT adds gfx1100 via runtime host warp-size
  query (`cpp/include/raft/util/cudart_utils.hpp`, `cuda_dev_essentials.cuh`) + fat-binary multi-
  arch. Both sides independently widened the warp mask to 64-bit and both use `__activemask()` for
  partial-wavefront shuffle (parity -- do not overclaim active_mask as MOAT-unique).
- **Bottom line:** hipRaft is genuinely missing the **CK distance/fused-NN backend** and **gfx1100**.
  Everything else is skew or also-deferred-by-MOAT.

## cudf -> hipDF

- **Bases:** MOAT 25.08, hipDF 25.02 (confirmed three ways incl. `CHANGELOG_UPSTREAM_CUDF_25.02.md`;
  product v2.0.0). Skew = 3 cycles.
- **Decisive axis -- compiled translation units:** hipDF builds the **full libcudf (~928 source
  entries)**; MOAT builds **160 scoped `.cu`** (`cpp/cmake/hip/cudf_hip_sources.cmake`, "scoped
  core"). MOAT compiles ZERO of: `io/` (parquet/orc/json/csv/avro), `binaryop/compiled`, `text`
  (NLP), `rolling`, `datetime`, `interop` (Arrow), `jit`, and most of `strings/`. These are REAL
  functional gaps in MOAT (all existed in 25.02 -- not skew). MOAT is a correctness/portability
  demonstrator over the analytics core (groupby/reduce/join/sort/copying/lists/dictionary/tdigest),
  not a feature-complete library.
- **ROCm-DS ahead (large):** hipDF ships the full **Python + `cudf.pandas`** layer as GA Production.
  MOAT ported no Python.
- **Architecture (premise refined):** hipDF DOES support gfx1100 -- but **experimental only**, behind
  `CUDF_USE_WARPSIZE_32`, "not production-ready, use at your own risk", and **mutually exclusive with
  wave64 in one build** (FATAL_ERROR on mixed arch). MOAT delivers gfx1100 **validated on real
  hardware in a single arch-unified source** (no compile-time wave switch) via the logical-32
  primitives `cpp/include/cudf/detail/utilities/hip/warp_primitives.cuh` (`ballot_32`/`tile_any_32`),
  `valid_if.cuh`, and the wave64 null-mask fix in `cpp/src/merge/merge.cu` (`MergeNullableBitmaskWave64`
  test). gfx1151 is in neither (MOAT blocked it, hipDF never targets it).
- **Bottom line:** hipDF is far ahead overall. MOAT's defensible niche is narrow but real:
  production-posture, arch-unified, validated RDNA3 wave32 vs hipDF's experimental/gated/separate-build.

## cuvs -> hipVS

- **Bases:** MOAT 25.08, hipVS 25.02 (README: "port of cuvs branch-25.02"; product 0.1.0). Skew is
  inert here -- the distance subsystem is identical across 25.02/25.08 (same 42 dispatch files).
- **Decisive axis -- ANN coverage:** hipVS builds essentially the **entire cuVS ANN surface**
  (brute_force, sparse_brute_force, IVF-Flat, IVF-PQ, **CAGRA** all dtypes, NN-Descent, Vamana/DiskANN,
  HNSW, refine, select_k, kmeans/clustering, spectral embed, quantize, stats, reachability, dynamic
  batching, multi-GPU). MOAT builds **only the pairwise-distance foundation** (43 distance `.cu`, SIMT
  fallback). MOAT is a strict subset -- roughly a dozen ANN algorithm families and a full C/Python/Rust
  API that MOAT entirely lacks.
- **Within distance (MOAT's scope):** MOAT has nothing hipVS lacks. Every MOAT distance file exists in
  hipVS on the same metric x dtype matrix. The SIMT-vs-CUTLASS backend difference is a perf skew.
- **ROCm-DS ahead:** full Python (`amd-hipvs` on AMD PyPI) + C + Rust bindings; MOAT C++ only.
- **CAGRA wave64 (notable):** the warp_size=32 wall MOAT cited as the reason to defer CAGRA is
  **already solved in hipVS** -- `cpp/src/neighbors/detail/cagra/device_common.hpp` replaces the
  hardcoded `warp_size=32` with `raft::warp_size()` and a dynamic `team_size_bitshift` (the `case 6`
  stride-32 reduction handles 64-lane wavefronts). So MOAT's deferral set has no technical moat on CDNA.
- **Architecture:** hipVS = gfx90a + gfx942 wave64, Linux, **no RDNA in the support matrix**. MOAT
  validated the distance slice on gfx1100 wave32.
- **Bottom line:** hipVS is far ahead. MOAT's only niche is gfx1100/RDNA3 (distance), and even that is
  narrow since hipVS's distance code would likely run on RDNA, just untested/unsupported.

## cugraph -> hipGRAPH + rocGRAPH

- **Bases:** MOAT 26.08, ROCm-DS ~cuGraph 24.06. Skew = ~2 years (~4 RAPIDS releases) -- the headline
  caveat; most feature deltas are version artifacts.
- **Relationship:** **rocGRAPH** is the AMD compute backend -- a HIP fork of cuGraph's C++ (its
  `library/src/` mirrors cuGraph one-to-one) plus a native `rocgraph_*` C API, carrying SG **and MG and
  MTMG**. **hipGRAPH** is a thin backend-agnostic marshalling facade (AMD via amd_detail -> rocGRAPH,
  or NVIDIA via nvidia_detail) plus the `pylibhipgraph` Python binding.
- **Algorithm coverage:** rocGRAPH carries the **full cuGraph 24.06 matrix INCLUDING** the four legacy
  algorithms MOAT deferred (spectral clustering, MST, hungarian, Force Atlas 2), plus MG/MTMG. MOAT
  carries the larger 26.08 modern-SG matrix but **GPU-validated only BFS/SSSP/PageRank** -- everything
  else merely compiles. rocGRAPH ships them all but is itself an unvalidated `1.0.0b1` preview.
- **C-API / Python:** ROCm-DS strictly ahead -- two layered C APIs (rocGRAPH + hipGRAPH) and a
  functional `pylibhipgraph` (beta). MOAT delivers neither (both deferred).
- **Architecture (MOAT's unambiguous, structural advantage):** rocGRAPH `CMakeLists.txt` defaults to
  CDNA/GCN wave64 only (`gfx803;gfx900;gfx906;gfx908;gfx90a;gfx942`) and **explicitly comments out
  gfx1030/gfx1100/gfx1101/gfx1102** ("because of wavefront_size=32"), with a FATAL_ERROR on mixed-wave
  builds. **ROCm-DS does not support RDNA3 at all in this release.** MOAT validated SG BFS/SSSP/PageRank
  on real gfx1100 (W7800) via genuine host/device wave-size dispatch
  (`cpp/include/cugraph/utilities/warp_size_ct.hpp` + arch derivation in `cpp/cmake/hip/cugraph_hip.cmake`)
  and a concrete wave64 ballot-mask fix in the frontier prims
  (`prims/detail/extract_transform_if_v_frontier_e.cuh` et al).
- **Bottom line:** ROCm-DS is broader (MG/MTMG, C-API, Python, all legacy algos) thanks to the 2-year
  skew and larger investment. MOAT's defensible niche is **validated RDNA3 (gfx1100) graph support**,
  which ROCm-DS explicitly disables -- everything else is duplicative and behind on version.

## cuml -> (no ROCm-DS port; propose hipML)

- ROCm-DS has no ML library. MOAT has a GPU-validated ROCm/HIP port of the **cuVS-free single-GPU C++
  slice** (base 25.08; validated_sha `23511cd5`, tree content == validated).
- **Validated on gfx90a AND gfx1100 (33/34 ctest both arches;** the one failure is a documented
  ROCm-7.2.1 in-process-sequential LARS state-leak, not a port defect): linear/ridge regression
  (GLM/OLS), coordinate descent, LARS, quasi-Newton, SGD, PCA, TSVD, Holt-Winters, genetic
  programming, kernel SHAP, the 21 cuvs-free PRIMS primitives, plus Stage-2 `pairwise_distance` +
  `random_projection` (proven against the cuvs distance subsystem).
- **Carried (compiles, not standalone-GPU-validated):** decision tree, random forest library code,
  ARIMA/auto-ARIMA, permutation SHAP, synthetic datasets.
- **Deferred (cuVS-dependent):** dbscan, hdbscan, kmeans, knn/kde, tsne, umap, SVM (needs the
  Gram-matrix sub-stage), distance metrics (silhouette/trustworthiness), spectral clustering; plus
  multi-GPU and the entire Python layer. **Correction to earlier notes:** the ten cuvs-free comparison
  metrics (accuracy, adjusted_rand, etc.) were NOT actually built -- the `metrics` module was deferred
  as a block; only `pairwise_distance` was added back. They are an easy near-term add.
- **Feasibility:** a hipML built on **hipMM + hipRAFT is feasible today** for the entire cuvs-free SG
  slice (`CUML_LINK_CUVS=OFF`). With hipVS present, Stage 2 is feasible -- and because **hipVS already
  ships ANN/neighbors** (further along than MOAT's distance-only cuvs fork), kmeans/knn/dbscan are
  likely portable into hipML SOONER on real ROCm-DS than MOAT's own slice allowed.
- **Architecture:** gfx90a + gfx1100 validated (single multi-arch `libcuml.so`); gfx1151 blocked (dep
  chain, not device code). Wave32 fixes cited in `src_prims/selection/kselection.cuh`,
  `linalg/batched/gemv.cuh`, `decisiontree/batched-levelalgo/builder.cuh`. Key non-wave ROCm fixes:
  degenerate width-1 double GEMM rerouted from hipBLASLt (Tensile SIGSEGV) to gemv in `glm/preprocess.cuh`
  and `glm/ridge.cuh`; hipSOLVER strict-lda fix in Holt-Winters.

## cucim -> (no ROCm-DS port; propose hipCIM)

- ROCm-DS has no image library. MOAT has a GPU-validated ROCm/HIP port of **both halves** (base 26.08;
  validated_sha `2885611`; `depends_on: []` -- fully self-contained).
- **Validated on gfx90a AND gfx1100:**
  - CuPy compute half (`cucim.skimage` + `cucim.core.operations`, 33 source files + 3 HIPRTC kernels)
    on `cupy-rocm-7-0`.
  - C++ `libcucim` core (host cudart surface -> HIP; Catch2 101/101) and CuImage TIFF/SVS slide IO
    (cuslide plugin, 1250/1250 openslide cross-check; device="cuda" byte-identical to CPU).
- **Scoped out (no ROCm equivalent, with working CPU/POSIX fallbacks):** nvjpeg, nvimgcodec/cuslide2,
  cuFile/GDS.
- **Feasibility:** zero RAPIDS deps -> the **simplest greenfield project to stand up**. CuPy half needs
  only ROCm CuPy + numpy + scikit-image; C++ half needs only `hip::host` + libjpeg-turbo. Portability
  fixes were HIPRTC preamble coverage guarded by `runtime.is_hip` (cites in
  `_vendored/_ndimage_filters_core.py`, `measure/_regionprops_gpu_utils.py`), arch-agnostic.

---

## Prioritized contribution backlog

All items target ROCm-DS per the chosen delivery model. **Any PR/issue against a ROCm-DS (AMD,
non-jeffdaily) repo is upstream-visible and requires explicit user approval plus AMD-internal
coordination -- this backlog is a proposal, not authorization to push.** Contributions must be
rebased onto each ROCm-DS repo's base (25.02 / 24.06), since MOAT's newer base is skew.

### Tier 1 -- Greenfield (net-new libraries ROCm-DS lacks entirely)

1. **hipCIM (cuCIM).** Highest-confidence, lowest-friction: zero deps, both halves validated incl.
   gfx1100. Propose as a new ROCm-DS project; Tier-1 scope = the CuPy `cucim.skimage` compute half,
   Tier-2 = `libcucim` C++ core + slide IO.
2. **hipML (cuML).** SG cuvs-free slice on hipMM + hipRAFT is buildable today; lean on ROCm-DS hipVS's
   existing ANN to deliver MORE than MOAT's slice (kmeans/knn/dbscan) on day one. Stage in tiers
   (linear/solvers/decomp/trees/timeseries/genetic/SHAP first; distance/ANN-backed next).

### Tier 2 -- RDNA3 (gfx1100) wave32 enablement (the architecture headline)

The single consistent gap across the existing five ROCm-DS libraries. MOAT has validated, real-hardware
gfx1100 support and concrete portability code for each:

3. **rocGRAPH:** un-disable gfx1100 (currently hard-commented-out with a mixed-wave FATAL_ERROR);
   contribute the `warp_size_ct.hpp` host/device wave-size dispatch + the frontier-prim ballot-mask
   fix. MOAT validated BFS/SSSP/PageRank on gfx1100 -- ROCm-DS supports no RDNA today.
4. **hipDF:** promote gfx1100 from experimental/separate-build to validated; offer the arch-unified
   logical-32 path (`ballot_32`/`tile_any_32`) as an alternative to the compile-time wave switch.
5. **hipVS:** extend the wave64-only support matrix to RDNA3 (distance validated by MOAT; CAGRA is
   already wave-dynamic upstream so may need only test coverage).
6. **hipRaft:** add the wave32 path (runtime host warp-size query) to its wave64-parity warp layer.
7. **hipMM:** trivial -- header-only, just document/enable the gfx1100 arch flag.

### Tier 3 -- Specific functional gap

8. **hipRaft CK distance backend.** Contribute MOAT's Composable Kernel reimplementation of the
   expanded-distance + fused-NN path (`dispatch_ck.cuh`, `dispatch_fused_nn_ck.cuh`) -- hipRaft's
   CUTLASS path is commented out with no substitute, so this is genuinely missing and gates
   cuvs/cuml distance performance on AMD.

### Not worth pursuing

- Re-porting rmm/cudf/cuvs functionality: ROCm-DS is equal-or-far-ahead (Python, full libcudf, full
  ANN). MOAT's newer base is skew, not a contribution.
- gfx1151 / Windows: blocked everywhere by RAPIDS's lack of upstream Windows support; not a near-term
  ROCm-DS target.
- cuSpatial / cuProj: also absent from ROCm-DS, but MOAT never ported them -- out of scope here (a
  separate greenfield decision if desired).
