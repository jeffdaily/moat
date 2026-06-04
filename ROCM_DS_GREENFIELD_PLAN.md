# ROCm-DS greenfield plan: hipCIM (cuCIM) and hipML (cuML)

Tier 1 of the ROCm-DS contribution backlog (see ROCM_DS_GAP_ANALYSIS.md). cuML and cuCIM have
NO ROCm-DS counterpart; MOAT has validated ports of both. Delivery model (jeff): contribute into
ROCm-DS by standing up NEW ROCm-DS projects. Decisions (jeff, this session): **build validated
candidates now and propose to AMD with evidence**; **both targets in parallel**.

## Engagement model

Greenfield is a NEW-PROJECT proposal, not a PR into an existing repo. The candidate repos are
built under `jeffdaily/` (within MOAT autonomy); the AMD-facing steps are gated:

- FREE (do now): create `jeffdaily/hipCIM` and `jeffdaily/hipML`, restructure our validated MOAT
  ports into ROCm-DS house style, build and GPU-validate.
- GATED (needs jeff): the internal AMD roadmap check (confirm hipCIM/hipML are not already in
  private AMD progress -- the same no-duplicate principle that started this effort), and any
  ROCm-DS-facing proposal (issue / repo donation). Show the validated candidate, then propose.

## Evidence base (already validated)

- cuCIM: `jeffdaily/cucim` @ `moat-port` (`2885611`), validated gfx90a + gfx1100, `depends_on: []`.
- cuML: `jeffdaily/cuml` @ `moat-port` (`23511cd`), validated gfx90a + gfx1100; cuvs-free SG slice
  (deps rmm, raft; cuvs-dependent algos deferred).

## House-style target (Family A, from hipMM/hipDF/hipRaft)

A ROCm-DS RAPIDS-derived repo keeps the upstream C++ namespaces (rmm:: in hipMM, cudf:: in hipDF)
but: renames the repo/product (cuCIM->hipCIM, cuML->hipML); adds `versions.json` (pins sibling
ROCm-DS dep forks at release/rocmds-25.10), `rapids_config.cmake`/`overrides.cmake` (the
`CMAKE_HIP_SOURCE_FILE_EXTENSIONS hip;cu` trick), `build.sh`; uses `__HIP_PLATFORM_AMD__` (not
USE_HIP) as the guard; puts AMD tests under `cpp/tests/amd/`; disables fork Actions; README/docs in
the hipMM/hipDF pattern. CK (if used) is header-only from the ROCm install.

## Track 1: hipCIM (cuCIM) -- simplest, do first on gfx90a

`depends_on: []`, so NO dep rewiring -- the big simplifier. Two halves, both validated:
- CuPy `cucim.skimage` compute half (33 sources + 3 HIPRTC kernels) on `cupy-rocm-7-0`.
- C++ `libcucim` core + CuImage TIFF/SVS slide IO (Catch2 101/101; cuslide 1250/1250 openslide
  cross-check).
Scoped out (no ROCm equivalent, CPU/POSIX fallbacks): nvjpeg, nvimgcodec/cuslide2, cuFile/GDS.

Steps: (1) create `jeffdaily/hipCIM`, Actions off; (2) seed from cucim @ moat-port; (3) rename to
hipCIM, add the house-style adaptation layer (versions.json trivial -- no RAPIDS deps; build.sh;
README); (4) move the validated HIP guard to `__HIP_PLATFORM_AMD__`; (5) build + run both halves on
gfx90a; (6) gfx1100 validation can follow on the gfx1100 host; (7) draft the AMD proposal (gated).

## Track 2: hipML (cuML) -- dependency-heavy, scope concurrently

Builds on the now-existing ROCm-DS stack: hipMM + hipRaft (+ hipVS for the distance/ANN tier).
The hard part is the dep wiring: `versions.json` + `get_{rmm,raft,cuvs}.cmake` must point at
hipMM/hipRaft/hipVS instead of rapidsai, and the build must resolve them via their rapids-cmake
CPM. Tiering (from the gap analysis):
- Tier 1 (hipMM + hipRaft): linear models, solvers (CD/LARS/SGD/QN), PCA/TSVD, decision tree +
  RF library, ARIMA/Holt-Winters, genetic, kernel SHAP, the cuvs-free PRIMS layer.
- Tier 2 (add hipVS): pairwise_distance + random_projection (proven), then -- leveraging hipVS's
  full ANN, which exceeds our own cuvs slice -- kmeans/knn/dbscan, distance metrics, etc.
- Tier 3: tree SHAP, FIL/RF tests, Python layer, multi-GPU.
A subagent is scoping the concrete restructure + dep-wiring spec; that spec lands here.

## Status

- hipCIM: NOT STARTED -> in progress on gfx90a this session.
- hipML: scoping (dep-wiring spec) in progress.
- AMD proposal: gated, pending validated candidates + jeff's internal roadmap check.
