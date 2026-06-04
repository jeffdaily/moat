# hipML standup spec (cuML -> ROCm-DS hipML)

Restructure + dependency-wiring spec for standing up `jeffdaily/hipML` from MOAT's validated
cuML port (`projects/cuml/src` @ moat-port `23511cd`, RAPIDS base 25.08). Companion to
ROCM_DS_GREENFIELD_PLAN.md. Source work is small; the build-system rewrite + the 25.08-vs-25.02
dep skew are the real effort.

## Central tension

MOAT cuML bypasses rapids-cmake (USE_HIP early-`return()` in `cpp/CMakeLists.txt:24-26`, then
`find_package(rmm/raft/cuvs)` against pre-installed `_deps/*/install`). Family-A repos keep
rapids-cmake and fetch the ROCm-DS dep forks via `rapids_cpm_*` + `rapids_cpm_package_override`.
Target = Family-A: adopt the rapids-cmake + CPM-override machinery so hipML builds in-tree like
its siblings. Keep the `cuml::` C++ namespace and `<cuml/...>` includes unchanged; only repo/
product naming, build files, and dep wiring change.

## Seed + scaffolding

Seed hipML from cuml @ moat-port (carries the validated source + MOAT fixes).
ADD (templates from `agent_space/rocm-ds/hipRaft-fork`): `versions.json`, `rapids_config.cmake`
(use hipMM's permissive `[0-9][0-9]?` regex), `overrides.cmake` (`CMAKE_HIP_SOURCE_FILE_EXTENSIONS
hip;cu`), README/CHANGELOG (hipMM doc style), `VERSION`.
MODIFY: `cpp/CMakeLists.txt` (restore rapids path, add `rapids_cpm_package_override(versions.json)`
+ overrides), `cpp/cmake/thirdparty/get_{rmm,raft,cuvs,cccl}.cmake` (repoint), 3 guard lines.
The standalone `cpp/cmake/hip/cuml_hip*.cmake` become dead under Family-A (delete or keep only any
`*.in` the rapids path still needs).

## versions.json (pin ROCm-DS forks; cuML needs raft+cuvs beyond hipRaft's set)

```json
{
  "packages": {
    "hipmm":         {"version":"3.0.0","git_url":"https://github.com/ROCm-DS/hipMM","git_tag":"release/rocmds-25.10"},
    "raft":          {"version":"25.02.00","git_url":"https://github.com/ROCm-DS/hipRaft","git_tag":"release/rocmds-25.10"},
    "cuvs":          {"version":"25.02.00","git_url":"https://github.com/ROCm-DS/hipVS","git_tag":"release/rocmds-25.10"},
    "rocmds_logger": {"version":"1.0.0","git_url":"https://github.com/ROCm-DS/rocmds-logger","git_tag":"release/rocmds-25.10"},
    "libhipcxx":     {"version":"2.7.0","git_url":"https://github.com/ROCm/libhipcxx","git_tag":"release/rocmds-25.10"},
    "hipco":         {"version":"0.3.0","git_url":"https://github.com/ROCm/hipCollections","git_tag":"release/rocmds-25.10"},
    "rocthrust":     {"version":"4.0.0"}
  }
}
```
The override KEY must match the CPM package name used in `rapids_cpm_find(<name>)` (hipMM is keyed
`hipmm`); verify raft/cuvs keys against ROCmDS-CMake's versions.json. cudf is NOT a C++ dep (0
`<cudf...>` includes) -- no hipDF pin. cuML uses header-only `raft::raft` -> get_raft sets
`RAFT_COMPILE_LIBRARY OFF`. get_cuvs points at `ROCm-DS/hipVS.git`, gated on `CUML_LINK_CUVS`.

## Guard migration (USE_HIP -> __HIP_PLATFORM_AMD__): 3 one-line edits

Footprint is tiny -- only 3 single-guard sites need changing (two others are already dual-guarded
`USE_HIP || __HIP_PLATFORM_AMD__`):
- `cpp/src/glm/preprocess.cuh:149` `#if defined(USE_HIP)` -> `__HIP_PLATFORM_AMD__`
- `cpp/src/glm/ridge.cuh:73` same
- `cpp/src/metrics/pairwise_distance.cu:89,197` `#if !defined(USE_HIP)` -> `!defined(__HIP_PLATFORM_AMD__)`
All 3 are in HIP TUs under the Family-A `.cu->HIP` rule, so the guard truth value is unchanged.
(Fallback 4b: just define USE_HIP in hipML's build -- zero source edits, but diverges from sibling
convention.)

## Tiering

- **Tier 1 (hipMM + hipRaft, `CUML_LINK_CUVS=OFF`):** reproduces MOAT's cuvs-free SG slice. Restrict
  to the validated set: `-DCUML_ALGORITHMS="linear_model;solvers;decomposition;ensemble;tsa;genetic;explainer;datasets"`
  (else the full glob pulls cuvs algos hipVS may not back).
- **Tier 2 (hipVS):** feasible -- hipVS's `cuvs::distance::pairwise_distance` signature (float/double,
  layout_c/f_contiguous = row/col_major) matches cuML's calls in `pairwise_distance.cu:41-84` exactly.
  Validate via `SG_RPROJ_TEST`. Stretch: hipVS also ships `cuvs/distance/grammian.hpp` (which MOAT's
  cuvs slice lacked), so SVM (deferred on MOAT) MIGHT become buildable -- check `nm -DC libcuvs.so`
  for GramMatrix symbols first.

## Risks (honest)

1. **25.08 vs 25.02 skew (dominant).** cuML 25.08 was validated against raft/cuvs 25.08; hipRaft/hipVS
   are 25.02. Any raft API drift the cuML slice touches breaks the compile. Highest-leverage gate:
   **confirm `raft::warp_size()` exists in hipRaft 25.02** -- the entire wave32/wave64 correctness story
   (kselection.cuh, device_utils.cu, batched/gemv.cuh, builder.cuh) depends on it. Also watch
   `raft::linalg::gemm/gemv` overloads (the hipBLASLt reroutes), mdspan layout aliases, and the
   `cuvs::distance::DistanceType` enum values (an enum reorder is a silent CORRECTNESS bug -> GPU
   validation mandatory). Expect an iterative raft-25.02 compat-fix loop; keep 25.08 source and patch
   gaps (don't downgrade to 25.02 and throw away MOAT's validation).
2. **Curated-list vs rapids-glob sources.** Use `-DCUML_ALGORITHMS=...` to reproduce MOAT's slice.
3. **VERSION/regex + get_raft version.** Use hipMM's permissive regex; hard-code `RAFT_VERSION 25.02`
   in get_raft (don't derive from hipML's VERSION).
4. **treelite** keep MOAT's pinned 4.4.1. **LARS** in-process known issue carries forward (not a defect).

## Ordered checklist (gfx90a execution session)

1. Seed jeffdaily/hipML from cuml@moat-port; confirm cuml:: intact.
2. Add Family-A scaffolding (versions.json, rapids_config permissive-regex, overrides, README/CHANGELOG, VERSION).
3. Wire `cpp/CMakeLists.txt` (drop USE_HIP early-return; add override + overrides.cmake).
4. Repoint get_{rmm,raft,cuvs}.cmake (raft header-only + RAFT_VERSION 25.02 hard-coded; cuvs gated).
5. Guard migration: the 3 edits above.
6. **PRE-FLIGHT GATE:** standalone-build hipMM+hipRaft+hipVS @ release/rocmds-25.10 on gfx90a; confirm
   hipRaft exports `raft::warp_size()` + header-only `raft::raft`; `nm -DC` hipVS libcuvs for
   pairwise_distance (Tier 2) and GramMatrix (SVM stretch). If the forks don't build at that tag, STOP + report.
7. Tier 1 build: `CUML_LINK_CUVS=OFF`, restricted CUML_ALGORITHMS, gfx90a, SINGLEGPU=ON, tests ON.
   Expect a raft-25.02 compat-fix loop.
8. Tier 1 validate (real gfx90a): MOAT baseline (SG OLS/RIDGE/CD/QN/SGD/PCA/TSVD/HOLTWINTERS/SHAP_KERNEL/
   GENETIC + 21 PRIMS incl KSELECTION wave64). LARS in-process fail = known issue.
9. Tier 2 build: `CUML_LINK_CUVS=ON`; verify DistanceType enum value-compat (correctness).
10. Tier 2 validate: SG_RPROJ_TEST 8/8 + non-regression.
11. Multi-arch: add gfx1100; `llvm-objdump --offloading` shows both code objects; re-confirm raft::warp_size wave32.
12. Stretch: if hipVS backs GramMatrix, attempt SVM (pure upside).

Bottom line: small source work (3 guard lines, cuml:: untouched), mechanical build rewrite; the
25.08-vs-25.02 dep skew is the real threat -- gate on `raft::warp_size()` in hipRaft 25.02 before
investing, and budget a compat-fix loop. Tier 1 is the right first milestone.

## Pre-flight gate result (2026-06-04) -- PASS, bounded compat set

Checked hipRaft 25.02 headers (`agent_space/rocm-ds/hipRaft-fork`) against cuML 25.08's needs. No
hard blocker; the deps are header libs at `release/rocmds-25.10`. Findings:

- `raft::WarpSize` (device constexpr) -- PRESENT (`util/cuda_dev_essentials.cuh`); cuML's 12
  device-context uses are fine.
- `raft::warp_size()` -- in hipRaft 25.02 it is `__device__`-only (`util/cudart_utils.hpp:68`); the
  host runtime query is `raft::host_warp_size(stream)` / `host_warp_size(device_id)` (NO no-arg
  form). Our raft 25.08 added a host overload `inline int warp_size(){return host_warp_size();}`
  that hipRaft 25.02 lacks -> cuML's 3 HOST call sites must use `host_warp_size(...)`. **Patch
  applied to the seed:**
  - `cpp/src_prims/selection/kselection.cuh:367` -> `raft::host_warp_size(stream)`
  - `cpp/src_prims/linalg/batched/gemv.cuh:120` -> `raft::host_warp_size(stream)`
  - `cpp/src/decisiontree/batched-levelalgo/builder.cuh:509` -> `raft::host_warp_size(builder_stream)`
- `raft::linalg::gemv` / `gemm` (the GLM hipBLASLt reroutes in preprocess.cuh/ridge.cuh) -- PRESENT
  with `raft::resources const& handle` signatures; verify exact args at build (low risk).
- Tier-2 only (deferred): `cuvs::distance::DistanceType` enum value-compat when CUML_LINK_CUVS=ON.

Seed done: `agent_space/rocm-ds/hipML-build` from cuml@moat-port (25.08) + the 3 warp_size patches.
NEXT: Family-A wiring (versions.json/get_* repoint/guard migration) then the Tier 1 build
(`CUML_LINK_CUVS=OFF`), iterating any remaining 25.02 raft-API drift the compiler surfaces.
