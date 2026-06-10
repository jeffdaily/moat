# colmap notes

## Colmap -- external community port (tracking entry) 2026-06-10

Colmap is MOAT's FOUNDATIONAL reference port (the "colmap model" = Strategy A in
PORTING_GUIDE: enable_language(HIP) + a single cuda_to_hip.h compat header). It is NOT a
MOAT-pipeline port -- it is tracked here for visibility.

- Main ROCm PR: colmap/colmap#4420 "Add ROCm/HIP support for patch_match_stereo (AMD GPU)",
  author iShengnan (NOT jeff); jeff contributed substantially. OPEN, base main, +349/-49.
  Validated on MI250 (gfx90a) per the port's testing.
- jeff is a contributor, not the PR author.
- Follow-up (jeff's, more feature support): jeffdaily/colmap:rocm-sift-gpu @ e41e06e0 -- adds
  SIFT GPU ROCm support; will become a separate PR.
- Lessons already in PORTING_GUIDE from colmap: CuTexObj rule-of-five handle bug, ComputeDOG
  out-of-bounds-read bug.

TODO: (1) address the review questions on PR #4420 (jeff helping); (2) open the rocm-sift-gpu
follow-up PR when ready.

## PR #4420 review status (reviewed 2026-06-10)

PR is MERGEABLE, REVIEW_REQUIRED (needs ahojnnes's approving review). jeff's PR#1 (the
colmap-model rework) + PR#2 (CI + regression fixes), both merged into iShengnan's fork,
resolved nearly all threads:
- ANSWERED: cuda.cc:71 device-sort regression (fixed in #2), gpu_mat.h:195 (void)cudaFree
  rationale (no throw from dtor), CMakeLists.txt:451 HIP_ENABLED option (#2), README.md:31
  reword (#2), NevesLucas rocm-sdk auto-detect (deferred to a follow-up; he was amenable).
- gemini-bot CMake nits (hardcoded /usr/include paths, absolute .so/librocrand paths,
  redundant block): OBSOLETE -- the #1 rework switched to imported targets (hip::host,
  hip::hiprand, roc::rocrand). A one-line "addressed in the rework" would close the bot threads.
- OUTSTANDING (substantive): ahojnnes "tested e2e / CUDA-HIP equivalent?" -- iShengnan answered
  honestly "not directly compared" (HIP runs patch_match_stereo e2e on a 109-image set + suite
  passes, but no side-by-side CUDA numerical diff). Draft reply prepared (pure backend
  substitution, no algo change; COLMAP GPU suite passes on gfx90a MI250 + gfx1100 7900XTX; e2e
  produces valid reconstructions; bit-exact CUDA-vs-HIP is not a meaningful gate since FP
  reductions/atomics reorder across GPUs; offer AMD depth-map stats for the maintainer to compare
  vs a CUDA run). NOT posted (jeff: move on).

NEXT when resumed: (1) post the equivalence reply to ahojnnes (jeff's call) + the gemini "addressed"
closes; (2) the jeffdaily/colmap:rocm-sift-gpu follow-up PR (SIFT GPU).
