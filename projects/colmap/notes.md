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
