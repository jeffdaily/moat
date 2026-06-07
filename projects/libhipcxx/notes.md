# libhipcxx -- ROCm/HIP semaphore enablement notes (lead: linux-gfx90a, ROCm 7.2.1, MI250X)

Fork: https://github.com/jeffdaily/libhipcxx branch `enable-semaphore`
(this is the work branch; the schema has no dedicated field for it, so it is
recorded both here and as `work_branch` in upstream.json and as
`fork_default_branch` in status.json). Upstream base: ROCm/libhipcxx
`amd-develop` @ fa4ccc6beb77bfaa59a6fbeeebc94a4f18678945. Work HEAD:
e2e8f70dd8737a5cf8b7b798b99654d06c8eb326.

Unlike a typical MOAT port, the AMD targets here all share the SAME fork branch
AND the SAME upstream-vs-base relationship; there is no Windows-specific delta
expected. Followers validate-first on `enable-semaphore` and only delta-port if
a wave32 failure shows up.

## (a) What the change is and why

The counting/binary semaphore implementation (`cuda::counting_semaphore`,
`cuda::binary_semaphore`, both `cuda::std::` and `cuda::` spellings) is already
present in libhipcxx and is built entirely on the `cuda::std::atomic`
wait/notify machinery that ROCm/HIP already supports. The ONLY thing stopping
it on AMD was a hard gate: the two public umbrella headers `<cuda/semaphore>`
and `<cuda/std/semaphore>` carried an `#error` for `__HIP_PLATFORM_AMD__`. The
change removes that `#error` so the existing implementation is reachable on AMD.
It does NOT add or rewrite any semaphore logic.

Why it matters:
- Restores the libhipcxx v2.2.0 capability (semaphores worked before the gate
  was added) and answers ROCm/libhipcxx issue #10.
- Consumers that need it: oneflow's `lru_cache` (uses
  `cuda::binary_semaphore<thread_scope_device>`); the hipCollections / cuCollections
  family and dgl pull semaphores through the same umbrella headers.

## (b) gfx90a baseline result (lead, validated)

MI250X, ROCm 7.2.1, `--offload-arch=gfx90a` (CDNA2, wave64). Real-GPU validated;
this is why linux-gfx90a is `completed` with validated_sha = the work HEAD.

PASS:
- Device producer/consumer: `cuda::binary_semaphore<thread_scope_device>`
  across two blocks (the oneflow type). See validation/sem_test.cpp.
- Umbrella headers compile + run on HIP/AMD post-change. See
  validation/sem_umbrella_smoke.cpp.
- Upstream conformance under `.upstream-tests/test/std/thread/thread.semaphore/`:
  version.pass.cpp, max.pass.cpp, try_acquire.pass.cpp, acquire.pass.cpp,
  release.pass.cpp.
- heterogeneous/semaphore.pass.cpp (host+device).
- thread_scope_device and thread_scope_system cross-block acquire/release.

FAIL / hazard on wave64 CDNA:
- `.../thread.semaphore/timed.pass.cpp` FAILS.
- thread_scope_block intra-wavefront producer/consumer can DEADLOCK (two threads
  of one block are lanes of one wave64 wavefront; the acquiring lane spins
  without yielding to the releasing lane). See validation/sem_block_probe.cpp.

Root cause of both: the same-wavefront forward-progress hazard on wave64. Lanes
in one wavefront share a program counter; a spin-wait acquire in one lane
prevents a sibling lane in the same wavefront from making the progress that
would release the semaphore. Cross-block / cross-wavefront cases are fine.

## KEY OPEN QUESTION for followers

Does the wave64 forward-progress hazard (timed.pass.cpp failure +
thread_scope_block intra-wavefront deadlock) ALSO occur on RDNA wave32
(linux-gfx1100, windows-gfx1201, windows-gfx1101)? On wave32 the two probe
threads are still lanes of one wavefront, but RDNA's execution model may let
them make independent progress.

ANSWER (linux-gfx1100, gfx1100 RDNA3 wave32, ROCm 7.2.1, W7800):
- sem_block_probe: PASS (no hang). Wave32 RDNA3 does NOT exhibit the
  forward-progress hazard. The two lanes of one wavefront make independent
  progress on RDNA3.
- timed.pass.cpp: FAIL with assertion error (NOT a hang). The chrono-based
  `try_acquire_until` fails due to the TSC clock inaccuracy warning visible at
  compile time (100 MHz assumption for gfx1100/gfx1101 per RDNA3 ISA). This is
  a chrono/TSC issue, not a semaphore forward-progress issue.

Conclusion for PR caveat: SUPERSEDED -- see CORRECTION 2026-06-07 below. Do NOT
claim "the hazard is CDNA/wave64-specific" in the PR.

CORRECTION 2026-06-07 (re-measured on gfx90a, MI250X, ROCm 7.2.x): the committed
validation/sem_block_probe.cpp does NOT hang on gfx90a. Re-running the exact
committed file AND a role-reversed twin (higher lane spins acquire, lower lane
releases) both PASS on wave64 (exit 0, fast). So the earlier "HANGS on wave64"
baseline was the probe's `// EXPECTED to hang` comment recorded as a result, not
a measurement; the gfx1100 "PASS" is therefore NOT evidence of a wave32-specific
capability -- the probe passes on both arches because this 2-lane pattern, as the
compiler lowers it (plus the s_sleep backoff), lets the releasing lane run on
both. AMD provides no intra-wavefront forward-progress GUARANTEE on any current
arch (no per-lane PC); a passing probe means "did not trigger the deadlock,"
never "safe." PR caveat MUST stay conservative: same-wavefront blocking
acquire()/release() is unsupported on ALL AMD arches (may deadlock; use
cross-wavefront placement or the non-blocking/timed APIs). The one solid, separable
gfx1100 finding stands: conf_timed fails because gfx1100's TSC clockrate is not
defined on amd-develop (only gfx90a/gfx908 @ 25 MHz are) -- port main's gfx1100
100 MHz TSC define. The block-scope probe is too weak to establish a per-arch
hazard claim either way and should not be the basis for the caveat.

This result directly shapes the upstream PR's caveat. Record it precisely
(per arch) in this file when each follower reports.

Remaining open: windows-gfx1101 (RDNA3 wave32) and windows-gfx1201 (RDNA4
wave32) -- expected to match gfx1100 but record explicitly.

## Validation 2026-06-07 (linux-gfx1100, gfx1100 RDNA3 wave32, ROCm 7.2.1)

GPU: AMD Radeon Pro W7800 48GB (gfx1100), HIP_VISIBLE_DEVICES=0
Fork HEAD: e2e8f70dd8737a5cf8b7b798b99654d06c8eb326

Build command:
```
bash projects/libhipcxx/validation/run_linux.sh gfx1100 0
```

NOTE: The validation scripts (sem_umbrella_smoke.cpp, sem_block_probe.cpp, run_linux.sh)
were fixed during this validation to address two issues that affect ALL arches
(not just gfx1100) that were not caught when the gfx90a lead was set up:
1. HIP does not allow non-trivially-constructible types as __shared__ variables;
   fixed to use aligned char storage + placement-new.
2. Upstream conformance tests (try_acquire/acquire/release/timed/heterogeneous)
   need `-I test/support -I test --include force_include_hip.h`; added to run_linux.sh.

Test results:
- sem_umbrella_smoke: PASS -- <cuda/semaphore> + <cuda/std/semaphore> compile and run
- sem_test: PASS -- device cuda::binary_semaphore<thread_scope_device> cross-block acquire/release
- sem_block_probe: PASS -- intra-wavefront thread_scope_block (wave32 forward-progress: NO HANG)
- conf_version: PASS
- conf_max: PASS
- conf_try_acquire: PASS
- conf_acquire: PASS
- conf_release: PASS
- conf_timed: FAIL -- device assertion on try_acquire_until (TSC clock 100 MHz assumption, not a hang)
- conf_heterogeneous: PASS

9 PASS / 1 FAIL (timed, TSC issue unrelated to semaphore change)
Wave32 result: NO forward-progress hazard. sem_block_probe PASSES without watchdog timeout.

## Per-arch TSC clockrate -- already present in base, NO code change needed (2026-06-07)

Investigated adding per-arch realtime TSC clockrate for RDNA (gfx1100/gfx1101/
gfx1201) to make `<cuda/std/chrono>` `try_acquire_for/until` and the conformance
`timed.pass.cpp` work. FINDING: the per-arch `_LIBCUDACXX_HIP_TSC_CLOCKRATE`
selection is ALREADY fully present in the amd-develop layout at
`include/cuda/std/detail/libcxx/include/__config` (the `#if defined(__gfx...)`
ladder around lines 347-393). It was added by the work's own base commit
fa4ccc6 ("Get rid of timing related error and emit warning instead", upstream
PR #22), so it is reachable at fork HEAD e2e8f70 without any new edit. This
covers more than upstream `main`'s single gfx1100 fix:
- gfx908 / gfx90a: 25 MHz (25000000) -- unchanged.
- gfx940 / gfx941 / gfx942 / gfx950: 100 MHz.
- catch-all `__GFX9__`: 100 MHz (warn under !NDEBUG).
- `__GFX10__` (gfx1030, RDNA2): 100 MHz (warn).
- `__GFX11__` (gfx1100 / gfx1101, RDNA3): 100 MHz (warn) -- equals main's fix.
- `__GFX12__` (gfx1200 / gfx1201, RDNA4): 100 MHz (warn) -- ASSUMED from the
  RDNA4 ISA per the existing comment; NOT confirmed on hardware here (no RDNA4).
  The RDNA4 ISA PDF was not machine-fetchable and no secondary source indicates
  the realtime clock changed from the 100 MHz used across gfx94x/RDNA3; ROCm's
  rocprofiler thread-trace decoder treats RT frequency as a runtime-queried
  value, not a fixed header constant. 100 MHz is the best-supported value and is
  already in place; the gfx1201 follower confirms actual timed behavior.
- unknown arch: meaningless clockrate 1 + the original FIXME/warning preserved.

Verified at HEAD e2e8f70 with ROCm 7.2.1 hipcc (no source change):
- Branch selection per arch via a `#pragma message` probe: gfx90a -> 25 MHz
  branch; gfx1100/gfx1101 -> `__GFX11__`; gfx1201 -> `__GFX12__`.
- Resolved literal via `static_assert(_LIBCUDACXX_HIP_TSC_CLOCKRATE == ...)`:
  gfx90a == 25000000; gfx1100/gfx1101/gfx1201 == 100000000 (all pass).
- Compiled the real `.upstream-tests/test/std/thread/thread.semaphore/
  timed.pass.cpp` with the playbook recipe (`-Iinclude -I.upstream-tests/test/
  support -I.upstream-tests/test --include .upstream-tests/test/
  force_include_hip.h`, `-std=c++17`): gfx90a compiles clean (object unchanged);
  gfx1100 compiles, only diagnostic is the expected `__GFX11__` "Assuming 100 MHz"
  `#warning`. (gfx90a device code object is byte-identical -- no change to that
  arch.) GPU-run not needed: no source delta; this is a verification, not a port.

CONSEQUENCE: head_sha stays e2e8f70; NO new fork commit was made and head was NOT
advanced (a redundant/cosmetic edit would flip the validated gfx90a + gfx1100
platforms to revalidate for zero functional benefit). The conf_timed gfx1100
FAILURE above is therefore NOT a missing-clockrate bug -- the 100 MHz define is
present -- it is a separate chrono/TSC accuracy or runtime issue. The semaphore
PR's chrono support for RDNA is already covered by the base; do not add a
duplicate gfx1100 TSC define in this PR.

## (c) Per-arch VALIDATION PLAYBOOK (the follower validator executes this)

Goal: on the target arch, (1) confirm the ungated headers compile and the
device semaphore works, then (2) answer the wave32 forward-progress question by
running the block-scope probe and timed test under a watchdog. Report which
tests pass and, crucially, whether sem_block_probe and timed.pass.cpp hang.

Test inventory (all in projects/libhipcxx/validation/, copied from the proven
gfx90a run so followers do not reconstruct them):
- sem_umbrella_smoke.cpp -- the two ungated public headers compile + run.
- sem_test.cpp           -- device producer/consumer (oneflow type), PASS on gfx90a.
- sem_block_probe.cpp    -- intra-wavefront thread_scope_block probe; HANGS on wave64.
- upstream conformance under the fork's
  `.upstream-tests/test/std/thread/thread.semaphore/`:
  version/max/try_acquire/acquire/release.pass.cpp (PASS on gfx90a),
  timed.pass.cpp (FAILS on wave64), heterogeneous/semaphore.pass.cpp (PASS).

### Linux (linux-gfx1100; gfx1100 RDNA3 wave32)

One command runs the whole suite (clones the fork into projects/libhipcxx/src/
if absent -- that path is gitignored):

```
bash projects/libhipcxx/validation/run_linux.sh gfx1100 0
```

(second arg is the HIP_VISIBLE_DEVICES gpu index). It watchdogs the block probe
and timed test at 30s; a watchdog exit code 124 means "hung on this arch" --
that is the wave32 hazard answer.

Important compile notes:
- sem_umbrella_smoke.cpp and sem_block_probe.cpp use aligned char storage +
  placement-new for __shared__ semaphores (HIP restriction on non-trivially-
  constructible __shared__ types).
- Upstream conformance tests (try_acquire/acquire/release/timed/heterogeneous)
  need `-I test/support -I test --include test/force_include_hip.h`. run_linux.sh
  handles this automatically.

VALIDATED RESULT on gfx1100 (2026-06-07): 9 PASS / 1 FAIL (timed, TSC issue).
sem_block_probe PASSES (no forward-progress hazard on wave32 RDNA3). See the
"Validation 2026-06-07" section above for the full result.

Pin one GPU via the second arg / HIP_VISIBLE_DEVICES. Carry the result forward
per the normal validator flow once recorded.

### Windows (windows-gfx1201 RDNA4 wave32; windows-gfx1101 RDNA3 wave32)

Same tests, same question; the Windows validator adapts the commands to the
host toolchain. Do NOT run run_linux.sh as-is on Windows.

- Build/run via the TheRock PyTorch venv (ROCm 7.14), all-clang-cl CMake-HIP, as
  used by other Windows MOAT validations (see CTranslate2 notes for the exact
  init-cache pattern and `--rocm-device-lib-path` gotcha; the same compiler
  invocation applies). For these standalone single-file tests the simplest path
  is to invoke the venv's `hipcc`/`clang++` directly with `--offload-arch` set to
  the target (gfx1201 or gfx1101) and `-I projects/libhipcxx/src/include`.
- ONE GPU PER PROCESS (mandatory on this host -- both GPUs visible in one
  process crashes the ROCm 7.14 HIP runtime). Pin `HIP_VISIBLE_DEVICES=1` for
  gfx1201 (RX 9070 XT) or `HIP_VISIBLE_DEVICES=0` for gfx1101 (PRO V710), and
  validate exactly one arch per process.
- Watchdog the block probe and timed test (Windows: a job/timeout wrapper or a
  short host-side timer) -- they are expected to hang on wave64; the question is
  whether wave32 RDNA4/RDNA3 also hangs.
- Record per-arch results (gfx1201 and gfx1101 separately) in this file.

## Install as a dependency

Header-only. Consumers that pull `<cuda/semaphore>` / `<cuda/std/semaphore>`
add the fork's `include/` to their include path:

```
git clone --branch enable-semaphore https://github.com/jeffdaily/libhipcxx \
  _deps/libhipcxx/src
# build flag: -I _deps/libhipcxx/src/include  (or CMAKE_PREFIX_PATH if installed)
```

No build/install step is required for the headers themselves.
