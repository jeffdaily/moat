#!/usr/bin/env bash
# libhipcxx semaphore follower validation (Linux). Clones jeffdaily/libhipcxx
# @ enable-semaphore if absent, then builds+runs the semaphore suite for one
# arch on one GPU. Reports which tests pass on wave32. See notes.md.
#
# Usage: bash run_linux.sh <offload-arch> [gpu_index]
#   bash run_linux.sh gfx1100 0
#
# The block-scope probe and timed test are EXPECTED to hang on wave64 CDNA
# (forward-progress hazard). The open question is whether they ALSO hang on
# wave32 RDNA -- run them under a watchdog and record the outcome.
set -uo pipefail

ARCH="${1:?usage: run_linux.sh <offload-arch> [gpu_index]}"
GPU="${2:-0}"
export HIP_VISIBLE_DEVICES="$GPU"

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$(cd "$HERE/.." && pwd)/src"   # projects/libhipcxx/src (gitignored fork checkout)
INC=""
FORK="https://github.com/jeffdaily/libhipcxx"
BR="enable-semaphore"

if [ ! -d "$SRC/.git" ]; then
  echo "== cloning $FORK @ $BR =="
  git clone --depth 1 --branch "$BR" "$FORK" "$SRC" || exit 2
fi
INC="$SRC/include"
echo "== libhipcxx semaphore validation: arch=$ARCH gpu=$GPU =="
echo "include: $INC"
echo "fork HEAD: $(git -C "$SRC" rev-parse HEAD)"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
fail=0

build_run() {  # <src> <bin> [watchdog_seconds]
  local s="$1" b="$2" wd="${3:-0}"
  echo; echo "--- $b ---"
  if ! hipcc -std=c++17 --offload-arch="$ARCH" -I "$INC" "$s" -o "$WORK/$b"; then
    echo "BUILD FAIL: $b"; fail=1; return
  fi
  if [ "$wd" -gt 0 ]; then
    timeout "$wd" "$WORK/$b"; local rc=$?
    [ "$rc" -eq 124 ] && echo "HANG (watchdog $wd s, rc=124) -- forward-progress hazard on this arch"
    [ "$rc" -ne 0 ] && fail=1
  else
    "$WORK/$b" || fail=1
  fi
}

# Headers-ungated smoke + device producer/consumer (both PASS on gfx90a).
build_run "$HERE/sem_umbrella_smoke.cpp" sem_umbrella_smoke
build_run "$HERE/sem_test.cpp"           sem_test

# Intra-wavefront block-scope probe: EXPECTED hang on wave64; the wave32 question.
build_run "$HERE/sem_block_probe.cpp"    sem_block_probe 30

# Upstream conformance tests. version/max/try_acquire/acquire/release PASS on
# gfx90a; timed.pass.cpp FAILS on wave64 (same-wavefront hazard) -- watchdog it.
TDIR="$SRC/.upstream-tests/test/std/thread/thread.semaphore"
for t in version max try_acquire acquire release; do
  f="$TDIR/$t.pass.cpp"
  [ -f "$f" ] && build_run "$f" "conf_$t"
done
build_run "$TDIR/timed.pass.cpp" conf_timed 30

# Heterogeneous (host+device) semaphore test (PASS on gfx90a).
HET="$SRC/.upstream-tests/test/std/thread/thread.semaphore/heterogeneous/semaphore.pass.cpp"
[ -f "$HET" ] && build_run "$HET" conf_heterogeneous

echo; echo "== done (arch=$ARCH): overall $([ $fail -eq 0 ] && echo PASS || echo 'PASS-with-expected-hazards/FAIL -- see above') =="
echo "Record per-test results in notes.md, especially whether sem_block_probe"
echo "and conf_timed hang on wave32 (they hang on wave64 CDNA)."
exit "$fail"
