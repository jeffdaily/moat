#!/usr/bin/env bash
# libhipcxx semaphore follower validation (Windows, TheRock ROCm). Windows analogue
# of run_linux.sh: builds+runs the semaphore suite for one arch on one GPU using the
# TheRock all-clang toolchain (clang++ -x hip), and reports which tests pass on
# wave32. See notes.md section (c).
#
# Usage: bash run_windows.sh <offload-arch> <gpu_index> [rocm_root]
#   bash run_windows.sh gfx1201 0
#
# Windows specifics vs run_linux.sh:
#  - invoke clang++ -x hip directly (no hipcc wrapper); pass --rocm-path and
#    --rocm-device-lib-path (TheRock clang does not auto-discover device libs on
#    Windows -- same gotcha as the CTranslate2 port).
#  - -DNOMINMAX: Windows SDK minwindef.h defines min()/max() macros that collide
#    with libcu++/libhipcxx's `static constexpr ptrdiff_t max()`. Build flag only;
#    no source change, so Linux/CUDA builds are unaffected.
#  - ONE GPU PER PROCESS: pin HIP_VISIBLE_DEVICES to exactly one arch. On the
#    current host gfx1201 (RX 9070 XT) is device 0 after the gfx1101 V710 went
#    offline -- pass gpu_index 0 for gfx1201.
#
# conf_timed is EXPECTED to fail on AMD (same-wavefront forward-progress hazard +
# 100 MHz TSC assumption); a conf_timed failure is NOT a port defect. sem_block_probe
# is run under a watchdog to record whether wave32 RDNA4 hangs (it does not on RDNA3).
set -uo pipefail

ARCH="${1:?usage: run_windows.sh <offload-arch> <gpu_index> [rocm_root]}"
GPU="${2:?usage: run_windows.sh <offload-arch> <gpu_index> [rocm_root]}"
# Canonical TheRock ROCm SDK root (ships rocm-core/rocm_version.h, needed by the
# upstream conformance tests via amd/amd_utils.h; the bare build/ tree omits it).
ROCM="${3:-B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel}"
export HIP_VISIBLE_DEVICES="$GPU"
CLANG="$ROCM/lib/llvm/bin/clang++.exe"

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$(cd "$HERE/.." && pwd)/src"
INC="$SRC/include"
TINC="$SRC/.upstream-tests/test/support"
TROOT="$SRC/.upstream-tests/test"
FI="$TROOT/force_include_hip.h"

echo "== libhipcxx semaphore validation (Windows): arch=$ARCH gpu=$GPU =="
echo "clang++ : $CLANG"
echo "rocm    : $ROCM"
echo "include : $INC"
echo "fork HEAD: $(git -C "$SRC" rev-parse HEAD)"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
fail=0

build_run() {  # <src> <bin> [watchdog_seconds] [extra_flags...]
  local s="$1" b="$2" wd="${3:-0}"
  local extra=()
  if [ $# -gt 3 ]; then extra=("${@:4}"); fi
  echo; echo "--- $b ---"
  if ! "$CLANG" -std=c++17 -x hip --offload-arch="$ARCH" -DNOMINMAX \
        --rocm-path="$ROCM" --rocm-device-lib-path="$ROCM/lib/llvm/amdgcn/bitcode" \
        -I "$INC" "${extra[@]}" "$s" -o "$WORK/$b.exe"; then
    echo "BUILD FAIL: $b"; fail=1; return
  fi
  if [ "$wd" -gt 0 ]; then
    timeout "$wd" "$WORK/$b.exe"; local rc=$?
    [ "$rc" -eq 124 ] && echo "HANG (watchdog $wd s, rc=124) -- forward-progress hazard on this arch"
    [ "$rc" -ne 0 ] && { echo "NON-ZERO rc=$rc"; fail=1; }
  else
    "$WORK/$b.exe" || fail=1
  fi
}

build_run "$HERE/sem_umbrella_smoke.cpp" sem_umbrella_smoke
build_run "$HERE/sem_test.cpp"           sem_test
build_run "$HERE/sem_block_probe.cpp"    sem_block_probe 30

TDIR="$TROOT/std/thread/thread.semaphore"
for t in version max try_acquire acquire release; do
  f="$TDIR/$t.pass.cpp"
  [ -f "$f" ] && build_run "$f" "conf_$t" 0 -I "$TINC" -I "$TROOT" -include "$FI"
done
build_run "$TDIR/timed.pass.cpp" conf_timed 30 -I "$TINC" -I "$TROOT" -include "$FI"

HET="$TROOT/heterogeneous/semaphore.pass.cpp"
[ -f "$HET" ] && build_run "$HET" conf_heterogeneous 0 -I "$TINC" -I "$TROOT" -include "$FI"

echo; echo "== done (arch=$ARCH): overall $([ $fail -eq 0 ] && echo PASS || echo 'see above (conf_timed failure is the expected AMD hazard)') =="
exit "$fail"
