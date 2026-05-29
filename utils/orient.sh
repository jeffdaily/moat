#!/usr/bin/env bash
# MOAT entrypoint. Pull the latest MOAT state, detect this host's AMD arch, pick
# the single next project + stage for this platform, and print a dispatch
# summary. Read-only on state except an advisory claim and follower-unblock
# bookkeeping. Run this (or /port-next) when starting a CLI in the MOAT repo.
set -uo pipefail
cd "$(dirname "$0")/.."

bash utils/setup_git.sh >/dev/null 2>&1 || true

# Sync the latest MOAT state. Best-effort: offline or local-only is fine.
if git remote 2>/dev/null | grep -q .; then
  git pull --rebase --autostash >/dev/null 2>&1 \
    || echo "orient: pull --rebase skipped (offline or local conflicts)" >&2
fi

if ! arch_out=$(bash utils/detect_arch.sh 2>/dev/null); then
  echo "== MOAT orient =="
  echo "platform : UNKNOWN (no AMD GPU detected)"
  echo "next     : NONE"
  exit 0
fi
eval "$arch_out"   # sets GFX_ARCH GFX_TRIPLE PLATFORM

VALID="linux-gfx90a linux-gfx1100 windows-gfx1151"
echo "== MOAT orient =="
echo "platform : $PLATFORM (gfx=$GFX_ARCH)"
case " $VALID " in
  *" $PLATFORM "*) ;;
  *)
    echo "next     : NONE (this host arch is not a MOAT target: $VALID)"
    exit 0 ;;
esac

python3 utils/moatlib.py unblock-followers >/dev/null 2>&1 || true

# Serialize select+claim so two same-host CLIs never grab the same project.
exec 9>"projects/.selection.lock"
if command -v flock >/dev/null 2>&1; then flock -w 10 9 || true; fi

NEXT=$(python3 utils/moatlib.py next-task "$PLATFORM" 2>/dev/null || echo NONE)
if [ "$NEXT" = "NONE" ] || [ -z "$NEXT" ]; then
  echo "next     : NONE actionable on $PLATFORM"
  if [ "$PLATFORM" = "linux-gfx90a" ]; then
    echo "hint     : adopt a project from data/candidates.json:"
    echo "           python3 utils/moatlib.py scaffold <owner/repo>"
  else
    echo "hint     : followers only pick projects whose linux-gfx90a is completed"
  fi
  exit 0
fi

read -r PROJECT STATE STAGE < <(echo "$NEXT" | python3 -c \
  'import sys,json;d=json.load(sys.stdin);print(d["project"],d["state"],d["stage"])')

# Advisory claim: we hold the selection lock and next-task already excluded
# live-claimed projects. The dispatched CLI should refresh this .claim while
# working so it stays live (heartbeat); a stale .claim is reclaimable.
printf '{"host":"%s","pid":%s,"platform":"%s","started":"%s"}\n' \
  "$(hostname)" "$$" "$PLATFORM" "$(date -u +%FT%TZ)" > "projects/$PROJECT/.claim"

echo "next     : projects/$PROJECT  state=$STATE  -> dispatch: $STAGE"
echo "triple   : $GFX_TRIPLE"
echo "action   : Use the $STAGE subagent on projects/$PROJECT"
echo "           (read CLAUDE.md Pipeline + Autonomy boundary first)"
