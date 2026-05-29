#!/usr/bin/env bash
# Mark session boundaries and optionally record approximate token/cost.
# Usage:
#   utils/session.sh <project> <platform> start|end
#   utils/session.sh <project> <platform> tokens --tokens N [--cost-usd X] [--source manual|otel]
# Token/cost are approximate (see statlib.py); start/end give session wall-clock.
set -uo pipefail
cd "$(dirname "$0")/.."
proj="${1:?project}"; platform="${2:?platform}"; event="${3:?start|end|tokens}"; shift 3
out="projects/${proj}/stats.jsonl"; mkdir -p "projects/${proj}"
ts=$(date -u +%FT%TZ); epoch=$(date +%s.%N)
case "$event" in
  start|end)
    jq -cn --arg ts "$ts" --argjson epoch "$epoch" --arg ev "$event" --arg p "$platform" \
       '{kind:"session",ts:$ts,epoch:$epoch,event:$ev,platform:$p}' >> "$out" ;;
  tokens)
    toks=0; cost=null; source="manual"
    while [ $# -gt 0 ]; do case "$1" in
      --tokens) toks="$2"; shift 2;;
      --cost-usd) cost="$2"; shift 2;;
      --source) source="$2"; shift 2;;
      *) shift;; esac; done
    jq -cn --arg ts "$ts" --arg p "$platform" --argjson toks "$toks" \
       --argjson cost "$cost" --arg src "$source" \
       '{kind:"tokens",ts:$ts,platform:$p,tokens:$toks,cost_usd:$cost,source:$src,approx:true}' >> "$out" ;;
  *) echo "session.sh: unknown event $event" >&2; exit 2;;
esac
