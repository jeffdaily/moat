#!/usr/bin/env python3
"""Triage discovery candidates: mark projects you will NOT port (already ported,
already supported, can't port, not a target) with a reason, and review the
remaining actionable candidates. Decisions persist in data/dispositions.json and
block accidental adoption (moatlib scaffold refuses a skipped project).

Usage:
  python3 utils/triage.py review [--top N] [--all]
  python3 utils/triage.py skip <owner/repo> --reason <r> [--note "..."]
  python3 utils/triage.py unskip <owner/repo>
  python3 utils/triage.py skipped

Reasons: already-ported, already-supported, cant-port, not-a-target, duplicate, other
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import moatlib  # noqa: E402

CANDIDATES = moatlib.REPO_ROOT / "data" / "candidates.json"


def load_candidates():
    if not CANDIDATES.exists():
        return []
    try:
        return json.loads(CANDIDATES.read_text())
    except json.JSONDecodeError:
        return []


def cmd_review(args):
    disp = moatlib.load_dispositions()
    cands = load_candidates()
    actionable = [c for c in cands if c["full_name"].lower() not in disp]
    disposed = len(cands) - len(actionable)
    shown = actionable if args.all else actionable[:args.top]
    print(f"# {len(actionable)} undecided candidates ({disposed} disposed of {len(cands)}); showing {len(shown)}")
    print(f"{'#':>3}  {'prio':>5}  {'stars':>7}  project -- description")
    for i, c in enumerate(shown, 1):
        desc = (c.get("description") or "")[:68]
        print(f"{i:>3}  {c['priority']:>5}  {c['stars']:>7}  {c['full_name']} -- {desc}")
    print(f"\nskip with: python3 utils/triage.py skip <owner/repo> "
          f"--reason <{'|'.join(moatlib.SKIP_REASONS)}> --note \"...\"")
    return 0


def cmd_skip(args):
    d = moatlib.set_disposition(args.repo, "skip", args.reason, args.note or "")
    print(f"skip {d['full_name']} ({d['reason']}) {d['note']}".rstrip())
    return 0


def cmd_unskip(args):
    print("unskipped" if moatlib.clear_disposition(args.repo) else "no disposition for that repo")
    return 0


def cmd_skipped(args):
    disp = moatlib.load_dispositions()
    if not disp:
        print("(no dispositions yet)")
        return 0
    for v in sorted(disp.values(), key=lambda x: (x["reason"], x["full_name"].lower())):
        note = f" -- {v['note']}" if v.get("note") else ""
        print(f"{v['disposition']:>5}  {v['reason']:<18}  {v['full_name']}{note}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="triage")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("review", help="show undecided candidates")
    r.add_argument("--top", type=int, default=40)
    r.add_argument("--all", action="store_true")
    r.set_defaults(fn=cmd_review)
    s = sub.add_parser("skip", help="mark a project not-to-port")
    s.add_argument("repo")
    s.add_argument("--reason", required=True, choices=moatlib.SKIP_REASONS)
    s.add_argument("--note")
    s.set_defaults(fn=cmd_skip)
    u = sub.add_parser("unskip", help="remove a disposition")
    u.add_argument("repo")
    u.set_defaults(fn=cmd_unskip)
    k = sub.add_parser("skipped", help="list all dispositions")
    k.set_defaults(fn=cmd_skipped)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
