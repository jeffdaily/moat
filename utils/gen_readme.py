#!/usr/bin/env python3
"""Regenerate the MOAT project table in README.md between sentinel markers,
preserving all hand-written prose outside them. Reads every
projects/*/status.json. Idempotent: same data -> identical bytes.

Usage:
  python3 utils/gen_readme.py            # rewrite README.md
  python3 utils/gen_readme.py --check    # exit 1 if README is stale (CI)"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import moatlib  # noqa: E402

REPO_ROOT = moatlib.REPO_ROOT
README = REPO_ROOT / "README.md"
START = "<!-- MOAT:TABLE:START -->"
END = "<!-- MOAT:TABLE:END -->"
EMPTY = ("_No projects adopted yet. Run `python3 utils/discover.py` then adopt "
         "rows from `data/candidates.json`._")

# Per-platform status state -> short table token.
CELL = {
    "unclaimed": "todo",
    "planned": "planned",
    "porting": "porting",
    "ported": "review",
    "changes-requested": "porting",
    "review-passed": "validating",
    "validating": "validating",
    "validation-failed": "porting",
    "validated": "done",
    "completed": "done",
    "port-ready": "validating",
    "delta-ported": "review",
    "revalidate": "revalidate",
    "blocked-needs-gfx90a": "gated",
    "pr-open": "pr-open",
}


def load_projects():
    out = []
    pdir = REPO_ROOT / "projects"
    if not pdir.exists():
        return out
    for d in sorted(pdir.iterdir()):
        sp = d / "status.json"
        if not sp.exists():
            continue
        try:
            rec = json.loads(sp.read_text())
        except json.JSONDecodeError:
            sys.stderr.write(f"gen_readme: skipping unparseable {sp}\n")
            continue
        # An open upstream PR lives in upstream.json (its presence implies the
        # port is upstreamed; there is no separate pr-open state to track).
        up = d / "upstream.json"
        if up.exists():
            try:
                u = json.loads(up.read_text())
                rec["pr_url"] = u.get("pr_url")
                rec["pr_number"] = u.get("pr_number")
            except json.JSONDecodeError:
                pass
        out.append(rec)
    return out


def cell(block):
    tok = CELL.get(block.get("state"), block.get("state", "?"))
    if block.get("blocked"):
        tok += " (blocked)"
    return tok


def owner_repo(url):
    return url.replace("https://github.com/", "").rstrip("/") if url else "?"


def render_table(projects):
    if not projects:
        return EMPTY
    projects = sorted(projects, key=lambda p: (-float(p.get("priority", 0)), p.get("name", "")))
    lines = ["| Project | Upstream | Fork | gfx90a | gfx1100 | gfx1151 | Upstream PR |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for p in projects:
        name = p.get("name", "?")
        up = owner_repo(p.get("upstream_url"))
        upstream = f"[{up}]({p.get('upstream_url')})"
        fork = f"[{name}]({p['fork_url']}/tree/{moatlib.PORT_BRANCH})" if p.get("fork_url") else "-"
        plats = p.get("platforms", {})
        cells = [cell(plats.get(k, {"state": "?"})) for k in moatlib.PLATFORMS]
        pr = f"[#{p['pr_number']}]({p['pr_url']})" if p.get("pr_url") else "-"
        lines.append(f"| {name} | {upstream} | {fork} | {cells[0]} | {cells[1]} | {cells[2]} | {pr} |")
    return "\n".join(lines)


def splice(readme_text, body):
    if START not in readme_text or END not in readme_text:
        raise SystemExit(f"gen_readme: README.md is missing {START} / {END} markers")
    head = readme_text[:readme_text.index(START) + len(START)]
    tail = readme_text[readme_text.index(END):]
    return f"{head}\n{body}\n{tail}"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="gen_readme")
    ap.add_argument("--check", action="store_true", help="exit 1 if README is stale")
    args = ap.parse_args(argv)
    current = README.read_text()
    new = splice(current, render_table(load_projects()))
    if args.check:
        if new != current:
            sys.stderr.write("gen_readme: README.md is out of date (run gen_readme.py)\n")
            return 1
        print("README.md table is up to date")
        return 0
    README.write_text(new)
    print("README.md table regenerated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
