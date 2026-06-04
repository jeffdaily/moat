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

# Per-platform state -> status glyph (legend is emitted by render_table).
EMOJI = {
    "completed": "✅",
    "validated": "✅",
    "pr-open": "✅",
    "port-ready": "🟡",       # follower queued: lead done, not yet validated here
    "revalidate": "🔄",       # was validated here; HEAD moved -> re-check
    "validating": "🔧",
    "review-passed": "🔧",
    "ported": "🔧",
    "delta-ported": "🔧",
    "porting": "🔧",
    "changes-requested": "🔧",
    "validation-failed": "🔧",
    "planned": "🔧",
    "unclaimed": "⬜",
    "blocked-needs-gfx90a": "⬜",   # follower gated on the lead (not yet startable)
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
        # Delivery tracking lives in upstream.json. pr_state (open/merged/closed)
        # drives the PR glyph; `outcome` records the terminal disposition for
        # projects whose success is NOT an upstream PR (e.g. we GPU-validated an
        # existing ROCm backend across archs). See outcome_cell() for the vocab.
        up = d / "upstream.json"
        if up.exists():
            try:
                u = json.loads(up.read_text())
                rec["pr_url"] = u.get("pr_url")
                rec["pr_number"] = u.get("pr_number")
                rec["pr_state"] = u.get("pr_state")
                rec["outcome"] = u.get("outcome")
                rec["outcome_note"] = u.get("outcome_note")
            except json.JSONDecodeError:
                pass
        out.append(rec)
    return out


def cell(block):
    """Status glyph for one platform. A blocked=true platform shows the blocked
    glyph regardless of its underlying state."""
    if block.get("blocked"):
        return "🚫"
    return EMOJI.get(block.get("state"), "❓")


def plat_header(key):
    """Stacked column header for a platform key: linux-gfx90a -> 'gfx90a<br>Linux'
    (arch on top of the OS)."""
    os_part, arch = key.split("-", 1)
    return f"{arch}<br>{os_part.capitalize()}"


def _validated_arch_count(p):
    """Number of platforms validated on real hardware (completed, not blocked)."""
    return sum(1 for b in p.get("platforms", {}).values()
               if b.get("state") == "completed" and not b.get("blocked"))


def outcome_cell(p):
    """The Outcome column: what this project actually delivered. An upstream PR
    (any state) is shown by its glyph + number. Projects without a PR carry an
    explicit `outcome` in upstream.json:
      validated  -- upstream already had a ROCm path; we GPU-validated it across
                    N archs (often extending coverage, e.g. first CDNA). 🔵
      fork       -- delivered as a working standalone fork; an upstream PR is not
                    appropriate (e.g. a kernel-experiments repo). 🍴
      superseded -- upstream/community already covers our archs; no value-add. ⚪
      blocked    -- non-viable. ⛔
    No PR and no outcome yet -> pending (—)."""
    if p.get("pr_url"):
        glyph = {"open": "🟢", "merged": "🟣", "closed": "🔴"}.get(p.get("pr_state") or "open", "🟢")
        return f"{glyph} [#{p['pr_number']}]({p['pr_url']})"
    oc = p.get("outcome")
    if oc == "validated":
        n = _validated_arch_count(p)
        return f"🔵 validated ({n} arch)" if n else "🔵 validated"
    if oc == "fork":
        fu = p.get("fork_url")
        return f"🍴 [fork]({fu}/tree/{moatlib.PORT_BRANCH})" if fu else "🍴 fork"
    if oc == "superseded":
        return "⚪ superseded"
    if oc == "blocked":
        return "⛔ blocked"
    return "—"


def render_table(projects):
    if not projects:
        return EMPTY
    # Alphabetical by name (case-insensitive) so the table is a lookup-by-name
    # reference; the per-row glyphs still convey status. (Popularity/priority order
    # buried manually-adopted projects at priority 0 and told no progress story.)
    projects = sorted(projects, key=lambda p: p.get("name", "").lower())
    legend = ("Status: ✅ done · 🔧 in progress · 🟡 queued (follower; lead done) · "
              "🔄 re-check (HEAD moved) · ⬜ todo/gated · 🚫 blocked · — n/a. "
              "Outcome: 🟣 PR merged · 🟢 PR open · 🔴 PR closed · 🔵 validated (existing ROCm confirmed on N archs) · "
              "🍴 fork-only · ⚪ superseded · — pending. "
              "The project name links to upstream, (fork) to our `moat-port` branch.")
    headers = ["Project"] + [plat_header(k) for k in moatlib.PLATFORMS] + ["Outcome"]
    aligns = ["---"] + [":---:"] * len(moatlib.PLATFORMS) + ["---"]
    lines = [legend, "",
             "| " + " | ".join(headers) + " |",
             "| " + " | ".join(aligns) + " |"]
    for p in projects:
        name = p.get("name", "?")
        up = p.get("upstream_url")
        if p.get("fork_url"):
            proj = f"[{name}]({up}) ([fork]({p['fork_url']}/tree/{moatlib.PORT_BRANCH}))"
        else:
            proj = f"[{name}]({up})"
        plats = p.get("platforms", {})
        cells = [cell(plats.get(k, {"state": "?"})) for k in moatlib.PLATFORMS]
        lines.append("| " + " | ".join([proj] + cells + [outcome_cell(p)]) + " |")
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
