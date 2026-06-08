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
            rec = json.loads(sp.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            sys.stderr.write(f"gen_readme: skipping unparseable {sp}\n")
            continue
        # Delivery tracking: PR fields can be in status.json (new workflow) or
        # upstream.json (legacy). status.json takes precedence. pr_state (open/merged/closed)
        # drives the PR glyph; `outcome` records the terminal disposition for
        # projects whose success is NOT an upstream PR (e.g. we GPU-validated an
        # existing ROCm backend across archs). See outcome_cell() for the vocab.

        # Check status.json first (new PR tracking)
        if "pr_url" in rec:
            # PR fields already in rec from status.json
            pass
        else:
            # Fall back to upstream.json (legacy)
            up = d / "upstream.json"
            if up.exists():
                try:
                    u = json.loads(up.read_text(encoding="utf-8"))
                    rec["pr_url"] = u.get("pr_url")
                    rec["pr_number"] = u.get("pr_number")
                    rec["pr_state"] = u.get("pr_state")
                    rec["outcome"] = u.get("outcome")
                    rec["outcome_note"] = u.get("outcome_note")
                except json.JSONDecodeError:
                    pass
        out.append(rec)
    return out


# Windows archs whose 'port-ready' (lead done, not yet validated here) reads as
# pending (—) rather than the 🟡 'queued' glyph. For now gfx1201 is the only
# active Windows validation target, so 🟡 'queued' appears there alone; gfx1101
# and gfx1151 are queued-but-not-active and show pending to keep the table honest.
PENDING_WINDOWS = {"windows-gfx1101", "windows-gfx1151"}

# gfx1151 is a retired/optional tier that never gates PR-readiness, so a blocked
# gfx1151 cell reads as n/a (—) -- EXCEPT the handful where we actually ran
# validation on the APU and it genuinely failed on the arch (wrong numerics,
# hangs, or host faults of an otherwise-correct port). Those stay 🚫. Curated
# here rather than parsed from blocked_reason free text (the substring matching
# that this replaces was applied inconsistently).
GFX1151_ATTEMPTED_FAIL = {"Gpufit", "alien", "lc0", "stdgpu"}


def cell(block, key, project_name):
    """Status glyph for one platform cell. Blocked platforms show 🚫, except the
    retired gfx1151 tier which shows n/a (—) unless it is a curated genuine
    attempt-and-fail. Windows 'port-ready' shows 🟡 only on gfx1201; gfx1101 and
    gfx1151 show pending (—)."""
    state = block.get("state")
    if block.get("blocked"):
        if key == "windows-gfx1151":
            return "🚫" if project_name in GFX1151_ATTEMPTED_FAIL else "—"
        return "🚫"
    if state == "port-ready" and key in PENDING_WINDOWS:
        return "—"
    return EMOJI.get(state, "❓")


def plat_header(key):
    """Stacked column header for a platform key: linux-gfx90a -> 'gfx90a<br>Linux'
    (arch on top of the OS). Windows archs carry a dagger -- they are a one-of tier
    (any one completed unlocks PR-readiness); the legend explains it."""
    os_part, arch = key.split("-", 1)
    marker = " †" if key in moatlib.WINDOWS_TIER else ""
    return f"{arch}{marker}<br>{os_part.capitalize()}"


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
        num = p.get("pr_number")
        if num is None:  # derive from the .../pull/<n> URL tail when not recorded
            tail = p["pr_url"].rstrip("/").rsplit("/", 1)[-1]
            num = tail if tail.isdigit() else "?"
        return f"{glyph} [#{num}]({p['pr_url']})"
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
    legend = ("Status: ✅ done · 🔧 in progress · 🟡 queued (gfx1201 follower; lead done) · "
              "🔄 re-check (HEAD moved) · ⬜ todo/gated · 🚫 blocked/failed · — n/a or pending. "
              "Outcome: 🟣 PR merged · 🟢 PR open · 🔴 PR closed · 🔵 validated (existing ROCm confirmed on N archs) · "
              "🍴 fork-only · ⚪ superseded · — pending. "
              "† The Windows archs (gfx1101 / gfx1201 / gfx1151) are a redundant tier -- any ONE completed (✅) "
              "satisfies the Windows requirement for PR-readiness; the two Linux archs (gfx90a then gfx1100) are each required. "
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
        cells = [cell(plats.get(k, {"state": "?"}), k, name) for k in moatlib.PLATFORMS]
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
    current = README.read_text(encoding="utf-8")
    new = splice(current, render_table(load_projects()))
    if args.check:
        if new != current:
            sys.stderr.write("gen_readme: README.md is out of date (run gen_readme.py)\n")
            return 1
        print("README.md table is up to date")
        return 0
    README.write_text(new, encoding="utf-8")
    print("README.md table regenerated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
