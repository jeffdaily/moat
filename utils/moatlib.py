#!/usr/bin/env python3
"""MOAT control-plane library: schema, per-platform state machine, cross-platform
gating + regression guard, validated status.json writes, and the single
git-sync write path. Also a small CLI used by orient.sh and the agents.

status.json is the source of truth. The three AMD targets share one fork branch,
so any HEAD advance re-validates the platforms that already passed (see
advance_head). State transitions are validated; illegal jumps raise."""

import argparse
import json
import subprocess
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECTS = REPO_ROOT / "projects"
SCHEMA_VERSION = 1

LEAD = "linux-gfx90a"
PLATFORMS = ["linux-gfx90a", "linux-gfx1100", "windows-gfx1151"]

# Per-platform pipeline. blocked-needs-gfx90a is the follower start state; the
# `blocked` boolean (needs user input) is orthogonal and set separately.
ALLOWED = {
    "unclaimed": {"planned"},
    "blocked-needs-gfx90a": {"port-ready"},
    "planned": {"porting"},
    "porting": {"ported"},
    "ported": {"review-passed", "changes-requested"},
    "changes-requested": {"porting", "delta-ported"},
    "review-passed": {"completed", "validation-failed"},
    "validation-failed": {"porting", "delta-ported"},
    "port-ready": {"completed", "validation-failed"},
    "delta-ported": {"review-passed", "changes-requested"},
    "revalidate": {"completed", "validation-failed"},
    "completed": {"revalidate"},
}
STATES = set(ALLOWED) | {s for v in ALLOWED.values() for s in v}

# Which agent handles each state, and selection priority (lower = sooner).
# Resume-before-start: drain work in flight before opening new fronts.
STAGE_FOR_STATE = {
    "unclaimed": "planner",
    "planned": "porter",
    "porting": "porter",
    "changes-requested": "porter",
    "validation-failed": "porter",
    "ported": "reviewer",
    "delta-ported": "reviewer",
    "review-passed": "validator",
    "port-ready": "validator",
    "revalidate": "validator",
}
SELECT_RANK = {
    "revalidate": 0,
    "validation-failed": 1,
    "changes-requested": 2,
    "porting": 3,
    "delta-ported": 4,
    "planned": 5,
    "ported": 6,
    "review-passed": 7,
    "port-ready": 8,
    "unclaimed": 9,
}
# States that take no agent action (terminal or gated or awaiting user).
INERT = {"completed", "pr-open", "blocked-needs-gfx90a"}


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def claim_ttl_seconds():
    """Read claim TTL from config/moat.toml; default 30 min. A .claim file
    untouched for longer than this is stale (its CLI crashed) and reclaimable."""
    cfg = REPO_ROOT / "config" / "moat.toml"
    minutes = 30
    if cfg.exists():
        try:
            minutes = tomllib.loads(cfg.read_text()).get("claims", {}).get("claim_ttl_minutes", 30)
        except (tomllib.TOMLDecodeError, OSError):
            pass
    return float(minutes) * 60.0


def claim_live(name):
    """True if projects/<name>/.claim exists and was refreshed within the TTL.
    Same-host coordination via the shared filesystem; .claim is gitignored."""
    cf = PROJECTS / name / ".claim"
    if not cf.exists():
        return False
    return (time.time() - cf.stat().st_mtime) < claim_ttl_seconds()


def _empty_stats():
    return {
        "tokens_total": 0,
        "tokens_approx": True,
        "wall_seconds": {"thinking": 0, "compile": 0, "test": 0, "misc": 0},
        "session_count": 0,
        "first_session_at": None,
        "last_session_at": None,
    }


def _platform_block(initial_state):
    return {
        "state": initial_state,
        "blocked": False,
        "blocked_reason": None,
        "validated_sha": None,
        "started_at": None,
        "completed_at": None,
        "updated_at": now_iso(),
        "stats": _empty_stats(),
    }


def status_path(name):
    return PROJECTS / name / "status.json"


def load_status(name):
    with open(status_path(name)) as f:
        obj = json.load(f)
    validate_status(obj)
    return obj


def save_status(name, obj):
    validate_status(obj)
    obj["updated_at"] = now_iso()
    p = status_path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
        f.write("\n")


def validate_status(obj):
    """Light hand-rolled validation (no jsonschema dependency). Raises ValueError."""
    for k in ("schema_version", "name", "upstream_url", "fork_default_branch",
              "priority", "ext_type", "platforms"):
        if k not in obj:
            raise ValueError(f"status.json missing required key: {k}")
    if obj["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {obj['schema_version']}")
    if set(obj["platforms"]) != set(PLATFORMS):
        raise ValueError(f"platforms must be exactly {PLATFORMS}")
    for plat, blk in obj["platforms"].items():
        if blk.get("state") not in STATES:
            raise ValueError(f"{plat}: invalid state {blk.get('state')!r}")
        if not isinstance(blk.get("blocked"), bool):
            raise ValueError(f"{plat}: blocked must be boolean")


# ---- state machine ---------------------------------------------------------

def set_state(name, platform, new_state, agent=None, save=True):
    """Validate and apply a transition with its side effects."""
    obj = load_status(name)
    blk = obj["platforms"][platform]
    cur = blk["state"]
    if new_state == cur:
        return obj
    if new_state not in ALLOWED.get(cur, set()):
        raise ValueError(f"{name}/{platform}: illegal transition {cur} -> {new_state}")
    blk["state"] = new_state
    ts = now_iso()
    blk["updated_at"] = ts
    if agent:
        blk["last_agent"] = agent  # informational; not in strict schema
    if new_state in ("porting", "port-ready", "delta-ported") and not blk.get("started_at"):
        blk["started_at"] = ts
    if new_state == "completed":
        blk["completed_at"] = ts
        blk["validated_sha"] = obj.get("head_sha")
        if platform == LEAD:
            _unblock_followers(obj)
    obj["platforms"][platform] = blk
    if save:
        save_status(name, obj)
    return obj


def set_blocked(name, platform, blocked, reason=None):
    obj = load_status(name)
    blk = obj["platforms"][platform]
    blk["blocked"] = bool(blocked)
    blk["blocked_reason"] = reason if blocked else None
    blk["updated_at"] = now_iso()
    save_status(name, obj)
    return obj


def _unblock_followers(obj):
    """Lead just completed: followers may begin (validate-first on the branch)."""
    for plat in PLATFORMS:
        if plat == LEAD:
            continue
        blk = obj["platforms"][plat]
        if blk["state"] == "blocked-needs-gfx90a":
            blk["state"] = "port-ready"
            blk["updated_at"] = now_iso()


def advance_head(name, new_sha):
    """A porter commit advanced the shared fork HEAD. Flip every platform that
    had passed at a different HEAD back to revalidate, so a CLI on that hardware
    re-checks it. This is the cross-platform regression guard."""
    obj = load_status(name)
    obj["head_sha"] = new_sha
    for plat in PLATFORMS:
        blk = obj["platforms"][plat]
        if blk["state"] == "completed" and blk.get("validated_sha") != new_sha:
            blk["state"] = "revalidate"
            blk["updated_at"] = now_iso()
    save_status(name, obj)
    return obj


def lead_done(obj):
    return obj["platforms"][LEAD]["state"] == "completed"


def actionable(obj, platform):
    """Is this platform pickable by an agent on this host right now?"""
    blk = obj["platforms"][platform]
    if blk["blocked"]:
        return False
    if blk["state"] in INERT:
        return False
    if platform != LEAD and not lead_done(obj):
        return False
    return blk["state"] in STAGE_FOR_STATE


def next_task(platform):
    """Pick the single next project for this platform. Returns dict or None."""
    cands = []
    if not PROJECTS.exists():
        return None
    for d in sorted(PROJECTS.iterdir()):
        sp = d / "status.json"
        if not sp.exists():
            continue
        try:
            obj = load_status(d.name)
        except (ValueError, json.JSONDecodeError):
            continue
        if platform not in obj["platforms"]:
            continue
        if not actionable(obj, platform):
            continue
        if claim_live(d.name):  # another CLI is already working it
            continue
        state = obj["platforms"][platform]["state"]
        cands.append((SELECT_RANK.get(state, 99), -float(obj.get("priority", 0)),
                      d.name, state))
    if not cands:
        return None
    cands.sort()
    rank, negprio, name, state = cands[0]
    return {"project": name, "state": state, "stage": STAGE_FOR_STATE[state],
            "priority": -negprio, "rank": rank}


def unblock_all_followers():
    """Flip blocked-needs-gfx90a -> port-ready wherever the lead is completed.
    Called by orient.sh before selection so followers become pickable."""
    changed = []
    if not PROJECTS.exists():
        return changed
    for d in sorted(PROJECTS.iterdir()):
        if not (d / "status.json").exists():
            continue
        try:
            obj = load_status(d.name)
        except (ValueError, json.JSONDecodeError):
            continue
        if not lead_done(obj):
            continue
        touched = False
        for plat in PLATFORMS:
            if plat == LEAD:
                continue
            blk = obj["platforms"][plat]
            if blk["state"] == "blocked-needs-gfx90a":
                blk["state"] = "port-ready"
                blk["updated_at"] = now_iso()
                touched = True
        if touched:
            save_status(d.name, obj)
            changed.append(d.name)
    return changed


# ---- dispositions (candidates we will NOT port, and why) -------------------

DISPOSITIONS = REPO_ROOT / "data" / "dispositions.json"
SKIP_REASONS = ["already-ported", "already-supported", "ported-elsewhere",
                "cant-port", "not-a-target", "duplicate", "other"]
# ported-elsewhere: AMD's ROCm/HIP support for this project (or an equivalent)
# lives in a separate repo, fork, or effort, so porting this one adds no value.


def load_dispositions():
    if DISPOSITIONS.exists():
        try:
            return json.loads(DISPOSITIONS.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_dispositions(d):
    DISPOSITIONS.parent.mkdir(parents=True, exist_ok=True)
    with open(DISPOSITIONS, "w") as f:
        json.dump(d, f, indent=2, sort_keys=True)
        f.write("\n")


def get_disposition(full_name):
    return load_dispositions().get(full_name.lower())


def set_disposition(full_name, disposition, reason, note=""):
    if disposition == "skip" and reason not in SKIP_REASONS:
        raise ValueError(f"reason must be one of {SKIP_REASONS}")
    d = load_dispositions()
    d[full_name.lower()] = {"full_name": full_name, "disposition": disposition,
                            "reason": reason, "note": note, "decided": now_iso()}
    save_dispositions(d)
    return d[full_name.lower()]


def clear_disposition(full_name):
    d = load_dispositions()
    if full_name.lower() in d:
        del d[full_name.lower()]
        save_dispositions(d)
        return True
    return False


# ---- scaffolding -----------------------------------------------------------

def scaffold_project(full_name, upstream_url=None, default_branch="main",
                     ext_type="unknown", priority=0.0, force=False):
    disp = get_disposition(full_name)
    if disp and disp.get("disposition") == "skip" and not force:
        raise ValueError(
            f"{full_name} is marked skip ({disp.get('reason')}): {disp.get('note', '')}. "
            f"Use force=True / --force to adopt anyway.")
    name = full_name.split("/")[-1]
    pdir = PROJECTS / name
    pdir.mkdir(parents=True, exist_ok=True)
    if status_path(name).exists():
        raise FileExistsError(f"{status_path(name)} already exists")
    upstream_url = upstream_url or f"https://github.com/{full_name}"
    obj = {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "upstream_url": upstream_url,
        "fork_url": None,
        "fork_default_branch": default_branch,
        "priority": float(priority),
        "ext_type": ext_type,
        "adopted_at": now_iso(),
        "updated_at": now_iso(),
        "head_sha": None,
        "platforms": {
            LEAD: _platform_block("unclaimed"),
            "linux-gfx1100": _platform_block("blocked-needs-gfx90a"),
            "windows-gfx1151": _platform_block("blocked-needs-gfx90a"),
        },
    }
    save_status(name, obj)
    upstream = {
        "full_name": full_name,
        "upstream_url": upstream_url,
        "default_branch": default_branch,
        "fork_url": None,
        "ext_type": ext_type,
        "base_sha": None,
    }
    with open(pdir / "upstream.json", "w") as f:
        json.dump(upstream, f, indent=2)
        f.write("\n")
    (pdir / "notes.md").write_text(f"# {name} notes\n")
    return name


# ---- git sync --------------------------------------------------------------

def _git(*args, check=True, cwd=None):
    return subprocess.run(["git", *args], cwd=str(cwd or REPO_ROOT),
                          capture_output=True, text=True, check=check)


def ensure_git_config():
    """Register the semantic status.json merge driver (idempotent). A fresh
    clone needs this for .gitattributes merge=moat-status to take effect."""
    drv = f"python3 {REPO_ROOT / 'utils' / 'merge_status.py'} %O %A %B %P"
    _git("config", "merge.moat-status.name", "MOAT status.json semantic merge", check=False)
    _git("config", "merge.moat-status.driver", drv, check=False)


def commit_and_push(paths, message, push=True, retries=3):
    """The single MOAT-repo write path: stage, commit-on-top, pull --rebase,
    push, bounded retry. Never amends, never force-pushes. No-op if nothing
    staged."""
    ensure_git_config()
    rels = [str(Path(p)) for p in (paths if isinstance(paths, (list, tuple)) else [paths])]
    _git("add", "--", *rels)
    staged = _git("diff", "--cached", "--name-only", check=False).stdout.strip()
    if not staged:
        return False
    _git("commit", "-m", message)
    if not push:
        return True
    for _ in range(retries):
        _git("pull", "--rebase", check=False)
        r = _git("push", check=False)
        if r.returncode == 0:
            return True
    sys.stderr.write("commit_and_push: push failed after retries; left committed locally\n")
    return False


# ---- CLI -------------------------------------------------------------------

def _print_json(obj):
    json.dump(obj, sys.stdout, indent=2)
    sys.stdout.write("\n")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="moatlib")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scaffold", help="create projects/<name>/{status,upstream}.json")
    s.add_argument("full_name")
    s.add_argument("--url")
    s.add_argument("--branch", default="main")
    s.add_argument("--ext", default="unknown", choices=["cmake", "torch-extension", "unknown"])
    s.add_argument("--priority", type=float, default=0.0)
    s.add_argument("--force", action="store_true", help="adopt even if marked skip")

    s = sub.add_parser("next-task", help="print next actionable project for a platform")
    s.add_argument("platform", choices=PLATFORMS)

    s = sub.add_parser("set-state")
    s.add_argument("name")
    s.add_argument("platform", choices=PLATFORMS)
    s.add_argument("new_state")
    s.add_argument("--agent")

    s = sub.add_parser("set-blocked")
    s.add_argument("name")
    s.add_argument("platform", choices=PLATFORMS)
    s.add_argument("reason")

    s = sub.add_parser("advance-head")
    s.add_argument("name")
    s.add_argument("sha")

    sub.add_parser("unblock-followers")
    s = sub.add_parser("validate")
    s.add_argument("name")
    s = sub.add_parser("show")
    s.add_argument("name")

    args = ap.parse_args(argv)

    if args.cmd == "scaffold":
        name = scaffold_project(args.full_name, args.url, args.branch, args.ext, args.priority, args.force)
        print(f"scaffolded projects/{name}")
    elif args.cmd == "next-task":
        t = next_task(args.platform)
        if t is None:
            print("NONE")
            return 0
        _print_json(t)
    elif args.cmd == "set-state":
        set_state(args.name, args.platform, args.new_state, agent=args.agent)
        print(f"{args.name}/{args.platform} -> {args.new_state}")
    elif args.cmd == "set-blocked":
        set_blocked(args.name, args.platform, True, args.reason)
        print(f"{args.name}/{args.platform} blocked: {args.reason}")
    elif args.cmd == "advance-head":
        advance_head(args.name, args.sha)
        print(f"{args.name} head_sha -> {args.sha}")
    elif args.cmd == "unblock-followers":
        changed = unblock_all_followers()
        print(" ".join(changed) if changed else "(none)")
    elif args.cmd == "validate":
        load_status(args.name)
        print(f"{args.name} status.json valid")
    elif args.cmd == "show":
        _print_json(load_status(args.name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
