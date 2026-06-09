#!/usr/bin/env python3
"""MOAT control-plane library: schema, per-platform state machine, cross-platform
gating + regression guard, validated status.json writes, and the single
git-sync write path. Also a small CLI used by orient.sh and the agents.

status.json is the source of truth. The three AMD targets share one fork branch,
so any HEAD advance re-validates the platforms that already passed (see
advance_head). State transitions are validated; illegal jumps raise."""

import argparse
import json
import re
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
# Windows is validated on distinct archs (gfx1101 RDNA3, gfx1201 RDNA4, plus the
# retired gfx1151 Strix Halo host) hosted on one multi-GPU machine; each is its own
# validation target, so work still flows to whichever Windows arch a host actually
# has. windows-gfx1151 is scheduled no new work; its already-completed blocks are
# preserved as real validation records, and per project its non-completed gfx1151
# block carries blocked=true, which `actionable` skips.
PLATFORMS = ["linux-gfx90a", "linux-gfx1100",
             "windows-gfx1101", "windows-gfx1201", "windows-gfx1151"]

# PR-readiness tiers (see pr_ready). The two Linux archs are each REQUIRED: a port
# validates on gfx90a, then gfx1100. The Windows archs form ONE redundant tier --
# they are interchangeable proofs that the port builds and runs on Windows ROCm, so
# only ONE Windows arch must reach `completed` to unlock the PR; the others may stay
# queued or blocked without gating it. This matches hardware reality: a Windows host
# can permanently lose an arch's GPU (gfx1101 on the current host), and requiring
# every Windows arch would then wedge the PR on hardware that no longer exists. The
# work pipeline is unchanged -- each arch still validates wherever its GPU is present.
# Satisfying the tier does NOT close it: an un-validated Windows arch stays selectable
# (`actionable` keeps returning True) even after a sibling satisfies the tier or the PR
# opens, so a host that later gains that GPU can validate it to complete the set. That
# is purely additive -- a validation changes no head_sha, so it disturbs neither the
# open PR nor the other platforms; it just enriches the coverage record.
PR_REQUIRED_PLATFORMS = ["linux-gfx90a", "linux-gfx1100"]
WINDOWS_TIER = ["windows-gfx1101", "windows-gfx1201", "windows-gfx1151"]
PORT_BRANCH = "moat-port"  # the topic branch that holds the port on each fork

# Per-platform pipeline. blocked-needs-gfx90a is the follower start state; the
# `blocked` boolean (needs user input) is orthogonal and set separately.
# pr-open and upstream-landed are LEAD-only states (followers stay at completed).
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
    "completed": {"revalidate", "pr-open"},
    "pr-open": {"upstream-landed"},
    "upstream-landed": {},  # terminal
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
INERT = {"completed", "pr-open", "upstream-landed", "blocked-needs-gfx90a"}


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
        # A real-GPU validation supersedes any prior carry-forward tag; drop the
        # stale annotation so the metadata reflects how this completion was reached.
        blk.pop("carry_forward", None)
        if platform == LEAD:
            _unblock_followers(obj)
        # Integrity backstop: completing while the fork has uncommitted source/build
        # edits means the validated content may not be in the branch. Warn loudly
        # (the validator must commit it first); pr_ready hard-blocks on the same.
        dirty = uncommitted_source_files(name)
        if dirty:
            sys.stderr.write(
                f"WARNING {name}: marking {platform} completed but the fork has "
                f"{len(dirty)} UNCOMMITTED source/build file(s) -- validated content "
                f"may not be in the branch (integrity gap). Commit or discard: "
                f"{', '.join(p for _, p in dirty[:6])}\n")
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


def _clean_pr_url(raw):
    """Extract the http(s) URL from a pr_url argument. Guards against a caller
    capturing extra stdout (e.g. a `Warning: N uncommitted changes` line) into
    the value via command substitution -- store only the URL, never the noise."""
    m = re.search(r"https?://\S+", raw or "")
    if not m:
        raise ValueError(f"set-pr-open: no http(s) URL found in pr_url {raw!r}")
    return m.group(0)


def set_pr_open(name, pr_url, pr_number):
    """Set lead platform to pr-open state and record PR metadata.
    Call this after successfully creating the upstream PR."""
    obj = load_status(name)
    obj["pr_url"] = _clean_pr_url(pr_url)
    obj["pr_number"] = int(pr_number)
    obj["pr_opened_at"] = now_iso()
    # Only the lead platform transitions to pr-open; followers stay at completed
    lead = obj["platforms"][LEAD]
    if lead["state"] != "completed":
        raise ValueError(f"{name}: can only open PR from completed state (lead is {lead['state']})")
    lead["state"] = "pr-open"
    lead["updated_at"] = now_iso()
    save_status(name, obj)
    return obj


def set_pr_merged(name):
    """Set lead platform to upstream-landed state and record merge timestamp.
    Call this after the upstream PR has been merged."""
    obj = load_status(name)
    if "pr_url" not in obj:
        raise ValueError(f"{name}: no PR recorded, cannot mark as merged")
    obj["pr_merged_at"] = now_iso()
    lead = obj["platforms"][LEAD]
    if lead["state"] != "pr-open":
        raise ValueError(f"{name}: can only merge from pr-open state (lead is {lead['state']})")
    lead["state"] = "upstream-landed"
    lead["updated_at"] = now_iso()
    lead["last_agent"] = "upstream-landed"
    save_status(name, obj)
    return obj


def _unblock_followers(obj):
    """Lead just completed: followers may begin (validate-first on the branch)."""
    for plat in PLATFORMS:
        if plat == LEAD:
            continue
        blk = obj["platforms"][plat]
        if blk.get("blocked"):  # retired/non-viable platform: never schedule it
            continue
        if blk["state"] == "blocked-needs-gfx90a":
            blk["state"] = "port-ready"
            blk["updated_at"] = now_iso()


def _fork_repo(name):
    return PROJECTS / name / "src"


# Tracked file kinds whose UNCOMMITTED modification in a fork is the integrity-gap
# fingerprint: a validation built against local source/build edits that were never
# committed, leaving the branch (and any PR off it) unbuildable. (Motivated by the
# baspacho/arrayfire 2026-06 gaps, incl. an arrayfire vcpkg.json -- so build
# MANIFESTS count, not just code.) Untracked files (build artifacts, scratch) do not.
_INTEGRITY_SUFFIXES = (
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".cu", ".cuh", ".hip",
    ".inl", ".inc", ".py", ".go", ".rs", ".jl", ".java", ".f", ".f90", ".f95",
    ".cmake", ".toml", ".json", ".cfg", ".pri", ".pro",
)
_INTEGRITY_NAMES = (
    "cmakelists.txt", "makefile", "setup.py", "setup.cfg", "pyproject.toml",
    "meson.build", "conanfile.py", "conanfile.txt", "vcpkg.json", "package.json",
    "cargo.toml", "build.gradle",
)


def uncommitted_source_files(name):
    """Tracked source/build files modified-but-not-committed in a project's fork
    clone -- the integrity-gap fingerprint (a validation that built against local
    edits never committed, so the branch/PR will not build). Returns a list of
    (status_code, path); untracked files (build artifacts, scratch) and ignored
    files are excluded. Never raises: a missing clone or unavailable git yields []
    (absence of a clone is not an integrity claim)."""
    repo = _fork_repo(name)
    if not Path(repo).is_dir():
        return []
    try:
        r = subprocess.run(["git", "status", "--porcelain"], cwd=str(repo),
                           capture_output=True, text=True, timeout=30)
    except Exception:
        return []
    if r.returncode != 0:
        return []
    out = []
    for line in r.stdout.splitlines():
        if len(line) < 4:
            continue
        code, path = line[:2], line[3:]
        if code.strip() in ("??", "!!"):  # untracked / ignored -> not an integrity gap
            continue
        # The fingerprint is validated CONTENT not committed: modified/added/renamed/
        # copied. Pure deletions (a dirty checkout dropping dev/benchmark files) leave
        # the branch buildable -- not an integrity gap.
        if not any(c in code for c in ("M", "A", "R", "C")):
            continue
        p = path.split(" -> ")[-1].strip().strip('"')  # handle rename "old -> new"
        base = p.rsplit("/", 1)[-1].lower()
        if base in _INTEGRITY_NAMES or any(base.endswith(s) for s in _INTEGRITY_SUFFIXES):
            out.append((code.strip(), p))
    return out


def _classify_safe(repo, old_sha, new_sha):
    """Classify a fork delta, returning None on any failure so the caller falls
    back to revalidation. Conservative by construction: machinery missing, repo
    absent, or sha unreachable all yield None, never a false carry-forward."""
    if not old_sha or not Path(repo).is_dir():
        return None
    try:
        d = str(Path(__file__).resolve().parent)
        if d not in sys.path:
            sys.path.insert(0, d)
        import changeclass
        return changeclass.classify(str(repo), old_sha, new_sha)
    except Exception:
        return None


def advance_head(name, new_sha, repo=None):
    """A porter commit advanced the shared fork HEAD. Each platform that had
    passed at a different HEAD is re-examined against the source delta from its
    validated_sha to new_sha (the cross-platform regression guard):

      - arch-independent inert (documentation-only, or a comment/format change
        with no __LINE__ hazard) cannot alter any target's compiled output, so
        validation carries forward (validated_sha bumped, stays completed).
      - everything else flips to revalidate. Rename/refactor deltas are inert but
        not arch-independent (an exported-symbol rename changes behavior with an
        identical instruction stream), so the validator confirms them per-arch
        with a binary-equivalence check before re-running GPU tests; unbuildable
        arches simply revalidate.

    On any classification failure the platform revalidates -- the safe default."""
    obj = load_status(name)
    obj["head_sha"] = new_sha
    repo = repo or _fork_repo(name)
    for plat in PLATFORMS:
        blk = obj["platforms"][plat]
        if blk["state"] != "completed" or blk.get("validated_sha") == new_sha:
            continue
        old = blk.get("validated_sha")
        verdict = _classify_safe(repo, old, new_sha)
        if verdict is not None and verdict.arch_independent:
            blk["validated_sha"] = new_sha
            blk["updated_at"] = now_iso()
            blk["carry_forward"] = {"from": old, "to": new_sha, "method": "source-class",
                                    "class": verdict.cls, "detail": verdict.detail[:200],
                                    "at": now_iso()}
        else:
            blk["state"] = "revalidate"
            blk["updated_at"] = now_iso()
    save_status(name, obj)
    return obj


def carry_forward(name, platform, new_sha, method, detail):
    """Carry one platform's validation forward to new_sha without a GPU re-run,
    because the change was proven behavior-preserving. method is 'source-class'
    (doc/comment-only) or 'binary-equiv' (compiled code objects identical on this
    arch, dynamic symbol table included). The validator calls this on a revalidate
    delta whose compiled output is unchanged; advance_head handles the
    arch-independent source classes itself. Records provenance for audit."""
    obj = load_status(name)
    blk = obj["platforms"][platform]
    if blk["state"] not in ("completed", "revalidate", "pr-open"):
        raise ValueError(f"{name}/{platform}: carry_forward needs completed/revalidate/pr-open, not {blk['state']}")
    ts = now_iso()
    # A pr-open lead stays pr-open (its upstream PR is still open at the advanced
    # branch); a behavior-preserving follow-up commit just advances its
    # validated_sha. completed/revalidate platforms transition to completed.
    if blk["state"] != "pr-open":
        blk["state"] = "completed"
        blk["completed_at"] = ts
    blk["validated_sha"] = new_sha
    blk["updated_at"] = ts
    blk["carry_forward"] = {"to": new_sha, "method": method, "detail": detail[:200], "at": ts}
    save_status(name, obj)
    return obj


def lead_done(obj):
    return obj["platforms"][LEAD]["state"] == "completed"


def project_lead_state(name):
    """Lead-platform state of another project, or None if it is not adopted."""
    try:
        return load_status(name)["platforms"][LEAD]["state"]
    except (FileNotFoundError, ValueError, json.JSONDecodeError, KeyError):
        return None


def unmet_deps(obj):
    """MOAT-internal upstream projects this one depends_on whose LEAD is not yet
    `completed` (or not adopted). A project is not portable until these are done:
    its build links/uses the ported dependency. See DEPENDENCIES.md."""
    return [d for d in obj.get("depends_on", []) if project_lead_state(d) != "completed"]


def actionable(obj, platform):
    """Is this platform pickable by an agent on this host right now?"""
    blk = obj["platforms"][platform]
    if blk["blocked"]:
        return False
    if blk["state"] in INERT:
        return False
    if platform != LEAD and not lead_done(obj):
        return False
    if unmet_deps(obj):  # deps-first ordering: wait until depended-on ports complete
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
            if blk.get("blocked"):  # retired/non-viable platform: never schedule it
                continue
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
SKIP_REASONS = ["already-supported", "ported-elsewhere",
                "cant-port", "not-a-target", "duplicate", "other"]
# already-supported: this upstream repo already supports ROCm/HIP, by any means
#   (CUDA path ported to HIP, or a native/designed-in backend); provenance is
#   irrelevant, what matters is that it runs on AMD.
# ported-elsewhere: AMD's ROCm/HIP support for this project (or an equivalent)
#   lives in a SEPARATE repo, fork, or effort; only use it with a found reference.


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
                     ext_type="unknown", priority=0.0, force=False, depends_on=None):
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
        "depends_on": list(depends_on or []),
        "platforms": {
            LEAD: _platform_block("unclaimed"),
            "linux-gfx1100": _platform_block("blocked-needs-gfx90a"),
            "windows-gfx1101": _platform_block("blocked-needs-gfx90a"),
            "windows-gfx1201": _platform_block("blocked-needs-gfx90a"),
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
        # --autostash so a concurrent agent's unstaged files in the shared
        # working tree don't abort our rebase (multi-agent MOAT runs).
        _git("pull", "--rebase", "--autostash", check=False)
        r = _git("push", check=False)
        if r.returncode == 0:
            return True
    sys.stderr.write("commit_and_push: push failed after retries; left committed locally\n")
    return False


def squash_carry_forward(name, new_sha, repo=None):
    """Advance head to a PR-prep squash, carrying every already-validated platform
    forward WITHOUT revalidation. Valid only when new_sha is a TREE-IDENTICAL
    collapse of the current validated head -- i.e. the squash combined
    already-validated commits and changed no content -- which is the case when the
    squash is done at PR-prep AFTER every platform is terminal (pr_ready). Then the
    squashed commit is known to work everywhere it already worked:
      - each `completed` platform: validated_sha advanced to new_sha, stays completed;
      - each `blocked` (non-viable, e.g. Windows-unportable) platform: left UNTOUCHED
        -- never flipped from non-viable to passing;
      - a redundant Windows-tier arch left un-validated because a sibling Windows arch
        already satisfied the one-of tier (see pr_ready): reported as `optional`, not
        a problem -- it is not required for the PR;
      - any other (actionable) state: left as-is (you should not be squashing yet).
    REFUSES if new_sha's tree differs from the current head's tree (then the squash
    introduced unvalidated content; validate it first / use advance_head). The
    carry-forward is recorded in the shared status.json, so other hosts see the new
    sha as already-validated and do not re-run -- the force-push history rewrite is
    irrelevant to them. Returns (ok, info)."""
    obj = load_status(name)
    repo = repo or _fork_repo(name)
    old_head = obj.get("head_sha")

    def _tree(sha):
        if not sha:
            return None
        r = subprocess.run(["git", "rev-parse", f"{sha}^{{tree}}"], cwd=str(repo),
                           capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None

    t_old, t_new = _tree(old_head), _tree(new_sha)
    if not t_old or not t_new or t_old != t_new:
        return (False, f"not a tree-identical squash (old tree {str(t_old)[:8]} != new {str(t_new)[:8]}); "
                       f"validate the new content first / use advance_head")
    obj["head_sha"] = new_sha
    win_satisfied = any(
        obj["platforms"][p].get("state") == "completed" and not obj["platforms"][p].get("blocked")
        for p in WINDOWS_TIER)
    carried, kept_blocked, skipped, optional = [], [], [], []
    for plat in PLATFORMS:
        blk = obj["platforms"][plat]
        if blk.get("blocked"):
            kept_blocked.append(plat)
        elif blk.get("state") == "completed":
            blk["validated_sha"] = new_sha
            blk["updated_at"] = now_iso()
            carried.append(plat)
        elif plat in WINDOWS_TIER and win_satisfied:
            optional.append((plat, blk.get("state")))  # redundant Windows arch; tier already satisfied
        else:
            skipped.append((plat, blk.get("state")))
    save_status(name, obj)
    return (True, {"carried": carried, "kept_blocked": kept_blocked,
                   "skipped": skipped, "optional": optional})


def pr_ready(name):
    """Is a port ready for its single upstream PR? Readiness has two tiers:

      * Linux archs (PR_REQUIRED_PLATFORMS: linux-gfx90a, linux-gfx1100) are each
        REQUIRED -- each must be `completed` (validated on real GPU) or `blocked`
        (a documented non-viable determination, which does NOT block the PR but
        must be scoped out of the PR body's claim).
      * Windows archs (WINDOWS_TIER: gfx1101, gfx1201, gfx1151) form ONE redundant
        tier: only ONE of them must be `completed` to unlock the PR. They are
        interchangeable proofs that the port builds and runs on Windows ROCm, so a
        single passing arch satisfies the tier; the rest may stay queued or blocked
        without gating it. If NONE has passed yet, the tier blocks (the still-
        actionable Windows archs are the blockers -- completing any one clears it);
        if every Windows arch is non-viable, the tier is non-viable and the PR
        scopes its claim to Linux.

    A REQUIRED platform in any actionable state (port-ready, revalidate, porting,
    planned, ported, review-passed, changes-requested, delta-ported,
    validation-failed, unclaimed, blocked-needs-gfx90a) means work is pending and
    BLOCKS the PR.

    Returns False if a PR already exists (lead platform in pr-open or upstream-landed state).

    Returns (ready, blocking, nonviable): ready bool; blocking list of
    (platform, state); nonviable list of platforms (documented non-viable, does not
    block). A Windows arch that is merely optional-and-not-yet-done -- because a
    sibling Windows arch already satisfied the tier -- appears in neither list."""
    obj = load_status(name)

    # Check if PR already exists (lead-only states)
    lead = obj["platforms"][LEAD]
    lead_state = lead.get("state")
    if lead_state == "pr-open":
        return (False, [(LEAD, "pr-open (PR already opened)")], [])
    if lead_state == "upstream-landed":
        return (False, [(LEAD, "upstream-landed (PR already merged)")], [])

    blocking, nonviable = [], []

    # Linux archs: each individually required.
    for plat in PR_REQUIRED_PLATFORMS:
        blk = obj["platforms"][plat]
        if blk.get("state") == "completed":
            continue
        if blk.get("blocked"):
            nonviable.append(plat)
        else:
            blocking.append((plat, blk.get("state")))

    # Windows tier: a single completed arch satisfies it.
    win = [(p, obj["platforms"][p]) for p in WINDOWS_TIER]
    if any(b.get("state") == "completed" for _, b in win):
        # Satisfied. Surface only the documented non-viable archs (for PR-body
        # scoping); leave optional-not-done archs out of both lists.
        nonviable.extend(p for p, b in win
                         if b.get("state") != "completed" and b.get("blocked"))
    else:
        actionable_win = [(p, b.get("state")) for p, b in win if not b.get("blocked")]
        if actionable_win:
            blocking.extend(actionable_win)  # completing any ONE clears the tier
        else:
            nonviable.extend(p for p, _ in win)

    # Integrity gate: the validated content must be COMMITTED. A fork with
    # uncommitted tracked source/build edits means a validation built against local
    # edits that are NOT in the branch -- the upstream PR would be unbuildable (the
    # baspacho/arrayfire 2026-06 gaps). Block the PR until the working tree is clean.
    dirty = uncommitted_source_files(name)
    if dirty:
        listed = ", ".join(p for _, p in dirty[:6]) + (" ..." if len(dirty) > 6 else "")
        blocking.append(("fork-uncommitted", f"{len(dirty)} uncommitted source/build file(s): {listed}"))

    return (not blocking, blocking, nonviable)


def record_tokens(name, tokens, source=None):
    """Append a token-usage record to projects/<name>/stats.jsonl. `tokens` is an
    agent/subagent output-token count for a unit of work (e.g. from a task
    completion notification); `source` labels what produced it. statlib sums these
    as the project's token total. Approximate by nature (output tokens, not full
    context) -- statlib always reports tokens as approx=True."""
    rec = {"kind": "tokens", "ts": now_iso(), "tokens": int(tokens)}
    if source:
        rec["source"] = source
    p = PROJECTS / name / "stats.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def commit_project(name, message, extra_paths=()):
    """Commit a project's control-plane artifacts together: status.json, notes.md,
    plan.md, and stats.jsonl (whichever exist), plus any extra_paths. Agents call
    this for every state transition so the per-phase telemetry in stats.jsonl
    (compile/test wall-clock etc., written by timeit.sh -- the README/blog metrics)
    is persisted WITH the transition and never accumulates uncommitted in the
    shared working tree. Prefer this over commit_and_push for project transitions."""
    paths = [f"projects/{name}/{fn}" for fn in
             ("status.json", "notes.md", "plan.md", "stats.jsonl")
             if (PROJECTS / name / fn).exists()]
    paths.extend(str(p) for p in extra_paths)
    return commit_and_push(paths, message)


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
    s.add_argument("--deps", nargs="*", default=[], help="MOAT project name(s) this depends on")

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

    s = sub.add_parser("carry-forward",
                       help="carry a platform's validation forward across a behavior-preserving change")
    s.add_argument("name")
    s.add_argument("platform")
    s.add_argument("sha")
    s.add_argument("method", choices=["source-class", "binary-equiv"])
    s.add_argument("detail")

    s = sub.add_parser("classify",
                       help="classify a fork delta: doc-only/comment-only/rename-only/mixed")
    s.add_argument("name")
    s.add_argument("old_sha")
    s.add_argument("new_sha")

    s = sub.add_parser("commit-project",
                       help="commit a project's status/notes/plan/stats together (telemetry-safe)")
    s.add_argument("name")
    s.add_argument("message")

    s = sub.add_parser("record-tokens", help="append a token-usage record to a project's stats.jsonl")
    s.add_argument("name")
    s.add_argument("tokens", type=int)
    s.add_argument("source", nargs="?", default=None)

    s = sub.add_parser("pr-ready", help="check PR readiness: both Linux archs completed/non-viable AND any one Windows arch completed")
    s.add_argument("name")

    s = sub.add_parser("audit-clean", help="report forks with uncommitted tracked source/build edits (integrity-gap fingerprint)")
    s.add_argument("name", nargs="?", default=None, help="one project, or omit to scan every fork")

    s = sub.add_parser("set-pr-open", help="mark lead as pr-open and record PR metadata after creating upstream PR")
    s.add_argument("name")
    s.add_argument("pr_url")
    s.add_argument("pr_number", type=int)

    s = sub.add_parser("set-pr-merged", help="mark lead as upstream-landed after PR merges")
    s.add_argument("name")

    s = sub.add_parser("squash-carry-forward",
                       help="advance head to a tree-identical PR-prep squash, carrying validated platforms forward (no revalidation)")
    s.add_argument("name")
    s.add_argument("new_sha")

    sub.add_parser("unblock-followers")
    s = sub.add_parser("validate")
    s.add_argument("name")
    s = sub.add_parser("show")
    s.add_argument("name")

    s = sub.add_parser("set-deps", help="record the MOAT projects a project depends on")
    s.add_argument("name")
    s.add_argument("deps", nargs="*")

    sub.add_parser("deps", help="print inter-project dependencies and what is blocked on them")

    args = ap.parse_args(argv)

    if args.cmd == "scaffold":
        name = scaffold_project(args.full_name, args.url, args.branch, args.ext, args.priority, args.force, args.deps)
        print(f"scaffolded projects/{name}" + (f" (depends_on={args.deps})" if args.deps else ""))
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
    elif args.cmd == "carry-forward":
        carry_forward(args.name, args.platform, args.sha, args.method, args.detail)
        print(f"{args.name}/{args.platform} carried forward -> {args.sha} ({args.method})")
    elif args.cmd == "classify":
        v = _classify_safe(_fork_repo(args.name), args.old_sha, args.new_sha)
        if v is None:
            print("class=unknown arch_independent=False (classification failed -> revalidate)")
        else:
            print(f"class={v.cls} arch_independent={v.arch_independent} inert={v.inert}")
            print(v.detail)
    elif args.cmd == "commit-project":
        ok = commit_project(args.name, args.message)
        print(f"committed projects/{args.name} (status/notes/plan/stats)" if ok else "(nothing to commit)")
    elif args.cmd == "record-tokens":
        r = record_tokens(args.name, args.tokens, args.source)
        print(f"recorded {r['tokens']} tokens for {args.name}" + (f" ({args.source})" if args.source else ""))
    elif args.cmd == "squash-carry-forward":
        ok, info = squash_carry_forward(args.name, args.new_sha)
        if not ok:
            print(f"REFUSED: {info}")
        else:
            msg = f"{args.name} -> {args.new_sha[:8]}: carried {info['carried']}; kept-blocked {info['kept_blocked']}"
            if info.get("optional"):
                msg += f"; optional Windows-tier (not required) {info['optional']}"
            if info["skipped"]:
                msg += f"; SKIPPED actionable {info['skipped']} (should not squash yet)"
            print(msg)
    elif args.cmd == "pr-ready":
        ready, blocking, nonviable = pr_ready(args.name)
        print(f"{args.name}: PR-ready={ready}")
        if blocking:
            print("  BLOCKING (Linux archs each required; Windows tier needs any ONE completed; "
                  "fork must be clean): " + ", ".join(f"{p}={s}" for p, s in blocking))
        if nonviable:
            print("  non-viable (does not block; scope the PR body): " + ", ".join(nonviable))
    elif args.cmd == "audit-clean":
        names = [args.name] if args.name else [d.name for d in sorted(PROJECTS.iterdir())
                                               if (d / "status.json").exists()]
        real_gap = False
        for n in names:
            files = uncommitted_source_files(n)
            if not files:
                continue
            try:
                states = {b.get("state") for b in load_status(n)["platforms"].values()}
            except Exception:
                states = set()
            # A real integrity gap = uncommitted edits on a fork that already has a
            # terminal (completed/pr) platform resting on them. Otherwise it is just
            # uncommitted WIP on an unfinished port.
            terminal = states & {"completed", "pr-open", "upstream-landed"}
            tag = "INTEGRITY GAP (has completed/pr platform)" if terminal else "uncommitted WIP (no completed platform yet)"
            if terminal:
                real_gap = True
            print(f"{n}: {len(files)} uncommitted source/build file(s) -- {tag}:")
            for code, p in files:
                print(f"    {code:3} {p}")
        if real_gap:
            sys.exit(1)
        else:
            print("OK: no fork with a completed/pr platform has uncommitted source edits" +
                  (f" ({args.name})" if args.name else ""))
    elif args.cmd == "set-pr-open":
        set_pr_open(args.name, args.pr_url, args.pr_number)
        print(f"{args.name}: PR opened -> {args.pr_url}")
    elif args.cmd == "set-pr-merged":
        set_pr_merged(args.name)
        obj = load_status(args.name)
        print(f"{args.name}: PR merged (lead -> upstream-landed)")
    elif args.cmd == "unblock-followers":
        changed = unblock_all_followers()
        print(" ".join(changed) if changed else "(none)")
    elif args.cmd == "validate":
        load_status(args.name)
        print(f"{args.name} status.json valid")
    elif args.cmd == "show":
        _print_json(load_status(args.name))
    elif args.cmd == "set-deps":
        obj = load_status(args.name)
        obj["depends_on"] = list(args.deps)
        obj["updated_at"] = now_iso()
        save_status(args.name, obj)
        print(f"{args.name} depends_on = {args.deps}")
    elif args.cmd == "deps":
        any_dep = False
        for d in sorted(PROJECTS.iterdir()):
            if not (d / "status.json").exists():
                continue
            try:
                obj = load_status(d.name)
            except (ValueError, json.JSONDecodeError):
                continue
            deps = obj.get("depends_on", [])
            if not deps:
                continue
            any_dep = True
            unmet = unmet_deps(obj)
            mark = "READY (deps complete)" if not unmet else ("WAITING on " + ", ".join(unmet))
            print(f"{d.name}: depends_on={deps} -> {mark}")
        if not any_dep:
            print("(no inter-project dependencies recorded)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
