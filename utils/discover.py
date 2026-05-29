#!/usr/bin/env python3
"""GitHub discovery + ranking for MOAT.

Honest limit: this is a best-effort union of targeted searches, NOT a census of
every CUDA repo. GitHub repo search caps at 1000 results per query;
`language:Cuda` only catches repos whose dominant language is Cuda (it misses
C++/Python repos that merely contain .cu files); `topic:cuda` only catches
self-tagged repos. Coverage grows by adding queries in config/discover.toml.

Output: data/candidates.json (ranked, filtered) and data/candidates.raw.jsonl
(every unique hit before filtering, for audit). Adopt rows with
`python3 utils/moatlib.py scaffold <owner/repo>`."""

import argparse
import json
import math
import subprocess
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "config" / "discover.toml"
DATA = REPO_ROOT / "data"
FIELDS = ("fullName,url,defaultBranch,stargazersCount,forksCount,pushedAt,"
          "isFork,isArchived,isDisabled,language,description")


def load_cfg():
    return tomllib.loads(CONFIG.read_text())


def run_query(query, limit):
    """One `gh search repos` call. Returns a list of repo dicts (gh handles
    paging up to the 1000 cap). Empty on failure; backs off on rate-limit."""
    cmd = ["gh", "search", "repos", *query.split(),
           "--sort", "stars", "--order", "desc",
           "--limit", str(limit), "--json", FIELDS]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(f"discover: query failed: {query}\n{r.stderr.strip()}\n")
        if "rate limit" in r.stderr.lower() or "API rate" in r.stderr:
            time.sleep(30)
        return []
    try:
        return json.loads(r.stdout or "[]")
    except json.JSONDecodeError:
        return []


def days_since(iso):
    if not iso:
        return 1e9
    try:
        t = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return 1e9
    return (datetime.now(timezone.utc) - t).total_seconds() / 86400.0


def gather(cfg, limit):
    """Run every query, dedupe by lowercased full_name, merge matched_queries,
    write the raw audit trail."""
    seen = {}
    DATA.mkdir(parents=True, exist_ok=True)
    raw = DATA / "candidates.raw.jsonl"
    min_stars = cfg["search"]["min_stars"]
    since_date = _since_date(cfg["search"]["since_months"])
    with open(raw, "w") as rawf:
        for q in cfg["search"]["queries"]:
            full_q = f"{q} stars:>={min_stars} pushed:>={since_date}"
            hits = run_query(full_q, limit)
            for h in hits:
                key = h["fullName"].lower()
                rawf.write(json.dumps(h) + "\n")
                if key in seen:
                    seen[key]["matched_queries"].append(q)
                else:
                    h["matched_queries"] = [q]
                    seen[key] = h
            time.sleep(cfg["search"]["throttle_seconds"])
    return list(seen.values())


def _since_date(months):
    days = int(months) * 30
    t = datetime.now(timezone.utc).timestamp() - days * 86400
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d")


def passes(h, cfg):
    s = cfg["search"]
    fn = h["fullName"].lower()
    if fn in {e.lower() for e in s["exclude"]}:
        return False
    if h.get("isArchived") or h.get("isDisabled"):
        return False
    if h.get("isFork") and fn not in {e.lower() for e in s["fork_exceptions"]}:
        return False
    if (h.get("stargazersCount") or 0) < s["min_stars"]:
        return False
    if days_since(h.get("pushedAt")) > int(s["since_months"]) * 30:
        return False
    return True


def score(h, w):
    stars = h.get("stargazersCount") or 0
    forks = h.get("forksCount") or 0
    dsp = days_since(h.get("pushedAt"))
    stars_term = w["stars"] * math.log10(stars + 1)
    forks_term = w["forks"] * math.log10(forks + 1)
    recency_term = w["recency"] * (0.5 ** (dsp / w["half_life_days"]))
    parts = {"stars_term": round(stars_term, 3),
             "forks_term": round(forks_term, 3),
             "recency_term": round(recency_term, 3)}
    return round(stars_term + forks_term + recency_term, 3), parts


def to_record(h, cfg):
    prio, parts = score(h, cfg["weights"])
    fn = h["fullName"]
    return {
        "full_name": fn,
        "url": h.get("url") or f"https://github.com/{fn}",
        "default_branch": h.get("defaultBranch"),
        "stars": h.get("stargazersCount") or 0,
        "forks": h.get("forksCount") or 0,
        "pushed_at": h.get("pushedAt"),
        "days_since_push": round(days_since(h.get("pushedAt")), 1),
        "is_fork": bool(h.get("isFork")),
        "fork_exception": h["fullName"].lower() in {e.lower() for e in cfg["search"]["fork_exceptions"]},
        "language": h.get("language"),
        "description": (h.get("description") or "").strip(),
        "matched_queries": sorted(set(h.get("matched_queries", []))),
        "priority": prio,
        "score_parts": parts,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(prog="discover")
    ap.add_argument("--limit", type=int, help="per-query result cap (<=1000)")
    ap.add_argument("--out", default=str(DATA / "candidates.json"))
    args = ap.parse_args(argv)
    cfg = load_cfg()
    limit = args.limit or cfg["search"]["per_query_limit"]
    raw_hits = gather(cfg, limit)
    recs = [to_record(h, cfg) for h in raw_hits if passes(h, cfg)]
    recs.sort(key=lambda r: (-r["priority"], r["full_name"].lower()))
    with open(args.out, "w") as f:
        json.dump(recs, f, indent=2)
        f.write("\n")
    sys.stderr.write(f"discover: {len(raw_hits)} unique hits, {len(recs)} after filters -> {args.out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
