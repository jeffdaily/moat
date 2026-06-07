# MOAT

MOAT (Moat Obliteration via Automated Translation) is a Claude-driven effort to port popular CUDA GitHub projects to ROCm/HIP, one repo at a time, across AMD targets: Linux gfx90a first, then Linux gfx1100, then Windows on gfx1101 and gfx1201. (gfx1151 was an earlier Windows target host, now retired: its validations are kept as records but it is scheduled no new work.) Each port lives on a fork in the jeffdaily org. This repo is the control plane that tracks progress, accumulates porting best practices, and lets any Claude CLI resume the work where the last one stopped.

# On startup, do this

1. Run `bash utils/orient.sh` (or `/port-next`). It pulls the latest MOAT state, detects this host's AMD arch, and prints the single next project to work on, the stage (planner, porter, reviewer, or validator), and a ready-to-paste dispatch line.
2. Dispatch the named subagent scoped to that project. State in `projects/<name>/status.json` gates every handoff; never skip a state.
3. Operate in auto mode within the Autonomy boundary below. Stop only at the two halts: the upstream-PR gate, and genuine blockers.

# Pipeline

`unclaimed -> planner -> porter -> reviewer -> validator -> completed -> [user gate] -> pr-open -> upstream-landed`. The reviewer can bounce back to the porter (changes-requested); the validator can bounce back to the porter (validation-failed). Follower platforms (Linux gfx1100, Windows gfx1101 and gfx1201) reuse the gfx90a fork branch, validate first, and only delta-port on failure; they do not re-plan from scratch. The AMD targets share one unified ROCm port, so a functional change re-validates the platforms that already passed. Behavior-preserving changes do NOT: the regression guard (`moatlib.advance_head`) classifies each `validated_sha -> new HEAD` delta and carries validation forward for documentation-only or comment/format-only changes (arch-independent, no GPU re-run on any platform); a symbol rename or refactor flips passed platforms to `revalidate`, where the validator may confirm it per-arch with a binary-equivalence check (`utils/codeobj_diff.py`: identical device code objects + exported symbols) and carry forward without re-running GPU tests, while unbuildable arches revalidate normally. Any classification uncertainty defaults to full revalidation. PR states (pr-open, upstream-landed) are lead-only; followers stay at completed. See PORTING_GUIDE.md for strategies and fault classes.

# Autonomy boundary

Auto mode is maximal within these bounds.

Allowed without asking: edit the working tree and the fork clone in `projects/<name>/src/`; build, hipify, compile, run tests including on GPU; install missing build dependencies via apt or conda when a standard package exists; inside the fork clone do git branch/add/commit/amend/rebase and `git push --force-with-lease` to the jeffdaily fork; `gh repo fork` and `gh` reads; update status.json, plan.md, notes.md, PORTING_GUIDE.md and push them to this MOAT repo.

Requires explicit approval (show the draft, wait for yes): any GitHub-visible action against an UPSTREAM repo (opening a PR, a PR or issue comment, a review) or anything visible outside the jeffdaily forks. After opening the upstream PR, run `python3 utils/moatlib.py set-pr-open <name> <pr_url> <pr_number>` to mark the lead platform as pr-open and record PR metadata. After the PR merges, run `python3 utils/moatlib.py set-pr-merged <name>` to mark it as upstream-landed.

Upstream-PR readiness gate: before opening the single upstream PR, readiness is checked in two tiers. The Linux archs (linux-gfx90a then linux-gfx1100) are each REQUIRED -- each must be `completed` (validated on real GPU) or `blocked` (a documented non-viable determination that does NOT block the PR but must be scoped out of the PR body's claim). The Windows archs (gfx1101, gfx1201, gfx1151) are ONE redundant tier: only ONE of them must be `completed` to unlock the PR (they are interchangeable proofs that the port builds and runs on Windows ROCm), so the others may stay queued or blocked without gating it; if none has passed, completing any one clears the tier. This reflects hardware reality -- a Windows host can permanently lose an arch's GPU (gfx1101 on the current host), and requiring every Windows arch would wedge the PR on hardware that no longer exists. A REQUIRED (Linux) platform in any actionable state (port-ready, revalidate, porting, ported, planned, review-passed, changes-requested, delta-ported, validation-failed) means validation is still pending and BLOCKS the PR. The work pipeline is unchanged: each arch still validates wherever its GPU is present (followers validate on their own gfx1100 Linux and gfx1101/gfx1201 Windows hosts). Satisfying the Windows tier does not close it: an un-validated Windows arch stays selectable even after a sibling satisfies the tier or the PR opens, so a host that later gains that GPU may elect to validate it and complete the set -- that is purely additive (a validation changes no head_sha, so it disturbs neither the open PR nor the other platforms). Check with `python3 utils/moatlib.py pr-ready <name>`.

PR-prep sequence (this ordering avoids needless follower revalidation): (1) make all PR-prep edits -- jargon scrub, CMake arch auto-detect, docs -- as commits ON TOP of the validated port; (2) get every platform terminal at that head (`pr-ready`); (3) THEN squash to one clean commit (a tree-identical collapse of the validated content) and run `python3 utils/moatlib.py squash-carry-forward <name> <squashed-sha>` -- it advances head_sha and carries every `completed` platform forward in status.json, so other hosts read already-validated-at-the-new-sha and never re-run (the force-push history rewrite is irrelevant; status.json is the authority), while `blocked`/non-viable platforms stay blocked (never flipped to passing). Squashing BEFORE the prep edits are validated on every platform makes the squash non-tree-identical, so `squash-carry-forward` refuses and the followers must revalidate the squashed sha -- so do prep-then-validate-then-squash, not squash-first. Do ALL MOAT cleanup (jargon, comments, docs) in the prep phase so nothing remains after the squash. If cleanup IS caught post-squash, it is a comment/doc (behavior-preserving) delta: commit it on top -- `advance_head` classifies it inert and carries every platform forward, no revalidation -- then re-squash to one commit (tree-identical to that cleanup state, so `squash-carry-forward` carries forward again). Only a FUNCTIONAL post-squash change correctly triggers revalidation.

Make progress without asking; ask jeff only when truly unavoidable (install missing build deps yourself via apt/conda; do not ask for those). If you are genuinely stuck on a project after a real attempt (an unresolvable dependency, a porting-strategy decision with no clear answer, missing hardware or access, or repeated validation failure with unclear cause after 3 porter attempts), set the platform `blocked=true` with a concrete reason and MOVE ON to the next project rather than waiting; jeff will have you summarize the blocked projects when he returns.

# Standing rules

- All forks and all work under the `jeffdaily` GitHub account. Do not switch to any AMD-internal account; MOAT is public.
- Never ghstack.
- The MOAT repo itself takes commits on top and is never force-pushed. Subproject forks may carry natural multi-commit history on the `moat-port` branch (the old single-curated-commit rule is retired -- it was overly restrictive); `git push --force-with-lease` is still allowed for rebases/cleanup, and bare `--force` without lease is forbidden everywhere. Squash to a tidy single commit only if/when you want it, e.g. right before opening the upstream PR. IMPORTANT: do not `git commit --amend` away a commit a platform has already validated (its `validated_sha`); amending orphans that commit on the remote and forces every passed platform to revalidate. Put follow-up edits in a NEW commit on top so `validated_sha` stays a reachable ancestor and the regression guard can classify the delta (see "MOAT repo synchronization" / cosmetic carry-forward). The port lives on a `moat-port` topic branch (the fork's default branch stays a clean upstream mirror); the single upstream PR is `moat-port` -> upstream default.
- Validation means exercising the change on real GPU. Lint is not validation. A CPU-only docker build smoketest proves compilation only, never GPU correctness, so it is never the sole validation gate.
- Commit titles: prefix `[ROCm]`, <= 72 chars. Body mentions Claude by name; no `Co-Authored-By: noreply` trailer.
- Prose: "ROCm" casing (code identifiers like USE_ROCM and arch names like gfx90a stay as-is). ASCII only, no em-dash (use -- or ; or parentheses). Do not manually line-wrap GitHub or markdown prose; let it reflow. No sycophancy.
- ROCm vs HIP (use the right word; they name different facets -- see PORTING_GUIDE "Naming: ROCm vs HIP"): HIP is the programming model -- the kernel-language dialect plus the `hipXxx` runtime API (the analogue of CUDA C++ and the CUDA runtime). ROCm is the platform/toolkit -- the compiler, runtime, driver, and the roc*/hip* domain libraries (the analogue of the CUDA Toolkit). The CODE port is "to HIP" (hipify, the cuda_to_hip.h shim, runtime symbols); the TARGET, build flag, and libraries are "ROCm" (USE_HIP/USE_ROCM, "ROCm 7.2.1", cuFFT->hipFFT). A pure language+runtime port is most precisely "a HIP port targeting ROCm." Do not call the platform "HIP" or the kernel dialect "ROCm". Commit prefix stays `[ROCm]` as the umbrella ("adds AMD support"); name a specific roc*/hip* library only when actually substituting it.
- No MOAT vocabulary in upstream-visible text (commit messages, code comments, PR titles/bodies): never "lead"/"follower", "Strategy A/B", "head_sha", "curated commit", "validated_sha", "revalidate", "moat-port" as jargon. Keep the technical rationale, drop the in-house labels. This text goes to external maintainers.

# MOAT repo synchronization

This repo is shared by every CLI, so keep it fresh: pull before deciding, push often. orient.sh runs `git pull --rebase` before selecting. Route every status.json transition and artifact write through `moatlib.commit_and_push` (commit on top, pull --rebase, merge, push, bounded retry); for a project transition prefer `moatlib.commit_project(name, msg)` (or `moatlib.py commit-project`), which also stages that project's `stats.jsonl` so the per-phase telemetry timeit.sh writes is persisted and never accumulates uncommitted. status.json conflicts resolve via the `merge=moat-status` driver; notes.md and the PORTING_GUIDE.md changelog use `merge=union`.

Telemetry: agents wrap build/test phases in `utils/timeit.sh` (wall-clock) and bracket runs with `utils/session.sh` (session wall). Tokens can only be recorded by the ORCHESTRATOR: when a dispatched subagent task completes, its notification reports `subagent_tokens` -- record it with `python3 utils/moatlib.py record-tokens <name> <tokens> "<agent role>"` so token cost is captured for the README/blog metrics (the subagent cannot self-report; the count exists only in the parent's completion notification). statlib.py aggregates compile/test wall, session/thinking wall, and tokens (always approx=True).

# Scratch space

Use `agent_space/` (gitignored, at repo root) for temporary scripts and throwaway experiments. Do not commit files from this directory.

# PR review

When asked to review work (the reviewer agent or otherwise), always use the /pr-review skill.

# Build

Check local memory for build configuration (env vars, incremental-build shortcuts) before building, and apply what you find. If nothing applies, search the project for build docs or analyze its build files (CMake, setup.py). Create repeatable build scripts as needed and record them in the project's notes.md.

Windows host (gfx1101 + gfx1201): this is a workstation with two discrete AMD GPUs (gfx1101 Radeon PRO V710 at HIP_VISIBLE_DEVICES=0, gfx1201 RX 9070 XT at HIP_VISIBLE_DEVICES=1). It is NOT power-constrained -- use normal build parallelism. IMPORTANT one-GPU-per-process rule: both GPUs visible in one process crashes the ROCm 7.14 HIP runtime (heterogeneous RDNA3+RDNA4 multi-GPU bug), so always pin `HIP_VISIBLE_DEVICES` to exactly one GPU for every build/test/validation, and validate one arch per process. Build/run via the TheRock PyTorch venv; full host setup in local memory [[windows-gfx1101-gfx1201-host]]. The retired gfx1151 Strix Halo APU host had a shared CPU/iGPU power rail and needed `-j6` caps; that constraint does NOT apply here (history in [[gfx1151-host-power-reboots]]).

# Testing

Find the project's automated tests from its docs or build files. Focus on GPU tests, but do not regress non-GPU tests. Full validation is required to mark success. Do NOT add GitHub Actions smoketest workflows to ports: a CPU-only GHA build cannot observe any GPU fault (so it is not a real gate), and every yml edit changes the fork HEAD sha, which trips the cross-platform regression guard and forces all platforms to revalidate -- not worth the churn or the failing-run email noise. Leave upstream workflows as they are; after `gh repo fork`, disable Actions on the jeffdaily fork so neither our changes nor inherited upstream CI run and email on it: `gh api -X PUT repos/jeffdaily/<fork>/actions/permissions -F enabled=false`. A local CPU-only docker build (image `rocm/dev-ubuntu-24.04:7.2.4-complete`) is fine as a manual compile check, never wired into the fork's Actions.

# Commit messages

Do not bullet-list individual changes. If the change is large, explain the order to review it; if short, omit the list. Include a Test Plan section with the literal commands run in fenced code blocks. If fixing a bug, explain the root cause and how the fix works. Disclose that the work was authored with an AI assistant. When amending, check that the message still describes the change.

# Coding style

- Minimize comments; code should be self-documenting. Comments carry non-obvious global context, not restatement of the code.
- No trivial 1-2 line single-use helpers unless they clearly improve readability.
- Prefer clear abstractions and explicit state. No dynamic setattr/getattr field juggling.
- Match the existing style of the project being modified.
- Assume the reader knows the project's domain but not this specific code.
- ASCII only in new comments. Leave preexisting Unicode in untouched comments alone.

If uncertain, choose the simpler, more concise implementation.

# Where things live

- PORTING_GUIDE.md -- living cross-project porting knowledge; the planner reads it every time.
- projects/<name>/ -- plan.md, notes.md, status.json, stats.jsonl per project.
- utils/ -- orient.sh (entrypoint), moatlib.py (state machine + sync), discover.py, gen_readme.py.
- .claude/agents/ -- planner, porter, reviewer, validator.
- data/candidates.json -- ranked discovery output.
- findings/<slug>/ -- prepared ROCm-component bug reports / reproducers (not auto-filed upstream).
- data/deferred.json -- the deferred-work registry (what we postponed and where to resume). Ask MOAT "what did we defer?" with `python3 utils/deferred.py list`; record a deferral with `utils/deferred.py add` (kinds: rocm-bug-report, feature-port, other). When you scope a sub-feature out of a port or prepare a findings/ bug report you do not file, register it here so it is not lost.

# How to add a project

Review candidates with `python3 utils/triage.py review`. Mark any you will not port using `python3 utils/triage.py skip <owner/repo> --reason <already-supported|ported-elsewhere|cant-port|not-a-target|duplicate|other> --note "..."` (or `triage.py verify <owner/repo>` to flag one for investigation without skipping); decisions persist in data/dispositions.json and scaffold refuses a skipped project. Adopt a remaining row with `python3 utils/moatlib.py scaffold <owner/repo>` (writes projects/<name>/{status.json,upstream.json}); orient.sh then picks it up.

# Project dependencies

Some targets build on other targets (RAPIDS: most build on rmm; cuml/cugraph on raft/cudf). A project's status.json `depends_on` lists the MOAT projects its build needs; the selector will not pick a project until those deps' lead platform is `completed` (deps-first ordering -- `moatlib.py deps` shows the graph). When porting a project that has deps, clone + build + install each ported dep (jeffdaily/<dep> @ moat-port) per its notes.md "## Install as a dependency" section into `_deps/<dep>/` (gitignored at the repo root) and point your build at it (e.g. `-DCMAKE_PREFIX_PATH=.../_deps/<dep>/install`). Record deps with `scaffold --deps ...` or `set-deps <name> <deps...>`. A base library other targets consume MUST document an "## Install as a dependency" section in its notes.md. Full workflow: DEPENDENCIES.md.
