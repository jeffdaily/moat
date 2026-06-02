# MOAT

MOAT (Moat Obliteration via Automated Translation) is a Claude-driven effort to port popular CUDA GitHub projects to ROCm/HIP, one repo at a time, across AMD targets: Linux gfx90a first, then Linux gfx1100, then Windows gfx1151. Each port lives on a fork in the jeffdaily org. This repo is the control plane that tracks progress, accumulates porting best practices, and lets any Claude CLI resume the work where the last one stopped.

# On startup, do this

1. Run `bash utils/orient.sh` (or `/port-next`). It pulls the latest MOAT state, detects this host's AMD arch, and prints the single next project to work on, the stage (planner, porter, reviewer, or validator), and a ready-to-paste dispatch line.
2. Dispatch the named subagent scoped to that project. State in `projects/<name>/status.json` gates every handoff; never skip a state.
3. Operate in auto mode within the Autonomy boundary below. Stop only at the two halts: the upstream-PR gate, and genuine blockers.

# Pipeline

`unclaimed -> planner -> porter -> reviewer -> validator -> [user gate] -> upstream PR`. The reviewer can bounce back to the porter (changes-requested); the validator can bounce back to the porter (validation-failed). Follower platforms (gfx1100, gfx1151) reuse the gfx90a fork branch, validate first, and only delta-port on failure; they do not re-plan from scratch. The three AMD targets share one unified ROCm port, so any change re-validates the platforms that already passed. See PORTING_GUIDE.md for strategies and fault classes.

# Autonomy boundary

Auto mode is maximal within these bounds.

Allowed without asking: edit the working tree and the fork clone in `projects/<name>/src/`; build, hipify, compile, run tests including on GPU; install missing build dependencies via apt or conda when a standard package exists; inside the fork clone do git branch/add/commit/amend/rebase and `git push --force-with-lease` to the jeffdaily fork; `gh repo fork` and `gh` reads; update status.json, plan.md, notes.md, PORTING_GUIDE.md and push them to this MOAT repo.

Requires explicit approval (show the draft, wait for yes): any GitHub-visible action against an UPSTREAM repo (opening a PR, a PR or issue comment, a review) or anything visible outside the jeffdaily forks. The upstream PR is additionally gated behind the `pr-approved-by-user` state.

Make progress without asking; ask jeff only when truly unavoidable (install missing build deps yourself via apt/conda; do not ask for those). If you are genuinely stuck on a project after a real attempt (an unresolvable dependency, a porting-strategy decision with no clear answer, missing hardware or access, or repeated validation failure with unclear cause after 3 porter attempts), set the platform `blocked=true` with a concrete reason and MOVE ON to the next project rather than waiting; jeff will have you summarize the blocked projects when he returns.

# Standing rules

- All forks and all work under the `jeffdaily` GitHub account. Do not switch to any AMD-internal account; MOAT is public.
- Never ghstack.
- The MOAT repo itself takes commits on top and is never force-pushed. Subproject forks are the deliberate exception: keep a clean curated commit (amend/rebase as needed) and `git push --force-with-lease`, with the commit title and body kept current with the latest port state. Bare `--force` without lease is forbidden everywhere. The port lives on a `moat-port` topic branch (the fork's default branch stays a clean upstream mirror); the single upstream PR is `moat-port` -> upstream default.
- Validation means exercising the change on real GPU. Lint is not validation. A CPU-only docker build smoketest proves compilation only, never GPU correctness, so it is never the sole validation gate.
- Commit titles: prefix `[ROCm]`, <= 72 chars. Body mentions Claude by name; no `Co-Authored-By: noreply` trailer.
- Prose: "ROCm" casing (code identifiers like USE_ROCM and arch names like gfx90a stay as-is). ASCII only, no em-dash (use -- or ; or parentheses). Do not manually line-wrap GitHub or markdown prose; let it reflow. No sycophancy.
- ROCm vs HIP (use the right word; they name different facets -- see PORTING_GUIDE "Naming: ROCm vs HIP"): HIP is the programming model -- the kernel-language dialect plus the `hipXxx` runtime API (the analogue of CUDA C++ and the CUDA runtime). ROCm is the platform/toolkit -- the compiler, runtime, driver, and the roc*/hip* domain libraries (the analogue of the CUDA Toolkit). The CODE port is "to HIP" (hipify, the cuda_to_hip.h shim, runtime symbols); the TARGET, build flag, and libraries are "ROCm" (USE_HIP/USE_ROCM, "ROCm 7.2.1", cuFFT->hipFFT). A pure language+runtime port is most precisely "a HIP port targeting ROCm." Do not call the platform "HIP" or the kernel dialect "ROCm". Commit prefix stays `[ROCm]` as the umbrella ("adds AMD support"); name a specific roc*/hip* library only when actually substituting it.
- No MOAT vocabulary in upstream-visible text (commit messages, code comments, PR titles/bodies): never "lead"/"follower", "Strategy A/B", "head_sha", "curated commit", "validated_sha", "revalidate", "moat-port" as jargon. Keep the technical rationale, drop the in-house labels. This text goes to external maintainers.

# MOAT repo synchronization

This repo is shared by every CLI, so keep it fresh: pull before deciding, push often. orient.sh runs `git pull --rebase` before selecting. Route every status.json transition and artifact write through `moatlib.commit_and_push` (commit on top, pull --rebase, merge, push, bounded retry); push each transition as it happens, not in a batch. status.json conflicts resolve via the `merge=moat-status` driver; notes.md and the PORTING_GUIDE.md changelog use `merge=union`.

# Scratch space

Use `agent_space/` (gitignored, at repo root) for temporary scripts and throwaway experiments. Do not commit files from this directory.

# PR review

When asked to review work (the reviewer agent or otherwise), always use the /pr-review skill.

# Build

Check local memory for build configuration (env vars, incremental-build shortcuts) before building, and apply what you find. If nothing applies, search the project for build docs or analyze its build files (CMake, setup.py). Create repeatable build scripts as needed and record them in the project's notes.md.

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

# How to add a project

Review candidates with `python3 utils/triage.py review`. Mark any you will not port using `python3 utils/triage.py skip <owner/repo> --reason <already-supported|ported-elsewhere|cant-port|not-a-target|duplicate|other> --note "..."` (or `triage.py verify <owner/repo>` to flag one for investigation without skipping); decisions persist in data/dispositions.json and scaffold refuses a skipped project. Adopt a remaining row with `python3 utils/moatlib.py scaffold <owner/repo>` (writes projects/<name>/{status.json,upstream.json}); orient.sh then picks it up.

# Project dependencies

Some targets build on other targets (RAPIDS: most build on rmm; cuml/cugraph on raft/cudf). A project's status.json `depends_on` lists the MOAT projects its build needs; the selector will not pick a project until those deps' lead platform is `completed` (deps-first ordering -- `moatlib.py deps` shows the graph). When porting a project that has deps, clone + build + install each ported dep (jeffdaily/<dep> @ moat-port) per its notes.md "## Install as a dependency" section into `_deps/<dep>/` (gitignored at the repo root) and point your build at it (e.g. `-DCMAKE_PREFIX_PATH=.../_deps/<dep>/install`). Record deps with `scaffold --deps ...` or `set-deps <name> <deps...>`. A base library other targets consume MUST document an "## Install as a dependency" section in its notes.md. Full workflow: DEPENDENCIES.md.
