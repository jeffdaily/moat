---
name: port-next
description: Resume MOAT. Detect this host's AMD platform, find the next actionable project, and dispatch the right subagent. Use when starting work in the MOAT repo or when asked to continue porting or "port next".
---

# port-next

Run the MOAT pipeline for the next available project on this host.

## Steps
1. Run `bash utils/orient.sh`. It pulls the latest state, detects the AMD arch, and prints the next project, its state, and the stage (planner / porter / reviewer / validator), or NONE.
2. If it names a project + stage, dispatch that subagent scoped to the project: "Use the <stage> subagent on projects/<name>". The subagent reads CLAUDE.md (Pipeline, Autonomy boundary, Standing rules), plan.md, notes.md, and PORTING_GUIDE.md.
3. Operate in auto mode within the Autonomy boundary. Stop only at the upstream-PR gate or a genuine blocker (set `blocked`, ask a specific question).
4. When the subagent finishes a stage (state advances), re-run this skill to pick up the next stage or project. Honor the pipeline depth in config/moat.toml: at most one heavy build/test stage at a time per host; a planner may run ahead for the next project.

## Notes
- All work and forks under the jeffdaily account. Never open or comment on an upstream PR without explicit approval.
- See CLAUDE.md for the full pipeline and rules.
