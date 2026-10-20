# Microduck auto-research

Your goal is an autonomous controller that completes the configured robot task without
live intervention: satisfy every condition in `TASK.md`. Solve the task by hand first to
learn what the controller must do. The final evaluation is the formal batch: your code,
`code/<project>/controller.py:main(ctx)`, run alone on held-out layouts after this session.

## Start here

Read `TASK.md`, then any existing `memory/PROGRESS.md`, and call `status` to learn how far
the work has got and what budget remains. Do not start a new episode merely because the
conversation resumed. The skill links below are entry points: read the skill for your
current step rather than the entire workspace first.

## Workflow

1. **Understand and solve the task yourself.** Read
   [manual-robot-control](.agents/skills/manual-robot-control/SKILL.md) and
   [robot-interface](.agents/skills/robot-interface/SKILL.md). Control the robot directly
   through the MCP tools, and confirm success with `evaluate` or `task_complete: true`
   before automating.
2. **Reproduce your decisions in an autonomous controller.** Read
   [autonomous-control](.agents/skills/autonomous-control/SKILL.md). Build
   observation-driven control, test functions and full scripts in exploration, inspect
   feedback, and debug. Prefer simple motion functions and heuristics; there is no VLM in
   this session, so perceive with your own image reading and with code (colour, depth,
   [object-localization](.agents/skills/object-localization/SKILL.md)).
3. **Test the controller on fresh layouts.** Every `start_episode` gives a new layout. Run
   the whole controller from a fresh start through a development wrapper, inspect where it
   fails, and fix it. Keep the controller under `code/<project>/` with a README. The formal
   batch runs `main(ctx)` on five held-out layouts with the robot tools only; an episode
   succeeds or fails by the condition in `TASK.md`, and errors count as failures.
4. **Finish.** When no exploration episode remains, update `memory/PROGRESS.md`, write your
   final report (controller path, what it covers, the layouts it was tested on, unresolved
   issues, artifact pointers) and end your turn.

## Episodes and budgets

`MCP_SESSION.md` lists this session's tools; the live tool schemas give arguments and limits.
`start_episode` starts an exploration episode on the next layout with the task's step limit;
every robot reply reports `episode_limits` and `episode_ended`. Call `finish` to close it when
done. Layouts differ between episodes: reacquire the scene rather than assuming the first
layout persists. The session has a fixed number of exploration episodes (`status`); there is
no wall-clock limit.

Ordinary Python uses `from api.runtime import Context`; `ctx.call(tool, **arguments)` calls
an MCP tool. Follow autonomous-control for development scripts and the controller.

## Workspace and artifacts

Use ordinary Bash, Python and file tools in this workspace, your working directory. It is
persistent and the only place you can write; the skills are under `.agents/skills`. There is
no internet or simulator access: do not seek simulator internals. Run records are under
`runtime/autonomous_controller/`, read-only: each motion call's frames are there.
Python has numpy and Pillow (`PIL`) for images.

Inspect selected images, action records and logs as needed; do not load whole recordings
into context.

## Intermediate notes and compaction

Write intermediate findings to `memory/` as you work, not only at the end. Keep
`memory/PROGRESS.md` current after useful experiments and each workflow step: current step
and episode, confirmed results, failed approaches and lessons, working motion parameters,
code, log and frame paths, remaining budget, and the next concrete step. Keep reusable robot
tips in `memory/TIPS.md` following
[experience-memory](.agents/skills/experience-memory/SKILL.md). Store evidence on disk and
reference its path instead of copying large traces or image bytes into notes.
