# rax-microduck

A MuJoCo-only backend for the [RAX](https://github.com/robot-agent-exploration/rax) robot-agent
harness, for Pollen's [Microduck](https://github.com/pollen-robotics/microduck) biped. It runs
end to end on a MacBook (Apple M3, 16 GB, no CUDA): a coding agent drives the duck through
RAX's tool contract, explores a task online, and compiles what worked into
`controller.py:main(ctx)`, which is then scored alone on five held-out layouts.

Every RAX branch needs Isaac Sim, a GPU MuJoCo setup or a real arm. This repo is the
"robot-agnostic skills + a new robot backend" exercise from RAX's
[adaptation guide](https://github.com/robot-agent-exploration/rax/blob/main/docs/REAL_ROBOT_ADAPTATION.md),
done for a small biped in CPU MuJoCo. [notes.md](notes.md) documents the backend interface the
harness expects and how the `robot-interface` skill template was filled.

## What works

- **`make demo`** boots the MCP service and runs the scripted reference controller for one
  episode. It walks to the cube and stops in about 4 s of wall time.
- **The backend**: `rax_microduck/service.py` serves RAX's lifecycle tools (`status`,
  `start_episode`, `evaluate`, `finish`) and the robot tools (`microduck_status`,
  `microduck_observe`, `microduck_step`, `microduck_hold`, `microduck_pixel_to_position`) over a
  Unix socket in RAX's reply shape (JSON text block + PNG image blocks, `episode_limits`,
  `episode_ended`, binary `task_complete`, whole errors). RAX's own `harness/mcp_bridge.py` and
  `api/runtime.py` are vendored unchanged, so Claude Code's `--mcp-config` and the agent's
  Python reach it the way they reach a RAX branch.
- **The robot**: pollen-robotics/microduck_rl's `robot_walk.xml`, the pretrained
  `alpha_walking.onnx` / `alpha_stand.onnx` policies as the low-level controller (61-D
  observation, 14 joint targets at 50 Hz, as the real duck's `robotd` runs them), a numpy port
  of the BAM XL330 servo model the policies were trained against, and the agent's action space
  of velocity command + head pose. Cameras: the head camera and a third-person follow camera,
  320x240 RGB and depth, rendered offscreen with MuJoCo's renderer on CGL (no GPU compute, no
  CUDA; it needs a logged-in macOS session, or EGL/OSMesa on Linux).
- **The task** `walk_to_cube`: a 5 cm blue cube 0.6-1.3 m away within 60 degrees of the start
  heading; success when the feet midpoint is within 10 cm of the cube and the robot has stood
  still for 0.5 s; failure on a fall, on pushing the cube more than 4 cm, on leaving a 3 m
  radius, or at 1500 steps (30 s). Exploration layouts (seeds 1000+) and the five formal layouts
  (seeds 20261019-20261023) are disjoint; `tests/` checks that and that a scripted oracle solves
  every formal layout.
- **The formal batch** (`make formal`, `rax_microduck/formal.py`): runs any `controller.py:main(ctx)`
  in a fresh subprocess per held-out layout with a timeout and writes `formal-scores.json`.
  The reference controller scores **4/5** (`results/reference_controller_formal.json`).
- **The agent workspace** (`rax_microduck/workspace.py`): RAX's universal skills
  (manual-robot-control, autonomous-control, experience-memory, object-localization, vendored
  from RAX main at the commit in `agent_skills/RAX_COMMIT`) with tool roles substituted, this
  robot's `robot-interface` skill with every number measured by `rax_microduck/characterize.py`,
  AGENTS.md, TASK.md, MCP_SESSION.md, MCP_CONTRACT.json and `api/runtime.py`.
- **Memory**: the service's peak resident size (`peak_rss_mb` in each session's `state.json`,
  from `getrusage`) was 0.6 GB for the exploration session, 0.75 GB for its formal batch and
  1.0 GB for `make demo`; the demo client peaks at 40-50 MB. Measured on the M3, well under the
  8 GB target.

## What does not work, or differs from RAX

- **The gait is a calibrated sim2sim approximation.** The policies were trained in mjlab
  (MuJoCo Warp) with BAM's full XL330 friction budget. With that budget applied by MuJoCo's CPU
  solver the walking net settles into a static pose and never lifts a foot; with plain PD
  servos it walks sloppily. The shipped setting scales the BAM friction budget to 0.2
  (`rax_microduck/actuator.py`, `FRICTION_SCALE`), which walks every layout without falling.
  Measured against the command: forward speed about half (0.3 m/s commanded gives 0.17 m/s),
  yaw rate about 0.7 of the command, strafing nearly ineffective, and commands below about
  0.2 m/s or 1.0 rad/s do not start a gait from standstill. All numbers are in
  `results/characterization.json` and in the robot-interface skill. This is not the real duck's
  gait and nothing here was validated on hardware.
- **Rare falls.** The reference controller's one formal failure is a fall 12 steps into a
  turn-in-place at 1.0 rad/s; the same command does not fall in 20 isolated trials. About one
  settle in forty stumbles on the initial contact, which `reset` now retries.
- **No host/simulator split, no sandbox, no dashboard, no Gemini, no episode reviews.** The
  service is one process (frontend + simulator). RAX's code-mode `register`/`rehearse`/`submit`
  are replaced by the formal runner. Motion replies disclose `end_reason` (`fell`,
  `cube_knocked`, ...), which RAX would hide.
- **Odometry is exact** (simulator truth), unlike the real duck's.
- **Rendering needs a window-server session** (CGL): it fails inside sandboxes without one,
  such as Claude Code's own command sandbox. `--no-render` keeps state-only sessions working.
- The model's own `head_camera` in `robot_walk.xml` points into the head shell; the scene
  builder re-aims it along the lens frame (`rax_microduck/scene.py`, `_fix_head_camera`).
- Claude Code's own skills link to `.agents/skills` through `.claude/skills`; the vlm-perception
  and camera-inspection skills are replaced by plain-text advice since there is no VLM tool.

## The exploration session

One session was run on 2026-10-19 and is recorded under `runs/explore_record/` (the
service's `mcp.jsonl` and `state.json`, the agent's whole workspace: notes, experiments,
controller; the saved frames are not committed).

**How it was run, honestly.** The Claude Code CLI's OAuth login on this machine had expired
(`claude -p` fails with "OAuth session expired and could not be refreshed"), so `make explore`
could not drive Claude Code through `--mcp-config`. The session was run instead by a Claude
agent (Claude Fable 5.1, spawned as a subagent of the session that built this repo) working in
the rendered RAX workspace with the kickoff prompt, with no MCP tools attached: it reached the
robot only through RAX's development path, `api.runtime.Context` over the same Unix socket,
which is how the agent's own scripts and controllers talk to the robot on every RAX branch.
It was told not to read anything outside the workspace. The MCP service therefore saw exactly
what it would see from a controller: 77 tool calls, 0 errors (4 `start_episode`, 36
`microduck_step`, 16 `microduck_hold`, 7 `microduck_observe`, 1 `microduck_pixel_to_position`,
4 `evaluate`, 4 `finish`). The run took 21 minutes and about 180k agent tokens.

**Outcome.** All four exploration episodes ended with `task_complete: true` (593, 302, 767 and
344 steps): episode 1 by hand, episodes 2 and 4 by the agent's controller unattended from
`start_episode`, episode 3 by the controller after the agent had first turned the robot away
from the cube by hand. The agent wrote `code/walk_to_cube/` (`controller.py`, `perception.py`,
`motion.py`, README): blue-mask perception with its own depth back-projection (checked against
`microduck_pixel_to_position`), fusion of head-camera, locked-odometry and overview estimates,
an approach at `vx = 0.3` with proportional yaw, an "inch" phase of short bursts and holds, and
a head/body scan for a cube out of view. It also measured for itself what the robot-interface
skill states (a yaw of 1.0 to turn from standstill, `vx = 0.2` not starting a gait, the hold's
yaw swing), and its final report lists what it did not verify (the scan branch ran only against
a mock; no cube beyond 0.93 m was seen).

**Formal batch** (`results/agent_controller_formal.json`, five held-out layouts the agent never
saw): **4/5**, 870, 429, 289 and 294 steps. The failure (layout 4) stalled 12 cm from the
cube: each turn-in-place swung the heading without closing the last 2 cm, the controller
alternated turns and holds for 900 steps and returned at step 1332 with `ok: false`. The
reference controller written by hand before the session also scores 4/5 (its failure is a fall).

## Layout

| Path | What |
| --- | --- |
| `rax_microduck/scene.py`, `actuator.py`, `policy.py`, `sim.py`, `tasks.py` | the simulator: scene, servo model, policies, episode, task and layouts |
| `rax_microduck/tools.py`, `service.py`, `bridge.py` | the RAX-contract MCP service and the stdio bridge |
| `rax_microduck/workspace.py`, `workspace_template/`, `agent_skills/` | the agent workspace and skills |
| `rax_microduck/reference_controller.py`, `demo.py`, `formal.py`, `explore.py`, `characterize.py` | the scripted baseline, `make demo`, the formal batch, the Claude Code launcher, the measurements |
| `assets/` | the two ONNX policies (from pollen-robotics/microduck_rl, Apache-2.0) and BAM's XL330 M6 parameters |
| `results/` | characterization, demo and formal records |
| `runs/explore_record/` | the recorded exploration session (MCP trace, state, the agent's workspace) |
| `notes.md` | the RAX backend interface notes |

## Setup

```bash
make setup     # uv venv (Python 3.12) + requirements, and a shallow clone of microduck_rl for the model
make test
make demo
make formal    # the reference controller on the five held-out layouts
make explore   # one Claude Code session through the harness (needs `claude login`)
```

`make explore` runs `claude -p` with `--mcp-config` pointing at the stdio bridge, Claude Code's
sandbox limited to the workspace, no web tools, and records `agent.jsonl`; then
`make formal CONTROLLER=runs/<session>/agent-workspace/code/<project>/controller.py` scores what
the agent wrote.

## Licenses

This repo's code is Apache-2.0. Vendored: RAX's universal skills, `mcp_bridge.py` and
`api/runtime.py` (Apache-2.0, `LICENSE.rax`); BAM's XL330 M6 parameters (Rhoban, Apache-2.0);
the Microduck policies and model from pollen-robotics/microduck_rl (Apache-2.0).
