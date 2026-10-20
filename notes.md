# Notes: the RAX backend interface, and how this backend fills it

Read from `robot-agent-exploration/rax` at commit `2652c73` (main) and branch
`dexjoco/native` (`bcd748c`), 2026-10-19: `docs/ARCHITECTURE.md`, `docs/DEXJOCO_BACKEND.md`,
`docs/REAL_ROBOT_ADAPTATION.md`, `skills/README.md`, `services/benchmarks.py`,
`services/mcp_contract.py`, `services/controller/gateway.py`, `services/dexjoco/{tools,backend,
session,server}.py`, `services/simulator/{rpc,protocol,io}.py`, `harness/mcp_bridge.py`,
`harness/claude_cli/auto_research.py`, `auto_research_agent/{AGENTS.md,api/runtime.py}`.

## What a RAX session is

One session = one agent (Codex CLI or Claude Code, frozen model) learning one task. The
launcher builds an **agent workspace** (AGENTS.md, TASK.md, MCP_SESSION.md, `.agents/skills/`,
`api/runtime.py`, `code/`, `memory/`), starts one **MCP frontend** on a Unix socket, and runs
the CLI natively in that workspace, sandboxed and offline. The CLI reaches the robot only
through a stdio bridge (`harness/mcp_bridge.py`) to that socket; the agent's own Python
reaches the same socket through `api.runtime.Context` (`ctx.call(tool, **arguments)`).
After exploration the agent's `code/<project>/controller.py:main(ctx)` is frozen and run alone
on held-out layouts (code mode), or live copies of the agent play them (agent formal).

## The tool contract the harness expects

`services/mcp_contract.py` defines one immutable contract per session, used for discovery and
dispatch alike. Tools come in two groups.

**Lifecycle tools** (the harness's, `services/controller/frontend.py`):

| Tool | Contract |
| --- | --- |
| `status` | task, episode state, exploration episodes left, bundles, storage, Gemini budget, `next_step` |
| `start_episode` | closes the previous episode, starts the next layout of the schedule; takes no layout or seed |
| `evaluate` | the trusted binary `task_complete` of the current exploration episode |
| `finish` | closes the current exploration episode (not a submission) |
| `register`, `rehearse`, `submit` | code mode only: freeze `code/<project>/`, run it unattended in a fresh episode, close exploration on one rehearsed bundle |

**Robot tools** come from the backend's registry entry (`services/benchmarks.py`): every name
carries the backend prefix (`dexjoco_*`), and the entry declares which are `mutating` (advance
physics), `unscoped` (pure computation, no episode), `geometry_tools` (removed by the rgb-only
profile), plus `status_tool` and `observe_tool`. The universal skills name tools by **role**
(`skills/README.md`, "Tool roles"):

| Role | Meaning | DexJoCo | This backend |
| --- | --- | --- | --- |
| `<status>` | robot status, limits, layouts, no physics | `dexjoco_status` | `microduck_status` |
| `<observe>` | current observation, cached after the last motion | `dexjoco_observe` | `microduck_observe` |
| `<pixel_to_position>` | depth back-projection of pixels (rgbd only) | `dexjoco_pixel_to_position` | `microduck_pixel_to_position` |
| `<move_pose>` / `<move_joints>` / `<base_move>` | motion: 1..N native steps, stops when the episode ends | `dexjoco_step` (1..150 action rows) | `microduck_step` (1..150 control steps of one velocity command) |
| `<hold>` | stand still for N steps | `dexjoco_hold` | `microduck_hold` |
| `<aim_camera>` | point a camera | - | the `head` argument of the motion tools |
| `<preview>` | dry run without touching the episode | `dexjoco_preview_run` | none |
| `<vlm>` | `gemini_generate` through the frontend | yes | none (no key) |

### Reply shape (`services/mcp_reply.py`, `gateway.py`)

Every reply is a JSON text block first, then image blocks; `attachments[*].content_index`
points from the JSON to its images (RGB 8-bit PNG; depth lossless uint16 PNG in millimetres,
0 invalid). An observation carries `episode_id`, `step_id`, `state`, `images`, `attachments`,
`depth`, `camera_intrinsics`, `camera_poses`, `instruction`. A motion reply adds `transition`
(what ran) and `frame_sequence` (every 10th step saved to the run tree; only the final frame
is attached). The gateway strips native reward, score, seeds, layout ids and host paths
(`_agent_safe`, `sanitized`), adds `episode_limits` (`episode_step_limit`,
`remaining_episode_steps`) and `episode_ended`; an ended episode discloses only the binary
`task_complete`. Errors pass through whole as JSON-RPC errors: `{type, operation, stage,
reason, episode_ended}`; an error never locks an episode.

### Simulator side

The backend's adapter (`services/<name>/benchmark.py`) implements hooks: `tool_definitions`,
`backend(config)`, `task_names`, `task_timeouts`, `validate_layouts`, `exploration_seeds`,
`task_context` (TASK.md), `refresh_workspace`, `episode_reviewer`, `unscoped_call`, ...  The
host-side backend validates requests before dispatch (`validate_request`, raising
`RejectedBeforeAction` so nothing moved) and talks to a **simulator process** over a msgpack RPC
(`services/simulator/`): ops `metadata`, `reset(layout_id)`, `observe`, `chunk_step(actions)`,
`hold(steps)`, `finish_pilot`, each identity-checked against `(episode_id, step_id)`. The
simulator writes operator-only files (`operator_state.json`, `evaluation_outcome.json`,
`summary.json`, `sensors.mp4`) that scoring and the dashboard read; the host never imports
the simulator package.

Layouts: exploration (collection 0) and formal (collection 1) draw from disjoint seeds; formal
layouts are never played in exploration. Scoring is binary success per episode; the formal
result is the success rate over a fixed number of held-out episodes, errors counting as
failures.

## How the robot-interface skill template is filled

`skills/templates/robot-interface/SKILL.md` is the one robot-specific skill every universal
skill links to (`../robot-interface/SKILL.md`). Its rules: every number comes from code,
configuration or a measurement on this robot (the adaptation guide's characterization
protocol), never an estimate; exact fields and per-call limits stay in the tool schemas, the
skill owns tool choice, interpretation and non-obvious behaviour; no task advice; delete
sections that do not apply. Its sections: the body and where it works; frames and units; tools
by level (sense, dry run, task space, joint space, gripper, base, hold); which tool for which
motion, **measured** (accuracy, time, when to avoid); neutral pose and recovery; gripper;
cameras and calibration; limits the backend enforces; arming and safety; reading errors; from
code.

This backend's fill (`workspace_template/skills/robot-interface/SKILL.md`): no gripper, no
arming (simulation), no dry run; the "measured" table comes from
`rax_microduck/characterize.py` (forward/backward/lateral speeds, turn rates from standstill
and while walking, stopping travel, standing drift, head tracking), and the camera section
from the rendered model. `object-localization`'s "This robot" table is filled the same way.

## Where this backend departs from the full harness

- One process, no host/simulator split: `rax_microduck/service.py` is both the MCP frontend
  (lifecycle tools, socket, recording) and the simulator (MuJoCo in-process). The RPC layer,
  supervisor, bubblewrap sandbox, dashboard, Gemini and episode reviews are not reproduced.
  The agent-facing contract (tools, reply shape, errors, workspace, bridge, `api.runtime`)
  is kept, so RAX's universal skills run unchanged.
- Code mode's `register`/`rehearse`/`submit` are replaced by `rax_microduck/formal.py`, which
  runs `controller.py:main(ctx)` on the five held-out layouts in fresh subprocesses with a
  timeout; the workspace says so (AGENTS.md).
- Motion replies disclose `end_reason` (`fell`, `cube_knocked`, `left_arena`, `step_limit`,
  `task_complete`) besides `task_complete`; RAX discloses only the binary outcome. A fallen
  biped is observable anyway, and the reason shortens exploration.
- The odometry pose in `state.base_pose` is exact (simulator ground truth); the real duck's
  odometry drifts. The cube's pose is never in the state.
