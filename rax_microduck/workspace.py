"""Render the agent workspace, as RAX's workspace generator does for a branch.

Layers: the vendored universal skills (``agent_skills/``, from RAX main; roles such as
``<observe>`` replaced by this robot's tools), this robot's own ``robot-interface`` and the
filled "This robot" table of object-localization (``workspace_template/skills``), the documents
(AGENTS.md, TASK.md, START_PROMPT.md, MCP_SESSION.md, MCP_CONTRACT.json), ``api/runtime.py``,
and the empty ``code/``, ``memory/``, ``experiments/`` folders. ``runtime/autonomous_controller``
links to the session's published run tree.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from . import tools as T
from .paths import ROOT
from .tasks import FORMAL_LAYOUTS, TASK

SKILLS = ("manual-robot-control", "autonomous-control", "experience-memory", "object-localization")
ROLES = {'ctx.call("<observe>")': 'ctx.call("microduck_observe")',
         'ctx.call("<move_pose>", **target)': 'ctx.call("microduck_step", twist=[0.3, 0.0, 0.0], steps=25)',
         "`<status>`": "`microduck_status`", "`<observe>`": "`microduck_observe`",
         "`<pixel_to_position>`": "`microduck_pixel_to_position`", "`<move_pose>`": "`microduck_step`",
         "`<move_joints>`": "`microduck_step`", "`<base_move>`": "`microduck_step`", "`<hold>`": "`microduck_hold`",
         "`<aim_camera>`": "the `head` argument of `microduck_step` / `microduck_hold`",
         "`<preview>`": "(this robot has no dry-run tool: test short motions in the episode instead)",
         "`<measured>`": "`<see the table>`",
         "[VLM](../vlm-perception/SKILL.md)": "VLM (none is available in this session: use your own reading of the "
                                               "images and colour or depth heuristics in code)",
         "[camera-inspection](../camera-inspection/SKILL.md)": "a closer look (walk nearer and look again)"}


def task_md():
    return "\n".join([
        f"# Task: {TASK.name}", "", f"**Instruction:** {TASK.instruction}", "",
        "## Rules", "",
        "- Success: the episode ends with the robot standing still (base speed under 0.06 m/s for 0.5 s) with the",
        "  midpoint of its feet within 10 cm (horizontal) of the cube's centre.",
        "- Failure, ending the episode at once: the robot falls; the cube is pushed more than 4 cm from where it",
        "  started; the robot leaves a 3 m radius; or the step limit is reached.",
        "- The cube is a 5 cm blue cube on the floor, 0.6-1.3 m away, within 60 degrees of the robot's start heading.",
        "", "## Episode", "",
        "- Robot: the Microduck biped under its pretrained walking policy (see the robot-interface skill).",
        f"- Cameras: {', '.join(f'`{c}`' for c in T.CAMERAS)}, both with depth.",
        f"- Step limit: {TASK.step_limit} steps of 20 ms of physics each (50 per second of robot time); the episode",
        "  ends there, on success, or on a failure condition.",
        f"- Exploration episodes use their own layouts; the formal batch plays {FORMAL_LAYOUTS} held-out layouts that",
        "  exploration never uses.", "",
        "## Success", "",
        "- An episode succeeds or fails. There is no partial credit: an episode that ends without the success",
        "  condition, hits the step limit, triggers a failure condition or errors does not succeed.", ""])


def contract():
    tools = ["status", "start_episode", "evaluate", "finish", *T.ROBOT_TOOLS]
    return {"mode": "auto-research", "phase": "exploration", "observation_profile": "rgbd", "exploration_envs": 1,
            "subagents": 0, "robot_tools": sorted(T.ROBOT_TOOLS), "gemini": False, "benchmark": "microduck",
            "role": "exploration", "tools": sorted(tools)}


def session_md():
    tools = ", ".join(f"`{t}`" for t in sorted(contract()["tools"]))
    return ("# Session tools\n\nRole: exploration. Observation profile: rgbd.\n\n"
            "RGB images, depth and camera calibration are supplied; get metric positions with "
            "[object-localization](.agents/skills/object-localization/SKILL.md).\n\n"
            f"Tools: {tools}. The live tool schemas give arguments and limits. `status` reports budgets and "
            "episode state; `microduck_status` reports the live robot and episode. There is no VLM tool.\n")


def _render_skill(source: Path, target: Path):
    target.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        if path.is_dir():
            continue
        text = path.read_text()
        for role, replacement in ROLES.items():
            text = text.replace(role, replacement)
        out = target / path.relative_to(source)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)


def build_workspace(workspace, *, published, episodes):
    workspace = Path(workspace)
    if workspace.exists():
        shutil.rmtree(workspace)
    skills = workspace / ".agents" / "skills"
    for name in SKILLS:
        _render_skill(ROOT / "agent_skills" / name, skills / name)
    shutil.copytree(ROOT / "workspace_template" / "skills" / "robot-interface", skills / "robot-interface")
    localization = skills / "object-localization" / "SKILL.md"
    text = localization.read_text()
    head, _, _ = text.partition("## This robot")
    localization.write_text(head + (ROOT / "workspace_template" / "skills" / "object-localization" / "this-robot.md").read_text())
    (workspace / "api").mkdir()
    shutil.copy(ROOT / "workspace_template" / "api" / "runtime.py", workspace / "api" / "runtime.py")
    for name in ("AGENTS.md", "START_PROMPT.md"):
        shutil.copy(ROOT / "workspace_template" / name, workspace / name)
    (workspace / "CLAUDE.md").write_text("@AGENTS.md\n")
    (workspace / "TASK.md").write_text(task_md())
    (workspace / "MCP_SESSION.md").write_text(session_md())
    (workspace / "MCP_CONTRACT.json").write_text(json.dumps(contract(), indent=2) + "\n")
    for folder in ("code", "memory", "experiments", "output"):
        (workspace / folder).mkdir()
    (workspace / ".claude").mkdir()
    (workspace / ".claude" / "skills").symlink_to("../.agents/skills", target_is_directory=True)
    (workspace / "runtime").mkdir()
    Path(published).mkdir(parents=True, exist_ok=True)
    (workspace / "runtime" / "autonomous_controller").symlink_to(Path(published).resolve(), target_is_directory=True)
    (workspace / "SESSION.json").write_text(json.dumps({"exploration_episodes": episodes}) + "\n")
    return workspace


if __name__ == "__main__":
    import sys
    build_workspace(sys.argv[1], published=Path(sys.argv[1]).parent / "published", episodes=4)
    print(sys.argv[1])
