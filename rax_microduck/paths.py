"""Where the robot model and policies live. Nothing heavy is imported here."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
POLICIES = ASSETS / "policies"
# The MuJoCo model (robot_walk.xml and its STL meshes) comes from pollen-robotics/microduck_rl,
# cloned by `make setup`; MICRODUCK_RL overrides the checkout.
MICRODUCK_RL = Path(os.environ.get("MICRODUCK_RL") or ROOT / "third_party" / "microduck_rl")
ROBOT_DIR = MICRODUCK_RL / "src" / "mjlab_microduck" / "robot" / "microduck"
ROBOT_XML = ROBOT_DIR / "robot_walk.xml"
WALK_POLICY = POLICIES / "alpha_walking.onnx"
STAND_POLICY = POLICIES / "alpha_stand.onnx"
RUNS = Path(os.environ.get("RAX_MICRODUCK_RUNS") or ROOT / "runs")


def require_model():
    if not ROBOT_XML.is_file():
        raise FileNotFoundError(
            f"{ROBOT_XML} is missing: run `make setup` (clones pollen-robotics/microduck_rl into third_party/) "
            "or set MICRODUCK_RL to a checkout")
    return ROBOT_XML
