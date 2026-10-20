"""The task, its layouts and its success check (pure Python, no simulator import)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Layout:
    collection: int
    layout_id: int
    seed: int
    robot_xy: tuple
    robot_yaw: float
    cube_xy: tuple
    cube_yaw: float


@dataclass(frozen=True)
class Task:
    name: str = "walk_to_cube"
    instruction: str = ("Walk to the blue cube and stop next to it: finish standing still with the midpoint "
                        "of your feet within 10 cm of the cube, without knocking the cube away.")
    step_limit: int = 1500              # control steps at 50 Hz: 30 s
    settle_steps: int = 50              # standing-policy steps before the agent acts (not counted)
    success_distance_m: float = 0.10    # feet midpoint to cube centre, horizontal
    cube_knock_m: float = 0.04          # the cube moved this far from its reset pose: failure
    still_steps: int = 25               # 0.5 s of near-zero base speed counts as stopped
    still_speed_mps: float = 0.06
    min_trunk_z_m: float = 0.065        # below this the trunk is on the floor
    min_upright_gz: float = 0.5         # projected gravity z must stay below -0.5 (tilt < 60 deg)
    arena_radius_m: float = 3.0
    cube_distance_m: tuple = (0.6, 1.3)
    cube_bearing_deg: float = 60.0      # cube within +-60 deg of the robot's start heading


TASK = Task()
EXPLORATION_COLLECTION, FORMAL_COLLECTION = 0, 1
FORMAL_LAYOUTS = 5
EXPLORATION_SEED_BASE = 1000
FORMAL_SEED_BASE = 20261019   # the held-out set: five fixed seeds, never used in exploration


def layout(collection, layout_id):
    if collection == FORMAL_COLLECTION:
        if not 0 <= layout_id < FORMAL_LAYOUTS:
            raise ValueError(f"Formal layouts are 0..{FORMAL_LAYOUTS - 1}")
        seed = FORMAL_SEED_BASE + layout_id
    elif collection == EXPLORATION_COLLECTION:
        if layout_id < 0:
            raise ValueError("Exploration layout ids are non-negative")
        seed = EXPLORATION_SEED_BASE + layout_id
    else:
        raise ValueError("collection must be 0 (exploration) or 1 (formal)")
    rng = np.random.default_rng(seed)
    yaw = float(rng.uniform(-math.pi, math.pi))
    distance = float(rng.uniform(*TASK.cube_distance_m))
    bearing = yaw + math.radians(float(rng.uniform(-TASK.cube_bearing_deg, TASK.cube_bearing_deg)))
    cube = (distance * math.cos(bearing), distance * math.sin(bearing))
    return Layout(collection, layout_id, seed, (0.0, 0.0), yaw, cube, float(rng.uniform(-math.pi, math.pi)))


def exploration_layouts(start, episodes):
    return [layout(EXPLORATION_COLLECTION, start + i) for i in range(episodes)]


def formal_layouts():
    return [layout(FORMAL_COLLECTION, i) for i in range(FORMAL_LAYOUTS)]
