import json
import math

import numpy as np
import pytest

from rax_microduck import tools as T
from rax_microduck.sim import MicroduckSim
from rax_microduck.tasks import FORMAL_LAYOUTS, TASK, exploration_layouts, formal_layouts, layout


def test_layouts_are_deterministic_and_held_out():
    assert layout(0, 3) == layout(0, 3)
    formal = formal_layouts()
    assert len(formal) == FORMAL_LAYOUTS
    exploration = exploration_layouts(0, 50)
    assert {f.seed for f in formal}.isdisjoint({e.seed for e in exploration})
    for lay in formal + exploration:
        d = math.hypot(*lay.cube_xy)
        assert TASK.cube_distance_m[0] <= d <= TASK.cube_distance_m[1]


@pytest.fixture(scope="module")
def sim():
    return MicroduckSim(render=False)


def test_stand_is_still_and_upright(sim):
    sim.reset(layout(0, 0))
    sim.command([0, 0, 0], steps=60)
    truth = sim.ground_truth()
    assert truth["still"] and truth["trunk_z_m"] > 0.1 and sim.end_reason is None


def test_walk_moves(sim):
    sim.reset(layout(0, 1))  # the cube is ahead, so walk backward, away from it
    p0 = np.array(sim.base_pose()["xy_m"]); yaw = sim.base_pose()["yaw_rad"]
    sim.command([-0.3, 0, 0], steps=250)
    dp = np.array(sim.base_pose()["xy_m"]) - p0
    assert dp @ np.array([math.cos(yaw), math.sin(yaw)]) < -0.4
    assert sim.end_reason is None


def test_oracle_walk_to_cube_succeeds(sim):
    """A scripted oracle that knows the cube pose reaches the success condition on every held-out layout."""
    outcomes = []
    for lay in formal_layouts():
        sim.reset(lay)
        for _ in range(120):
            if sim.terminated:
                break
            truth = sim.ground_truth(); base = truth["base"]
            dx, dy = truth["cube_xy_m"][0] - base["xy_m"][0], truth["cube_xy_m"][1] - base["xy_m"][1]
            d = math.hypot(dx, dy); err = (math.atan2(dy, dx) - base["yaw_rad"] + math.pi) % (2 * math.pi) - math.pi
            if d <= 0.08:
                sim.command([0, 0, 0], steps=30)
            elif abs(err) > 0.4:
                sim.command([0, 0, math.copysign(1.0, err)], steps=15)
            else:
                sim.command([0.3 if d > 0.35 else 0.15, 0, max(-0.6, min(0.6, 1.2 * err))], steps=15)
        outcomes.append((lay.layout_id, sim.end_reason, sim.step_id))
    assert all(reason == "task_complete" for _, reason, _ in outcomes), outcomes


def test_fall_is_detected(sim):
    sim.reset(layout(0, 2))
    sim.data.qpos[3:7] = [math.cos(0.7), 0, math.sin(0.7), 0]  # pitch the trunk 80 degrees forward
    sim.command([0, 0, 0], steps=100)
    assert sim.end_reason == "fell"


def test_tool_validation():
    T.validate_request(T.STEP, {"twist": [0.1, 0, 0], "steps": 10})
    with pytest.raises(T.RejectedBeforeAction):
        T.validate_request(T.STEP, {"twist": [0.1, 0], "steps": 10})
    with pytest.raises(T.RejectedBeforeAction):
        T.validate_request(T.HOLD, {"steps": 0})
    with pytest.raises(T.RejectedBeforeAction):
        T.validate_request(T.OBSERVE, {"x": 1})
    names = [t["name"] for t in T.tool_definitions()]
    assert names == list(T.ROBOT_TOOLS)
    json.dumps(T.tool_definitions())


def test_pixel_to_world_roundtrip():
    intrinsics = {"matrix": [[200.0, 0, 160.0], [0, 200.0, 120.0], [0, 0, 1]], "width": 320, "height": 240}
    pose = {"position_m": [0, 0, 1.0], "quaternion_wxyz": [1, 0, 0, 0]}  # looking down -z from 1 m up
    depth = np.full((240, 320), 1.0, dtype=np.float32)
    point = T.pixel_to_world(depth, intrinsics, pose, 160, 120, radius=0)
    assert point["valid"] and np.allclose(point["position_m"], [0, 0, 0], atol=1e-6)
