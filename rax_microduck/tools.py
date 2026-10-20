"""Robot tool schemas and host-side validation for the Microduck backend (pure Python).

The tool set follows RAX's backend contract (docs/REAL_ROBOT_ADAPTATION.md, skills/README.md
"Tool roles"): a status tool, an observation tool, motion tools whose replies carry the
episode's step_id, and a pixel-to-position tool for the rgbd profile.
"""
from __future__ import annotations

import math

import numpy as np

STATUS, OBSERVE = "microduck_status", "microduck_observe"
STEP, HOLD = "microduck_step", "microduck_hold"
PIXEL_TO_POSITION = "microduck_pixel_to_position"
ROBOT_TOOLS = (STATUS, OBSERVE, STEP, HOLD, PIXEL_TO_POSITION)
MUTATING = frozenset({STEP, HOLD})
GEOMETRY_TOOLS = frozenset({PIXEL_TO_POSITION})
LIFECYCLE_TOOLS = ("status", "start_episode", "evaluate", "finish")
MAX_STEPS = 150          # control steps per motion call (3 s)
FRAME_STRIDE = 10        # every 10th control step is saved as a frame (plus the final one)
CAMERAS = ("head_camera", "overview")
CONTROL_TEXT = "50 Hz, 20 ms of physics per step"


class RejectedBeforeAction(ValueError):
    """A request refused before any physics ran."""


def reject_unless(condition, message):
    if not condition:
        raise RejectedBeforeAction(message)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _vector(n, description):
    return {"type": "array", "items": {"type": "number"}, "minItems": n, "maxItems": n, "description": description}


def action_layout():
    return {"twist": {"fields": ["vx_mps", "vy_mps", "yaw_rate_radps"], "limits": [[-0.4, 0.4], [-0.3, 0.3], [-1.0, 1.0]],
                      "frame": "trunk (robot) frame: +x forward, +y left, yaw counter-clockwise from above",
                      "semantics": "a velocity command to the pretrained walking policy; it is held for every "
                                   "step of the call. Below 0.05 in magnitude the standing policy runs instead."},
            "head": {"fields": ["neck_pitch", "head_pitch", "head_yaw", "head_roll"], "unit": "rad, delta from the home pose",
                     "limits": [[-1.1, 1.1], [-1.1, 1.1], [-1.4, 1.4], [-0.31, 0.31]],
                     "semantics": "head pose targets tracked by the policy; positive head_yaw turns the camera left, "
                                  "positive pitch tilts it down. The head camera moves with the head."},
            "step": "one control step is 20 ms of physics; the policy runs once per step"}


def tool_definitions():
    return [
        {"name": STATUS, "description": (
            "Episode status, step budget, the action layout (velocity command and head pose), cameras and "
            "instruction. No physics."),
         "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
        {"name": OBSERVE, "description": (
            "Current observation: the head camera and third-person overview RGB (320x240) with depth attachments "
            "and calibration, the measured robot state (odometry base pose, projected gravity, gyro, joint "
            "positions, the current command) and the instruction. No physics; cached after the last motion."),
         "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
        {"name": STEP, "description": (
            f"Walk: hold a velocity command (and optional head pose) for 1..{MAX_STEPS} control steps "
            f"({CONTROL_TEXT}) of the pretrained walking policy. twist = [vx m/s, vy m/s, yaw rate rad/s] in the "
            "robot frame, clipped to +-[0.4, 0.3, 1.0]; head = 4 joint deltas from home (rad). Each step counts "
            f"against the episode limit; stops early when the episode ends. Returns the final observation and "
            f"saves every {FRAME_STRIDE}th frame. Measured response is in the robot-interface skill."),
         "inputSchema": {"type": "object", "required": ["twist", "steps"], "additionalProperties": False, "properties": {
             "twist": _vector(3, "[vx, vy, yaw_rate] in the robot frame"),
             "head": _vector(4, "[neck_pitch, head_pitch, head_yaw, head_roll] deltas from home, rad (default 0)"),
             "steps": {"type": "integer", "minimum": 1, "maximum": MAX_STEPS}}}},
        {"name": HOLD, "description": (
            f"Stand still for 1..{MAX_STEPS} control steps: a zero velocity command under the standing policy, "
            "with an optional head pose. Steps count against the episode limit. Use it to stop, to settle before "
            "an observation, and to let the task's 'standing still' condition register."),
         "inputSchema": {"type": "object", "required": ["steps"], "additionalProperties": False, "properties": {
             "steps": {"type": "integer", "minimum": 1, "maximum": MAX_STEPS},
             "head": _vector(4, "[neck_pitch, head_pitch, head_yaw, head_roll] deltas from home, rad (default 0)")}}},
        {"name": PIXEL_TO_POSITION, "description": (
            "Back-project 1..32 pixels of the current observation's depth images into world-frame points "
            "(metres; the world frame is the odometry frame of base_pose). Pass the observation's step_id."),
         "inputSchema": {"type": "object", "required": ["observation_step_id", "selections"], "additionalProperties": False,
                         "properties": {
                             "observation_step_id": {"type": "integer"},
                             "selections": {"type": "array", "minItems": 1, "maxItems": 32, "items": {
                                 "type": "object", "required": ["camera", "u", "v"], "additionalProperties": False,
                                 "properties": {"camera": {"type": "string", "enum": list(CAMERAS)},
                                                "u": {"type": "number"}, "v": {"type": "number"}}}},
                             "depth_window_radius": {"type": "integer", "minimum": 0, "maximum": 5}}}},
    ]


def validate_request(name, arguments):
    reject_unless(isinstance(arguments, dict), "arguments must be an object")
    allowed = {STATUS: (), OBSERVE: (), STEP: ("twist", "head", "steps"), HOLD: ("steps", "head"),
               PIXEL_TO_POSITION: ("observation_step_id", "selections", "depth_window_radius")}
    reject_unless(name in allowed, f"Unknown Microduck tool: {name}")
    unknown = set(arguments) - set(allowed[name])
    reject_unless(not unknown, f"Unknown arguments: {sorted(unknown)}")
    if name in (STEP, HOLD):
        steps = arguments.get("steps")
        reject_unless(type(steps) is int and 1 <= steps <= MAX_STEPS, f"steps must be an integer 1..{MAX_STEPS}")
        if name == STEP:
            twist = arguments.get("twist")
            reject_unless(isinstance(twist, list) and len(twist) == 3 and all(_finite(v) for v in twist),
                          "twist must be [vx, vy, yaw_rate], three finite numbers")
        head = arguments.get("head")
        if head is not None:
            reject_unless(isinstance(head, list) and len(head) == 4 and all(_finite(v) for v in head),
                          "head must be four finite numbers")
    elif name == PIXEL_TO_POSITION:
        reject_unless(type(arguments.get("observation_step_id")) is int, "observation_step_id must be an integer")
        selections = arguments.get("selections")
        reject_unless(isinstance(selections, list) and 1 <= len(selections) <= 32, "selections must have 1..32 items")
        for item in selections:
            reject_unless(isinstance(item, dict) and set(item) == {"camera", "u", "v"}, "selection needs camera, u, v")
            reject_unless(item["camera"] in CAMERAS, f"unknown camera {item['camera']!r}")
            reject_unless(_finite(item["u"]) and _finite(item["v"]), "pixel coordinates must be finite numbers")
        radius = arguments.get("depth_window_radius", 1)
        reject_unless(type(radius) is int and 0 <= radius <= 5, "depth_window_radius must be 0..5")


# ---------------------------------------------------------------- geometry (as RAX services/dexjoco/tools.py)
def quaternion_matrix(q_wxyz):
    w, x, y, z = (float(v) for v in q_wxyz)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def pixel_to_world(depth_m, intrinsics, pose, u, v, radius=1):
    """Back-project one pixel of a linear-depth image (metres along the optical axis) into the world."""
    depth = np.asarray(depth_m, dtype=np.float64)
    height, width = depth.shape[:2]
    reject_unless(0 <= u < width and 0 <= v < height, f"pixel ({u}, {v}) outside {width}x{height}")
    iu, iv = int(round(u)), int(round(v))
    window = depth[max(0, iv - radius):iv + radius + 1, max(0, iu - radius):iu + radius + 1]
    valid = window[np.isfinite(window) & (window > 0)]
    if valid.size == 0:
        return {"valid": False, "reason": "no valid depth near the pixel"}
    z = float(np.median(valid))
    K = np.asarray(intrinsics["matrix"], dtype=np.float64)
    point_camera = np.array([(u - K[0, 2]) / K[0, 0] * z, -(v - K[1, 2]) / K[1, 1] * z, -z])
    point = quaternion_matrix(pose["quaternion_wxyz"]) @ point_camera + np.asarray(pose["position_m"], dtype=np.float64)
    return {"valid": True, "position_m": point.tolist(), "depth_m": z, "frame": "world",
            "depth_spread_m": float(valid.max() - valid.min())}
