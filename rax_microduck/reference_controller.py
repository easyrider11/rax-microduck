"""The scripted reference controller: walk to the blue cube using only the agent's tools.

``main(ctx)`` is in RAX's controller shape (autonomous-control skill): it observes, finds the
cube by colour in the head camera, back-projects its pixels with microduck_pixel_to_position,
walks toward the point with the odometry pose, and holds still once close. It is the baseline
that ``make demo`` runs and that validates the success check; the agent writes its own.
"""
from __future__ import annotations

import base64
import io
import json
import math

import numpy as np

STOP_DISTANCE_M = 0.11      # trunk-to-cube: holding from a 0.2 m/s walk travels a few more centimetres
SLOW_DISTANCE_M = 0.35
MAX_ITERATIONS = 80
SEARCH_YAW_RATE = 1.0       # smaller yaw rates do not start a gait from standstill (characterize.py)


def _text(reply):
    return json.loads(next(b["text"] for b in reply["content"] if b["type"] == "text"))


def _image(reply, index):
    from PIL import Image
    block = reply["content"][index]
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(block["data"]))))


def find_cube(rgb):
    """Centroid (u, v) and pixel count of blue pixels, or None."""
    r, g, b = rgb[..., 0].astype(int), rgb[..., 1].astype(int), rgb[..., 2].astype(int)
    mask = (b > 120) & (b > r + 60) & (b > g + 40)
    count = int(mask.sum())
    if count < 6:
        return None
    vs, us = np.nonzero(mask)
    return float(us.mean()), float(vs.mean()), count, mask


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def locate_cube(ctx, obs, reply):
    """World-frame cube point from the head camera, or None when it is not in view."""
    attachments = {(a["camera"], a["kind"]): a["content_index"] for a in obs["attachments"]}
    rgb = _image(reply, attachments[("head_camera", "rgb")])
    found = find_cube(rgb)
    if found is None:
        return None, {"visible": False}
    u, v, count, mask = found
    vs, us = np.nonzero(mask)
    # A few pixels inside the blob (not its edges), back-projected and averaged.
    picks = [(float(us[i]), float(vs[i])) for i in np.linspace(0, len(us) - 1, min(5, len(us))).astype(int)]
    geometry = _text(ctx.call("microduck_pixel_to_position", observation_step_id=obs["step_id"],
                              selections=[{"camera": "head_camera", "u": pu, "v": pv} for pu, pv in picks],
                              depth_window_radius=0))
    points = [s["position_m"] for s in geometry["selections"] if s.get("valid")]
    if not points:
        return None, {"visible": True, "pixels": count, "depth": "invalid"}
    point = np.median(np.asarray(points), axis=0)
    return point, {"visible": True, "pixels": count, "pixel": [u, v], "point": point.tolist()}


def head_for(distance):
    """Tilt the head down as the cube gets close: at the home pose it leaves the camera's view below 0.5 m."""
    if distance is None or distance > 0.7:
        return [0.0, 0.0, 0.0, 0.0]
    return [0.0, 0.6 if distance > 0.35 else 0.9, 0.0, 0.0]


def main(ctx):
    log = []
    last_point = None
    searched = 0
    for iteration in range(MAX_ITERATIONS):
        reply = ctx.call("microduck_observe")
        obs = _text(reply)
        if obs.get("episode_ended"):
            return {"ok": bool(obs.get("task_complete")), "measured": {"end_reason": obs.get("end_reason")},
                    "why": "episode ended", "log": log}
        base = obs["state"]["base_pose"]
        point, info = locate_cube(ctx, obs, reply)
        if point is not None:
            last_point = point
        if last_point is None:
            # Not seen yet: look down for a moment, then turn in place (small yaw rates do not start a gait).
            head = [0.0, 0.6 if searched % 2 else 0.0, 0.0, 0.0]
            step = _text(ctx.call("microduck_step", twist=[0.0, 0.0, SEARCH_YAW_RATE], head=head, steps=15))
            searched += 1
            log.append({"iteration": iteration, "action": "search", "head": head, **info})
            if step.get("episode_ended"):
                return {"ok": bool(step.get("task_complete")), "measured": step.get("transition"),
                        "why": "ended while searching", "log": log}
            continue
        target = last_point
        dx, dy = target[0] - base["xy_m"][0], target[1] - base["xy_m"][1]
        distance = math.hypot(dx, dy)
        heading_error = wrap(math.atan2(dy, dx) - base["yaw_rad"])
        head = head_for(distance)
        if distance <= STOP_DISTANCE_M:
            hold = _text(ctx.call("microduck_hold", head=head, steps=40))
            log.append({"iteration": iteration, "action": "hold", "distance": round(distance, 3), **info})
            if hold.get("episode_ended"):
                return {"ok": bool(hold.get("task_complete")), "measured": {"distance_m": distance, "end_reason": hold.get("end_reason")},
                        "why": "held still next to the cube", "log": log}
            continue
        if abs(heading_error) > 0.45:
            twist, steps, action = [0.0, 0.0, math.copysign(1.0, heading_error)], 15, "turn"
        elif distance > SLOW_DISTANCE_M:
            twist, steps, action = [0.3, 0.0, float(np.clip(1.2 * heading_error, -0.6, 0.6))], 25, "walk"
        else:
            twist, steps, action = [0.2, 0.0, float(np.clip(1.2 * heading_error, -0.6, 0.6))], 12, "approach"
        step = _text(ctx.call("microduck_step", twist=twist, head=head, steps=steps))
        log.append({"iteration": iteration, "action": action, "twist": twist, "steps": steps, "distance": round(distance, 3),
                    "heading_error": round(heading_error, 3), **info})
        if step.get("episode_ended"):
            return {"ok": bool(step.get("task_complete")), "measured": {"distance_m": distance, "end_reason": step.get("end_reason")},
                    "why": "episode ended during motion", "log": log}
    return {"ok": False, "measured": {}, "why": "iteration limit", "log": log}
