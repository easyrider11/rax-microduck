"""Measure every control mode (RAX's characterization protocol) and print JSON for the robot skill."""
from __future__ import annotations

import json
import math
import sys

import numpy as np

from .sim import MicroduckSim
from .tasks import layout


def _run(sim, twist, steps, head=None):
    pa = np.array(sim.base_pose()["xy_m"]); ya = sim.base_pose()["yaw_rad"]
    executed = sim.command(twist, head, steps=steps)
    pb = np.array(sim.base_pose()["xy_m"]); yb = sim.base_pose()["yaw_rad"]
    f = np.array([math.cos(ya), math.sin(ya)]); dp = pb - pa
    return {"forward_m": float(dp @ f), "left_m": float(f[0] * dp[1] - f[1] * dp[0]),
            "yaw_deg": float(math.degrees((yb - ya + math.pi) % (2 * math.pi) - math.pi)), "steps": executed,
            "ended": sim.end_reason}


def main():
    seeds = (0, 1, 2)
    sim = MicroduckSim(render=False)
    report = {}

    def average(name, twist, steps, warmup=25):
        rows = []
        for seed in seeds:
            sim.reset(layout(0, seed))
            sim.command(twist, steps=warmup)
            rows.append(_run(sim, twist, steps))
        report[name] = {k: round(float(np.mean([r[k] for r in rows])), 3) for k in ("forward_m", "left_m", "yaw_deg")}
        report[name]["seconds"] = steps / 50
        report[name]["falls"] = sum(r["ended"] == "fell" for r in rows)
        return report[name]

    for vx in (0.1, 0.2, 0.3, 0.4):
        average(f"forward_{vx}", [vx, 0, 0], 250)
    for vx in (-0.15, -0.3):
        average(f"backward_{abs(vx)}", [vx, 0, 0], 250)
    for vy in (0.15, 0.3):
        average(f"left_{vy}", [0, vy, 0], 250)
    for wz in (0.3, 0.6, 1.0):
        average(f"turn_{wz}", [0, 0, wz], 150)
    average("turn_while_walking_0.2_0.5", [0.2, 0, 0.5], 150)
    # Stopping: walk at 0.3, then hold; distance travelled and steps until 'still'.
    stops = []
    for seed in seeds:
        sim.reset(layout(0, seed)); sim.command([0.3, 0, 0], steps=150)
        pa = np.array(sim.base_pose()["xy_m"]); n = 0
        while not sim.ground_truth()["still"] and n < 150:
            sim.command([0, 0, 0], steps=1); n += 1
        stops.append((float(np.linalg.norm(np.array(sim.base_pose()["xy_m"]) - pa)), n))
    report["stop_from_0.3"] = {"travel_m": round(float(np.mean([s[0] for s in stops])), 3),
                               "steps_until_still": int(np.mean([s[1] for s in stops]))}
    # Drift when standing: hold 500 steps.
    drift = []
    for seed in seeds:
        sim.reset(layout(0, seed)); drift.append(_run(sim, [0, 0, 0], 500))
    report["stand_10s"] = {k: round(float(np.mean([r[k] for r in drift])), 3) for k in ("forward_m", "left_m", "yaw_deg")}
    # Head: command a yaw and pitch, measure the reached joint deltas.
    sim.reset(layout(0, 0)); sim.command([0, 0, 0], head=[0, 0.5, 0.8, 0], steps=50)
    q = sim.state()["joint_positions_rad"]
    report["head_tracking"] = {"commanded": {"head_pitch": 0.5, "head_yaw": 0.8},
                               "reached_delta": {"neck_pitch": round(q["neck_pitch"] - 0.3491, 3), "head_pitch": round(q["head_pitch"] - 0.3491, 3),
                                                 "head_yaw": round(q["head_yaw"], 3), "head_roll": round(q["head_roll"], 3)}, "seconds": 1.0}
    # Walking with the head turned.
    sim.reset(layout(0, 0)); sim.command([0.2, 0, 0], head=[0, 0, 0.8, 0], steps=25)
    report["walk_0.2_head_yaw_0.8"] = _run(sim, [0.2, 0, 0], 250, head=[0, 0, 0.8, 0])
    json.dump(report, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
