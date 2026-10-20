# PROGRESS (walk_to_cube, exploration) — FINAL

Budget: 4/4 exploration episodes used, 0 remaining. Controller: code/walk_to_cube/controller.py:main(ctx).

## Episode results (trusted `evaluate` after each)
| ep | episode_id | how run | start (base yaw, cube d / bearing) | task_complete | end_reason | steps |
|---|---|---|---|---|---|---|
| 1 | walk_to_cube_exploration_001_52ae7004 | manual (dev approach loop to 0.12 m, then hand bursts) | yaw -0.05, 0.93 m / 0.01 rad | TRUE | task_complete | 593 |
| 2 | walk_to_cube_exploration_002_abe93f59 | controller v1, unattended, fresh start | yaw 0.26, 0.56 m / -0.37 rad | TRUE | task_complete | 302 |
| 3 | walk_to_cube_exploration_003_a6703b09 | manual 85-step turn away first (err 2.69 rad), then controller v1.1 | yaw -1.38, 0.74 m / +1.16 rad (overview only) | TRUE | task_complete | 767 |
| 4 | walk_to_cube_exploration_004_de0aa739 | controller v1.2, unattended, fresh start | yaw -2.22, 0.69 m / +0.18 rad | TRUE | task_complete | 344 |
Note: start yaw is NOT always ~0 (ep3: -1.38, ep4: -2.22); the controller uses the measured pose and does not assume it.

## What the controller does (code/walk_to_cube)
1. perception.py: blue mask -> depth back-projection (verified == microduck_pixel_to_position) -> cube centre
   (top-face mean when the head cam sees mostly the top face, else near-face mean + 2.5 cm away from camera);
   fusion: unclipped head view > locked xy from a close unclipped head view (odometry is exact) > overview > clipped head.
2. motion.approach: turn in place (yaw 1.0, ~0.7 rad/s) if |bearing| > 0.45; else vx=0.3 calls (50 steps far, 25 below 0.45 m)
   with yaw_rate = err / call_seconds (clip 0.5); stop at d <= 0.30 m and hold 50 (stop travel 4-8 cm).
3. motion.inch: vx=0.3 bursts (30/20/15 steps by distance) + hold 50 (~3-7 cm per cycle), turn in place first if |err| > 0.8;
   the episode ends itself (task_complete) during a hold once base-to-cube < ~0.10 m. Success seen at d = 0.064, 0.084, 0.089, 0.089.
4. scan_for_cube (head sweeps +-0.9 rad, then 60-step body turns): only exercised offline with a mock (experiments/mock_test.py).

## Lessons
- Gait start from standstill is slow: vx=0.2 does nothing in 25 steps; vx=0.3 x20 -> ~2 cm, the following hold adds ~2 cm.
- Holds after a burst swing yaw by up to 0.5 rad (standing transition); re-perceive after every hold, never dead-reckon heading.
- Head cam blob is clipped at the image bottom below ~0.2 m even at pitch 1.1: lock the cube xy from 0.25-0.5 m and use odometry.
- Overview camera is fixed in the world (-0.55, 0.45, 0.42); it saw the cube in every layout, incl. 66 deg bearing at 0.74 m.
- Not success at d(base->cube centre) 0.108-0.117; success at <= 0.089. Cube never pushed in any episode (toes stay clear to d ~0.064).

## Evidence
- Consoles: output/ep2_console.txt, ep3_console.txt, ep4_console.txt; run logs output/run_*.json.
- ep1 frames/analysis: experiments/img/ep1_*, inch_*; harness frames under runtime/autonomous_controller/runs/.
## Unresolved
- Scan branch (no camera sees the cube) untested on the robot. Head-camera horizontal half-FOV is 35 deg; layouts up to 60 deg rely on the overview or the scan.
- Far head estimate vs close estimate differ by ~1 cm (ep2: 0.617 vs 0.605); within tolerance but unexplained.
- Fall risk never observed (no falls in 4 episodes) but not stress-tested.
