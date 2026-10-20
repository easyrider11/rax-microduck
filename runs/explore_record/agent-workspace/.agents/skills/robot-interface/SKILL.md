---
name: robot-interface
description: The simulated Pollen Microduck biped - its velocity-command and head-pose action space, its odometry frame and state, its two cameras with depth, and the measured response of every control mode. Read before composing any robot action or writing robot code.
---

# Microduck robot interface

The robot is Pollen's Microduck, a 25 cm, 800 g biped (5-DoF legs, a 4-DoF neck and head
with the head camera), simulated in MuJoCo. A pretrained walking policy drives the legs; you
do not command joints. Every command reaches the robot only through these tools; there is no
other access to the simulator.

## Frames and units

- **Robot frame:** at the trunk, +x forward, +y left, +z up. Units: metres, radians.
- **World (odometry) frame:** the robot starts each episode at its origin facing +x. The state
  reports `base_pose` in it: `xy_m`, `z_m`, `yaw_rad` (counter-clockwise from +x). In this
  backend the odometry is exact (simulator ground truth, no drift).
- **Floor:** z = 0. The trunk stands at z = 0.116 m; the head camera is 0.23 m up.
- `camera_poses` of each observation are in the world frame (OpenGL convention: a camera looks
  along its -z, +y up; image row 0 at the top); see object-localization.

## Tools by level

| Level | Tool | Use it for |
|---|---|---|
| Sense | `microduck_status` | action layout, limits, episode step budget, whether the episode ended |
| | `microduck_observe` | both cameras (RGB and depth), calibration, the measured state; cached after the last motion |
| | `microduck_pixel_to_position` | pixels of the current observation to world-frame points (metres) |
| Walk | `microduck_step` | hold a velocity command `[vx, vy, yaw_rate]` and a head pose for 1..150 steps |
| Stand | `microduck_hold` | zero velocity under the standing policy for 1..150 steps |

Session tools (`status`, `start_episode`, `evaluate`, `finish`) are described in AGENTS.md.
One control step is 20 ms; a 150-step call is 3 s of robot time and runs in well under a
second of wall time. Steps count against the episode limit (1500 steps, 30 s).

## The action

`twist = [vx m/s, vy m/s, yaw_rate rad/s]` in the robot frame, clipped to
`[-0.4..0.4, -0.3..0.3, -1.0..1.0]`; `head = [neck_pitch, head_pitch, head_yaw, head_roll]`
in radians as deltas from the home pose, clipped to `[±1.1, ±1.1, ±1.4, ±0.31]` (default 0).
A twist of magnitude 0.05 or less switches to the standing policy (`microduck_hold` is
exactly that). Commands are not deltas: the same twist repeated keeps the same gait.

## Measured response (this simulator, 2026-10-19, averaged over three layouts)

Each row: the command held for 5 s after a 0.5 s start, from standing, measured on the
odometry pose. The policy tracks commands loosely; plan from these numbers, not from the
command values.

| Command | Measured | Notes |
|---|---|---|
| `vx = 0.1` | no motion | too small to start a gait from standstill |
| `vx = 0.2` | 0.52 m forward, 9° yaw drift | about 0.10 m/s |
| `vx = 0.3` | 0.85 m forward, 6° drift | about 0.17 m/s; the most stable gait |
| `vx = 0.4` | 1.03 m forward, 0.17 m sideways, 19° drift | fastest, drifts most |
| `vx = -0.15` | no motion | |
| `vx = -0.3` | 0.92 m backward, 21° drift | |
| `vy = 0.15` | no motion | |
| `vy = 0.3` | 0.29 m left, 9° drift | strafing is slow |
| `yaw = 0.3` or `0.6` (3 s, from standing) | 1-2° | **turning in place needs `yaw = 1.0`** |
| `yaw = 1.0` (3 s) | 122° | about 0.7 rad/s |
| `vx = 0.2, yaw = 0.5` (3 s) | 93° while walking | small yaw rates work once walking |
| stop with `microduck_hold` from `vx = 0.3` | 8 cm travelled, still after 57 steps (1.1 s) | |
| stand 10 s | 0 cm, 0.2° | standing is stable |
| `head = [0, 0.5, 0.8, 0]` | head_pitch +0.57, head_yaw +0.86 after 1 s | the head tracks; walking with the head turned is fine |

- Heading drifts a few degrees per metre: re-localize the target after every call rather
  than dead-reckoning a long path.
- Short bursts near a target: at `vx = 0.2` a 12-step call (0.24 s) moves about 2-3 cm plus
  the stopping travel.
- The robot never fell in these measurements; falls do happen on collisions and on sharp
  reversals. A fall ends the episode (`end_reason: fell`).

## Cameras

- `head_camera`: 320x240 RGB and depth, on the head (moves with the head pose and with the
  gait), looking forward and about 11° down at the home head pose; vertical field of view 55°.
  The floor appears from about 0.3 m ahead; a 5 cm object is visible to about 1.5 m.
- `overview`: 320x240 RGB and depth, a third-person camera that follows the trunk from
  behind-left of the start heading (its orientation is fixed in the world, so after a turn
  the robot is seen from another side).
- Depth attachments are uint16 PNGs in millimetres (0 = invalid); `camera_intrinsics` and
  `camera_poses` come with every observation.
- Colours: the robot's own feet, ankles and lower head shell are orange and show in the `overview`
  camera; the floor is a grey checkerboard, the sky near-white.

Motion replies attach the final frame of each camera and save every 10th step under
`runtime/autonomous_controller/runs/<run>/frames/<sequence>/` (paths in `frame_sequence`).

## Limits the backend enforces

| Limit | Value | Effect |
|---|---|---|
| twist, head | the clip ranges above | clipped silently; `transition.commanded_twist` reports what ran |
| steps per call | 1..150 | refused before moving |
| episode steps | 1500 | the episode ends at the limit |
| arena | 3 m from the origin | the episode ends (`left_arena`) |
| fall | trunk below 6.5 cm or tilted beyond 60° | the episode ends (`fell`) |

A refusal is a wall: fix the cause named in the error.

## Reading errors

A failed tool returns `{type, operation, stage, reason, episode_ended}`:
`RejectedBeforeAction` at stage `validation` means nothing moved; `PermissionError` means the
episode has ended or the budget is spent; any other type is a simulator failure. After an
error, call `microduck_status` and observe before the next motion.

## From code

Python reaches the same tools through `api.runtime.Context`:
`ctx.call("microduck_step", twist=[0.3, 0, 0], steps=25)`. Calls are serialized with your
direct calls; check every reply and stop on the first unexpected one.
