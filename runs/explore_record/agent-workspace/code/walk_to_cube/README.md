# walk_to_cube controller

Entry: `controller.py:main(ctx)` (no new Context; uses the supplied one). Modules:
- `perception.py`: parse replies, blue-cube mask, depth back-projection, centre estimate (top-face mean
  when the head camera sees the top face; otherwise the near-face mean pushed half a cube away), fusion
  with a locked estimate from an unclipped close head view (close views are clipped at the image bottom).
- `motion.py`: `approach` (vx=0.3 calls with yaw_rate = bearing_err / call duration, turn in place when the
  error > 0.45 rad, stop at 0.30 m then hold), `inch` (20-30 step vx=0.3 bursts + 50-step holds, ~4 cm each,
  until the episode ends on success), `scan_for_cube` (head sweeps then body turns).

Run in development from the workspace root:
`RAX_MCP_SOCKET=... python experiments/run_controller.py` (needs a running episode).
Output log: `output/run_<t>.json`. Tested: ep1 by hand (components), ep2+ unattended (see memory/PROGRESS.md).
Tested layouts (all task_complete=true by `evaluate`):
- ep1 0.93 m / bearing 0.0 rad (by hand with the same approach loop); ep2 0.56 m / -0.37 rad (unattended, 302 steps);
- ep3 0.74 m / +1.16 rad, after a manual 1.2 rad turn away (controller from bearing error 2.69 rad, 767 steps);
- ep4 0.69 m / +0.18 rad (unattended, 344 steps). Start yaw varied from -2.2 to +0.26 rad.
Untested on the robot: the scan branch (cube in neither camera), cube at 1.3 m, falls. Mock test: experiments/mock_test.py.
