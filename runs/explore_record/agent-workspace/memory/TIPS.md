| phase | preconditions | action | effect | evidence |
|---|---|---|---|---|
| localize | cube in head cam, blob unclipped | blue mask (b>100,b>r+60,b>g+40), 25th-pct depth band, back-project; top-face mean if top dominates, else near-face mean + 2.5 cm from camera | +-1 cm vs truth; == microduck_pixel_to_position | ep1 step 0, experiments/s04_analyze.py |
| localize close (<0.2 m) | head pitch 1.1, blob touches bottom edge | do NOT trust; use xy locked at 0.25-0.5 m + exact odometry | avoids +2 cm bias | ep1 inch_*, ep2 approach3 |
| big bearing error | cube only in overview, |err|>0.45 | microduck_step [0,0,+-1.0] steps=err/0.7*50 (<=90) | ~1.0 rad per 90 steps, little translation | ep3 approach0-2 |
| approach far (d>0.45) | cube localized, |err|<0.45 | microduck_step [0.3,0,yaw=err/1.0 s clipped 0.5] steps 50, head pitch 0.6 below 0.7 m | ~0.15-0.17 m/call, yaw tracks | ep2/ep4 approach |
| stop walking | d<=0.30 at vx 0.3 | microduck_hold 50 head [0,1.1,0,0] | 4-8 cm travel, yaw may swing up to 0.5 rad | ep2-4 approach_stop |
| inch | standing, 0.10<d<0.20 | microduck_step [0.3,0,yaw] 30/20/15 steps then hold 50 | 3-7 cm per cycle; 15-step burst ~1 cm + stop | ep1 step 527-593, ep4 inch |
| success | base within <=0.089 of cube centre, standing | hold 50 | episode ends task_complete by itself | ep1-4 |
