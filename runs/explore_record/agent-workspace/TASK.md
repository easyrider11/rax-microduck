# Task: walk_to_cube

**Instruction:** Walk to the blue cube and stop next to it: finish standing still with the midpoint of your feet within 10 cm of the cube, without knocking the cube away.

## Rules

- Success: the episode ends with the robot standing still (base speed under 0.06 m/s for 0.5 s) with the
  midpoint of its feet within 10 cm (horizontal) of the cube's centre.
- Failure, ending the episode at once: the robot falls; the cube is pushed more than 4 cm from where it
  started; the robot leaves a 3 m radius; or the step limit is reached.
- The cube is a 5 cm blue cube on the floor, 0.6-1.3 m away, within 60 degrees of the robot's start heading.

## Episode

- Robot: the Microduck biped under its pretrained walking policy (see the robot-interface skill).
- Cameras: `head_camera`, `overview`, both with depth.
- Step limit: 1500 steps of 20 ms of physics each (50 per second of robot time); the episode
  ends there, on success, or on a failure condition.
- Exploration episodes use their own layouts; the formal batch plays 5 held-out layouts that
  exploration never uses.

## Success

- An episode succeeds or fails. There is no partial credit: an episode that ends without the success
  condition, hits the step limit, triggers a failure condition or errors does not succeed.
