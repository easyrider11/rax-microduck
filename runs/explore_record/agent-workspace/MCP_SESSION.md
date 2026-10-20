# Session tools

Role: exploration. Observation profile: rgbd.

RGB images, depth and camera calibration are supplied; get metric positions with [object-localization](.agents/skills/object-localization/SKILL.md).

Tools: `evaluate`, `finish`, `microduck_hold`, `microduck_observe`, `microduck_pixel_to_position`, `microduck_status`, `microduck_step`, `start_episode`, `status`. The live tool schemas give arguments and limits. `status` reports budgets and episode state; `microduck_status` reports the live robot and episode. There is no VLM tool.
