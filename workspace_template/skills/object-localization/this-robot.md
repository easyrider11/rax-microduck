## This robot

| Fact | Value |
|---|---|
| Cameras, resolution, what moves each | `head_camera` 320x240, on the head (moves with head pose and gait); `overview` 320x240, follows the trunk from behind-left, fixed orientation |
| Where a frame's images, depth, intrinsics and camera pose are saved | the reply of `microduck_observe` / motion tools: image blocks (`attachments[*].content_index`), `depth` (uint16 PNG, millimetres, 0 invalid), `camera_intrinsics`, `camera_poses`; every 10th motion step's RGB under `runtime/autonomous_controller/runs/<run>/frames/<sequence>/` |
| Depth units and invalid value; where depth is unreliable | metres along the optical axis (`microduck_pixel_to_position`), PNG in millimetres; the sky has no depth; edge pixels take the background's depth |
| Robot frame and units of returned points | world (odometry) frame, metres; `state.base_pose` gives the robot's pose in it |
| Cross-view tolerance | a few centimetres between `head_camera` and `overview` on the same cube pixel (simulator depth is exact up to PNG rounding) |
| Known plane heights | floor z = 0; the cube is 5 cm, its centre at z = 0.025 m |
| Robot-body pixels to ignore | the overview sees the robot itself in its centre-right; the head camera sees no body parts at the home head pose |
