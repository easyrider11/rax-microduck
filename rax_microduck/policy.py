"""The pretrained Microduck locomotion policies (ONNX) and their observation contract.

Layout and conventions are those of pollen-robotics/microduck ``duck-control/src/obs.rs`` and
``model.rs`` and of the ONNX metadata (``observation_names``, ``default_joint_pos``,
``action_scale``): obs[1, 61] -> actions[1, 14], joint targets = home + action.
"""
from __future__ import annotations

import numpy as np

from .paths import STAND_POLICY, WALK_POLICY

JOINTS = ("left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
          "neck_pitch", "head_pitch", "head_yaw", "head_roll",
          "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle")
HEAD_JOINTS = JOINTS[5:9]
HOME = np.array([0.0, -0.0873, -0.4579, -0.0049, 0.4530,
                 0.3491, 0.3491, 0.0, 0.0,
                 0.0, 0.0873, 0.4579, 0.0049, -0.4530])
OBS_LEN, ACTION_LEN = 61, 14
# Training command ranges (microduck_velocity_env_cfg.py); the backend clips to them.
TWIST_LIMITS = np.array([0.4, 0.3, 1.0])            # vx (m/s), vy (m/s), yaw rate (rad/s)
HEAD_LIMITS = np.array([1.10, 1.10, 1.40, 0.31])     # deltas from home: neck_pitch, head_pitch, head_yaw, head_roll
# robotd switches to the standing network at or below this twist magnitude (policy.rs).
STANDING_THRESHOLD = 0.05


class Policies:
    def __init__(self, walk_path=WALK_POLICY, stand_path=STAND_POLICY, *, noise=True, seed=0):
        import onnxruntime as ort
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.prev_joint_vel = np.zeros(ACTION_LEN)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self.walk = ort.InferenceSession(str(walk_path), options, providers=["CPUExecutionProvider"])
        self.stand = ort.InferenceSession(str(stand_path), options, providers=["CPUExecutionProvider"]) \
            if stand_path and stand_path.is_file() else None
        self.last_action = np.zeros(ACTION_LEN, dtype=np.float32)

    def reset(self):
        self.last_action[:] = 0.0
        self.prev_joint_vel[:] = 0.0

    @staticmethod
    def will_stand(twist):
        return float(np.linalg.norm(twist)) <= STANDING_THRESHOLD

    def observation(self, gyro, gravity, joint_pos, joint_vel, twist, head):
        obs = np.zeros(OBS_LEN, dtype=np.float32)
        # Training observed joint velocity one control step late (Dynamixel moving average).
        lagged_vel, self.prev_joint_vel = self.prev_joint_vel, np.array(joint_vel, dtype=float)
        obs[0:3] = gyro
        obs[3:6] = gravity
        obs[6:20] = joint_pos - HOME
        obs[20:34] = lagged_vel
        obs[34:48] = self.last_action
        if self.noise:  # the training-time uniform observation noise (microduck_velocity_env_cfg.py)
            u = self.rng.uniform
            obs[0:3] += u(-0.03, 0.03, 3)
            obs[3:6] += u(-0.01, 0.01, 3)
            obs[6:20] += u(-0.001, 0.001, 14)
            obs[20:34] += u(-0.25, 0.25, 14)
        obs[48:51] = twist
        obs[51:55] = head
        # 55..61: body pose command [x, y, z, roll, pitch, yaw]; nominal stance is all zero.
        return obs

    def act(self, obs, twist):
        net = self.stand if (self.stand is not None and self.will_stand(twist)) else self.walk
        action = net.run(None, {"obs": obs[None, :]})[0][0].astype(np.float32)
        self.last_action = action
        return HOME + action  # action_scale 1.0, default-offset targets
