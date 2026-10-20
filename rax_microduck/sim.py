"""One Microduck episode in MuJoCo: physics, the low-level walking policy, cameras, the task check.

Everything the agent may know about an episode comes out of ``observation()``; the cube's
pose and the success check stay in ``ground_truth()`` for the operator side.
"""
from __future__ import annotations

import math
import os
import time

import mujoco
import numpy as np

from . import scene
from .actuator import BamXL330
from .policy import ACTION_LEN, HEAD_LIMITS, JOINTS, Policies, TWIST_LIMITS
from .tasks import Layout, TASK

MAX_STEPS_PER_CALL = 150


def _yaw_of(quat_wxyz):
    w, x, y, z = quat_wxyz
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _quat_z(yaw):
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


class MicroduckSim:
    def __init__(self, *, render=True, width=scene.IMAGE_WIDTH, height=scene.IMAGE_HEIGHT, actuator=None):
        self.actuator = actuator or BamXL330()
        self.model = scene.build_model(actuator=self.actuator)
        self.data = mujoco.MjData(self.model)
        self.actuator.bind(self.model, JOINTS)
        self.policies = Policies()
        self.width, self.height = width, height
        self.renderer = None
        self.render_enabled = render
        m = self.model
        self.joint_qpos = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in JOINTS])
        self.joint_qvel = np.array([m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in JOINTS])
        self.actuators = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in JOINTS])
        self.gyro = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
        self.gyro_adr = m.sensor_adr[self.gyro]
        self.trunk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        self.cube_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.cube_qpos = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cube_freejoint")]
        self.feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s) for s in ("left_foot", "right_foot")]
        self.stand_key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "STAND")
        self.layout = None
        self.step_id = 0
        self.twist = np.zeros(3)
        self.head = np.zeros(4)
        self.speed_history = []
        self.cube_start = np.zeros(2)
        self.terminated = False
        self.success = False
        self.end_reason = None

    # ------------------------------------------------------------------ episode
    def reset(self, layout: Layout):
        """Place the robot (standing, feet on the floor) and the cube, then let the standing policy
        settle for TASK.settle_steps. About one settle in forty stumbles on the initial contact; a
        settle that leaves the robot not upright is retried (the observation noise differs), so
        every episode starts standing."""
        m, d = self.model, self.data
        for attempt in range(5):
            mujoco.mj_resetDataKeyframe(m, d, self.stand_key)
            d.qpos[0:2] = layout.robot_xy
            d.qpos[2] = 0.12
            d.qpos[3:7] = _quat_z(layout.robot_yaw)
            d.qpos[self.cube_qpos:self.cube_qpos + 3] = [layout.cube_xy[0], layout.cube_xy[1], scene.CUBE_HALF]
            d.qpos[self.cube_qpos + 3:self.cube_qpos + 7] = _quat_z(layout.cube_yaw)
            d.qvel[:] = 0.0
            d.ctrl[:] = 0.0
            mujoco.mj_forward(m, d)
            d.qpos[2] -= float(d.site_xpos[self.feet][:, 2].min()) - 0.001  # feet on the floor, not dropped onto it
            mujoco.mj_forward(m, d)
            self.actuator.reset()
            self.policies.reset()
            self.layout = layout
            self.step_id = 0
            self.twist = np.zeros(3)
            self.head = np.zeros(4)
            self.speed_history = []
            self.terminated = self.success = False
            self.end_reason = None
            for _ in range(TASK.settle_steps):
                self._control_step(np.zeros(3), np.zeros(4), count=False)
            self.settle_attempts = attempt + 1
            if self._upright():
                break
        self.cube_start = d.xpos[self.cube_body, :2].copy()

    # ------------------------------------------------------------------ control
    def _sensors(self):
        d = self.data
        gyro = d.sensordata[self.gyro_adr:self.gyro_adr + 3]
        rot = d.xmat[self.trunk].reshape(3, 3)
        gravity = rot.T @ np.array([0.0, 0.0, -1.0])
        return gyro, gravity

    def _control_step(self, twist, head, *, count=True):
        m, d = self.model, self.data
        gyro, gravity = self._sensors()
        obs = self.policies.observation(gyro, gravity, d.qpos[self.joint_qpos], d.qvel[self.joint_qvel], twist, head)
        targets = self.policies.act(obs, twist)
        for _ in range(scene.SUBSTEPS):
            self.actuator.step(m, d, targets)
            mujoco.mj_step(m, d)
        if count:
            self.step_id += 1
            self.speed_history.append(float(np.linalg.norm(d.cvel[self.trunk, 3:5])))
            self.speed_history = self.speed_history[-TASK.still_steps:]
            self._check_end()

    def command(self, twist, head=None, steps=1):
        """Run `steps` control steps (20 ms each) under a velocity command and head pose."""
        if self.terminated:
            raise RuntimeError("Episode already ended")
        twist = np.clip(np.asarray(twist, dtype=float), -TWIST_LIMITS, TWIST_LIMITS)
        head = np.zeros(4) if head is None else np.clip(np.asarray(head, dtype=float), -HEAD_LIMITS, HEAD_LIMITS)
        self.twist, self.head = twist, head
        executed = 0
        for _ in range(int(steps)):
            self._control_step(twist, head)
            executed += 1
            if self.terminated:
                break
        return executed

    # --------------------------------------------------------------------- task
    def base_pose(self):
        d = self.data
        pos = d.xpos[self.trunk]
        quat = d.xquat[self.trunk]
        return {"xy_m": [float(pos[0]), float(pos[1])], "z_m": float(pos[2]), "yaw_rad": float(_yaw_of(quat))}

    def _feet_mid(self):
        return self.data.site_xpos[self.feet].mean(axis=0)

    def ground_truth(self):
        d = self.data
        _, gravity = self._sensors()
        cube = d.xpos[self.cube_body]
        feet = self._feet_mid()
        return {
            "cube_xy_m": [float(cube[0]), float(cube[1])],
            "cube_moved_m": float(np.linalg.norm(cube[:2] - self.cube_start)),
            "feet_mid_xy_m": [float(feet[0]), float(feet[1])],
            "distance_m": float(np.linalg.norm(cube[:2] - feet[:2])),
            "trunk_z_m": float(d.xpos[self.trunk, 2]),
            "gravity_z": float(gravity[2]),
            "speed_mps": self.speed_history[-1] if self.speed_history else 0.0,
            "still": self._still(),
            "base": self.base_pose(),
        }

    def _still(self):
        return (len(self.speed_history) >= TASK.still_steps
                and max(self.speed_history) <= TASK.still_speed_mps)

    def _upright(self):
        _, gravity = self._sensors()
        return self.data.xpos[self.trunk, 2] > TASK.min_trunk_z_m and gravity[2] < -TASK.min_upright_gz

    def _check_end(self):
        truth = self.ground_truth()
        if not self._upright():
            self.terminated, self.success, self.end_reason = True, False, "fell"
        elif truth["cube_moved_m"] > TASK.cube_knock_m:
            self.terminated, self.success, self.end_reason = True, False, "cube_knocked"
        elif np.linalg.norm(self.data.xpos[self.trunk, :2]) > TASK.arena_radius_m:
            self.terminated, self.success, self.end_reason = True, False, "left_arena"
        elif truth["distance_m"] <= TASK.success_distance_m and truth["still"]:
            self.terminated, self.success, self.end_reason = True, True, "task_complete"
        elif self.step_id >= TASK.step_limit:
            self.terminated, self.success, self.end_reason = True, False, "step_limit"

    # -------------------------------------------------------------- observation
    def _renderer(self):
        if self.renderer is None:
            os.environ.setdefault("MUJOCO_GL", "cgl" if os.uname().sysname == "Darwin" else "egl")
            self.renderer = mujoco.Renderer(self.model, self.height, self.width)
        return self.renderer

    def render(self, camera, depth=False):
        r = self._renderer()
        r.update_scene(self.data, camera=camera)
        if depth:
            r.enable_depth_rendering()
            try:
                return r.render().astype(np.float32)
            finally:
                r.disable_depth_rendering()
        return r.render().copy()

    def state(self):
        d = self.data
        gyro, gravity = self._sensors()
        return {"base_pose": self.base_pose(),
                "projected_gravity": [float(v) for v in gravity],
                "gyro_radps": [float(v) for v in gyro],
                "joint_positions_rad": {name: float(d.qpos[i]) for name, i in zip(JOINTS, self.joint_qpos)},
                "commanded_twist": [float(v) for v in self.twist],
                "commanded_head": [float(v) for v in self.head]}

    def observation(self, *, images=True, depth=True):
        obs = {"step_id": self.step_id, "state": self.state(), "instruction": TASK.instruction}
        if images and self.render_enabled:
            obs["camera_intrinsics"], obs["camera_poses"] = {}, {}
            for camera in scene.CAMERAS:
                obs[camera] = self.render(camera)
                if depth:
                    obs[f"{camera}_depth_m"] = self.render(camera, depth=True)
                obs["camera_intrinsics"][camera] = scene.camera_intrinsics(self.model, camera, self.width, self.height)
                obs["camera_poses"][camera] = scene.camera_pose(self.model, self.data, camera)
        return obs

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
