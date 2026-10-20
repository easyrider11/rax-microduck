"""The XL330 servo as the walking policy was trained against it: BAM model M6, in numpy.

A straight port of ``bam.mjlab.BamActuator`` (Rhoban's BAM, Apache-2.0; parameters in
``assets/xl330_m6.json``) and of microduck_rl's actuator settings (``kp_fw=200``, command
delay 3-6 physics steps, 6.5-8.2 V): a firmware P-controller on position error gives a PWM
duty cycle, a DC-motor equation with back-EMF gives the torque, and a load- and
velocity-dependent friction budget is written into MuJoCo's ``dof_frictionloss`` and
``dof_damping`` each physics step so the constraint solver applies it.
"""
from __future__ import annotations

import json
from collections import deque

import mujoco
import numpy as np

from .paths import ASSETS

XL330_ENCODER_COUNTS_PER_REV = 4096
XL330_KP_DIVISOR = 256
XL330_PWM_LIMIT = 885
ERROR_GAIN = (XL330_ENCODER_COUNTS_PER_REV / (2 * np.pi)) / (XL330_KP_DIVISOR * XL330_PWM_LIMIT)
MAX_CURRENT = 1.75
KP_FW = 200.0         # microduck's firmware stiffness (microduck_constants.py)
VIN = 7.4             # the middle of the training range 6.5-8.2 V (a fresh 2S pack sags under load)
VIN_DROP_GAIN = 0.1   # V/Nm, the middle of the training range 0.0-0.2
VIN_MIN = 6.0
DELAY_STEPS = 4       # physics steps of command latency (training randomized 3-6)
# Sim2sim calibration (see README, "What the actuator model is"): with the full BAM friction budget
# applied by MuJoCo's CPU solver the walking policy settles into a static pose and never lifts a
# foot; at 0.2 x the budget it walks every layout without falling. Measured 2026-10-19.
FRICTION_SCALE = 0.2
STIFF_SOLREF_FRICTION = (-5.0e4, -2.0e2)
STIFF_SOLIMP_FRICTION = (0.99, 0.9999, 0.001, 0.5, 2.0)


class BamXL330:
    def __init__(self, params_path=ASSETS / "xl330_m6.json", *, kp_fw=KP_FW, delay_steps=DELAY_STEPS,
                 friction_scale=FRICTION_SCALE, vin=VIN, max_current=MAX_CURRENT, stiff_friction=True):
        self.p = json.loads(params_path.read_text())
        self.kp_fw, self.delay_steps, self.friction_scale = kp_fw, delay_steps, friction_scale
        self.vin, self.max_current = vin, max_current
        self.stiff_friction = stiff_friction
        self.kt, self.R = self.p["kt"], self.p["R"]
        self.force_limit = 8.2 * self.kt / self.R
        self.prev_torque = None
        self.delay = None
        self.dof_ids = None
        self.act_ids = None

    # ------------------------------------------------------------ model edits
    def edit_spec(self, spec, joint_names):
        """Turn the XML position actuators into torque motors with the motor's armature."""
        targets = set(joint_names)
        for act in spec.actuators:
            if act.target in targets:
                act.set_to_motor()
                act.forcelimited = True
                act.forcerange = (-self.force_limit, self.force_limit)
                act.gear = [1.0, 0, 0, 0, 0, 0]
        for joint in spec.joints:
            if joint.name in targets:
                joint.armature = float(self.p["armature"])
                joint.damping = np.zeros((3, 1))
                joint.frictionloss = 0.0
                if self.stiff_friction:
                    joint.solref_friction = STIFF_SOLREF_FRICTION
                    joint.solimp_friction = STIFF_SOLIMP_FRICTION

    def bind(self, model, joint_names):
        self.dof_ids = np.array([model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
                                 for j in joint_names])
        self.act_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in joint_names])
        self.reset()

    def reset(self):
        n = len(self.dof_ids)
        self.prev_torque = np.zeros(n)
        self.delay = deque([None] * self.delay_steps, maxlen=self.delay_steps + 1)

    # -------------------------------------------------------------- per step
    def _friction_force(self, data):
        """Force of our own dof-friction constraints on the previous solve, per DOF."""
        out = np.zeros(data.qfrc_constraint.shape[0])
        n = data.nefc
        if n:
            types = data.efc_type[:n]
            mask = types == mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF
            np.add.at(out, data.efc_id[:n][mask], data.efc_force[:n][mask])
        return out

    def step(self, model, data, targets):
        """Write this physics step's motor torques (ctrl) and friction budget for `targets` (rad)."""
        p = self.p
        self.delay.append(np.asarray(targets, dtype=float).copy())
        target = self.delay[0]
        if target is None:
            target = data.qpos[model.jnt_qposadr[model.dof_jntid[self.dof_ids]]]
        q = data.qpos[model.jnt_qposadr[model.dof_jntid[self.dof_ids]]]
        dq = data.qvel[self.dof_ids]
        # Battery sag with the previous step's load.
        vin = max(self.vin - VIN_DROP_GAIN * float(np.abs(self.prev_torque).sum()), VIN_MIN)
        # Firmware control law: position error -> duty cycle, current-limited, PWM-limited.
        duty = (target - q) * self.kp_fw * ERROR_GAIN
        span = self.R * self.max_current / vin
        center = self.kt * dq / vin
        duty = np.clip(duty, center - span, center + span)
        duty = np.clip(duty, -1.0, 1.0)
        volts = vin * duty
        torque = self.kt * volts / self.R - (self.kt ** 2) * dq / self.R
        # Friction budget from the previous solve's loads (BAM M6).
        external = (-data.qfrc_bias[self.dof_ids] + data.qfrc_constraint[self.dof_ids]
                    - self._friction_force(data)[self.dof_ids])
        motor_prev = data.qfrc_actuator[self.dof_ids]
        stribeck = np.exp(-np.power(np.abs(dq) / p["dtheta_stribeck"], p["alpha"]))
        frictionloss = np.full_like(dq, p["friction_base"]) + stribeck * p["friction_stribeck"]
        gearbox = np.abs(external * p["load_friction_external"] - motor_prev * p["load_friction_motor"])
        frictionloss += gearbox
        gearbox_s = np.abs(external * p["load_friction_external_stribeck"] - motor_prev * p["load_friction_motor_stribeck"])
        frictionloss += stribeck * gearbox_s
        abs_ext, abs_mot = np.abs(external), np.abs(motor_prev)
        drive = (abs_mot > abs_ext).astype(float)
        quad = drive * p["load_friction_external_quad"] * abs_ext ** 2 + (1 - drive) * p["load_friction_motor_quad"] * abs_mot ** 2
        frictionloss += stribeck * quad
        model.dof_frictionloss[self.dof_ids] = frictionloss * self.friction_scale
        model.dof_damping[self.dof_ids] = p["friction_viscous"] * self.friction_scale
        data.ctrl[self.act_ids] = torque
        self.prev_torque = torque
        return torque
