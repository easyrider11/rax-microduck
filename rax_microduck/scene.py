"""Build the task scene: the Microduck walk model plus a floor, a coloured cube and cameras.

The robot model is pollen-robotics/microduck_rl's ``robot_walk.xml`` (the one the walking
policy was trained on), loaded through ``mujoco.MjSpec`` and left unchanged; everything else
is added programmatically so no modified copy of the robot file is needed.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from .actuator import BamXL330
from .paths import require_model
from .policy import JOINTS

# Control runs at 50 Hz (the policy's rate). The XML's own timestep is 2 ms; 10 substeps = 20 ms.
CONTROL_HZ = 50
SUBSTEPS = 4
PHYSICS_TIMESTEP = 0.005   # as in training (mjlab: 5 ms x 4)

CUBE_HALF = 0.025          # 5 cm cube
CUBE_RGBA = (0.08, 0.25, 0.95, 1.0)   # blue: the robot's shell is orange, the floor grey
CUBE_MASS = 0.05

# Fixed-size RGB/depth images; small keeps replies cheap for the agent.
IMAGE_WIDTH, IMAGE_HEIGHT = 320, 240
CAMERAS = ("head_camera", "overview")


def _look_quat(forward, up=(0.0, 0.0, 1.0)):
    """MuJoCo camera quaternion (w, x, y, z) looking along `forward`.

    MuJoCo cameras look down their -Z axis with +Y up.
    """
    f = np.asarray(forward, dtype=float)
    f /= np.linalg.norm(f)
    u = np.asarray(up, dtype=float)
    right = np.cross(f, u)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, f)
    rot = np.stack([right, true_up, -f], axis=1)  # columns: x, y, z axes of the camera frame
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.reshape(9))
    return quat


def build_model(cube_xy=(1.0, 0.0), actuator: BamXL330 | None = None) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(require_model()))
    spec.option.timestep = PHYSICS_TIMESTEP
    spec.option.iterations, spec.option.ls_iterations = 10, 20
    if actuator is not None:
        actuator.edit_spec(spec, JOINTS)
    # Checkerboard floor so the agent can see motion and distance in images.
    tex = spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                           builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                           rgb1=[0.82, 0.82, 0.82], rgb2=[0.55, 0.6, 0.65],
                           mark=mujoco.mjtMark.mjMARK_EDGE, markrgb=[0.3, 0.3, 0.3],
                           width=300, height=300)
    mat = spec.add_material(name="grid", texrepeat=[20, 20], texuniform=True, reflectance=0.05)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    sky = spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                           builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                           rgb1=[0.93, 0.93, 0.9], rgb2=[1.0, 1.0, 1.0], width=256, height=1536)
    del tex, sky
    spec.visual.headlight.ambient = [0.45, 0.45, 0.45]
    spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    world = spec.worldbody
    world.add_light(pos=[0, 0, 3.0], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.5, 0.5, 0.5])
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
                   material="grid", friction=[1.0, 0.005, 0.0001])
    cube = world.add_body(name="cube", pos=[cube_xy[0], cube_xy[1], CUBE_HALF])
    cube.add_freejoint(name="cube_freejoint")
    cube.add_geom(name="cube", type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUBE_HALF] * 3,
                  rgba=list(CUBE_RGBA), mass=CUBE_MASS, friction=[0.8, 0.005, 0.0001])
    # A third-person camera that follows the trunk (translation only, fixed orientation):
    # behind-left of the robot's start heading, looking down at it.
    offset = np.array([-0.55, 0.45, 0.42])
    world.add_camera(name="overview", mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACK,
                     targetbody="trunk_base", pos=offset.tolist(),
                     quat=_look_quat(-offset + np.array([0.15, 0.0, 0.0])).tolist(), fovy=60)
    _fix_head_camera(spec)
    model = spec.compile()
    return model


HEAD_CAMERA_FOVY = 55.0


def _fix_head_camera(spec):
    """Point the model's head camera along the lens.

    robot_walk.xml carries a `head_camera` whose quaternion looks into the head shell (depth
    reads 1.2 cm everywhere). The `head_camera` site next to it is the lens frame: at the home
    pose its +x axis points forward and 11 degrees down, its +z up. A MuJoCo camera looks along
    its -z with +y up, so the camera frame is x = -y_site, y = z_site, z = -x_site.
    """
    site = next(s for s in spec.sites if s.name == "head_camera")
    camera = next(c for c in spec.cameras if c.name == "head_camera")
    site_rot = np.zeros(9)
    mujoco.mju_quat2Mat(site_rot, np.asarray(site.quat, dtype=np.float64))
    site_rot = site_rot.reshape(3, 3)
    cam_in_site = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # columns: x_cam, y_cam, z_cam
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, (site_rot @ cam_in_site).reshape(9))
    camera.pos = np.asarray(site.pos, dtype=np.float64)
    camera.quat = quat
    camera.fovy = HEAD_CAMERA_FOVY


def camera_intrinsics(model, camera, width=IMAGE_WIDTH, height=IMAGE_HEIGHT):
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    fovy = float(model.cam_fovy[cid])
    focal = (height / 2.0) / math.tan(math.radians(fovy) / 2.0)
    return {"matrix": [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
            "width": width, "height": height, "fovy_deg": fovy}


def camera_pose(model, data, camera):
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.asarray(data.cam_xmat[cid], dtype=np.float64))
    return {"position_m": [float(v) for v in data.cam_xpos[cid]], "quaternion_wxyz": [float(v) for v in quat],
            "frame": "world", "convention": "OpenGL camera: looks along -Z, +Y up; image row 0 at the top"}
