"""Perception for walk_to_cube: parse a robot reply, find the blue cube, estimate its centre in world xy.

Geometry: depth PNG (mm) + intrinsics + OpenGL camera pose of the same observation; verified
against microduck_pixel_to_position (<1 mm, ep1 step 0).
"""
import base64, io, json, math
import numpy as np
from PIL import Image

CUBE_SIZE = 0.05            # m (TASK.md)
TOP_Z = 0.04                # points above this are on the cube's top face (top at 0.05)
DEPTH_BAND = 0.08           # m: keep blob points within this of the nearest-quartile depth (drop edge pixels)
MIN_PTS = {"head_camera": 6, "overview": 4}
MIN_TOP_PTS = 300          # use the top-face mean only when the top face dominates (also > 50% of the blob)
BORDER = 2                  # px: a blob touching the image border is clipped


def parse_reply(reply):
    """-> (meta dict, {(kind, camera): np.ndarray}) using attachments[*].content_index."""
    blocks = reply["content"]
    meta = json.loads(blocks[0]["text"])
    imgs = {}
    for att in meta.get("attachments", []):
        b = blocks[att["content_index"]]
        if b.get("type") != "image":
            continue
        im = Image.open(io.BytesIO(base64.b64decode(b["data"])))
        if att["kind"] == "rgb":
            imgs[("rgb", att["camera"])] = np.array(im.convert("RGB"))
        else:
            imgs[("depth", att["camera"])] = np.array(im).astype(np.uint16)
    return meta, imgs


def blue_mask(rgb):
    r, g, b = [rgb[..., i].astype(int) for i in range(3)]
    return (b > 100) & (b > r + 60) & (b > g + 40)


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def backproject(u, v, depth_m, K, pose):
    p = depth_m * np.linalg.solve(np.array(K, dtype=float), [u, v, 1.0])  # pinhole: x right, y down, z fwd
    p_gl = np.array([p[0], -p[1], -p[2]])                                 # OpenGL camera axes
    return quat_to_R(pose["quaternion_wxyz"]) @ p_gl + np.array(pose["position_m"])


def backproject_many(us, vs, depths_m, K, pose):
    Kinv = np.linalg.inv(np.array(K, dtype=float))
    pix = np.stack([us, vs, np.ones_like(us, dtype=float)], axis=0)
    p = (Kinv @ pix) * depths_m[None, :]
    p_gl = np.stack([p[0], -p[1], -p[2]], axis=0)
    return (quat_to_R(pose["quaternion_wxyz"]) @ p_gl).T + np.array(pose["position_m"])


def locate_cube(meta, imgs, camera):
    """Cube centre estimate from one camera, or None.
    Returns {xy, method, pixels, top_pts, clipped, uv}."""
    rgb, depth = imgs.get(("rgb", camera)), imgs.get(("depth", camera))
    if rgb is None or depth is None:
        return None
    m = blue_mask(rgb)
    if int(m.sum()) < MIN_PTS[camera]:
        return None
    vs, us = np.nonzero(m)
    d = depth[vs, us].astype(float) / 1000.0
    ok = d > 0
    if ok.sum() < 3:
        return None
    near = np.percentile(d[ok], 25)
    keep = ok & (np.abs(d - near) < DEPTH_BAND)
    us_k, vs_k, d_k = us[keep].astype(float), vs[keep].astype(float), d[keep]
    K = meta["camera_intrinsics"][camera]["matrix"]
    pose = meta["camera_poses"][camera]
    pts = backproject_many(us_k, vs_k, d_k, K, pose)
    h, w = m.shape
    clipped = bool(vs.max() >= h - 1 - BORDER or us.min() <= BORDER or us.max() >= w - 1 - BORDER or vs.min() <= BORDER)
    top = pts[pts[:, 2] > TOP_Z]
    cam = np.array(pose["position_m"])
    if camera == "head_camera" and len(top) >= MIN_TOP_PTS and len(top) > 0.5 * len(pts):
        centre = top[:, :2].mean(axis=0)
        method = "top_face"
    else:
        side = pts[pts[:, 2] <= TOP_Z] if (pts[:, 2] <= TOP_Z).sum() >= 3 else pts
        surf = side[:, :2].mean(axis=0)
        horiz = surf - cam[:2]
        n = np.linalg.norm(horiz)
        centre = surf + horiz / n * (CUBE_SIZE / 2) if n > 1e-6 else surf
        method = "side_push"
    return {"xy": [float(centre[0]), float(centre[1])], "method": method, "pixels": int(m.sum()),
            "top_pts": int(len(top)), "clipped": clipped, "uv": [float(us.mean()), float(vs.mean())],
            "depth_m": float(near)}


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def perceive(meta, imgs, lock=None, prev=None):
    """Fixed-output scene description.
    lock: previously locked cube xy (from an unclipped close head view) or None.
    Fusion: unclipped head view with >=30 px -> head (and may update lock); else lock; else overview; else clipped head."""
    if "state" in meta:
        bp = meta["state"]["base_pose"]
        robot = {"xy": [float(bp["xy_m"][0]), float(bp["xy_m"][1])], "yaw": float(bp["yaw_rad"])}
    elif prev is not None:
        robot = dict(prev["robot"])          # an episode-ended reply may carry no state
    else:
        robot = {"xy": [0.0, 0.0], "yaw": 0.0}
    head = locate_cube(meta, imgs, "head_camera")
    over = locate_cube(meta, imgs, "overview")
    cube, source = None, None
    if head is not None and not head["clipped"] and head["pixels"] >= 30:
        cube, source = head["xy"], "head_camera"
    elif lock is not None:
        cube, source = list(lock), "lock"
    elif over is not None:
        cube, source = over["xy"], "overview"
    elif head is not None:
        cube, source = head["xy"], "head_clipped"
    out = {"robot": robot, "cube_xy": cube, "source": source, "head": head, "overview": over,
           "step_id": meta.get("step_id"), "episode_ended": bool(meta.get("episode_ended")),
           "task_complete": bool(meta.get("task_complete")), "end_reason": meta.get("end_reason"),
           "remaining": meta.get("episode_limits", {}).get("remaining_episode_steps")}
    if cube is not None:
        dx, dy = cube[0] - robot["xy"][0], cube[1] - robot["xy"][1]
        out["dist"] = math.hypot(dx, dy)
        out["bearing_err"] = wrap(math.atan2(dy, dx) - robot["yaw"])
    return out
