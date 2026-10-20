import json, sys, numpy as np
from PIL import Image
from rax import Context, call

def load(tag):
    meta = json.load(open(f"experiments/img/{tag}.json"))
    rgb_h = np.array(Image.open(f"experiments/img/{tag}_0.png").convert("RGB"))
    rgb_o = np.array(Image.open(f"experiments/img/{tag}_1.png").convert("RGB"))
    d_h = np.array(Image.open(f"experiments/img/{tag}_2.png"))
    d_o = np.array(Image.open(f"experiments/img/{tag}_3.png"))
    return meta, rgb_h, rgb_o, d_h, d_o

def blue_mask(rgb):
    r, g, b = [rgb[..., i].astype(int) for i in range(3)]
    return (b > 100) & (b > r + 60) & (b > g + 40)

def stats(name, rgb, d):
    m = blue_mask(rgb)
    n = int(m.sum())
    print(f"{name}: shape {rgb.shape}, depth dtype {d.dtype} shape {d.shape}, blue pixels {n}")
    if n:
        vs, us = np.nonzero(m)
        print(f"  u range {us.min()}-{us.max()}, v range {vs.min()}-{vs.max()}, centroid u={us.mean():.1f} v={vs.mean():.1f}")
        print(f"  depth at blue px: min {d[m].min()} max {d[m].max()} median {np.median(d[m])} mm")
        # colour sample
        print("  mean colour of blue px:", rgb[m].mean(axis=0).round(1))
    return m

def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def backproject(u, v, depth_m, K, pose):
    p = depth_m * np.linalg.solve(np.array(K), [u, v, 1.0])   # pinhole: x right, y down, z forward
    p_gl = np.array([p[0], -p[1], -p[2]])                     # OpenGL: y up, looks along -z
    R = quat_to_R(pose["quaternion_wxyz"]); t = np.array(pose["position_m"])
    return R @ p_gl + t

if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "ep1_start"
    meta, rgb_h, rgb_o, d_h, d_o = load(tag)
    print("unique colours sample head (corners):", rgb_h[0,0], rgb_h[239,160], rgb_h[120,160])
    mh = stats("head", rgb_h, d_h)
    mo = stats("overview", rgb_o, d_o)
    sel = []
    for cam, m, d in (("head_camera", mh, d_h), ("overview", mo, d_o)):
        if m.sum() == 0: continue
        vs, us = np.nonzero(m)
        u, v = int(round(us.mean())), int(round(vs.mean()))
        K = meta["camera_intrinsics"][cam]["matrix"]; pose = meta["camera_poses"][cam]
        pt = backproject(u, v, d[v, u] / 1000.0, K, pose)
        print(f"{cam}: centroid px ({u},{v}) depth {d[v,u]} mm -> local backproject world {pt.round(3)}")
        sel.append({"camera": cam, "u": u, "v": v})
    if "--tool" in sys.argv and sel:
        with Context() as ctx:
            r = call(ctx, "microduck_pixel_to_position", observation_step_id=meta["step_id"], selections=sel, depth_window_radius=1)
            print("tool:", json.dumps(r, indent=1, default=str)[:2500])
