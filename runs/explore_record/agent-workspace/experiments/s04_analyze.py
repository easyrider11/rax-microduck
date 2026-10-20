import sys, os, json, numpy as np
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "code/walk_to_cube")
from PIL import Image
from api.runtime import Context
from perception import perceive, blue_mask, backproject, parse_reply
from motion import step, hold, observe

def cloud(meta, imgs, cam, mask_fn):
    rgb, dep = imgs[("rgb", cam)], imgs[("depth", cam)]
    m = mask_fn(rgb); vs, us = np.nonzero(m)
    K = meta["camera_intrinsics"][cam]["matrix"]; pose = meta["camera_poses"][cam]
    pts = [backproject(u, v, dep[v, u] / 1000.0, K, pose) for u, v in zip(us, vs) if dep[v, u] > 0]
    return np.array(pts), (us, vs)

def orange(rgb):
    r, g, b = [rgb[..., i].astype(int) for i in range(3)]
    return (r > 150) & (g > 60) & (g < 170) & (b < 90) & (r > g + 40)

def analyze(name, meta, imgs):
    p = perceive(meta, imgs)
    print(f"== {name}: step {p['step_id']} ended {meta.get('episode_ended')} {meta.get('end_reason')} robot {np.round(p['robot']['xy'],3)} yaw {p['robot']['yaw']:.3f}")
    for cam in ("head_camera", "overview"):
        pts, (us, vs) = cloud(meta, imgs, cam, blue_mask)
        if len(pts) == 0: print(f"  {cam}: no blue"); continue
        top = pts[pts[:, 2] > 0.04]
        near = pts[pts[:, 2] < 0.03]
        print(f"  {cam}: n={len(pts)} u {us.min()}-{us.max()} v {vs.min()}-{vs.max()} | xy bbox x {pts[:,0].min():.3f}-{pts[:,0].max():.3f} y {pts[:,1].min():.3f}-{pts[:,1].max():.3f} z {pts[:,2].min():.3f}-{pts[:,2].max():.3f}"
              f" | top(z>0.04) n={len(top)} mean {np.round(top[:,:2].mean(0),3) if len(top) else None} | cur est {np.round(p[cam.split('_')[0]]['xy'],3) if p[cam.split('_')[0]] else None}")
    pts, _ = cloud(meta, imgs, "overview", orange)
    if len(pts):
        feet = pts[pts[:, 2] < 0.02]
        print(f"  feet(z<2cm) n={len(feet)}: x {feet[:,0].min():.3f}-{feet[:,0].max():.3f} y {feet[:,1].min():.3f}-{feet[:,1].max():.3f} (base {np.round(p['robot']['xy'],3)})")
    return p

if __name__ == "__main__":
    tag = sys.argv[1]; n = int(sys.argv[2]); vx = float(sys.argv[3]); hp = float(sys.argv[4])
    with Context() as ctx:
        meta, imgs = observe(ctx); analyze("observe", meta, imgs)
        json.dump({k: v for k, v in meta.items() if k != "attachments"}, open(f"experiments/img/{tag}_obs.json", "w"), default=str)
        if n > 0:
            meta, imgs = step(ctx, [vx, 0, 0], n, [0, hp, 0, 0]); analyze(f"burst vx={vx} n={n}", meta, imgs)
            if not meta.get("episode_ended"):
                meta, imgs = hold(ctx, 50, [0, hp, 0, 0]); analyze("hold50", meta, imgs)
            for (kind, cam), arr in imgs.items(): Image.fromarray(arr).save(f"experiments/img/{tag}_{kind}_{cam}.png")
            json.dump({k: v for k, v in meta.items() if k != "attachments"}, open(f"experiments/img/{tag}_final.json", "w"), default=str)
        print("ENDED:", meta.get("episode_ended"), meta.get("end_reason"), meta.get("episode_limits"))
