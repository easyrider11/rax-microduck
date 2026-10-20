import sys, os, json, numpy as np
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "code/walk_to_cube"); sys.path.insert(0, "experiments")
from PIL import Image
from api.runtime import Context
from perception import perceive, blue_mask, backproject
from motion import step, hold, observe
from s04_analyze import cloud, orange

def est(meta, imgs):
    out = {}
    for cam in ("head_camera", "overview"):
        if ("rgb", cam) not in imgs: continue
        pts, (us, vs) = cloud(meta, imgs, cam, blue_mask)
        if len(pts) == 0: out[cam] = None; continue
        top = pts[pts[:, 2] > 0.04]
        out[cam] = {"n": len(pts), "top_n": len(top), "top_mean": np.round(top[:, :2].mean(0), 3).tolist() if len(top) else None,
                    "v_max": int(vs.max()), "u_rng": [int(us.min()), int(us.max())]}
    pts, _ = cloud(meta, imgs, "head_camera", orange)
    out["head_orange"] = {"n": len(pts), "x_rng": [round(pts[:,0].min(),3), round(pts[:,0].max(),3)], "y_rng": [round(pts[:,1].min(),3), round(pts[:,1].max(),3)], "z_rng": [round(pts[:,2].min(),3), round(pts[:,2].max(),3)]} if len(pts) else None
    return out

n_burst = int(sys.argv[1]); vx = float(sys.argv[2]); head = [float(x) for x in sys.argv[3].split(",")]; max_iter = int(sys.argv[4])
with Context() as ctx:
    meta, imgs = observe(ctx)
    for i in range(max_iter):
        p = perceive(meta, imgs)
        print(f"[{i}] step {p['step_id']} ended {meta.get('episode_ended')} {meta.get('end_reason')} base {np.round(p['robot']['xy'],3)} yaw {p['robot']['yaw']:.3f} rem {p['remaining']}")
        print("    ", json.dumps(est(meta, imgs), default=str))
        if meta.get("episode_ended"): break
        meta, imgs = step(ctx, [vx, 0, 0], n_burst, head)
        pb = perceive(meta, imgs)
        print(f"    burst -> base {np.round(pb['robot']['xy'],3)} ended {meta.get('episode_ended')} {meta.get('end_reason')}")
        if meta.get("episode_ended"): 
            print("    ", json.dumps(est(meta, imgs), default=str)); break
        meta, imgs = hold(ctx, 50, head)
        for (kind, cam), arr in imgs.items(): Image.fromarray(arr).save(f"experiments/img/inch_{i}_{kind}_{cam}.png")
    print("FINAL ended", meta.get("episode_ended"), meta.get("end_reason"), meta.get("episode_limits"))
    print("final meta keys:", [k for k in meta.keys() if k not in ("state","camera_intrinsics","camera_poses","attachments","depth")])
    print("final extra:", json.dumps({k: meta[k] for k in meta if k in ("end_reason","transition","result","success","task_complete","termination")}, default=str)[:1500])
