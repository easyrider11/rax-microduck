import json, sys, numpy as np
from PIL import Image
sys.path.insert(0, "code/walk_to_cube")
from perception import parse_reply, perceive, backproject, locate_cube
from rax import Context, call

# 1) feet offset from the start overview frame: orange pixels -> world points
meta = json.load(open("experiments/img/ep1_start.json"))
rgb = np.array(Image.open("experiments/img/ep1_start_1.png").convert("RGB"))
dep = np.array(Image.open("experiments/img/ep1_start_3.png"))
r, g, b = [rgb[..., i].astype(int) for i in range(3)]
orange = (r > 150) & (g > 60) & (g < 170) & (b < 90) & (r > g + 40)
vs, us = np.nonzero(orange)
print("orange px:", orange.sum(), "u", us.min(), us.max(), "v", vs.min(), vs.max())
K = meta["camera_intrinsics"]["overview"]["matrix"]; pose = meta["camera_poses"]["overview"]
pts = np.array([backproject(u, v, dep[v, u] / 1000.0, K, pose) for u, v in zip(us, vs) if dep[v, u] > 0])
low = pts[pts[:, 2] < 0.05]   # feet-level points
print("all orange pts: x %.3f..%.3f y %.3f..%.3f z %.3f..%.3f" % (pts[:,0].min(), pts[:,0].max(), pts[:,1].min(), pts[:,1].max(), pts[:,2].min(), pts[:,2].max()))
if len(low):
    print("feet-level (z<5cm) pts: n=%d mean xy (%.3f, %.3f) x %.3f..%.3f y %.3f..%.3f" % (len(low), low[:,0].mean(), low[:,1].mean(), low[:,0].min(), low[:,0].max(), low[:,1].min(), low[:,1].max()))
print("base_pose:", meta["state"]["base_pose"])

# 2) one measured walk call
with Context() as ctx:
    reply = ctx.call("microduck_step", twist=[0.3, 0.0, 0.0], steps=50)
    m, imgs = parse_reply(reply)
    json.dump({k: v for k, v in m.items() if k not in ("attachments",)}, open("experiments/img/ep1_walk1.json", "w"), indent=1, default=str)
    for (kind, cam), arr in imgs.items():
        Image.fromarray(arr).save(f"experiments/img/ep1_walk1_{kind}_{cam}.png")
    p = perceive(m, imgs)
    print("transition:", json.dumps(m.get("transition"), default=str)[:800])
    print("episode_ended:", m.get("episode_ended"), "end_reason:", m.get("end_reason"), "limits:", m.get("episode_limits"))
    print("frame_sequence:", m.get("frame_sequence"))
    print("perceive:", json.dumps(p, indent=1, default=str))
