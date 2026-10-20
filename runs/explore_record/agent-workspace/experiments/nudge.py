"""nudge.py vx vy yaw steps head_pitch hold_steps  -- one burst then a hold, printing perception."""
import sys, os, json
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "code/walk_to_cube")
from PIL import Image
from api.runtime import Context
from perception import perceive
from motion import step, hold, observe
vx, vy, yaw, n, hp, hn = [float(x) for x in sys.argv[1:7]]
tag = sys.argv[7] if len(sys.argv) > 7 else "nudge"
def show(name, meta, imgs):
    p = perceive(meta, imgs)
    print(name, "step", p["step_id"], "ended", meta.get("episode_ended"), "reason", meta.get("end_reason"),
          "robot", [round(x, 3) for x in p["robot"]["xy"]], "yaw", round(p["robot"]["yaw"], 3),
          "cube", p["cube_xy"] and [round(x, 3) for x in p["cube_xy"]], "src", p["source"],
          "dist", p.get("dist") and round(p["dist"], 3), "head_px", p["head"] and p["head"]["pixels"], "over_px", p["overview"] and p["overview"]["pixels"])
    for (kind, cam), arr in imgs.items():
        Image.fromarray(arr).save(f"experiments/img/{tag}_{name}_{kind}_{cam}.png")
    return meta
with Context() as ctx:
    head = [0.0, hp, 0.0, 0.0]
    if int(n) > 0:
        m, im = step(ctx, [vx, vy, yaw], int(n), head); show("burst", m, im)
        if m.get("episode_ended"): print("ENDED:", json.dumps({k: m.get(k) for k in ("end_reason", "episode_ended", "episode_limits")}, default=str)); sys.exit()
    if int(hn) > 0:
        m, im = hold(ctx, int(hn), head); show("hold", m, im)
    print("ENDED:", m.get("episode_ended"), m.get("end_reason"), m.get("episode_limits"))
