"""ep3: start, turn ~1.2 rad AWAY from the cube by hand (so the head camera loses it), then run the controller."""
import os, sys, json, math
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "code/walk_to_cube")
from api.runtime import Context
from perception import parse_reply, perceive
from motion import step, hold
import controller
with Context() as ctx:
    meta, imgs = parse_reply(ctx.call("start_episode"))
    p = perceive(meta, imgs)
    print("start:", meta["episode_id"], "base", p["robot"], "cube", p["cube_xy"], "src", p["source"], "d", p.get("dist"), "err", p.get("bearing_err"))
    e = p.get("bearing_err") or 0.0
    sgn = -1.0 if e >= 0 else 1.0          # turn away from the cube
    meta, imgs = step(ctx, [0.0, 0.0, sgn], 85, None)
    p = perceive(meta, imgs)
    print("after turn:", "base", p["robot"], "cube", p["cube_xy"], "src", p["source"], "d", p.get("dist"), "err", p.get("bearing_err"), "head", p["head"] and (p["head"]["pixels"], p["head"]["clipped"]), "over", p["overview"] and p["overview"]["pixels"])
    r = controller.main(ctx)
    print(json.dumps(r, indent=1, default=str))
    print("evaluate:", ctx.call("evaluate", agent_assessment="controller v1.1 after a manual 1.2 rad turn away from the cube")["content"][0]["text"])
    print("finish:", ctx.call("finish")["content"][0]["text"])
