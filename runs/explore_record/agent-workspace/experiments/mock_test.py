"""Offline test of the scan branch and the ended-without-state reply using saved ep1 frames and a mock ctx."""
import os, sys, json, base64, io, copy
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "code/walk_to_cube")
import numpy as np
from PIL import Image
import controller
from motion import Tracker, scan_for_cube
from perception import blue_mask

meta = json.load(open("experiments/img/ep1_start.json"))
rgb_h = np.array(Image.open("experiments/img/ep1_start_0.png").convert("RGB")); rgb_o = np.array(Image.open("experiments/img/ep1_start_1.png").convert("RGB"))
dep_h = open("experiments/img/ep1_start_2.png", "rb").read(); dep_o = open("experiments/img/ep1_start_3.png", "rb").read()
def nocube(rgb):
    r = rgb.copy(); r[blue_mask(rgb)] = [200, 200, 200]; return r
def png(arr):
    b = io.BytesIO(); Image.fromarray(arr).save(b, format="PNG"); return b.getvalue()
def reply(with_cube, ended=False, state=True):
    m = copy.deepcopy(meta); m.pop("_image_paths", None)
    if ended: m["episode_ended"] = True; m["task_complete"] = True; m["end_reason"] = "task_complete"
    if not state:
        m.pop("state"); m.pop("attachments"); m.pop("camera_poses"); m.pop("camera_intrinsics")
        return {"content": [{"type": "text", "text": json.dumps(m)}]}
    h, o = (rgb_h, rgb_o) if with_cube else (nocube(rgb_h), nocube(rgb_o))
    blocks = [{"type": "text", "text": json.dumps(m)}]
    for data in (png(h), png(o), dep_h, dep_o):
        blocks.append({"type": "image", "data": base64.b64encode(data).decode(), "mimeType": "image/png"})
    return {"content": blocks}

class Mock:
    def __init__(self, script): self.script, self.calls = list(script), []
    def call(self, tool, **a):
        self.calls.append((tool, a))
        return self.script.pop(0) if self.script else reply(True)

# 1) scan: no cube in 3 looks, then the cube appears on the first body turn
mock = Mock([reply(False), reply(False), reply(False), reply(True)])
tr = Tracker({}, print)
m0, i0 = __import__("perception").parse_reply(reply(False))
p, _, _ = scan_for_cube(mock, tr, m0, i0)
print("scan result source:", p["source"], "calls:", [c[0] for c in mock.calls])
assert p["cube_xy"] is not None

# 2) main with: observe(no cube) -> scan finds it -> walking replies -> an ended reply without state
mock = Mock([reply(False), reply(False), reply(True)] + [reply(True)] * 3 + [reply(True, ended=True, state=False)])
r = controller.main(mock)
print("main report:", json.dumps(r, default=str)[:400])
assert r["ok"] and "traceback" not in r, r
print("calls:", [c[0] for c in mock.calls])
print("MOCK TESTS PASSED")
