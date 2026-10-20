import json
from rax import Context, call, short
with Context() as ctx:
    m = call(ctx, "start_episode", tag="ep1_start")
    print("== start_episode keys:", list(m.keys()))
    print(json.dumps(short(m), indent=1, default=str)[:4000])
    print("attachments:", json.dumps(m.get("attachments"), default=str)[:1500])
    print("intrinsics:", json.dumps(m.get("camera_intrinsics"), default=str)[:800])
    print("poses:", json.dumps(m.get("camera_poses"), default=str)[:1200])
    print("images:", m["_image_paths"])
    ms = call(ctx, "microduck_status", tag="ep1_mstatus")
    print("== microduck_status"); print(json.dumps(ms, indent=1, default=str)[:3000])
