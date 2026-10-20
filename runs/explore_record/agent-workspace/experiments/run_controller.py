import os, sys, json
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "code/walk_to_cube")
from api.runtime import Context
import controller
with Context() as ctx:
    if "--start" in sys.argv:
        r0 = ctx.call("start_episode"); print("started:", json.loads(r0["content"][0]["text"])["episode_id"])
    r = controller.main(ctx)
    print(json.dumps(r, indent=1, default=str))
    if "--eval" in sys.argv:
        print("evaluate:", ctx.call("evaluate", agent_assessment="controller v1 unattended")["content"][0]["text"])
        print("finish:", ctx.call("finish")["content"][0]["text"])
