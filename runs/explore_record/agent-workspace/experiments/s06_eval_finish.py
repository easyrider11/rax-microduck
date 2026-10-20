import json, sys
from rax import Context, call
with Context() as ctx:
    print("evaluate:", json.dumps(call(ctx, "evaluate", agent_assessment=sys.argv[1] if len(sys.argv) > 1 else "manual run"), default=str))
    print("finish:", json.dumps(call(ctx, "finish"), default=str)[:800])
    print("status:", json.dumps(call(ctx, "status")["exploration"], default=str))
