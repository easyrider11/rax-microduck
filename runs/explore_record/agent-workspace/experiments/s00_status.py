import json, sys
from rax import Context, call, short
with Context() as ctx:
    print("== status"); print(json.dumps(call(ctx, "status"), indent=1, default=str))
    tools = ctx.request("tools/list", {})
    print("== tools")
    for t in tools.get("tools", []):
        print(t["name"], ":", t.get("description", "")[:600])
        print("  schema:", json.dumps(t.get("inputSchema", {}), default=str)[:1500])
