"""The formal batch: run a controller.py:main(ctx) alone on the five held-out layouts.

The RAX harness runs frozen bundles in a bubblewrap sandbox with a wall limit; here each episode
runs the controller in a fresh subprocess with a timeout, in the same spirit, without the
Linux-only sandbox.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from .client import Service, text
from .paths import ROOT, RUNS
from .tasks import FORMAL_COLLECTION, FORMAL_LAYOUTS

WRAPPER = """
import importlib.util, json, sys
sys.path.insert(0, {project!r})
spec = importlib.util.spec_from_file_location("submitted_controller", {controller!r})
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
sys.path.insert(0, {workspace!r})
from api.runtime import Context
with Context() as ctx:
    result = module.main(ctx)
print("CONTROLLER_RESULT " + json.dumps({{k: v for k, v in (result or {{}}).items() if k != "log"}}, default=str))
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", required=True, help="path to controller.py defining main(ctx)")
    parser.add_argument("--root", default=None)
    parser.add_argument("--episodes", type=int, default=FORMAL_LAYOUTS)
    parser.add_argument("--timeout", type=int, default=600, help="wall seconds per episode")
    args = parser.parse_args(argv)
    controller = Path(args.controller).resolve()
    root = Path(args.root or RUNS / f"formal_{time.strftime('%Y%m%dT%H%M%S')}")
    workspace = ROOT / "workspace_template"
    results = []
    with Service(root, episodes=args.episodes, collection=FORMAL_COLLECTION) as service:
        for index in range(args.episodes):
            with service.context() as ctx:
                first = text(ctx.call("start_episode"))
            started = time.time()
            code = WRAPPER.format(project=str(controller.parent), controller=str(controller), workspace=str(workspace))
            try:
                run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=args.timeout,
                                     env={"RAX_MCP_SOCKET": str(service.socket), "PATH": "/usr/bin:/bin"}, cwd=str(root))
                stdout, stderr, timed_out = run.stdout, run.stderr, False
            except subprocess.TimeoutExpired as exc:
                stdout, stderr, timed_out = exc.stdout or "", exc.stderr or "", True
            with service.context() as ctx:
                evaluation = text(ctx.call("evaluate"))
                closed = text(ctx.call("finish"))
            record = {"episode": index, "episode_id": first["episode_id"], "success": bool(evaluation["task_complete"]),
                      "end_reason": closed.get("end_reason"), "steps": closed["step_id"], "timed_out": timed_out,
                      "wall_seconds": round(time.time() - started, 1),
                      "controller_result": next((line[len("CONTROLLER_RESULT "):] for line in (stdout or "").splitlines()
                                                 if line.startswith("CONTROLLER_RESULT ")), None),
                      "stderr_tail": (stderr or "").strip().splitlines()[-5:]}
            results.append(record)
            print(json.dumps(record))
            (root / f"episode_{index:03d}.log").write_text((stdout or "") + "\n--- stderr ---\n" + (stderr or ""))
    successes = sum(r["success"] for r in results)
    scores = {"controller": str(controller), "episodes": len(results), "successes": successes,
              "success_rate": successes / max(1, len(results)), "results": results}
    (root / "formal-scores.json").write_text(json.dumps(scores, indent=2) + "\n")
    print(json.dumps({k: scores[k] for k in ("episodes", "successes", "success_rate")}))


if __name__ == "__main__":
    main()
