"""`make demo`: boot the service, run the scripted reference controller for one episode, report."""
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

from . import reference_controller
from .client import Service, text
from .paths import RUNS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--layout", type=int, default=0, help="exploration layout id")
    parser.add_argument("--collection", type=int, default=0)
    args = parser.parse_args(argv)
    root = Path(args.root or RUNS / f"demo_{time.strftime('%Y%m%dT%H%M%S')}")
    started = time.time()
    with Service(root, episodes=1, collection=args.collection, start_layout=args.layout) as service:
        with service.context() as ctx:
            status = text(ctx.call("status"))
            print("status:", json.dumps({k: status[k] for k in ("task", "exploration", "next_step")}))
            first = text(ctx.call("start_episode"))
            print(f"episode {first['episode_id']} started; cameras {first['images']}; base {first['state']['base_pose']}")
            result = reference_controller.main(ctx)
            evaluation = text(ctx.call("evaluate", agent_assessment=result["why"]))
            final = text(ctx.call("microduck_status"))
            closed = text(ctx.call("finish"))
        peak_service = service.peak_rss_mb()
    summary = {"root": str(root), "task_complete": evaluation["task_complete"], "end_reason": closed.get("end_reason"),
               "steps": final["step_id"], "controller": {"ok": result["ok"], "why": result["why"], "iterations": len(result["log"])},
               "wall_seconds": round(time.time() - started, 1), "service_rss_mb_at_end": peak_service,
               "demo_process_peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 1)}
    (root / "demo_summary.json").write_text(json.dumps({**summary, "controller_log": result["log"]}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if evaluation["task_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
