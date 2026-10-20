"""walk_to_cube controller: main(ctx). See README.md."""
import json, os, sys, time, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motion import Tracker, observe, approach, inch, hold, HEAD_DOWN


def main(ctx, printer=print):
    log = {"t0": time.time()}
    tr = Tracker(log, printer)
    report = {"ok": False, "measured": {}, "why": ""}
    try:
        meta, imgs = observe(ctx)
        p, meta, imgs = approach(ctx, tr, meta, imgs)
        report["measured"]["after_approach"] = {k: p.get(k) for k in ("robot", "cube_xy", "dist", "step_id")}
        if not p["episode_ended"]:
            p, meta, imgs = inch(ctx, tr, meta, imgs)
        if not p["episode_ended"]:
            meta, imgs = hold(ctx, 100, HEAD_DOWN)
            p = tr.see(meta, imgs, "final_hold")
        report["measured"]["final"] = {k: p.get(k) for k in ("robot", "cube_xy", "dist", "step_id", "episode_ended", "end_reason", "task_complete")}
        report["ok"] = bool(p.get("task_complete"))
        report["why"] = f"episode_ended={p['episode_ended']} end_reason={p.get('end_reason')}"
    except Exception as e:
        report["why"] = f"{type(e).__name__}: {e}"
        report["traceback"] = traceback.format_exc()
    finally:
        try:
            os.makedirs("output", exist_ok=True)
            with open(os.path.join("output", f"run_{int(log['t0'])}.json"), "w") as f:
                json.dump({"log": log, "report": report}, f, indent=1, default=str)
        except Exception:
            pass
    return report
