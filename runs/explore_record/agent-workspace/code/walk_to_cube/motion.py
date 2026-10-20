"""Motion primitives and the approach/inch loops for walk_to_cube (use the supplied ctx only)."""
import math, json
from perception import parse_reply, perceive, wrap

# Tunables. Tested: ep1 (cube 0.93 m ahead, bearing ~0).
VX = 0.3                     # the only command that reliably starts and keeps a gait (~0.17 m/s)
FAR_D = 0.45                 # m: above -> 50-step calls; below -> 25-step calls
WALK_STOP_D = 0.30           # m: stop walking here; the hold travels <= 8 cm from a steady gait
INCH_LONG_D = 0.18           # m: above -> 30-step bursts, below -> 20-step bursts (~4 cm incl. stop)
INCH_SHORT_D = 0.125         # m: below -> 15-step bursts (~1 cm + stop travel), to avoid overshooting into the cube
INCH_STOP_D = 0.055          # m: never burst when nearer than this (success registered at ~0.064 in ep1)
LOCK_MAX_D = 0.55            # m: lock the cube xy from an unclipped head view nearer than this
YAW_MAX = 0.5                # rad/s while walking; yaw_rate = bearing_err / call_seconds
INCH_TURN_THRESH = 0.8       # rad: in the inch phase, turn in place first when the error exceeds this (d > 0.10)
TURN_THRESH = 0.45           # rad: bigger error and d > 0.25 -> turn in place at yaw 1.0 (~0.7 rad/s)
HEAD_HOME = [0.0, 0.0, 0.0, 0.0]
HEAD_MID = [0.0, 0.6, 0.0, 0.0]      # d < 0.7
HEAD_DOWN = [0.0, 1.1, 0.0, 0.0]     # d < 0.3
MIN_REMAINING = 80           # steps: below this, just hold


def observe(ctx):
    return parse_reply(ctx.call("microduck_observe"))


def step(ctx, twist, steps, head=None):
    args = {"twist": [float(x) for x in twist], "steps": int(steps)}
    if head is not None:
        args["head"] = [float(x) for x in head]
    return parse_reply(ctx.call("microduck_step", **args))


def hold(ctx, steps, head=None):
    args = {"steps": int(steps)}
    if head is not None:
        args["head"] = [float(x) for x in head]
    return parse_reply(ctx.call("microduck_hold", **args))


def head_for(d):
    if d is None:
        return HEAD_HOME
    return HEAD_DOWN if d < 0.3 else (HEAD_MID if d < 0.7 else HEAD_HOME)


class Tracker:
    """Holds the cube lock and logs every perception."""
    def __init__(self, log, printer):
        self.lock, self.log, self.printer = None, log, printer
        self.prev = None

    def see(self, meta, imgs, tag):
        p = perceive(meta, imgs, lock=self.lock, prev=self.prev)
        self.prev = p
        h = p["head"]
        if h is not None and not h["clipped"] and h["pixels"] >= 30 and p.get("dist") is not None and p["dist"] < LOCK_MAX_D:
            self.lock = list(h["xy"])
        p["lock"] = self.lock
        self.log.setdefault("perceptions", []).append({"tag": tag, **{k: v for k, v in p.items()}})
        self.printer(f"[{tag}] step {p['step_id']} rem {p['remaining']} base ({p['robot']['xy'][0]:.3f},{p['robot']['xy'][1]:.3f}) yaw {p['robot']['yaw']:.3f} "
                     f"cube {None if p['cube_xy'] is None else [round(x, 3) for x in p['cube_xy']]} src {p['source']} "
                     f"d {None if p.get('dist') is None else round(p['dist'], 3)} err {None if p.get('bearing_err') is None else round(p['bearing_err'], 3)} "
                     f"head={None if h is None else (h['pixels'], h['method'], h['clipped'])} ended {p['episode_ended']} {p['end_reason'] or ''}")
        return p


def scan_for_cube(ctx, tr, meta, imgs):
    """Cube not visible: look left/right with the head, then turn the body. Returns (p, meta, imgs)."""
    for body_turn in range(5):
        for hy in (0.9, -0.9):
            meta, imgs = hold(ctx, 40, [0.0, 0.3, hy, 0.0])
            p = tr.see(meta, imgs, f"scan_head{hy:+.1f}")
            if p["episode_ended"] or p["cube_xy"] is not None:
                return p, meta, imgs
        meta, imgs = step(ctx, [0.0, 0.0, 1.0], 60, HEAD_HOME)
        p = tr.see(meta, imgs, f"scan_turn{body_turn}")
        if p["episode_ended"] or p["cube_xy"] is not None:
            return p, meta, imgs
    raise RuntimeError("scan: cube not found after head sweeps and body turns")


def approach(ctx, tr, meta, imgs, max_calls=40):
    """Walk until base-to-cube distance <= WALK_STOP_D, then hold. Returns (p, meta, imgs)."""
    for i in range(max_calls):
        p = tr.see(meta, imgs, f"approach{i}")
        if p["episode_ended"]:
            return p, meta, imgs
        if p["cube_xy"] is None:
            p, meta, imgs = scan_for_cube(ctx, tr, meta, imgs)
            continue
        if p["remaining"] is not None and p["remaining"] < MIN_REMAINING:
            return p, meta, imgs
        d, e = p["dist"], p["bearing_err"]
        if d <= WALK_STOP_D:
            meta, imgs = hold(ctx, 50, HEAD_DOWN)
            return tr.see(meta, imgs, "approach_stop"), meta, imgs
        if abs(e) > TURN_THRESH and d > 0.25:
            n = int(min(90, max(10, abs(e) / 0.7 * 50)))
            meta, imgs = step(ctx, [0.0, 0.0, math.copysign(1.0, e)], n, HEAD_HOME)
            continue
        n = 50 if d > FAR_D else 25
        yaw = max(-YAW_MAX, min(YAW_MAX, e / (n * 0.02)))
        meta, imgs = step(ctx, [VX, 0.0, yaw], n, head_for(d))
    raise RuntimeError("approach: max_calls reached")


def inch(ctx, tr, meta, imgs, max_bursts=12):
    """Short vx=0.3 bursts + holds until the episode ends (success) or too close. Returns (p, meta, imgs)."""
    p = tr.see(meta, imgs, "inch_start")
    for i in range(max_bursts):
        if p["episode_ended"]:
            return p, meta, imgs
        if p["cube_xy"] is None:
            raise RuntimeError("inch: lost the cube")
        if p["remaining"] is not None and p["remaining"] < MIN_REMAINING:
            return p, meta, imgs
        d, e = p["dist"], p["bearing_err"]
        if d <= INCH_STOP_D:
            meta, imgs = hold(ctx, 100, HEAD_DOWN)
            return tr.see(meta, imgs, "inch_toolclose_hold"), meta, imgs
        if abs(e) > INCH_TURN_THRESH and d > 0.10:
            n = int(min(40, max(10, abs(e) / 0.7 * 50)))
            meta, imgs = step(ctx, [0.0, 0.0, math.copysign(1.0, e)], n, HEAD_DOWN)
            p = tr.see(meta, imgs, f"inch{i}_turn")
            if p["episode_ended"]:
                return p, meta, imgs
            meta, imgs = hold(ctx, 30, HEAD_DOWN)
            p = tr.see(meta, imgs, f"inch{i}_turnhold")
            continue
        n = 30 if d > INCH_LONG_D else (20 if d > INCH_SHORT_D else 15)
        yaw = max(-YAW_MAX, min(YAW_MAX, e / (n * 0.02)))
        meta, imgs = step(ctx, [VX, 0.0, yaw], n, HEAD_DOWN)
        p = tr.see(meta, imgs, f"inch{i}_burst")
        if p["episode_ended"]:
            return p, meta, imgs
        meta, imgs = hold(ctx, 50, HEAD_DOWN)
        p = tr.see(meta, imgs, f"inch{i}_hold")
    return p, meta, imgs
