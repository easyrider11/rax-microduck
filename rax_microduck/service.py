"""The session's MCP service: RAX's lifecycle tools plus the Microduck robot tools, on a Unix socket.

Newline-delimited JSON-RPC 2.0, as RAX's frontend (services/controller/frontend.py) speaks it:
Claude Code reaches it through ``rax_microduck.bridge`` (stdio <-> socket) and Python through
``api.runtime.Context``. One dispatcher lock serializes calls; replies are a JSON text block
followed by image blocks that ``attachments[*].content_index`` point to. Scores stay on the
operator side under ``<root>/native``; the agent sees ``episode_ended`` and the binary
``task_complete``.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import queue
import resource
import socket
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

import numpy as np

from . import tools as T
from .paths import RUNS
from .policy import HEAD_LIMITS, TWIST_LIMITS
from .tasks import EXPLORATION_COLLECTION, FORMAL_COLLECTION, FORMAL_LAYOUTS, TASK, layout

MCP_PROTOCOL = "2025-06-18"
ROBOT_DESCRIPTION = ("Pollen Microduck: a 25 cm, 800 g biped with 5-DoF legs and a 4-DoF head carrying the head "
                     "camera, driven by a pretrained walking policy that takes velocity commands")


def _png(array):
    from PIL import Image
    stream = io.BytesIO()
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(stream, format="PNG")
    return stream.getvalue()


def _depth_png(depth):
    from PIL import Image
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    millimeters = np.zeros(depth.shape, dtype=np.uint16)
    if valid.any():
        millimeters[valid] = np.rint(np.clip(depth[valid] * 1000.0, 0, 65535)).astype(np.uint16)
    stream = io.BytesIO()
    Image.fromarray(millimeters).save(stream, format="PNG")
    values = depth[valid]
    stats = {"shape": [int(depth.shape[1]), int(depth.shape[0])], "encoding": "uint16_png", "unit": "millimeters",
             "meters_per_unit": 0.001, "invalid_value": 0, "invalid_pixels": int((~valid).sum()),
             "min_meters": float(values.min()) if values.size else None,
             "max_meters": float(values.max()) if values.size else None}
    return stream.getvalue(), stats


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


class ToolError(Exception):
    def __init__(self, reason, *, stage="request", kind="SimulatorError"):
        super().__init__(reason)
        self.stage, self.kind = stage, kind


class Session:
    """One auto-research session: an episode budget over one layout collection, one simulator."""

    def __init__(self, root, *, episodes, collection=EXPLORATION_COLLECTION, start_layout=0,
                 workspace_prefix="runtime/autonomous_controller", render=True):
        from .sim import MicroduckSim
        self.root = Path(root)
        self.published = self.root / "published"
        self.native = self.root / "native"
        for directory in (self.published / "runs", self.native):
            directory.mkdir(parents=True, exist_ok=True)
        self.prefix = workspace_prefix
        self.sim = MicroduckSim(render=render)
        self.collection = int(collection)
        self.episodes_total = int(episodes)
        self.next_layout = int(start_layout)
        self.episodes_used = 0
        self.episode = None          # live or last episode record
        self.history = []
        self.raw = None              # (episode_id, step_id, observation) for pixel geometry
        self.lock = threading.RLock()
        self.audit_path = self.root / "mcp.jsonl"
        self.started_at = time.time()
        self._save_state()

    # ----------------------------------------------------------------- state
    def _save_state(self):
        _write_json(self.root / "state.json", {
            "task": TASK.name, "collection": self.collection, "episodes_total": self.episodes_total,
            "episodes_used": self.episodes_used, "next_layout": self.next_layout,
            "episode": self.episode, "history": self.history,
            "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 1),
            "updated_at_unix_s": time.time()})

    def _audit(self, record):
        with self.audit_path.open("a") as stream:
            stream.write(json.dumps({"t": time.time(), **record}) + "\n")

    @property
    def active(self):
        return self.episode is not None and not self.episode["ended"]

    # ------------------------------------------------------------- definitions
    def definitions(self):
        lifecycle = [
            {"name": "status", "description": (
                "Session status: the task, the exploration episode budget, the live episode (step, limit, ended) "
                "and what to do next. No physics."),
             "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
            {"name": "start_episode", "description": (
                "Start the next exploration episode on the next layout (closes the previous one). The robot "
                "stands at the origin of the odometry frame facing +x; the cube is placed anew. Takes no layout or "
                "seed. Returns the first observation."),
             "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
            {"name": "evaluate", "description": (
                "The trusted binary task_complete of the current episode (true only once the task's success "
                "condition held and the episode ended on it)."),
             "inputSchema": {"type": "object", "properties": {"agent_assessment": {"type": "string"}},
                             "additionalProperties": False}},
            {"name": "finish", "description": "Close the current exploration episode (it is not a submission).",
             "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
        ]
        return lifecycle + T.tool_definitions()

    # ---------------------------------------------------------------- packets
    def _limits(self):
        return {"episode_step_limit": TASK.step_limit,
                "remaining_episode_steps": max(0, TASK.step_limit - self.sim.step_id)}

    def _end_fields(self):
        if not self.episode:
            return {"episode_ended": False}
        fields = {"episode_ended": bool(self.episode["ended"])}
        if self.episode["ended"]:
            fields["task_complete"] = bool(self.episode["success"])
            fields["end_reason"] = self.episode["end_reason"]
        return fields

    def _encode(self, observation, *, depth):
        content, attachments, depth_stats = [], [], {}
        for camera in T.CAMERAS:
            if camera in observation:
                content.append({"type": "image", "data": base64.b64encode(_png(observation[camera])).decode(),
                                "mimeType": "image/png"})
                attachments.append({"kind": "rgb", "camera": camera, "content_index": len(content)})
        if depth:
            for camera in T.CAMERAS:
                key = f"{camera}_depth_m"
                if key in observation:
                    data, stats = _depth_png(observation[key])
                    content.append({"type": "image", "data": base64.b64encode(data).decode(), "mimeType": "image/png"})
                    depth_stats[camera] = dict(stats, content_index=len(content))
                    attachments.append({"kind": "depth", "camera": camera, "content_index": len(content)})
        return content, attachments, depth_stats

    def _observation_packet(self, observation, *, transition=None):
        self.raw = (self.episode["episode_id"], self.sim.step_id, observation)
        content, attachments, depth = self._encode(observation, depth=True)
        packet = {"episode_id": self.episode["episode_id"], "step_id": self.sim.step_id, "task": TASK.name,
                  "instruction": TASK.instruction, "state": observation["state"],
                  "images": [a["camera"] for a in attachments if a["kind"] == "rgb"], "attachments": attachments,
                  "depth": depth, "camera_intrinsics": observation.get("camera_intrinsics"),
                  "camera_poses": observation.get("camera_poses"), "episode_limits": self._limits(), **self._end_fields()}
        if transition is not None:
            packet["transition"] = transition
        return packet, content

    def _save_frames(self, frames, transition):
        run_id = self.episode["run_id"]
        sequence_id = f"seq_{self.sim.step_id:06d}_{uuid.uuid4().hex[:8]}"
        directory = self.published / "runs" / run_id / "frames" / sequence_id
        directory.mkdir(parents=True)
        logical = f"{self.prefix}/runs/{run_id}/frames/{sequence_id}"
        records = []
        for index, frame in enumerate(frames):
            files = []
            for camera in T.CAMERAS:
                if camera not in frame:  # --no-render sessions keep state only
                    continue
                name = f"frame_{index:06d}_{camera}_rgb.png"
                (directory / name).write_bytes(_png(frame[camera]))
                files.append({"kind": "rgb", "camera": camera, "path": f"{logical}/{name}"})
            records.append({"frame_index": index, "step_id": frame["step_id"], "state": frame["state"], "files": files})
        manifest = {"sequence_id": sequence_id, "artifact_id": run_id, "frequency_hz": 50 / T.FRAME_STRIDE,
                    "frame_count": len(records), "transition": transition, "frames": records}
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return {"sequence_id": sequence_id, "frequency_hz": 50 / T.FRAME_STRIDE, "frame_stride_steps": T.FRAME_STRIDE,
                "frame_count": len(records), "manifest_path": f"{logical}/manifest.json",
                "frame_path_pattern": f"{logical}/frame_{{frame_index:06d}}_{{camera}}_rgb.png",
                "frame_index_range": [0, len(records) - 1],
                "streams": [{"camera": c, "kind": "rgb"} for c in T.CAMERAS], "returned_images": "final_frame_only"}

    # ---------------------------------------------------------------- lifecycle
    def status(self):
        episode = None
        if self.episode:
            episode = {k: self.episode[k] for k in ("episode_id", "run_id", "ended")}
            episode.update(step_id=self.sim.step_id, **self._limits(), **self._end_fields())
        remaining = self.episodes_total - self.episodes_used
        if self.active:
            next_step = "Act in the live episode; call evaluate after visible success and finish when done."
        elif remaining > 0:
            next_step = "Call start_episode to begin the next exploration episode."
        else:
            next_step = "No exploration episodes remain: write memory/PROGRESS.md and your final report, then end."
        return {"task": TASK.name, "instruction": TASK.instruction, "robot": ROBOT_DESCRIPTION,
                "exploration": {"episodes_total": self.episodes_total, "episodes_used": self.episodes_used,
                                "exploration_remaining": remaining, "live": self.active},
                "episode": episode, "formal": {"held_out_layouts": FORMAL_LAYOUTS,
                                               "note": "the formal batch runs code/<project>/controller.py:main(ctx) "
                                                       "alone on held-out layouts after the session"},
                "next_step": next_step, "session_seconds": round(time.time() - self.started_at, 1)}

    def start_episode(self):
        if self.episodes_used >= self.episodes_total:
            raise ToolError("Exploration budget exhausted: no episode may start", stage="budget", kind="PermissionError")
        if self.active:
            self._close("reset")
        lay = layout(self.collection, self.next_layout)
        self.next_layout += 1
        self.episodes_used += 1
        mode = "formal" if self.collection == FORMAL_COLLECTION else "exploration"
        episode_id = f"{TASK.name}_{mode}_{self.episodes_used:03d}_{uuid.uuid4().hex[:8]}"
        run_id = f"{TASK.name}_{mode}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{uuid.uuid4().hex[:6]}"
        self.sim.reset(lay)
        self.episode = {"episode_id": episode_id, "run_id": run_id, "ended": False, "success": False, "end_reason": None,
                        "started_at_unix_s": time.time(), "steps": 0, "layout": {"collection": lay.collection, "id": lay.layout_id}}
        _write_json(self.native / episode_id / "reset.json", {"episode_id": episode_id, "layout": lay.__dict__,
                                                                "ground_truth": self.sim.ground_truth()})
        (self.published / "runs" / run_id).mkdir(parents=True, exist_ok=True)
        self._save_state()
        packet, content = self._observation_packet(self.sim.observation())
        packet.update(reset={"episode_id": episode_id, "step_id": 0}, artifact_id=run_id,
                      environment_spec={"benchmark": "microduck", "episode_mode": mode, "episode_step_limit": TASK.step_limit,
                                        "control_frequency_hz": 50})
        return packet, content

    def _close(self, reason):
        if not self.episode or self.episode.get("closed"):
            return
        ep = self.episode
        ep["closed"], ep["close_reason"] = True, reason
        ep["ended"] = ep["ended"] or True
        ep["end_reason"] = ep["end_reason"] or reason
        truth = self.sim.ground_truth()
        _write_json(self.native / ep["episode_id"] / "summary.json", {
            **{k: v for k, v in ep.items()}, "step_id": self.sim.step_id, "ground_truth": truth,
            "native_success": bool(ep["success"]), "native_score": 1.0 if ep["success"] else 0.0})
        self.history.append({k: ep[k] for k in ("episode_id", "run_id", "success", "end_reason", "steps")})
        self._save_state()

    def evaluate(self, agent_assessment=None):
        if not self.episode:
            raise ToolError("No episode to evaluate", kind="RuntimeError")
        value = {"status": "completed", "task_complete": bool(self.episode["success"]),
                 "episode_id": self.episode["episode_id"], "step_id": self.sim.step_id, "task": TASK.name,
                 "evaluation_kind": "binary_task_completion", "decided_at_unix_s": time.time(), **self._end_fields()}
        if agent_assessment:
            value["agent_assessment"] = str(agent_assessment)[:2000]
        return value

    def finish(self):
        if not self.episode:
            raise ToolError("No episode to finish", kind="RuntimeError")
        self._close("finish")
        return {"episode_id": self.episode["episode_id"], "step_id": self.sim.step_id, "closed": True, **self._end_fields()}

    # -------------------------------------------------------------- robot tools
    def _require_episode(self):
        if not self.episode:
            raise ToolError("No active episode; call start_episode first", kind="RuntimeError")

    def robot_status(self):
        return {"active": self.active, "task": TASK.name, "instruction": TASK.instruction, "robot": ROBOT_DESCRIPTION,
                "episode_id": self.episode["episode_id"] if self.episode else None,
                "step_id": self.sim.step_id if self.episode else None,
                "action_layout": T.action_layout(), "cameras": list(T.CAMERAS), "image_size": [self.sim.width, self.sim.height],
                "max_steps_per_call": T.MAX_STEPS, "control_frequency_hz": 50, "physics_seconds_per_step": 0.02,
                "twist_limits": TWIST_LIMITS.tolist(), "head_limits": HEAD_LIMITS.tolist(),
                "episode_limits": self._limits() if self.episode else None, **self._end_fields()}

    def observe(self):
        self._require_episode()
        if self.raw is not None and self.raw[:2] == (self.episode["episode_id"], self.sim.step_id):
            observation = self.raw[2]
        else:
            observation = self.sim.observation()
        return self._observation_packet(observation)

    def motion(self, twist, head, steps, kind):
        self._require_episode()
        if self.episode["ended"]:
            raise ToolError("Episode has ended; further motion is blocked", kind="PermissionError")
        frames, executed = [], 0
        for _ in range(steps):
            self.sim.command(twist, head, steps=1)
            executed += 1
            self.episode["steps"] = self.sim.step_id
            if self.sim.terminated:
                self.episode.update(ended=True, success=bool(self.sim.success), end_reason=self.sim.end_reason)
                break
            if self.sim.step_id % T.FRAME_STRIDE == 0:
                frame = self.sim.observation(depth=False)
                frames.append(frame)
        final = self.sim.observation()
        frames.append(final)
        transition = {"kind": kind, "steps_requested": steps, "steps_executed": executed, "step_id": self.sim.step_id,
                      "commanded_twist": [float(v) for v in np.clip(twist, -TWIST_LIMITS, TWIST_LIMITS)],
                      "commanded_head": [float(v) for v in np.clip(head, -HEAD_LIMITS, HEAD_LIMITS)], **self._end_fields()}
        sequence = self._save_frames(frames, transition)
        if self.sim.terminated:
            self._close(self.sim.end_reason)
        packet, content = self._observation_packet(final, transition=transition)
        packet["frame_sequence"] = sequence
        return packet, content

    def pixel_to_position(self, arguments):
        self._require_episode()
        if self.raw is None or self.raw[0] != self.episode["episode_id"]:
            raise ToolError("Observe first; no current observation", kind="ValueError")
        episode, step, observation = self.raw
        T.reject_unless(arguments["observation_step_id"] == self.sim.step_id == step,
                        f"Stale pixel selection: observation_step_id={arguments['observation_step_id']}, "
                        f"current step_id={self.sim.step_id}; observe again and reselect")
        results = []
        for item in arguments["selections"]:
            camera = item["camera"]
            depth = observation.get(f"{camera}_depth_m")
            if depth is None:
                results.append({"camera": camera, "valid": False, "reason": "no depth for this camera"})
                continue
            point = T.pixel_to_world(depth, observation["camera_intrinsics"][camera], observation["camera_poses"][camera],
                                     float(item["u"]), float(item["v"]), int(arguments.get("depth_window_radius", 1)))
            results.append({"camera": camera, "u": item["u"], "v": item["v"], **point})
        return {"episode_id": episode, "observation_step_id": step, "selections": results}

    # ---------------------------------------------------------------- dispatch
    def call(self, name, arguments):
        if not isinstance(arguments, dict):
            raise ToolError("Arguments must be an object", stage="validation", kind="ValueError")
        if name == "status":
            return self.status(), []
        if name == "start_episode":
            return self.start_episode()
        if name == "evaluate":
            return self.evaluate(arguments.get("agent_assessment")), []
        if name == "finish":
            return self.finish(), []
        if name in T.ROBOT_TOOLS:
            try:
                T.validate_request(name, arguments)
            except T.RejectedBeforeAction as exc:
                raise ToolError(str(exc), stage="validation", kind="RejectedBeforeAction") from exc
            if name == T.STATUS:
                return self.robot_status(), []
            if name == T.OBSERVE:
                return self.observe()
            if name == T.STEP:
                return self.motion(np.asarray(arguments["twist"], dtype=float),
                                   np.asarray(arguments.get("head") or [0, 0, 0, 0], dtype=float), arguments["steps"], "step")
            if name == T.HOLD:
                return self.motion(np.zeros(3), np.asarray(arguments.get("head") or [0, 0, 0, 0], dtype=float),
                                   arguments["steps"], "hold")
            if name == T.PIXEL_TO_POSITION:
                try:
                    return self.pixel_to_position(arguments), []
                except T.RejectedBeforeAction as exc:
                    raise ToolError(str(exc), stage="validation", kind="RejectedBeforeAction") from exc
        raise ToolError(f"Tool is not available in this MCP contract: {name}", kind="PermissionError")

    def close(self):
        if self.active:
            self._close("close")
        self.sim.close()
        self._save_state()


def short_socket_path():
    """AF_UNIX paths are limited to about 100 bytes on macOS, so sockets live in $TMPDIR or /tmp."""
    base = os.environ.get("TMPDIR", "")
    if not base or len(base) > 60 or not os.path.isdir(base):
        base = "/tmp"
    return os.path.join(base, f"rax-microduck-{uuid.uuid4().hex[:8]}.sock")


class Server:
    """Clients get a thread each, but every tool call runs on one robot thread: MuJoCo's offscreen
    OpenGL context is bound to the thread that created it, and RAX serializes calls anyway."""

    def __init__(self, session, socket_path):
        self.session = session
        self.path = str(socket_path)
        self.jobs = queue.Queue()
        threading.Thread(target=self._robot_thread, daemon=True, name="robot").start()

    def _robot_thread(self):
        while True:
            function, done, box = self.jobs.get()
            try:
                box["result"] = function()
            except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
                box["error"] = exc
            done.set()

    def on_robot_thread(self, function):
        if threading.current_thread().name == "robot":
            return function()
        done, box = threading.Event(), {}
        self.jobs.put((function, done, box))
        done.wait()
        if "error" in box:
            raise box["error"]
        return box["result"]

    def handle(self, request):
        session = self.session
        ident = request.get("id") if isinstance(request, dict) else None
        method = request.get("method") if isinstance(request, dict) else None
        if method == "notifications/initialized" or (ident is None and method and method.startswith("notifications/")):
            return None
        try:
            if not isinstance(request, dict) or request.get("jsonrpc") != "2.0":
                raise ToolError("Invalid JSON-RPC message", kind="ValueError")
            if method == "initialize":
                result = {"protocolVersion": MCP_PROTOCOL, "capabilities": {"tools": {}},
                          "serverInfo": {"name": "rax-microduck", "version": "0.1"}}
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": session.definitions()}
            elif method in ("prompts/list", "resources/list", "resources/templates/list"):
                result = {method.split("/")[0] if method != "resources/templates/list" else "resourceTemplates": []}
            elif method == "tools/call":
                params = request.get("params") or {}
                name, arguments = params.get("name"), params.get("arguments", {})
                with session.lock:
                    session._audit({"event": "mcp_request", "tool": name, "arguments": arguments})
                    started = time.time()
                    value, images = self.on_robot_thread(lambda: session.call(name, arguments))
                    text = json.dumps(value, allow_nan=False)
                    session._audit({"event": "mcp_result", "tool": name, "seconds": round(time.time() - started, 3),
                                    "result": value if name not in T.MUTATING | {T.OBSERVE, "start_episode"} else
                                    {k: v for k, v in value.items() if k not in ("attachments", "camera_intrinsics", "camera_poses")},
                                    "images": len(images)})
                    result = {"content": [{"type": "text", "text": text}, *images]}
            else:
                raise ToolError(f"Unsupported MCP method: {method}", kind="PermissionError")
            return {"jsonrpc": "2.0", "id": ident, "result": result}
        except Exception as exc:  # noqa: BLE001 - every failure is reported whole to the agent
            failure = {"type": getattr(exc, "kind", type(exc).__name__), "operation": method if method != "tools/call" else
                       (request.get("params") or {}).get("name"), "stage": getattr(exc, "stage", "native_execution"),
                       "reason": str(exc), **session._end_fields()}
            if not isinstance(exc, ToolError):
                failure["traceback"] = traceback.format_exc().strip().splitlines()[-6:]
            session._audit({"event": "mcp_error", **failure})
            return {"jsonrpc": "2.0", "id": ident, "error": {"code": -32000, "message": failure["reason"], "data": failure}}

    def _serve_client(self, connection):
        with connection, connection.makefile("rb") as reader:
            for line in reader:
                line = line.strip()
                if not line:
                    continue
                try:
                    request = json.loads(line)
                except ValueError:
                    reply = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
                else:
                    reply = self.handle(request)
                if reply is not None:
                    try:
                        connection.sendall((json.dumps(reply, allow_nan=False) + "\n").encode())
                    except OSError:
                        return

    def serve_forever(self):
        if os.path.exists(self.path):
            os.unlink(self.path)
        with socket.socket(socket.AF_UNIX) as listener:
            listener.bind(self.path)
            os.chmod(self.path, 0o600)
            listener.listen(16)
            print(json.dumps({"event": "ready", "socket": self.path, "pid": os.getpid()}), flush=True)
            try:
                while True:
                    connection, _ = listener.accept()
                    threading.Thread(target=self._serve_client, args=(connection,), daemon=True).start()
            finally:
                self.session.close()
                if os.path.exists(self.path):
                    os.unlink(self.path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="session directory (state.json, mcp.jsonl, published/, native/)")
    parser.add_argument("--socket", default=None, help="Unix socket path (default <root>/robot.sock)")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--collection", type=int, choices=(EXPLORATION_COLLECTION, FORMAL_COLLECTION), default=EXPLORATION_COLLECTION)
    parser.add_argument("--start-layout", type=int, default=0)
    parser.add_argument("--no-render", action="store_true", help="state only, no cameras (tests)")
    args = parser.parse_args(argv)
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    socket_path = Path(args.socket or short_socket_path())
    session = Session(root, episodes=args.episodes, collection=args.collection, start_layout=args.start_layout,
                      render=not args.no_render)
    (root / "socket.txt").write_text(str(socket_path) + "\n")
    Server(session, socket_path).serve_forever()


if __name__ == "__main__":
    main()
