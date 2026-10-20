"""One Claude Code exploration session through the harness (RAX's `agent_cli: claude` launch).

Prepares a session directory with the rendered agent workspace, starts the MCP service, points
Claude Code at it through the stdio bridge (``--mcp-config``, ``--strict-mcp-config``), runs the
kickoff prompt headless and records the stream to ``agent.jsonl``; afterwards the submitted
controller (``code/*/controller.py``) can be run on the held-out layouts with ``formal``.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .client import Service
from .paths import ROOT, RUNS
from .workspace import build_workspace


def settings(workspace, socket_dir):
    """Claude Code settings for the session: edits in the workspace need no approval, sandboxed
    commands neither; no web tools; the macOS sandbox confines writes to the workspace."""
    return {"permissions": {"defaultMode": "acceptEdits", "allow": ["mcp__auto_research", "Bash", "Read", "Edit", "Write",
                                                                     "Glob", "Grep", "MultiEdit"],
                            "deny": ["WebFetch", "WebSearch", "Agent"]},
            "sandbox": {"enabled": True, "autoAllowBashIfSandboxed": True,
                        "filesystem": {"allowWrite": [str(workspace), str(socket_dir)]},
                        "network": {"allowedDomains": [], "allowAllUnixSockets": True, "allowLocalBinding": False}}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--claude", default=shutil.which("claude"))
    args = parser.parse_args(argv)
    if not args.claude:
        raise SystemExit("Claude Code (`claude`) is not on PATH")
    root = Path(args.root or RUNS / f"explore_{time.strftime('%Y%m%dT%H%M%S')}")
    root.mkdir(parents=True, exist_ok=True)
    workspace = build_workspace(root / "agent-workspace", published=root / "published", episodes=args.episodes)
    home = root / "claude-home"
    home.mkdir(exist_ok=True)
    with Service(root, episodes=args.episodes) as service:
        (home / "mcp.json").write_text(json.dumps({"mcpServers": {"auto_research": {
            "type": "stdio", "command": sys.executable, "args": ["-I", str(ROOT / "rax_microduck" / "bridge.py"), str(service.socket)]}}}))
        (home / "settings.json").write_text(json.dumps(settings(workspace, service.socket.parent), indent=2))
        prompt = (workspace / "START_PROMPT.md").read_text().strip()
        command = [args.claude, "-p", "--mcp-config", str(home / "mcp.json"), "--strict-mcp-config",
                   "--settings", str(home / "settings.json"), "--disallowedTools", "WebFetch", "WebSearch", "Agent",
                   "--output-format", "stream-json", "--verbose"]
        if args.model:
            command += ["--model", args.model]
        if args.max_turns:
            command += ["--max-turns", str(args.max_turns)]
        command += ["--", prompt]
        env = {**os.environ, "RAX_MCP_SOCKET": str(service.socket), "MCP_TOOL_TIMEOUT": "600000",
               "BASH_MAX_TIMEOUT_MS": "600000", "DISABLE_AUTOUPDATER": "1", "PYTHONPATH": str(ROOT)}
        (root / "launch.json").write_text(json.dumps({"command": command, "cwd": str(workspace), "started_at": time.time()}, indent=2))
        started = time.time()
        with (root / "agent.jsonl").open("wb") as out, (root / "agent.stderr").open("wb") as err:
            code = subprocess.call(command, cwd=str(workspace), env=env, stdout=out, stderr=err, stdin=subprocess.DEVNULL)
        summary = {"returncode": code, "wall_seconds": round(time.time() - started, 1), "service_rss_mb": service.peak_rss_mb(),
                   "state": json.loads((root / "state.json").read_text())}
    (root / "launch_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
