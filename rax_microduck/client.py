"""Start the service as a subprocess and talk to it with RAX's api.runtime.Context."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .paths import ROOT
from .service import short_socket_path

RUNTIME = ROOT / "workspace_template" / "api" / "runtime.py"


def load_context_class():
    spec = importlib.util.spec_from_file_location("rax_runtime", RUNTIME)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Context


class Service:
    """The MCP service in its own process, like RAX's frontend; `socket` is its path."""

    def __init__(self, root, *, episodes, collection=0, start_layout=0, render=True, log=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.socket = Path(short_socket_path())
        command = [sys.executable, "-m", "rax_microduck.service", "--root", str(self.root), "--socket", str(self.socket),
                   "--episodes", str(episodes), "--collection", str(collection), "--start-layout", str(start_layout)]
        if not render:
            command.append("--no-render")
        self.log = open(log or self.root / "service.log", "ab")
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=self.log, cwd=str(ROOT),
                                        env={**os.environ, "PYTHONPATH": str(ROOT)})
        line = self.process.stdout.readline().decode()
        if '"ready"' not in line:
            raise RuntimeError(f"service did not start: {line!r}; see {self.log.name}")
        self.ready = json.loads(line)

    def context(self):
        os.environ["RAX_MCP_SOCKET"] = str(self.socket)
        return load_context_class()()

    def peak_rss_mb(self):
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(self.process.pid)], capture_output=True, text=True).stdout
        return round(int(out.strip() or 0) / 1024, 1)

    def stop(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.log.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.stop()


def text(reply):
    return json.loads(next(b["text"] for b in reply["content"] if b["type"] == "text"))
