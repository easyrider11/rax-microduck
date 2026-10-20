import json

import pytest

from rax_microduck.service import Server, Session


@pytest.fixture
def server(tmp_path):
    session = Session(tmp_path / "session", episodes=2, render=False)
    yield Server(session, tmp_path / "robot.sock")
    session.close()


def call(server, name, **arguments):
    reply = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}})
    if "error" in reply:
        return reply["error"]
    return json.loads(reply["result"]["content"][0]["text"])


def test_contract_and_lifecycle(server):
    tools = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"]
    assert {t["name"] for t in tools} == {"status", "start_episode", "evaluate", "finish", "microduck_status",
                                          "microduck_observe", "microduck_step", "microduck_hold", "microduck_pixel_to_position"}
    status = call(server, "status")
    assert status["exploration"]["exploration_remaining"] == 2 and status["episode"] is None
    error = call(server, "microduck_step", twist=[0.2, 0, 0], steps=10)
    assert error["data"]["type"] == "RuntimeError" and "start_episode" in error["message"]
    first = call(server, "start_episode")
    assert first["step_id"] == 0 and first["episode_limits"]["episode_step_limit"] == 1500 and not first["episode_ended"]
    moved = call(server, "microduck_step", twist=[0.2, 0, 0], steps=30)
    assert moved["step_id"] == 30 and moved["transition"]["steps_executed"] == 30
    assert moved["frame_sequence"]["frame_count"] == 4  # steps 10, 20, 30 and the final frame
    assert "success" not in json.dumps(moved) and "layout" not in json.dumps(moved)
    held = call(server, "microduck_hold", steps=5)
    assert held["step_id"] == 35 and held["transition"]["commanded_twist"] == [0.0, 0.0, 0.0]
    bad = call(server, "microduck_step", twist=[0.2, 0, 0], steps=0)
    assert bad["data"]["type"] == "RejectedBeforeAction" and bad["data"]["stage"] == "validation"
    evaluation = call(server, "evaluate")
    assert evaluation["task_complete"] is False
    closed = call(server, "finish")
    assert closed["closed"] and closed["episode_ended"]
    second = call(server, "start_episode")
    assert second["episode_id"] != first["episode_id"]
    assert call(server, "status")["exploration"]["exploration_remaining"] == 0
    call(server, "finish")
    exhausted = call(server, "start_episode")
    assert exhausted["data"]["type"] == "PermissionError"
    state = json.loads((server.session.root / "state.json").read_text())
    assert state["episodes_used"] == 2 and len(state["history"]) == 2
