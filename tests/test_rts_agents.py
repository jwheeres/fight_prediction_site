"""Commander and server tests. Nothing here touches the network."""

from __future__ import annotations

import json
import time
import urllib.request

from rts import config as cfg
from rts.commanders import LLMCommander, ScriptedCommander, make_commander
from rts.match import Match
from rts.server import MatchRunner
from rts.view import build_view


def _view(world=None):
    from rts.engine import World
    world = world or World(seed=1)
    return build_view(world, 0, ["match started"])


def _await(commander, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        decision = commander.poll()
        if decision is not None:
            return decision
        time.sleep(0.01)
    raise AssertionError("commander never answered")


# -- offline behaviour ------------------------------------------------------

def test_no_api_key_falls_back_instead_of_dying():
    commander = LLMCommander("RED", api_key="")
    commander.request(_view())
    decision = _await(commander)
    assert decision.source == "fallback"
    assert "ANTHROPIC_API_KEY" in decision.error
    assert isinstance(decision.orders, list)


def test_api_failure_falls_back_and_reports_why():
    commander = LLMCommander("RED", api_key="sk-test")
    commander._call_api = lambda _: (_ for _ in ()).throw(RuntimeError("HTTP 401: nope"))
    commander.request(_view())
    decision = _await(commander)
    assert decision.source == "fallback"
    assert "HTTP 401" in decision.error


def test_call_budget_stops_spending():
    commander = LLMCommander("RED", api_key="sk-test", max_calls=1)
    commander._call_api = lambda _: json.dumps({"talk": "hi", "orders": []})
    commander.request(_view())
    assert _await(commander).source == "llm"
    commander.request(_view())
    decision = _await(commander)
    assert decision.source == "fallback" and "budget" in decision.error


def test_model_reply_is_parsed_into_orders():
    commander = LLMCommander("RED", api_key="sk-test")
    commander._call_api = lambda _: (
        'Here you go:\n```json\n'
        '{"plan":"boom","talk":"watch this","orders":['
        '{"cmd":"train","unit":"villager","count":3}]}\n```')
    commander.request(_view())
    decision = _await(commander)
    assert decision.source == "llm"
    assert decision.plan == "boom" and decision.talk == "watch this"
    assert decision.orders == [{"cmd": "train", "unit": "villager", "count": 3}]


def test_garbage_reply_falls_back():
    commander = LLMCommander("RED", api_key="sk-test")
    commander._call_api = lambda _: "I would like to build a castle please"
    commander.request(_view())
    assert _await(commander).source == "fallback"


def test_orders_are_capped_even_if_the_model_asks_for_more():
    commander = LLMCommander("RED", api_key="sk-test")
    commander._call_api = lambda _: json.dumps(
        {"orders": [{"cmd": "say", "text": str(i)} for i in range(40)]})
    commander.request(_view())
    assert len(_await(commander).orders) == cfg.MAX_ORDERS_PER_TURN


def test_the_prompt_describes_the_actual_balance_numbers():
    """If the tables and the prompt drift apart the agent plays a game that
    doesn't exist, so the prompt is generated from config."""
    system = LLMCommander("RED", api_key="sk-test").system
    for unit in cfg.UNITS:
        assert unit in system
    assert str(cfg.UNITS["knight"]["cost"]["gold"]) in system
    assert "population" in system.lower()


# -- the sim never waits for an agent ---------------------------------------

def test_a_slow_agent_does_not_stall_the_simulation():
    slow = LLMCommander("RED", api_key="sk-test")
    slow._call_api = lambda _: (time.sleep(0.4) or json.dumps({"orders": []}))
    match = Match([slow, ScriptedCommander("BLUE", "rush")], seed=1, think_interval=10)
    start = time.time()
    for _ in range(300):
        match.step()
    assert match.world.tick == 300
    assert time.time() - start < 2.0, "the sim blocked on the agent"


def test_a_busy_agent_is_not_asked_again():
    commander = LLMCommander("RED", api_key="sk-test")
    commander._call_api = lambda _: (time.sleep(0.3) or json.dumps({"orders": []}))
    commander.request(_view())
    assert commander.busy
    commander.request(_view())  # ignored while thinking
    assert commander.calls == 1


# -- factory ----------------------------------------------------------------

def test_make_commander_specs():
    assert isinstance(make_commander("rush", "RED"), ScriptedCommander)
    assert isinstance(make_commander("llm", "RED"), LLMCommander)
    assert make_commander("llm:a cowardly duke", "RED").persona == "a cowardly duke"
    assert isinstance(make_commander("nonsense-personality", "RED"), ScriptedCommander)


# -- server -----------------------------------------------------------------

def test_runner_snapshot_and_restart():
    runner = MatchRunner("rush", "boom", seed=3)
    for _ in range(50):
        runner.match.step()
    snap = runner.snapshot()
    assert snap["tick"] == 50 and snap["speed"] == 1.0
    json.dumps(snap)
    runner.restart()
    assert runner.match.world.tick == 0 and runner.seed == 4


def test_http_endpoints():
    from http.server import ThreadingHTTPServer
    import threading
    import rts.server as server_module

    runner = MatchRunner("rush", "boom", seed=1, speed=8)
    server_module.RUNNER = runner
    runner.start()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server_module.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        with urllib.request.urlopen(base + "/") as response:
            assert b"<canvas" in response.read()
        with urllib.request.urlopen(base + "/state") as response:
            assert "entities" in json.loads(response.read())
        request = urllib.request.Request(
            base + "/control", method="POST",
            data=json.dumps({"action": "speed", "value": 4}).encode(),
            headers={"content-type": "application/json"})
        with urllib.request.urlopen(request) as response:
            assert json.loads(response.read())["speed"] == 4.0
    finally:
        httpd.shutdown()
        runner.stop()


def test_call_api_against_a_stub_messages_endpoint():
    """Exercises the real HTTP path -- headers, payload shape, response
    parsing -- without talking to Anthropic."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    seen: dict = {}

    class Stub(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_POST(self):
            seen["path"] = self.path
            seen["key"] = self.headers.get("x-api-key")
            seen["version"] = self.headers.get("anthropic-version")
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            body = json.dumps({"content": [{"type": "text", "text": json.dumps(
                {"plan": "expand", "talk": "easy",
                 "orders": [{"cmd": "train", "unit": "villager", "count": 2}]})}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        commander = LLMCommander("RED", api_key="sk-test", model="test-model")
        commander.__dict__["_url"] = None
        import rts.commanders as commanders_module
        original = commanders_module.ANTHROPIC_URL
        commanders_module.ANTHROPIC_URL = f"http://127.0.0.1:{httpd.server_address[1]}/v1/messages"
        try:
            commander.request(_view())
            decision = _await(commander)
        finally:
            commanders_module.ANTHROPIC_URL = original
    finally:
        httpd.shutdown()

    assert decision.source == "llm", decision.error
    assert decision.orders == [{"cmd": "train", "unit": "villager", "count": 2}]
    assert decision.latency_ms >= 0
    assert seen["path"] == "/v1/messages"
    assert seen["key"] == "sk-test" and seen["version"] == "2023-06-01"
    assert seen["body"]["model"] == "test-model"
    assert seen["body"]["messages"][-1]["role"] == "user"
    assert "town centre" in seen["body"]["system"]


def test_llm_commander_can_play_a_whole_match_through_a_stub():
    """A full match driven entirely through the LLM code path.

    The stub reads the situation report and answers with real orders, so
    prompt-building, JSON parsing, order application, the feedback loop and
    the snapshot path all run against agent-shaped input -- with no network.
    """
    scripts = {"RED": ScriptedCommander("RED", "balanced"),
               "BLUE": ScriptedCommander("BLUE", "rush")}

    def reply_for(name):
        def fake_call(user_content):
            decision = scripts[name].decide(json.loads(user_content))
            return ("```json\n" + json.dumps({"plan": decision.plan,
                                              "talk": decision.talk,
                                              "orders": decision.orders}) + "\n```")
        return fake_call

    commanders = [LLMCommander("RED", api_key="sk-test"),
                  LLMCommander("BLUE", api_key="sk-test")]
    for commander in commanders:
        commander._call_api = reply_for(commander.name)
    match = Match(commanders, seed=2, think_interval=30)

    seen: set[str] = set()
    turns = 0
    while not match.world.finished and match.world.tick < cfg.MATCH_TICK_LIMIT:
        match.step()
        for decision in match.last_decision:
            if decision is not None:
                seen.update(decision.results)
                turns += 1
    json.dumps(match.snapshot())

    assert match.world.finished, match.world.tick
    assert turns > 100, "the agents were barely asked for orders"
    assert all(d is not None and d.source == "llm" for d in match.last_decision)
    assert any(r.startswith("training") for r in seen)
    assert any(r.startswith("started building") for r in seen)
    assert any("attacking with" in r for r in seen)
    # And the agents must be told when something did not work.
    assert any(r.startswith("rejected") for r in seen), seen
