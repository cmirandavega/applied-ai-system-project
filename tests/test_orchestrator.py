"""
Tests for the agent orchestrator (src/orchestrator.py).

The Groq client is fully mocked — these tests never make a real API call. Each
test scripts the sequence of tool calls the model "decides" to make, then asserts
the orchestrator drove the catalog/expansion/ranking helpers correctly.

Covered:
  • normal run   — search_catalog finds enough candidates, model skips expansion
  • thin catalog — model calls expand_catalog_tool before ranking
  • MAX_TURNS    — model never ranks; cap is enforced and a warning is logged
"""
import json
import os
import sys

import pytest

# Make sure the project root is on sys.path when running from any directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import orchestrator
from src import expander
from src import recommender
from src import logger as logger_module


# ── Fake Groq client ──────────────────────────────────────────────────────────
class _FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments  # JSON string, like the real API


class _FakeToolCall:
    def __init__(self, id, name, arguments):
        self.id = id
        self.type = "function"
        self.function = _FakeFunction(name, arguments)


class _FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, message):
        self.message = message


class _FakeResponse:
    def __init__(self, message):
        self.choices = [_FakeChoice(message)]


class FakeGroqClient:
    """Scripts a sequence of assistant turns.

    Each element of ``script`` is either:
      • a list of ``(tool_name, args_dict)`` tuples → the model calls those tools
      • a ``str`` → the model returns a plain text message (no tool call)

    Once the script is exhausted, the LAST turn repeats (so a one-element script
    of a tool call simulates a model that loops forever without ranking).
    """

    def __init__(self, script):
        self.script = script
        self.calls = 0
        self.received_messages = []
        # Expose the chat.completions.create(...) surface the orchestrator uses.
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.received_messages.append(kwargs.get("messages"))
        idx = self.calls
        self.calls += 1
        turn = self.script[idx] if idx < len(self.script) else self.script[-1]

        if isinstance(turn, str):
            return _FakeResponse(_FakeMessage(content=turn, tool_calls=None))

        tool_calls = [
            _FakeToolCall(f"call_{idx}_{i}", name, json.dumps(args))
            for i, (name, args) in enumerate(turn)
        ]
        return _FakeResponse(_FakeMessage(content="", tool_calls=tool_calls))


# ── Fixtures / helpers ──────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def isolate_log(monkeypatch, tmp_path):
    """Redirect the log file to a temp path for every test so runs don't pollute
    the repo's logs/recommender.log. Tests that need to read the log override
    LOG_FILE themselves after this runs."""
    monkeypatch.setattr(logger_module, "LOG_FILE", str(tmp_path / "default.log"))
    monkeypatch.setattr(logger_module, "_configured", False)


def make_song(id, title, genre="pop", mood="happy", energy=0.8):
    return {
        "id": id,
        "title": title,
        "artist": f"Artist {id}",
        "genre": genre,
        "mood": mood,
        "energy": energy,
        "tempo_bpm": 120.0,
        "valence": 0.7,
        "danceability": 0.6,
        "acousticness": 0.2,
    }


PROFILE = {
    "favorite_genre": "pop",
    "favorite_mood": "happy",
    "target_energy": 0.8,
    "likes_acoustic": False,
}


# ── Test 1: normal run, model skips expansion ─────────────────────────────────
def test_normal_run_skips_expansion(monkeypatch):
    """search_catalog returns plenty of candidates → model goes straight to rank."""
    # 8 matching songs → retrieve_candidates returns >= MIN_GOOD_CANDIDATES (6).
    catalog = [make_song(i, f"Song {i}") for i in range(1, 9)]

    # Guard: expand must NOT be called on this path.
    def _fail_expand(*a, **k):
        raise AssertionError("expand_catalog should not be called on the normal path")

    monkeypatch.setattr(expander, "expand_catalog", _fail_expand)

    client = FakeGroqClient(
        [
            [("search_catalog", {})],
            [("rank_songs", {"k": 5})],
        ]
    )

    results, trace = orchestrator.run_orchestration(
        PROFILE, catalog, "unused.csv", k=5, client=client
    )

    assert client.calls == 2, "expected exactly 2 Groq calls (search + rank)"
    assert len(results) == 5
    tools_used = [e["tool"] for e in trace if e["type"] == "tool_call"]
    assert tools_used == ["search_catalog", "rank_songs"]
    assert "expand_catalog_tool" not in tools_used
    # search_catalog should report that the catalog had enough candidates.
    search_entry = next(e for e in trace if e.get("tool") == "search_catalog")
    assert search_entry["result"]["enough"] is True


# ── Test 2: thin catalog, model expands before ranking ────────────────────────
def test_thin_catalog_triggers_expansion(monkeypatch):
    """A near-empty catalog → model calls expand_catalog_tool, then ranks."""
    catalog = [make_song(1, "Lonely Track")]  # only 1 song → thin

    # After expansion, the reloaded catalog should contain the new songs.
    enlarged = catalog + [make_song(i, f"Generated {i}") for i in range(2, 8)]
    new_songs = enlarged[1:]

    expand_calls = {"n": 0}

    def _fake_expand(profile, existing, csv_path, target=3):
        expand_calls["n"] += 1
        return (new_songs, len(new_songs))

    # Orchestrator reloads via recommender.load_songs after expanding.
    monkeypatch.setattr(expander, "expand_catalog", _fake_expand)
    monkeypatch.setattr(recommender, "load_songs", lambda path: enlarged)

    client = FakeGroqClient(
        [
            [("search_catalog", {})],
            [("expand_catalog_tool", {"target": 5})],
            [("rank_songs", {"k": 5})],
        ]
    )

    results, trace = orchestrator.run_orchestration(
        PROFILE, catalog, "unused.csv", k=5, client=client
    )

    assert expand_calls["n"] == 1, "expand_catalog should be called exactly once"
    assert client.calls == 3
    tools_used = [e["tool"] for e in trace if e["type"] == "tool_call"]
    assert tools_used == ["search_catalog", "expand_catalog_tool", "rank_songs"]
    # Expansion should have grown the candidate pool the ranking drew from.
    expand_entry = next(e for e in trace if e.get("tool") == "expand_catalog_tool")
    assert expand_entry["result"]["new_songs_added"] == len(new_songs)
    assert len(results) == 5


# ── Test 3: MAX_TURNS cap is enforced and logged ──────────────────────────────
def test_max_turns_cap_and_fallback(monkeypatch, tmp_path):
    """Model loops on search_catalog forever → cap stops it, fallback ranks,
    and a WARNING is written to the log."""
    catalog = [make_song(i, f"Song {i}") for i in range(1, 9)]

    # Redirect the log file so we can assert the warning was written.
    log_file = tmp_path / "orchestrator_test.log"
    monkeypatch.setattr(logger_module, "LOG_FILE", str(log_file))
    monkeypatch.setattr(logger_module, "_configured", False)

    # Model always asks to search again — never ranks.
    client = FakeGroqClient([[("search_catalog", {})]])

    results, trace = orchestrator.run_orchestration(
        PROFILE, catalog, "unused.csv", k=5, client=client, max_turns=3
    )

    # Hard cap honored: exactly max_turns Groq calls, no more.
    assert client.calls == 3, "MAX_TURNS must cap the number of Groq API calls"

    # The run must still produce results via the fallback ranking.
    assert len(results) == 5
    assert any(e["type"] == "cap_hit" for e in trace)
    assert any(e["type"] == "fallback" for e in trace)

    # A warning tagged [AGENT] must be in the log.
    logger_module.log_orchestrator_warning  # sanity: symbol exists
    content = log_file.read_text(encoding="utf-8")
    assert "[AGENT]" in content
    assert "Safety cap hit" in content


# ── Test 4: object-level hit_cap flag and run log ─────────────────────────────
def test_orchestrator_object_state(monkeypatch, tmp_path):
    """The AgentOrchestrator exposes hit_cap and writes a tagged run log entry."""
    catalog = [make_song(i, f"Song {i}") for i in range(1, 9)]

    log_file = tmp_path / "run.log"
    monkeypatch.setattr(logger_module, "LOG_FILE", str(log_file))
    monkeypatch.setattr(logger_module, "_configured", False)

    client = FakeGroqClient([[("search_catalog", {})], [("rank_songs", {})]])
    orch = orchestrator.AgentOrchestrator(
        PROFILE, catalog, "unused.csv", k=3, client=client
    )
    results, trace = orch.run()

    assert orch.hit_cap is False
    assert orch.api_calls == 2
    assert len(results) == 3
    content = log_file.read_text(encoding="utf-8")
    assert "[AGENT]" in content
    assert "Tools:" in content
