"""
Tests for the agent orchestrator (src/orchestrator.py).

The Groq client is fully mocked — these tests never make a real API call. Each
test scripts the sequence of tool calls the model "decides" to make, then asserts
the orchestrator drove the catalog/expansion/ranking helpers correctly.

Covered:
  • normal run   — search_catalog finds enough candidates, model skips expansion
  • thin catalog — model calls expand_catalog_tool before ranking
  • MAX_TURNS    — model never ranks; cap is enforced and a warning is logged

Failure modes / edge cases (model or environment misbehaving):
  • hallucinated tool name        — _dispatch returns an error payload, loop continues
  • rank_songs with no search     — falls back to ranking the full catalog
  • malformed tool-call arguments — invalid JSON args degrade to {} instead of raising
  • Groq API error mid-loop       — propagates so app.py's try/except can surface it
  • expansion adds zero songs     — still ranks whatever candidates exist
  • duplicate tool calls          — candidates are reassigned, not accumulated
  • empty catalog                 — every stage degrades to an empty result cleanly
  • k=0 requested                 — returns an empty list, not the default k
  • trace entry shapes            — locks in the keys app.py's UI renderer depends on
  • batched tool calls            — parallel_tool_calls=False is requested, but if the
                                     model still returns 2+ tool_calls in one turn only
                                     the first executes; the rest are discarded + logged
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


# ── Test 5: hallucinated / unknown tool name ──────────────────────────────────
def test_model_calls_unknown_tool(monkeypatch):
    """A hallucinated/invalid tool name must not crash the loop — _dispatch
    returns an error payload and the orchestrator keeps going until a real
    terminal tool (rank_songs) is called."""
    catalog = [make_song(i, f"Song {i}") for i in range(1, 9)]

    client = FakeGroqClient(
        [
            [("delete_catalog", {"confirm": True})],  # not in TOOLS
            [("rank_songs", {"k": 5})],
        ]
    )

    results, trace = orchestrator.run_orchestration(
        PROFILE, catalog, "unused.csv", k=5, client=client
    )

    bad_entry = next(e for e in trace if e.get("tool") == "delete_catalog")
    assert bad_entry["result"] == {"error": "unknown tool: delete_catalog"}
    assert client.calls == 2, "the loop must continue past the bad call to rank_songs"
    assert len(results) == 5


# ── Test 6: rank_songs called without search_catalog first ───────────────────
def test_rank_songs_called_without_search_first(monkeypatch):
    """If the model skips search_catalog and calls rank_songs immediately,
    self.candidates is still empty — the tool must fall back to ranking the
    full catalog (self.all_songs) rather than returning nothing or crashing."""
    catalog = [make_song(i, f"Song {i}") for i in range(1, 9)]

    client = FakeGroqClient([[("rank_songs", {"k": 5})]])

    results, trace = orchestrator.run_orchestration(
        PROFILE, catalog, "unused.csv", k=5, client=client
    )

    assert client.calls == 1
    tools_used = [e["tool"] for e in trace if e["type"] == "tool_call"]
    assert tools_used == ["rank_songs"]
    assert len(results) == 5, "should have ranked against the full catalog, not nothing"


# ── Test 7: malformed JSON in tool-call arguments ─────────────────────────────
def test_malformed_tool_arguments(monkeypatch):
    """Invalid JSON in tool_call.function.arguments (e.g. a truncated/unquoted
    payload) must be caught and degrade to {} rather than raising an uncaught
    JSONDecodeError."""
    catalog = [make_song(i, f"Song {i}") for i in range(1, 9)]

    class _BadArgsClient:
        def __init__(self):
            self.calls = 0
            self.chat = self
            self.completions = self

        def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                bad_call = _FakeToolCall("call_0", "search_catalog", "{k: 5")
                return _FakeResponse(_FakeMessage(content="", tool_calls=[bad_call]))
            rank_call = _FakeToolCall("call_1", "rank_songs", json.dumps({"k": 5}))
            return _FakeResponse(_FakeMessage(content="", tool_calls=[rank_call]))

    client = _BadArgsClient()
    results, trace = orchestrator.run_orchestration(
        PROFILE, catalog, "unused.csv", k=5, client=client
    )

    assert client.calls == 2
    search_entry = next(e for e in trace if e.get("tool") == "search_catalog")
    assert search_entry["args"] == {}, "malformed JSON must degrade to an empty dict"
    assert len(results) == 5


# ── Test 8: Groq API error mid-loop ────────────────────────────────────────────
def test_groq_api_error_mid_loop(monkeypatch):
    """If the Groq client raises mid-loop (rate limit, connection error, etc.),
    the orchestrator does not swallow it — it propagates so app.py's
    try/except around run_orchestration can surface a clean st.error(). This
    locks in that the two stay in sync: orchestrator raises, UI catches."""
    catalog = [make_song(i, f"Song {i}") for i in range(1, 9)]

    class _FlakyClient:
        def __init__(self):
            self.calls = 0
            self.chat = self
            self.completions = self

        def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                tc = _FakeToolCall("call_0", "search_catalog", json.dumps({}))
                return _FakeResponse(_FakeMessage(content="", tool_calls=[tc]))
            raise RuntimeError("rate limit exceeded")

    client = _FlakyClient()
    with pytest.raises(RuntimeError, match="rate limit exceeded"):
        orchestrator.run_orchestration(PROFILE, catalog, "unused.csv", k=5, client=client)
    assert client.calls == 2


# ── Test 9: expansion succeeds but adds zero new songs ────────────────────────
def test_expand_returns_zero_new_songs(monkeypatch):
    """expand_catalog succeeding but finding zero new matching songs is a real
    possible outcome (not an error) — the orchestrator must still produce a
    ranked result from whatever candidates exist, without looping or crashing."""
    catalog = [make_song(1, "Lonely Track")]  # thin catalog

    def _fake_expand(profile, existing, csv_path, target=3):
        return ([], 0)

    monkeypatch.setattr(expander, "expand_catalog", _fake_expand)
    monkeypatch.setattr(recommender, "load_songs", lambda path: catalog)  # unchanged

    client = FakeGroqClient(
        [
            [("search_catalog", {})],
            [("expand_catalog_tool", {"target": 3})],
            [("rank_songs", {"k": 5})],
        ]
    )

    results, trace = orchestrator.run_orchestration(
        PROFILE, catalog, "unused.csv", k=5, client=client
    )

    expand_entry = next(e for e in trace if e.get("tool") == "expand_catalog_tool")
    assert expand_entry["result"]["new_songs_added"] == 0
    assert client.calls == 3
    assert len(results) == 1, "only the original song was ever available to rank"


# ── Test 10: duplicate tool call despite the system prompt's instruction ─────
def test_duplicate_tool_call_despite_instruction(monkeypatch):
    """The system prompt asks the model to call search_catalog once, but
    nothing enforces that. A duplicate call must not accumulate state
    (candidates get reassigned, not appended) and must not blow past
    MAX_TURNS unexpectedly."""
    catalog = [make_song(i, f"Song {i}") for i in range(1, 9)]

    client = FakeGroqClient(
        [
            [("search_catalog", {})],
            [("search_catalog", {})],  # duplicate, against instructions
            [("rank_songs", {"k": 5})],
        ]
    )

    results, trace = orchestrator.run_orchestration(
        PROFILE, catalog, "unused.csv", k=5, client=client, max_turns=6
    )

    search_entries = [e for e in trace if e.get("tool") == "search_catalog"]
    assert len(search_entries) == 2
    assert (
        search_entries[0]["result"]["candidate_count"]
        == search_entries[1]["result"]["candidate_count"]
    ), "repeating search_catalog must reassign candidates, not accumulate them"
    assert client.calls == 3
    assert client.calls <= 6, "duplicate calls must not blow past MAX_TURNS"
    assert len(results) == 5


# ── Test 11: empty catalog ────────────────────────────────────────────────────
def test_empty_catalog(monkeypatch):
    """An empty catalog must not raise anywhere in the tool chain — retrieval,
    ranking, and dispatch should all degrade to an empty result cleanly."""
    client = FakeGroqClient(
        [
            [("search_catalog", {})],
            [("rank_songs", {"k": 5})],
        ]
    )

    results, trace = orchestrator.run_orchestration(
        PROFILE, [], "unused.csv", k=5, client=client
    )

    assert results == []
    search_entry = next(e for e in trace if e.get("tool") == "search_catalog")
    assert search_entry["result"]["candidate_count"] == 0
    assert search_entry["result"]["catalog_size"] == 0


# ── Test 12: rank_songs called with k=0 ───────────────────────────────────────
def test_zero_recommendations_requested(monkeypatch):
    """rank_songs invoked with k=0 must return an empty list. (Guards against
    a real bug where `k or self.k` treated an explicit 0 as missing and
    silently substituted the default k instead.)"""
    catalog = [make_song(i, f"Song {i}") for i in range(1, 9)]

    client = FakeGroqClient(
        [
            [("search_catalog", {})],
            [("rank_songs", {"k": 0})],
        ]
    )

    results, trace = orchestrator.run_orchestration(
        PROFILE, catalog, "unused.csv", k=5, client=client
    )

    assert results == []
    rank_entry = next(e for e in trace if e.get("tool") == "rank_songs")
    assert rank_entry["result"]["ranked"] == []


# ── Test 13: trace entry structure for every entry type ──────────────────────
def test_trace_structure_for_each_tool_type(monkeypatch):
    """Lock in the trace entry shape for every entry type so a future refactor
    can't silently break the UI's trace rendering — app.py branches on
    entry['type'] and reads specific keys per type."""
    catalog = [make_song(1, "Lonely Track")]
    enlarged = catalog + [make_song(i, f"Generated {i}") for i in range(2, 8)]
    new_songs = enlarged[1:]

    monkeypatch.setattr(
        expander, "expand_catalog", lambda *a, **k: (new_songs, len(new_songs))
    )
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

    tool_call_entries = [e for e in trace if e["type"] == "tool_call"]
    assert len(tool_call_entries) == 3
    for entry in tool_call_entries:
        assert set(entry.keys()) == {"turn", "type", "tool", "args", "result"}
        assert isinstance(entry["turn"], int)
        assert isinstance(entry["tool"], str)
        assert isinstance(entry["args"], dict)
        assert isinstance(entry["result"], dict)

    # A run that hits MAX_TURNS must produce well-formed cap_hit + fallback entries.
    cap_client = FakeGroqClient([[("search_catalog", {})]])
    _, cap_trace = orchestrator.run_orchestration(
        PROFILE, catalog, "unused.csv", k=5, client=cap_client, max_turns=2
    )
    cap_entry = next(e for e in cap_trace if e["type"] == "cap_hit")
    assert set(cap_entry.keys()) == {"type", "reason", "api_calls", "max_turns"}
    fallback_entry = next(e for e in cap_trace if e["type"] == "fallback")
    assert set(fallback_entry.keys()) == {"type", "reason", "pool_size", "ranked_count"}

    # A run ending in plain text (no tool call) must produce a well-formed
    # final_message entry.
    text_client = FakeGroqClient(
        [[("search_catalog", {})], "All done, no more tools needed."]
    )
    _, text_trace = orchestrator.run_orchestration(
        PROFILE, catalog, "unused.csv", k=5, client=text_client
    )
    final_entry = next(e for e in text_trace if e["type"] == "final_message")
    assert set(final_entry.keys()) == {"turn", "type", "content"}
    assert isinstance(final_entry["content"], str)


# ── Test 14: model batches multiple tool_calls despite parallel_tool_calls=False ──
def test_forces_sequential_tool_calls(monkeypatch, tmp_path):
    """parallel_tool_calls=False is requested on every Groq call, but nothing
    guarantees the model honors it — it can still legally return 2+ tool_calls
    in a single message. Only the first must execute; the rest are discarded
    and a warning is logged, so a run can never silently rank on unobserved
    state (the exact bug this guards against)."""
    catalog = [make_song(i, f"Song {i}") for i in range(1, 9)]

    log_file = tmp_path / "batch_warning.log"
    monkeypatch.setattr(logger_module, "LOG_FILE", str(log_file))
    monkeypatch.setattr(logger_module, "_configured", False)

    client = FakeGroqClient(
        [
            # Turn 1: the model illegally batches search_catalog + rank_songs
            # together. rank_songs must NOT run here — it hasn't seen the
            # search result yet.
            [("search_catalog", {}), ("rank_songs", {"k": 5})],
            # Turn 2: now that search_catalog's result is visible, the model
            # (correctly) ranks.
            [("rank_songs", {"k": 5})],
        ]
    )

    results, trace = orchestrator.run_orchestration(
        PROFILE, catalog, "unused.csv", k=5, client=client
    )

    # Only search_catalog from turn 1 actually executed.
    turn1_tool_calls = [e for e in trace if e["type"] == "tool_call" and e["turn"] == 1]
    assert len(turn1_tool_calls) == 1
    assert turn1_tool_calls[0]["tool"] == "search_catalog"

    # The discarded rank_songs call is recorded, not silently dropped.
    discard_entry = next(e for e in trace if e["type"] == "batched_tool_calls_discarded")
    assert discard_entry["turn"] == 1
    assert discard_entry["kept"] == "search_catalog"
    assert discard_entry["discarded"] == ["rank_songs"]

    # The run took a genuine second turn to rank — it didn't rank on turn 1's
    # unobserved batch.
    assert client.calls == 2
    assert len(results) == 5

    content = log_file.read_text(encoding="utf-8")
    assert "[AGENT]" in content
    assert "batched" in content.lower()
