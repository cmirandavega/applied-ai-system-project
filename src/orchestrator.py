"""
Agent orchestration for VibeMatch 2.0.

This is an *alternate* path to the fixed pipeline (extract_profile →
retrieve_candidates → hardcoded needs_expansion → expand_catalog →
recommend_songs). Instead of a hardcoded branch deciding whether to expand the
catalog, an LLM tool-calling loop decides — turn by turn — whether to call:

    • search_catalog       – filter the existing catalog for candidates
    • expand_catalog_tool  – generate & append new songs, then re-search
    • rank_songs           – score candidates and produce the final top-k

The model chooses which tools to call and in what order. The fixed pipeline in
app.py remains the default; this is purely additive.

Safety: ``max_turns`` is a hard cap on the number of Groq API calls per run. If
it is reached before the model produces a final ranking, we fall back to ranking
whatever candidates we have and log a warning so runaway loops are impossible.

Testability: the Groq client is injectable (``client=``) so tests can mock it
and never touch the real API. The catalog / expansion helpers are called through
their modules (``retriever``, ``expander``, ``recommender``) so tests can
monkeypatch ``expander.expand_catalog`` (the only helper that hits an API).
"""
import json
import os
from typing import List, Dict, Tuple, Optional

from src import retriever
from src import expander
from src import recommender
from src import logger as rec_logger

# Tool-calling capable Groq model. Overridable via env for experimentation.
MODEL = os.environ.get("GROQ_ORCHESTRATOR_MODEL", "llama-3.3-70b-versatile")

# Hard cap on Groq API calls per orchestration run. Prevents runaway loops.
MAX_TURNS = 6

# Below this many candidates, the model is nudged (via the system prompt) to
# consider expanding the catalog. This is guidance for the model, not a branch.
MIN_GOOD_CANDIDATES = 6


# ── Tool schemas (OpenAI / Groq function-calling format) ─────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_catalog",
            "description": (
                "Search the EXISTING song catalog for candidates matching the "
                "user's profile (genre, mood, energy). Always call this FIRST. "
                "Returns how many candidate songs were found."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "expand_catalog_tool",
            "description": (
                "Generate NEW songs that match the user's profile, append them to "
                "the catalog, then re-search. Only call this if search_catalog "
                f"returned too few candidates (fewer than {MIN_GOOD_CANDIDATES}) "
                "to produce good recommendations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "integer",
                        "description": "How many new songs to generate (1-5).",
                        "minimum": 1,
                        "maximum": 5,
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rank_songs",
            "description": (
                "Score the current candidate songs against the user's profile and "
                "produce the FINAL top-k ranked recommendations. Call this LAST, "
                "once you have enough candidates. This ends the run."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "k": {
                        "type": "integer",
                        "description": "How many recommendations to return.",
                    }
                },
                "required": [],
            },
        },
    },
]


def _default_client():
    """Construct the real Groq client. Imported lazily so this module (and the
    tests that mock the client) don't require the groq package or an API key."""
    from groq import Groq

    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not set — agent orchestration requires the Groq backend."
        )
    return Groq(api_key=key)


class AgentOrchestrator:
    """Runs the tool-calling loop for a single recommendation request."""

    def __init__(
        self,
        profile: Dict,
        all_songs: List[Dict],
        catalog_path: str,
        k: int = 5,
        client=None,
        model: str = MODEL,
        max_turns: int = MAX_TURNS,
    ):
        self.profile = profile
        self.all_songs = list(all_songs)
        self.catalog_path = catalog_path
        self.k = k
        self.client = client if client is not None else _default_client()
        self.model = model
        self.max_turns = max_turns

        # Mutable run state
        self.candidates: List[Dict] = []
        self.results: Optional[List[Tuple[Dict, float, str]]] = None
        self.trace: List[Dict] = []
        self.api_calls: int = 0
        self.hit_cap: bool = False

    # ── Prompts ──────────────────────────────────────────────────────────────
    def _system_prompt(self) -> str:
        return (
            "You are the orchestrator for a music recommender. You have three "
            "tools: search_catalog, expand_catalog_tool, and rank_songs. Your job "
            "is to produce a good final ranked list for the user.\n\n"
            "Guidelines:\n"
            "1. ALWAYS call search_catalog first to see how many candidates the "
            "existing catalog offers.\n"
            f"2. If search_catalog returns fewer than {MIN_GOOD_CANDIDATES} "
            "candidates, call expand_catalog_tool once to generate more, then you "
            "may search again if useful.\n"
            f"3. If search_catalog already returns {MIN_GOOD_CANDIDATES} or more "
            "candidates, do NOT expand — go straight to ranking.\n"
            "4. When you have enough candidates, call rank_songs to produce the "
            "final recommendations. Do not call any tool after rank_songs.\n"
            "Be efficient: use as few tool calls as possible."
        )

    def _user_prompt(self) -> str:
        p = self.profile
        return (
            "Produce the top recommendations for this user profile:\n"
            f"  favorite_genre: {p.get('favorite_genre')}\n"
            f"  favorite_mood:  {p.get('favorite_mood')}\n"
            f"  target_energy:  {p.get('target_energy')}\n"
            f"  likes_acoustic: {p.get('likes_acoustic')}\n"
            f"The catalog currently has {len(self.all_songs)} songs. "
            f"Return {self.k} recommendations."
        )

    # ── Tool implementations ──────────────────────────────────────────────────
    def _tool_search_catalog(self, args: Dict) -> Dict:
        self.candidates = retriever.retrieve_candidates(self.profile, self.all_songs)
        return {
            "candidate_count": len(self.candidates),
            "catalog_size": len(self.all_songs),
            "enough": len(self.candidates) >= MIN_GOOD_CANDIDATES,
            "sample": [f"{s['title']} — {s['artist']}" for s in self.candidates[:5]],
        }

    def _tool_expand_catalog(self, args: Dict) -> Dict:
        try:
            target = int(args.get("target", 3) or 3)
        except (TypeError, ValueError):
            target = 3
        target = max(1, min(5, target))

        new_songs, count = expander.expand_catalog(
            self.profile, self.all_songs, self.catalog_path, target=target
        )
        # Reload from disk so the appended songs are represented consistently,
        # then re-run retrieval over the enlarged catalog.
        self.all_songs = recommender.load_songs(self.catalog_path)
        self.candidates = retriever.retrieve_candidates(self.profile, self.all_songs)
        return {
            "new_songs_added": count,
            "new_titles": [f"{s['title']} — {s['artist']}" for s in new_songs],
            "candidate_count": len(self.candidates),
            "catalog_size": len(self.all_songs),
        }

    def _tool_rank_songs(self, args: Dict) -> Dict:
        # `k or self.k` would treat an explicit k=0 as "missing" and silently
        # substitute self.k — check for None instead so k=0 is honored.
        raw_k = args.get("k", self.k)
        try:
            k = int(raw_k) if raw_k is not None else self.k
        except (TypeError, ValueError):
            k = self.k
        pool = self.candidates if self.candidates else self.all_songs
        self.results = recommender.recommend_songs(self.profile, pool, k=k)
        return {
            "ranked": [
                {"title": s["title"], "artist": s["artist"], "score": round(score, 4)}
                for s, score, _ in self.results
            ]
        }

    def _dispatch(self, name: str, args: Dict) -> Dict:
        if name == "search_catalog":
            return self._tool_search_catalog(args)
        if name == "expand_catalog_tool":
            return self._tool_expand_catalog(args)
        if name == "rank_songs":
            return self._tool_rank_songs(args)
        return {"error": f"unknown tool: {name}"}

    # ── Main loop ──────────────────────────────────────────────────────────────
    def run(self) -> Tuple[List[Tuple[Dict, float, str]], List[Dict]]:
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": self._user_prompt()},
        ]

        # Assume the cap was reached until we break out cleanly.
        cap_reached = True

        while self.api_calls < self.max_turns:
            self.api_calls += 1
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0,
            )
            msg = response.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None)

            if not tool_calls:
                # Model answered with text and no tool call → it considers itself done.
                self.trace.append(
                    {
                        "turn": self.api_calls,
                        "type": "final_message",
                        "content": (getattr(msg, "content", "") or "").strip(),
                    }
                )
                cap_reached = False
                break

            # Echo the assistant's tool-call message back into the conversation.
            messages.append(
                {
                    "role": "assistant",
                    "content": getattr(msg, "content", "") or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )

            reached_terminal = False
            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except (json.JSONDecodeError, TypeError):
                    args = {}

                result = self._dispatch(name, args)
                self.trace.append(
                    {
                        "turn": self.api_calls,
                        "type": "tool_call",
                        "tool": name,
                        "args": args,
                        "result": result,
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": name,
                        "content": json.dumps(result),
                    }
                )
                if name == "rank_songs":
                    reached_terminal = True

            if reached_terminal:
                cap_reached = False
                break

        # If the while condition went false without a clean break, the cap was hit.
        if cap_reached:
            self.hit_cap = True
            self.trace.append(
                {
                    "type": "cap_hit",
                    "reason": "MAX_TURNS reached",
                    "api_calls": self.api_calls,
                    "max_turns": self.max_turns,
                }
            )
            rec_logger.log_orchestrator_warning(
                "MAX_TURNS reached before rank_songs", self.api_calls, self.max_turns
            )

        # Ensure the user always gets *something*: if the model never ranked,
        # rank whatever candidates we have (or the full catalog).
        if self.results is None:
            pool = self.candidates if self.candidates else self.all_songs
            self.results = recommender.recommend_songs(self.profile, pool, k=self.k)
            self.trace.append(
                {
                    "type": "fallback",
                    "reason": "no rank_songs call — ranked current candidate pool",
                    "pool_size": len(pool),
                    "ranked_count": len(self.results),
                }
            )

        rec_logger.log_orchestrator_run(self.profile, self.results, self.trace, self.api_calls)
        return self.results, self.trace


def run_orchestration(
    profile: Dict,
    all_songs: List[Dict],
    catalog_path: str,
    k: int = 5,
    client=None,
    model: str = MODEL,
    max_turns: int = MAX_TURNS,
) -> Tuple[List[Tuple[Dict, float, str]], List[Dict]]:
    """Convenience wrapper: build an orchestrator and run it.

    Returns ``(results, trace)`` where ``results`` matches recommend_songs'
    output — a list of ``(song_dict, score, explanation_string)`` tuples — and
    ``trace`` is the ordered list of the agent's tool-call decisions.
    """
    orch = AgentOrchestrator(
        profile,
        all_songs,
        catalog_path,
        k=k,
        client=client,
        model=model,
        max_turns=max_turns,
    )
    return orch.run()
