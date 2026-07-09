"""
Reliability tests for the VibeMatch 2.0 agent, retriever, and logger modules.
"""
import os
import sys

import pytest

# Make sure the project root is on sys.path when running from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import extract_profile, DEFAULT_PROFILE, VALID_GENRES, VALID_MOODS
from src.retriever import retrieve_candidates
from src.recommender import load_songs, recommend_songs
import src.logger as logger_module


# ---------------------------------------------------------------------------
# Agent tests
# ---------------------------------------------------------------------------

def test_agent_consistency():
    """Same input → same genre + mood across 5 consecutive calls (temperature=0)."""
    user_input = "I want something chill and acoustic for late-night studying"
    results = [extract_profile(user_input) for _ in range(5)]
    genres = [r["favorite_genre"] for r in results]
    moods = [r["favorite_mood"] for r in results]
    assert all(g == genres[0] for g in genres), f"Genres inconsistent across runs: {genres}"
    assert all(m == moods[0] for m in moods), f"Moods inconsistent across runs: {moods}"


def test_empty_input_returns_default():
    assert extract_profile("") == DEFAULT_PROFILE


def test_whitespace_only_input_returns_default():
    assert extract_profile("   \t\n  ") == DEFAULT_PROFILE


def test_nonsense_input_returns_valid_profile():
    """Garbage input must still yield a structurally valid, in-range profile."""
    result = extract_profile("asdfghjkl qwerty 12345 !@#$%")
    assert set(result.keys()) == {"favorite_genre", "favorite_mood", "target_energy", "likes_acoustic"}
    assert result["favorite_genre"] in VALID_GENRES
    assert result["favorite_mood"] in VALID_MOODS
    assert 0.0 <= result["target_energy"] <= 1.0
    assert isinstance(result["likes_acoustic"], bool)


# ---------------------------------------------------------------------------
# Retriever tests
# ---------------------------------------------------------------------------

def test_rag_retrieves_more_than_one_candidate_for_niche_genre():
    """
    latin/playful/0.71 matches at least one song on each filter axis; the
    energy window (±0.25) also pulls in several other tracks — so the filtered
    list should contain more than one song.
    """
    songs = load_songs("data/songs.csv")
    user_prefs = {
        "favorite_genre": "latin",
        "favorite_mood": "playful",
        "target_energy": 0.71,
        "likes_acoustic": False,
    }
    candidates = retrieve_candidates(user_prefs, songs)
    assert len(candidates) > 1, (
        f"Expected > 1 candidate for latin/playful/0.71, got {len(candidates)}"
    )


def test_rag_always_includes_exact_genre_match():
    """
    rock/intense/0.91 — Storm Runner is an exact match on all three axes and
    must appear in the candidate set.
    """
    songs = load_songs("data/songs.csv")
    user_prefs = {
        "favorite_genre": "rock",
        "favorite_mood": "intense",
        "target_energy": 0.91,
        "likes_acoustic": False,
    }
    candidates = retrieve_candidates(user_prefs, songs)
    genres = [s["genre"] for s in candidates]
    assert "rock" in genres, (
        f"Expected 'rock' in candidate genres, got: {genres}"
    )


# ---------------------------------------------------------------------------
# Logger tests
# ---------------------------------------------------------------------------

def test_log_file_is_written_after_request(monkeypatch, tmp_path):
    """
    Redirect LOG_FILE to a temp path, reset _configured, run the full
    log_request pipeline, then assert the log file exists and contains
    the original input string.
    """
    log_file = tmp_path / "test_recommender.log"

    monkeypatch.setattr(logger_module, "LOG_FILE", str(log_file))
    monkeypatch.setattr(logger_module, "_configured", False)

    profile = {
        "favorite_genre": "pop",
        "favorite_mood": "happy",
        "target_energy": 0.8,
        "likes_acoustic": False,
    }

    songs = load_songs("data/songs.csv")
    results = recommend_songs(profile, songs, k=3)

    user_input = "test log input"
    logger_module.log_request(user_input, profile, results)

    assert log_file.exists(), "Log file was not created"
    content = log_file.read_text(encoding="utf-8")
    assert user_input in content, (
        f"Expected {user_input!r} in log file, got:\n{content}"
    )
