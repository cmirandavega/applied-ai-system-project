from typing import List, Dict

ENERGY_WINDOW = 0.25
MIN_CANDIDATES = 6


def retrieve_candidates(user_prefs: dict, songs: List[Dict]) -> List[Dict]:
    if not songs:
        return []

    seen_ids: set = set()
    candidates: List[Dict] = []

    for song in songs:
        genre_match = song["genre"] == user_prefs["favorite_genre"]
        mood_match = song["mood"] == user_prefs["favorite_mood"]
        energy_match = abs(song["energy"] - user_prefs["target_energy"]) <= ENERGY_WINDOW

        if genre_match or mood_match or energy_match:
            if song["id"] not in seen_ids:
                seen_ids.add(song["id"])
                candidates.append(song)

    if len(candidates) < MIN_CANDIDATES:
        return songs

    return candidates


def retrieval_summary(user_prefs: dict, songs: List[Dict], candidates: List[Dict]) -> str:
    if candidates is songs:
        return (
            "Full catalog used as fallback — fewer than the minimum candidate threshold "
            "matched the genre, mood, or energy filters. All songs were scored."
        )

    lines = [
        f"RAG Retrieval: {len(candidates)} of {len(songs)} songs passed the filters:\n"
    ]
    for song in candidates:
        matched: List[str] = []
        if song["genre"] == user_prefs["favorite_genre"]:
            matched.append(f"genre={song['genre']}")
        if song["mood"] == user_prefs["favorite_mood"]:
            matched.append(f"mood={song['mood']}")
        if abs(song["energy"] - user_prefs["target_energy"]) <= ENERGY_WINDOW:
            matched.append(f"energy≈{song['energy']:.2f}")
        lines.append(f"  • {song['title']} by {song['artist']}  —  matched: {', '.join(matched)}")

    return "\n".join(lines)
