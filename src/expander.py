import csv
import json
import os
from typing import List, Dict, Tuple

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

_client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY", ""))

CATALOG_PATH = "data/songs.csv"
TARGET_COUNT = 3   # small request to stay within free-tier token limits
MIN_CANDIDATES = 6
_EXPAND_MODEL = "gemini-2.0-flash"

VALID_GENRES = [
    "pop", "lofi", "rock", "ambient", "jazz", "synthwave",
    "indie pop", "hip-hop", "folk", "electronic", "soul",
    "metal", "latin", "blues", "classical",
]

VALID_MOODS = [
    "happy", "chill", "intense", "relaxed", "moody", "focused",
    "energetic", "melancholic", "romantic", "playful", "nostalgic",
    "introspective",
]


def needs_expansion(candidates: List[Dict], all_songs: List[Dict], user_prefs: dict = None) -> bool:
    if candidates is all_songs or len(candidates) < MIN_CANDIDATES:
        return True
    if user_prefs is not None:
        genre = user_prefs["favorite_genre"]
        genre_count = sum(1 for s in all_songs if s["genre"] == genre)
        if genre_count < MIN_CANDIDATES:
            return True
    return False


def _validate_song(raw: dict) -> "dict | None":
    title = str(raw.get("title", "")).strip()
    artist = str(raw.get("artist", "")).strip()

    if not title or not artist:
        return None

    genre = raw.get("genre", "pop")
    if genre not in VALID_GENRES:
        genre = "pop"

    mood = raw.get("mood", "happy")
    if mood not in VALID_MOODS:
        mood = "happy"

    try:
        energy = float(raw.get("energy", 0.5))
        energy = max(0.0, min(1.0, energy))
    except (TypeError, ValueError):
        energy = 0.5

    try:
        tempo_bpm = float(raw.get("tempo_bpm", 120))
        tempo_bpm = max(60.0, min(200.0, tempo_bpm))
    except (TypeError, ValueError):
        tempo_bpm = 120.0

    try:
        valence = float(raw.get("valence", 0.5))
        valence = max(0.0, min(1.0, valence))
    except (TypeError, ValueError):
        valence = 0.5

    try:
        danceability = float(raw.get("danceability", 0.5))
        danceability = max(0.0, min(1.0, danceability))
    except (TypeError, ValueError):
        danceability = 0.5

    try:
        acousticness = float(raw.get("acousticness", 0.5))
        acousticness = max(0.0, min(1.0, acousticness))
    except (TypeError, ValueError):
        acousticness = 0.5

    return {
        "title": title,
        "artist": artist,
        "genre": genre,
        "mood": mood,
        "energy": energy,
        "tempo_bpm": tempo_bpm,
        "valence": valence,
        "danceability": danceability,
        "acousticness": acousticness,
    }


def _extract_json(text: str) -> str:
    """Pull the first JSON array out of a model response, regardless of wrapping."""
    # Already bare JSON
    stripped = text.strip()
    if stripped.startswith("["):
        return stripped
    # Markdown fenced block: ```json ... ``` or ``` ... ```
    if "```" in stripped:
        parts = stripped.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].lstrip()
            if part.startswith("["):
                return part
    # Last resort: find the first '[' and last ']'
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return stripped


def _already_in_catalog(song: dict, existing: List[Dict]) -> bool:
    title_lower = song["title"].lower()
    return any(s["title"].lower() == title_lower for s in existing)


def _append_to_csv(songs: List[Dict], csv_path: str) -> None:
    fieldnames = [
        "id", "title", "artist", "genre", "mood",
        "energy", "tempo_bpm", "valence", "danceability", "acousticness",
    ]
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for song in songs:
            writer.writerow(song)


def expand_catalog(
    user_prefs: dict,
    existing_songs: List[Dict],
    csv_path: str,
    target: int = TARGET_COUNT,
) -> Tuple[List[Dict], int]:
    prompt = (
        f"Generate {target} real songs that closely match this music profile:\n"
        f"  Genre:    {user_prefs['favorite_genre']}\n"
        f"  Mood:     {user_prefs['favorite_mood']}\n"
        f"  Energy:   {user_prefs['target_energy']} (scale 0.0–1.0)\n"
        f"  Acoustic: {user_prefs['likes_acoustic']}\n\n"
        f"Valid genres: {', '.join(VALID_GENRES)}\n"
        f"Valid moods:  {', '.join(VALID_MOODS)}\n\n"
        "Return ONLY a raw JSON array. Each element must have exactly these fields:\n"
        "  title        (string — real song title)\n"
        "  artist       (string — real artist name)\n"
        "  genre        (string — must be from the valid genres list)\n"
        "  mood         (string — must be from the valid moods list)\n"
        "  energy       (float 0.0–1.0)\n"
        "  tempo_bpm    (float 60–200)\n"
        "  valence      (float 0.0–1.0)\n"
        "  danceability (float 0.0–1.0)\n"
        "  acousticness (float 0.0–1.0)\n\n"
        "No markdown, no explanation — output ONLY the JSON array."
    )

    response = _client.models.generate_content(
        model=_EXPAND_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=1024,
        ),
    )

    raw_text = response.text.strip()
    raw_songs = json.loads(_extract_json(raw_text))

    max_id = max((s["id"] for s in existing_songs), default=0)
    new_songs: List[Dict] = []

    for raw in raw_songs:
        validated = _validate_song(raw)
        if validated is None:
            continue
        if _already_in_catalog(validated, existing_songs + new_songs):
            continue
        max_id += 1
        validated["id"] = max_id
        new_songs.append(validated)

    if new_songs:
        _append_to_csv(new_songs, csv_path)

    return (new_songs, len(new_songs))
