import os
import json

from dotenv import load_dotenv

load_dotenv()

_GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
_GEMINI_KEY = os.environ.get("GOOGLE_API_KEY", "")

if _GROQ_KEY:
    from groq import Groq
    _groq_client = Groq(api_key=_GROQ_KEY)
    _backend = "groq"
else:
    from google import genai
    from google.genai import types
    _gemini_client = genai.Client(api_key=_GEMINI_KEY)
    _backend = "gemini"

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

DEFAULT_PROFILE = {
    "favorite_genre": "pop",
    "favorite_mood": "happy",
    "target_energy": 0.5,
    "likes_acoustic": False,
}

SYSTEM_PROMPT = (
    "You are a music preference extractor. "
    "Read the user's message and return ONLY a raw JSON object with exactly these four keys:\n"
    '  "favorite_genre": one of ' + str(VALID_GENRES) + "\n"
    '  "favorite_mood": one of ' + str(VALID_MOODS) + "\n"
    '  "target_energy": a float between 0.0 and 1.0\n'
    '  "likes_acoustic": a boolean (true or false)\n\n'
    "Rules:\n"
    "- Output ONLY the raw JSON object. No markdown fences, no explanation, no extra text.\n"
    "- favorite_genre MUST be one of the valid genres listed above.\n"
    "- favorite_mood MUST be one of the valid moods listed above.\n"
    "- If the input is ambiguous, make a reasonable inference.\n"
)


def _validate_and_sanitize(profile: dict) -> dict:
    result = dict(DEFAULT_PROFILE)

    genre = profile.get("favorite_genre", "")
    if genre in VALID_GENRES:
        result["favorite_genre"] = genre

    mood = profile.get("favorite_mood", "")
    if mood in VALID_MOODS:
        result["favorite_mood"] = mood

    try:
        energy = float(profile.get("target_energy", DEFAULT_PROFILE["target_energy"]))
        result["target_energy"] = max(0.0, min(1.0, energy))
    except (TypeError, ValueError):
        result["target_energy"] = DEFAULT_PROFILE["target_energy"]

    result["likes_acoustic"] = bool(profile.get("likes_acoustic", DEFAULT_PROFILE["likes_acoustic"]))

    return result


def _call_groq(user_input: str) -> str:
    response = _groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ],
        temperature=0,
        max_tokens=256,
    )
    return response.choices[0].message.content.strip()


def _call_gemini(user_input: str) -> str:
    response = _gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=user_input,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0,
            max_output_tokens=256,
        ),
    )
    return response.text.strip()


def _strip_fences(raw: str) -> str:
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else parts[0]
        if raw.startswith("json"):
            raw = raw[4:].lstrip()
    return raw


def extract_profile(user_input: str) -> dict:
    if not user_input or not user_input.strip():
        return dict(DEFAULT_PROFILE)

    try:
        raw_json = _call_groq(user_input) if _backend == "groq" else _call_gemini(user_input)
        raw_json = _strip_fences(raw_json)
        profile = json.loads(raw_json)
        return _validate_and_sanitize(profile)
    except Exception as e:
        print(f"Warning: Failed to extract profile ({_backend}): {type(e).__name__}: {e}")
        return dict(DEFAULT_PROFILE)
