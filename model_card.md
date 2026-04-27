# Model Card: VibeMatch 2.0

## 1. Model Name

**VibeMatch 2.0**

Upgraded from the original **Music Recommender Simulation (VibeMatch 1.0)** built in Module 3. The core scoring formula is unchanged; everything around it — natural language input, RAG retrieval, a curated real-song catalog, and a Streamlit UI — is new.

---

## 2. Intended Use

VibeMatch 2.0 suggests songs from a catalog based on a plain-English description of what the user wants to hear.

- Accepts any natural language input (e.g. "something chill for studying" or "heavy music for the gym")
- Returns up to 10 ranked recommendations with per-feature score breakdowns
- Recommends only from a static catalog of 559 real, well-known songs
- Built for educational exploration of AI-assisted recommender systems
- Not intended for production use or real-world music discovery at scale

---

## 3. How the Model Works

The system has three stages:

**Stage 1 — Profile Extraction (LLM)**
The user's plain-English input is sent to a large language model (Groq `llama-3.1-8b-instant` if `GROQ_API_KEY` is set, otherwise Gemini `gemini-2.0-flash`). The LLM returns a structured JSON object with four fields: `favorite_genre`, `favorite_mood`, `target_energy` (0–1), and `likes_acoustic` (true/false). If the LLM fails or the input is empty, the system falls back to a default profile (pop / happy / 0.5 energy / non-acoustic).

**Stage 2 — RAG Retrieval**
Instead of scoring the entire catalog, a pre-filter selects candidate songs that match any of three conditions: exact genre match, exact mood match, or energy within ±0.25 of the target. If fewer than 6 candidates are found, the full catalog is used as a fallback. With 559 real songs across 15 genres, every genre has well above the minimum threshold.

**Stage 3 — Scoring and Ranking**
Every candidate is scored out of 5.0 using the same weighted formula from the original project:

| Component | Max Points | Method |
|---|---|---|
| Genre match | +2.0 | Exact string match |
| Mood match | +1.0 | Exact string match |
| Energy similarity | +1.0 | Gaussian: exp(−diff² / 0.045) |
| Acousticness fit | +0.5 | Linear preference match |
| Valence bonus | +0.5 | song.valence × 0.5 |

Songs are ranked highest to lowest. The top k results (user-selected, 1–10) are displayed with a full score breakdown per song.

---

## 4. Data

The catalog contains **559 real songs** stored in `data/songs.csv`. Each song has 10 fields: id, title, artist, genre, mood, energy, tempo\_bpm, valence, danceability, and acousticness.

Songs were drawn from Billboard charts and well-known genre catalogs. Audio features (energy, tempo, valence, danceability, acousticness) are estimated based on each song's known musical character rather than sourced from a streaming API.

**Genre distribution:**

| Genre | Count | Notes |
|---|---|---|
| pop | 74 | Taylor Swift, Adele, Ed Sheeran, BTS |
| hip-hop | 51 | Drake, Kendrick Lamar, Eminem, Kanye West |
| rock | 51 | Queen, Nirvana, Metallica, Foo Fighters |
| latin | 50 | Shakira, Bad Bunny, Celia Cruz, Buena Vista Social Club |
| soul | 35 | Stevie Wonder, Marvin Gaye, Aretha Franklin, SZA |
| jazz | 31 | Miles Davis, Frank Sinatra, Norah Jones |
| indie pop | 30 | Tame Impala, Lorde, The 1975 |
| blues | 30 | B.B. King, Stevie Ray Vaughan, Robert Johnson |
| classical | 30 | Beethoven, Debussy, Mozart, Chopin |
| electronic | 30 | Daft Punk, Avicii, Skrillex, Alan Walker |
| folk | 30 | Bob Dylan, Fleet Foxes, Bon Iver |
| metal | 29 | Metallica, Slayer, Black Sabbath, Slipknot |
| lofi | 27 | jinsang, Idealism, Philanthrope |
| ambient | 26 | Brian Eno, Sigur Rós, Aphex Twin |
| synthwave | 26 | Kavinsky, The Midnight, Carpenter Brut |

**Genre curation note:** The `latin` genre was curated to contain only Spanish/Portuguese-language songs. English-language artists with a Latin-influenced sound (Pitbull, Jason Derulo, Major Lazer) were moved to `pop` or `electronic`, and 24 Spanish-language melancholic and romantic songs were added (Buena Vista Social Club, Maná, Chavela Vargas, Camila, etc.) to improve coverage of that mood space.

---

## 5. Strengths

- **Natural language input** — users no longer need to know genre names or energy values; a sentence is enough
- **Explainable scoring** — every recommendation shows exactly which features contributed and by how much
- **Real, recognizable songs** — recommendations are artists and titles users actually know, making results immediately meaningful
- **No LLM dependency for song data** — the catalog is fully static; the system produces results even if the LLM is slow or unavailable (using the default profile fallback)
- **RAG pre-filtering** — only relevant songs are scored, mirroring how production recommenders reduce the candidate space before ranking
- **Dual LLM backend** — automatically uses Groq if available, falls back to Gemini; isolates the system from single-provider quota issues

---

## 6. Limitations and Bias

**Genre lock-in remains.** The scoring formula still awards 2.0 out of 5.0 points for a single exact genre match. No combination of energy, mood, acousticness, and valence can fully compensate for a genre miss. This was the core limitation of v1.0 and is unchanged.

**Silent LLM failure.** When the profile extraction LLM call fails (quota error, network issue, unparseable response), the system silently returns a default pop/happy profile and continues. The user sees recommendations but has no way of knowing their input was ignored. This was observed when the Gemini API returned `limit: 0` quota errors on every call.

**Western music bias.** The catalog was assembled from Billboard charts and mainstream streaming catalogs, which skew heavily toward Western popular music. Genres like Afrobeats, K-pop, bhangra, or regional folk traditions from non-Western cultures are absent. Searches for those styles will fall back to the full catalog and return irrelevant results.

**Genre label subjectivity.** Genre assignment is a judgment call. An artist like Pitbull sits at the intersection of latin, pop, and hip-hop. The current catalog places English-language Pitbull tracks in `pop`, but a user who considers Pitbull a latin artist may find this unintuitive. There is no universally correct answer.

**Audio features are estimated.** Energy, valence, danceability, and acousticness values were assigned by hand based on knowledge of each song, not sourced from Spotify's audio analysis API. Some values may be inaccurate, which affects scoring precision.

**likes\_acoustic is still binary.** The user's acoustic preference is True or False. A user who likes "slightly acoustic" music cannot express a nuanced preference.

---

## 7. Evaluation

### Reliability tests (automated)

Nine automated tests cover the agent, retriever, and logger:

| Test | Result |
|---|---|
| Agent returns same genre/mood across 5 calls for same input | Pass |
| Empty input returns default profile without API call | Pass |
| Whitespace-only input returns default profile | Pass |
| Nonsense input returns structurally valid profile | Pass |
| RAG retrieves >1 candidate for latin/playful/0.71 | Pass |
| RAG always includes exact genre match for rock/intense/0.91 | Pass |
| Log file is written and contains input string | Pass |
| Recommender returns sorted results | Pass |
| Recommender returns non-empty explanation | Pass |

### Manual smoke tests (5 inputs, Groq backend)

| Input | Extracted profile | Top result |
|---|---|---|
| "Something chill and acoustic for late-night studying" | lofi / chill / 0.50 | Midnight Coding — LoRoom |
| "Upbeat Latin music to dance to at a party" | latin / energetic / 0.80 | Hips Don't Lie — Shakira |
| "Heavy and intense, I am working out" | rock / intense / 0.80 | Alive — Pearl Jam |
| "Romantic jazz for a dinner date" | jazz / romantic / 0.50 | Fly Me to the Moon — Sinatra |
| "Old school blues, something nostalgic" | blues / nostalgic / 0.50 | Sweet Home Chicago — Robert Johnson |
| "Melancholic Spanish songs" | latin / melancholic / 0.50 | No Me Doy Por Vencido — Luis Fonsi |

All six inputs returned genre-accurate, recognizable results. Before the latin genre curation fix, "Melancholic Spanish songs" returned Jason Derulo as a top result — a known-English artist that slipped in due to a mislabeled genre tag.

### What the Gemini quota failure revealed

During testing, all Gemini API calls returned `limit: 0` quota errors. Because `extract_profile()` catches exceptions silently, the app continued showing recommendations — based on the default pop profile — with no visible indication of failure. This failure mode was invisible without the sanity-check CLI test. It highlighted that LLM-integrated systems need explicit failure visibility, not just try/except blocks that swallow errors, and that depending on a single LLM provider is a reliability risk.

---

## 8. Future Work

- **Soften genre matching.** Allow partial credit for musically related genres (rock/metal, lofi/ambient). A genre miss currently costs the full 2.0 points with no middle ground.
- **Surface LLM failures explicitly.** Replace the silent default-profile fallback with a visible warning in the UI so users know when their input was not processed.
- **Make acoustic preference a float.** Replace the True/False `likes_acoustic` field with a 0–1 scale so users can express nuanced preferences.
- **Add artist diversity.** The top k results can currently include multiple songs from the same artist (e.g. three Frank Sinatra songs for a jazz query). A rule capping results at 2 songs per artist would improve variety.
- **Source audio features from Spotify API.** The current estimated features introduce scoring imprecision. Replacing them with Spotify's actual audio analysis data would make the scoring formula meaningfully more accurate.
- **Score valence against user preference.** Currently all users receive a bonus for high-valence songs. Users who prefer melancholic or dark music are effectively penalized by the formula.
- **Expand non-Western genres.** The catalog has no Afrobeats, K-pop, cumbia, or regional folk traditions. Adding representative songs from underrepresented genres would make the system useful to a wider audience.

---

## 9. Personal Reflection

The jump from v1.0 to v2.0 showed how much of a recommender's behavior is determined by what happens before the scoring formula runs. In v1.0, the bottleneck was catalog coverage — niche genre users hit a wall after the first result. In v2.0, the bottleneck shifted to data quality: when genre labels are wrong, the formula is running on the right math but the wrong data, and the results are confidently incorrect.

The decision to replace LLM-generated catalog expansion with a curated static catalog of real songs was one of the most consequential changes. The original expansion approach was elegant in theory — the system could grow to cover any genre on demand. In practice, it introduced a second LLM failure point, generated fictional artists that felt out of place next to real songs, and produced genre labels that didn't always match what users expected. A catalog of 559 real songs, carefully labeled, produces more trustworthy results than an infinitely expandable catalog of plausible-sounding fiction.

The other major lesson was about provider dependency. Hard-coding a single LLM provider (Gemini) and a specific model name (`llama3-8b-8192`) both turned into silent failures — one due to quota limits, one due to model deprecation. Building the system to detect which key is available and route accordingly made it meaningfully more robust without much additional complexity.
