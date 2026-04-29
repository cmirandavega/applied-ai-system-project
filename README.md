# VibeMatch 2.0 — AI-Powered Music Recommender

## Original Project: Music Recommender Simulation (Module 3)

The original project, **Music Recommender Simulation**, was a rule-based CLI recommender built in Module 3. It represented songs as structured data objects and scored each one against a hard-coded user taste profile using a weighted formula across five audio features: genre, mood, energy, acousticness, and valence. The system could rank an 18-song catalog and explain every recommendation with a per-feature score breakdown, but it required profiles to be written directly in Python code and could not understand natural language or expand beyond its fixed catalog.

---

## Title and Summary

**VibeMatch 2.0** upgrades the original simulation into a full AI-assisted recommendation system. A user types a plain-English description of what they want to hear — *"something chill and acoustic for late-night studying"* — and the system extracts a structured taste profile using an LLM, retrieves the most relevant songs from a static catalog of 559 real songs using a RAG pre-filter, scores and ranks the candidates, and presents ranked results with full score breakdowns in a Streamlit web UI.

The project demonstrates how a simple rule-based recommender can be made genuinely useful by layering natural language understanding and retrieval-augmented generation on top of a transparent scoring core — without sacrificing explainability. The catalog is built from real, well-known songs across 15 genres, acting as a reliable static fallback that never requires an LLM to generate fictional music data.

---

## Architecture Overview

```
User Input (natural language)
        │
        ▼
┌─────────────────┐
│   src/agent.py  │  LLM extracts structured UserProfile dict
│ (Groq / Gemini) │  {genre, mood, energy, likes_acoustic}
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│  src/retriever.py    │  RAG pre-filter: genre OR mood OR energy ±0.25
│  retrieve_candidates │  Returns candidate subset (≥6) or full catalog
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  src/recommender.py  │  Scores every candidate (max 5.0 pts)
│  recommend_songs()   │  Genre +2.0 | Mood +1.0 | Energy +1.0
└────────┬─────────────┘         │ Acousticness +0.5 | Valence +0.5
         │
         ▼
┌──────────────────────┐
│  src/logger.py       │  Logs input, profile, and top result to
│                      │  logs/recommender.log
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│     app.py           │  Streamlit UI — profile metrics, ranked results,
│  (Streamlit UI)      │  score bars, RAG reasoning expander
└──────────────────────┘
```

The system uses a **Retrieval-Augmented Generation** pattern: the retriever narrows the scoring space before the recommender runs. The LLM is used once — to understand the user's natural language input — and all song matching is done against a static catalog of real songs. The core scoring logic remains a deterministic, explainable formula with no black-box decisions.

---

## Catalog

`data/songs.csv` contains **559 real songs** across all 15 supported genres. Songs were sourced from Billboard charts and well-known genre catalogs, with estimated audio features (energy, tempo, valence, danceability, acousticness) based on each song's known character.

| Genre | Songs | Example Artists |
|---|---|---|
| pop | 74 | Taylor Swift, Ed Sheeran, Adele, BTS |
| hip-hop | 51 | Drake, Kendrick Lamar, Eminem, Kanye West |
| rock | 51 | Queen, Nirvana, Metallica, Foo Fighters |
| latin | 50 | Shakira, Bad Bunny, Celia Cruz, Buena Vista Social Club |
| soul | 35 | Stevie Wonder, Marvin Gaye, Aretha Franklin, SZA |
| jazz | 31 | Miles Davis, Frank Sinatra, Norah Jones, Duke Ellington |
| indie pop | 30 | Tame Impala, Lorde, The 1975, Vampire Weekend |
| blues | 30 | B.B. King, Stevie Ray Vaughan, Robert Johnson |
| classical | 30 | Beethoven, Debussy, Mozart, Chopin |
| electronic | 30 | Daft Punk, Avicii, Skrillex, Alan Walker |
| folk | 30 | Bob Dylan, Fleet Foxes, Bon Iver, Tracy Chapman |
| metal | 29 | Metallica, Slayer, Black Sabbath, Slipknot |
| lofi | 27 | jinsang, Idealism, Philanthrope, Kupla |
| ambient | 26 | Brian Eno, Sigur Rós, Aphex Twin, Max Richter |
| synthwave | 26 | Kavinsky, The Midnight, Carpenter Brut, Gunship |

Every genre has well above the 6-song retrieval threshold, so every query returns real, relevant results without needing to generate fictional songs.

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd applied-ai-system-project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure your API key

Add your key to a `.env` file in the project root. **Groq is recommended** — it has a reliable free tier and no regional restrictions.

**Option A — Groq** (recommended):
1. Go to [console.groq.com](https://console.groq.com) → **API Keys** → **Create API key**
2. Add to `.env`:
   ```
   GROQ_API_KEY=gsk_...
   ```

**Option B — Google Gemini** (free tier, some regional restrictions):
1. Go to [aistudio.google.com](https://aistudio.google.com) → **Get API key** → **Create API key in new project**
2. Add to `.env`:
   ```
   GOOGLE_API_KEY=AIza...
   ```

The system automatically uses Groq if `GROQ_API_KEY` is present, and falls back to Gemini otherwise.

### 4. Run the Streamlit app

```bash
python -m streamlit run app.py
```

Opens at `http://localhost:8501`

### 5. Run the CLI (original interface)

```bash
python -m src.main
```

### 6. Run tests

```bash
python -m pytest tests/ -v
```

Expected: **9 passed**

---

## Sample Interactions

### Example 1 — Natural language → profile extraction → recommendations

**Input:**
> "I want something chill and acoustic for late-night studying"

**Extracted profile:**
| Genre | Mood | Energy | Acoustic |
|---|---|---|---|
| lofi | chill | 0.35 | Yes |

**Top recommendations:**
```
1. Midnight Coding — LoRoom             Score: 4.65 / 5.0
2. Library Rain — Paper Lanterns        Score: 4.41 / 5.0
3. Lay It On Me — Tennyson              Score: 4.37 / 5.0
```

**RAG retrieval:** 295 of 559 songs passed the filters (genre=lofi OR mood=chill OR energy ≈ 0.35).

---

### Example 2 — Melancholic Spanish songs

**Input:**
> "Melancholic Spanish songs"

**Extracted profile:**
| Genre | Mood | Energy | Acoustic |
|---|---|---|---|
| latin | melancholic | 0.50 | — |

**Top recommendations:**
```
1. No Me Doy Por Vencido — Luis Fonsi   Score: 4.52 / 5.0
2. Por Tu Amor — Marc Anthony           Score: 4.50 / 5.0
3. Tal Vez — Ricky Martin               Score: 4.39 / 5.0
4. Vivir Sin Aire — Maná                Score: 4.38 / 5.0
5. La Tortura — Shakira                 Score: 4.32 / 5.0
```

The latin genre catalog was curated to separate Spanish-language songs from English-language artists who were previously mislabeled as latin. All top results are Spanish-language.

---

### Example 3 — High-energy rock

**Input:**
> "Heavy and intense, I'm working out"

**Extracted profile:**
| Genre | Mood | Energy | Acoustic |
|---|---|---|---|
| rock | intense | 0.80 | No |

**Top recommendations:**
```
1. Alive — Pearl Jam                    Score: 4.74 / 5.0
2. Welcome to the Jungle — Guns N' Roses  Score: 4.72 / 5.0
3. We Will Rock You — Queen             Score: 4.69 / 5.0
```

---

### Example 4 — Romantic jazz dinner

**Input:**
> "Romantic jazz for a dinner date"

**Extracted profile:**
| Genre | Mood | Energy | Acoustic |
|---|---|---|---|
| jazz | romantic | 0.50 | Yes |

**Top recommendations:**
```
1. Fly Me to the Moon — Frank Sinatra   Score: 4.29 / 5.0
2. The Way You Look Tonight — Frank Sinatra  Score: 4.15 / 5.0
3. Moonlight Serenade — Glenn Miller    Score: 4.02 / 5.0
```

---

## Design Decisions

### Why a deterministic scoring formula instead of a learned model?
Transparency was the priority. Every recommendation can be traced to an exact score contribution from each feature. A neural collaborative filter would likely produce better rankings on a large dataset but would give no explanation for why a song appeared. For a project exploring how recommenders work, explainability beats accuracy.

### Why RAG instead of scoring all 559 songs every time?
Two reasons. First, it mirrors how production recommenders work — they retrieve a candidate set before ranking, rather than scoring millions of tracks. Second, it keeps the scoring loop fast and focused even as the catalog grows.

### Why a static catalog of real songs instead of LLM-generated expansion?
The original design used an LLM to generate new fictional songs on demand when a genre was under-represented. This was replaced with a curated catalog of 559 real, well-known songs because:
- Real songs produce meaningful, recognizable recommendations
- The catalog is large enough to serve every genre without generation
- LLM-generated songs introduced quality issues — inconsistent audio features, fictional artists that felt out of place, and mislabeled genres
- Removing the expander eliminates an entire failure mode (LLM quota errors during expansion)

### Why Groq over Gemini?
Gemini's free tier has `limit: 0` quota issues on new API keys in certain regions, which causes the system to silently fall back to default profiles. Groq's free tier is more reliable. The system detects which key is present and routes accordingly, so both work.

### Why curate genre assignments in the catalog?
The initial catalog included English-language artists (Jason Derulo, Pitbull, Major Lazer) under the `latin` genre because they have a Latin-influenced sound. This caused Spanish-language searches to return English results. Genre labels were corrected so that `latin` only contains Spanish/Portuguese-language songs — which is what users expect when they search for "Spanish" or "Latin" music.

### Trade-offs made
| Decision | Benefit | Cost |
|---|---|---|
| Static real-song catalog | Reliable, recognizable results | Catalog doesn't grow automatically |
| Removed LLM expander | Eliminates expansion failure mode | Niche genres not in catalog return full-catalog fallback |
| Groq as primary LLM | Reliable free tier | Adds a second API dependency |
| Genre label curation | Spanish searches return Spanish songs | Manual effort to maintain genre accuracy |
| No `@st.cache_data` on `get_songs()` | Catalog changes appear immediately | Reads disk on every button click |

---

## Testing Summary

### What worked

- **9/9 tests pass** across both test files (`test_recommender.py`, `test_reliability.py`)
- Groq backend (`llama-3.1-8b-instant`) correctly extracts structured profiles from natural language
- RAG retrieval correctly includes exact genre matches and filters by energy window
- Logger test with `monkeypatch` correctly redirects the log file to a temp path and verifies content
- Empty and whitespace inputs reliably return `DEFAULT_PROFILE` without hitting the API
- All 5 smoke-test inputs returned genre-accurate, recognizable song recommendations

### What didn't work

- **Gemini free tier quota** — `limit: 0` errors appeared on the existing API key, causing every query to silently return the default pop/happy profile. Switching to Groq resolved this.
- **Groq model deprecation** — the initially configured model (`llama3-8b-8192`) had been decommissioned. Updating to `llama-3.1-8b-instant` fixed it.
- **English songs in the latin genre** — the initial catalog included English-language artists under `latin`, causing searches for Spanish music to surface Jason Derulo and Pitbull. Genre labels were corrected.

### What was learned

- `limit: 0` in a Google API quota error means the project has zero allocated quota — it is not a "you've used up your quota" message, and waiting does not help. Creating a fresh API key in a new Google project is the correct fix.
- LLM model names have lifecycles. Hard-coding a specific model ID without checking the provider's deprecation list is a silent breakage waiting to happen.
- Genre labels in a music catalog are subjective but consequential. An artist like Pitbull belongs in `latin` from a cultural standpoint but his English-language tracks don't match what users mean when they search for "Spanish songs." Label granularity matters.

---

## Responsible AI

### Limitations and biases in the system

The scoring formula treats genre and mood as binary matches — a song either matches or it doesn't, with no partial credit. This makes the system inherently biased toward genres that are well represented in the catalog. The catalog skews toward Western popular music; genres like Afrobeats, K-pop, bhangra, or regional folk traditions are absent.

There is also a bias embedded in the profile extraction step. When the LLM cannot confidently interpret an input, it defaults to `pop / happy / 0.5 energy` — the most statistically common profile in Western streaming data. Ambiguous or culturally specific inputs silently degrade to a generic Western pop profile rather than asking for clarification.

### Could this AI be misused?

The main risk is **API key exposure** — if a user accidentally commits their `.env` file, the key is public. The `.gitignore` entry for `.env` addresses this, but does not prevent a user from force-adding the file.

Since catalog expansion was removed, the previous catalog poisoning risk (LLM-generated songs with extreme feature values appended without review) no longer applies.

### What surprised you while testing reliability

The most surprising finding was how quietly the system failed. When the Gemini API returned a quota error, `extract_profile()` caught the exception and silently returned the default pop/happy profile. The app continued to show recommendations — plausible-looking ones — with no indication that the LLM had not run at all. This underscored that reliability testing needs to probe failure paths explicitly, not just the happy path.

A second surprise was discovering that `llama3-8b-8192` on Groq had been silently decommissioned. The error only surfaced at runtime; there was no static warning. This is a general lesson about depending on external model identifiers — they change without notice.

### Collaboration with AI during this project

This project was built collaboratively with Claude Code (Anthropic). The AI handled file creation, module wiring, test writing, catalog generation, and debugging based on a detailed specification.

**One instance where the AI gave a helpful suggestion:** When the Gemini API returned `limit: 0` quota errors, the AI correctly identified that this was a project-level configuration issue rather than a temporary rate limit, and recommended Groq as an alternative. It also correctly diagnosed that `llama3-8b-8192` had been decommissioned and updated the model to `llama-3.1-8b-instant`.

**One instance where the AI's suggestion was flawed:** The initial catalog generation included English-language artists (Jason Derulo, Pitbull, Major Lazer) in the `latin` genre because they have a Latin-influenced sound. This only became apparent when a real user query for "melancholic Spanish songs" surfaced Jason Derulo as a top result. The genre assignment needed manual review and correction — something the AI could not anticipate without user feedback.

---

## Reflection

Building VibeMatch 2.0 made the gap between a rule-based system and an AI-assisted one very concrete. The original recommender was correct and explainable, but completely rigid. Adding an LLM extraction layer made the same scoring logic accessible to anyone who can type a sentence.

The harder lesson was about failure modes. In the original project, failures were obvious — the code either ran or it didn't. In an LLM-integrated system, failures are quiet: the agent returns a default profile, the UI still shows recommendations, and the user has no idea their input was ignored. Designing for visible, fast failures — surfacing error messages, choosing a reliable LLM provider, failing immediately instead of sleeping and retrying — turned out to be as important as the feature logic itself.

Replacing LLM-generated catalog expansion with a curated real-song catalog also changed how I think about data quality. A larger, real catalog is more valuable than an infinitely expandable fictional one, because the recommendations mean something to users. When someone searches for "melancholic Spanish songs" and gets songs they actually recognize — Shakira, Maná, Marc Anthony — that is a fundamentally better outcome than getting a list of convincing-sounding fictional artists.

https://www.loom.com/share/de41919e78554a6e962772b9350d2640 
