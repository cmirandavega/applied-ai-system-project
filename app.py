import streamlit as st

from src.recommender import load_songs, recommend_songs
from src.agent import extract_profile
from src.retriever import retrieve_candidates, retrieval_summary
from src.orchestrator import run_orchestration
from src import logger as rec_logger

CATALOG_PATH = "data/songs.csv"

st.set_page_config(page_title="VibeMatch 2.0", page_icon="🎵", layout="centered")
st.title("🎵 VibeMatch 2.0")
st.caption("Your AI-powered music recommender")


def get_songs():
    """Load songs fresh from disk — intentionally not cached so expansions appear immediately."""
    return load_songs(CATALOG_PATH)


# ── Inputs ──────────────────────────────────────────────────────────────────
user_input = st.text_input(
    "What kind of music are you in the mood for?",
    placeholder="e.g. Something chill and acoustic for late-night studying…",
)

num_recs = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)

use_agent = st.checkbox(
    "Use agent orchestration",
    value=False,
    help=(
        "Let an LLM tool-calling loop decide whether to expand the catalog, "
        "instead of the fixed hardcoded pipeline. Both produce the same shape of "
        "results so you can compare side by side."
    ),
)

# ── Main flow ────────────────────────────────────────────────────────────────
if st.button("Find my music"):

    # ── Step 1: Extract profile ───────────────────────────────────────────
    try:
        profile = extract_profile(user_input)
    except Exception as exc:
        st.error(f"Could not extract your music profile: {exc}")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Genre", profile["favorite_genre"].title())
    col2.metric("Mood", profile["favorite_mood"].title())
    col3.metric("Energy", f"{profile['target_energy']:.2f}")
    col4.metric("Acoustic", "Yes" if profile["likes_acoustic"] else "No")

    st.divider()

    # ── Step 2: Load catalog ──────────────────────────────────────────────
    all_songs = get_songs()

    if use_agent:
        # ── Agent orchestration path (model-decided branching) ────────────
        st.caption("🤖 Agent orchestration mode")
        try:
            results, trace = run_orchestration(
                profile, all_songs, CATALOG_PATH, k=num_recs
            )
        except Exception as exc:
            st.error(f"Agent orchestration failed: {exc}")
            st.stop()
        candidates = None  # retrieval reasoning is shown via the trace instead
    else:
        # ── Fixed pipeline path (hardcoded heuristic) ─────────────────────
        candidates = retrieve_candidates(profile, all_songs)
        results = recommend_songs(profile, candidates, k=num_recs)
        rec_logger.log_request(user_input, profile, results)
        trace = None

    # ── Step 4: Display results ───────────────────────────────────────────
    st.subheader(f"Top {len(results)} Recommendation{'s' if len(results) != 1 else ''}")

    for rank, (song, score, explanation) in enumerate(results, 1):
        st.markdown(f"**{rank}. {song['title']}** — *{song['artist']}*")
        st.progress(score / 5.0, text=f"Score: {score:.4f} / 5.0")

        with st.expander("Score breakdown"):
            for reason in explanation.split(" | "):
                if reason.strip():
                    st.write(reason)

    # ── Step 5: Reasoning expanders ───────────────────────────────────────
    if use_agent:
        # Surface the agent's tool-call decisions so the behavior is inspectable.
        with st.expander("Agent Tool-Call Trace"):
            for i, entry in enumerate(trace, 1):
                etype = entry.get("type")
                if etype == "tool_call":
                    st.markdown(
                        f"**{i}. 🛠 `{entry['tool']}`** (turn {entry.get('turn')})"
                    )
                    if entry.get("args"):
                        st.write(f"args: {entry['args']}")
                    st.write(entry.get("result"))
                elif etype == "final_message":
                    st.markdown(f"**{i}. 💬 final message** (turn {entry.get('turn')})")
                    st.write(entry.get("content") or "(empty)")
                elif etype == "cap_hit":
                    st.markdown(f"**{i}. ⚠️ safety cap hit**")
                    st.warning(
                        f"{entry.get('reason')} — "
                        f"{entry.get('api_calls')} Groq calls (max {entry.get('max_turns')})."
                    )
                elif etype == "fallback":
                    st.markdown(f"**{i}. ↩️ fallback ranking**")
                    st.write(entry.get("reason"))
    else:
        with st.expander("RAG Retrieval Reasoning"):
            st.text(retrieval_summary(profile, all_songs, candidates))
