import streamlit as st

from src.recommender import load_songs, recommend_songs
from src.agent import extract_profile
from src.retriever import retrieve_candidates, retrieval_summary
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

    # ── Step 2: Load catalog + retrieve candidates ────────────────────────
    all_songs = get_songs()
    candidates = retrieve_candidates(profile, all_songs)

    # ── Step 3: Recommend + log ───────────────────────────────────────────
    results = recommend_songs(profile, candidates, k=num_recs)
    rec_logger.log_request(user_input, profile, results)

    # ── Step 4: Display results ───────────────────────────────────────────
    st.subheader(f"Top {len(results)} Recommendation{'s' if len(results) != 1 else ''}")

    for rank, (song, score, explanation) in enumerate(results, 1):
        st.markdown(f"**{rank}. {song['title']}** — *{song['artist']}*")
        st.progress(score / 5.0, text=f"Score: {score:.4f} / 5.0")

        with st.expander("Score breakdown"):
            for reason in explanation.split(" | "):
                if reason.strip():
                    st.write(reason)

    # ── Step 6: RAG retrieval reasoning ──────────────────────────────────
    with st.expander("RAG Retrieval Reasoning"):
        st.text(retrieval_summary(profile, all_songs, candidates))
