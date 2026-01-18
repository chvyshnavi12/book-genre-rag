import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import streamlit as st

from rag.rag_pipeline import run_rag
from src.prediction.predict_pipeline import predict_genre_with_scores

# -----------------------------
# Page config (dull colors)
# -----------------------------
st.set_page_config(
    page_title="Book Genre & Recommendation System",
    layout="centered"
)

st.markdown(
    """
    <style>
    body {
        background-color: #f5f5f5;
    }
    .stButton button {
        background-color: #6c757d;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Session state
# -----------------------------
if "preferences" not in st.session_state:
    st.session_state.preferences = []

if "prefs_saved" not in st.session_state:
    st.session_state.prefs_saved = False

# -----------------------------
# Title
# -----------------------------
st.title("📚 Book Recommendation & Genre Match System")

# =====================================================
# 1️⃣ USER PREFERENCES
# =====================================================
if not st.session_state.prefs_saved:
    st.subheader("🎯 Select Your Preferred Genres")

    genres = [
        "Fantasy", "Romance", "Sci-Fi", "Thriller",
        "Mystery", "Horror", "Classic", "Young Adult"
    ]

    selected = st.multiselect(
        "Choose one or more genres:",
        genres
    )

    if st.button("Save Preferences"):
        if selected:
            st.session_state.preferences = selected
            st.session_state.prefs_saved = True
            st.success("Preferences saved successfully ✅")
        else:
            st.warning("Please select at least one genre")

    st.stop()

# =====================================================
# 2️⃣ MAIN OPTIONS
# =====================================================
st.subheader("🔍 Choose an Action")

option = st.radio(
    "What do you want to do?",
    [
        "📖 Recommend books using title / description",
        "📊 Check genre matching percentage for a book"
    ]
)

# =====================================================
# OPTION A: RECOMMENDATION (RAG)
# =====================================================
if option.startswith("📖"):
    st.subheader("📚 Book Recommendation")

    query = st.text_area(
        "Enter book title, description, or summary:",
        height=150
    )

    if st.button("Recommend Books"):
        if query.strip():
            with st.spinner("Generating recommendations..."):
                result = run_rag(query)

            st.markdown("### 🧠 Recommendations")
            st.write(result)

        else:
            st.warning("Please enter some text")

# =====================================================
# OPTION B: GENRE MATCH PERCENTAGE
# =====================================================
else:
    st.subheader("📊 Preference Match Score")

    book_name = st.text_input("Enter a book name:")

    if st.button("Check Match"):
        if book_name.strip():
            with st.spinner("Analyzing book..."):
                predictions = predict_genre_with_scores(book_name)

                from src.prediction.match_score import compute_preference_match
                match_score = compute_preference_match(
                    predictions,
                    st.session_state.preferences
                )

            st.markdown("### 🎯 Overall Preference Match")

            # Progress bar
            st.progress(match_score / 100)

            st.write(f"**Match Score:** {match_score}%")

            st.caption(
                "This score represents how well the book matches "
                "your preferred genres based on genre prediction."
            )

        else:
            st.warning("Please enter a book name")


# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("AI-powered Book Genre Prediction & Recommendation System")
