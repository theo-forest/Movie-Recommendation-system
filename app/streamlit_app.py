"""
Streamlit front-end for the Movie Recommendation System.

Start the API first:
    uvicorn src.api:app --reload

Then run the app:
    streamlit run app/streamlit_app.py
"""
import requests
import streamlit as st

API_BASE = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 Movie Recommendation System")
st.caption(
    "Powered by SVD collaborative filtering, TF-IDF content-based filtering, "
    "and a hybrid blend of both."
)

# ---------------------------------------------------------------------------
# Sidebar — controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    mode = st.radio("Recommendation mode", ["By User ID", "By Movie"])
    model_choice = st.selectbox(
        "Model",
        ["hybrid", "collaborative", "content"],
        help=(
            "hybrid = 70% CF + 30% CBF  |  "
            "collaborative = SVD only  |  "
            "content = TF-IDF similarity"
        ),
    )
    n_recs = st.slider("Number of recommendations", 5, 20, 10)

    st.markdown("---")
    st.subheader("API Health")
    try:
        health = requests.get(f"{API_BASE}/health", timeout=3).json()
        for model_name, loaded in health.get("models", {}).items():
            icon = "✅" if loaded else "❌"
            st.write(f"{icon} {model_name.replace('_', ' ').title()}")
    except Exception:
        st.error("API not reachable — start the FastAPI server first.")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
col_input, col_results = st.columns([1, 2])

with col_input:
    st.subheader("Input")

    if mode == "By User ID":
        user_id = st.number_input("User ID (1 – 610)", min_value=1, max_value=610, value=1)
        movie_id = None
        submit_label = f"Get recommendations for user {user_id}"
    else:
        st.markdown("**Search for a movie:**")
        query = st.text_input("Movie title keyword", "")
        movie_id = None

        if query:
            try:
                results = requests.get(
                    f"{API_BASE}/movies/search",
                    params={"q": query, "limit": 20},
                    timeout=5,
                ).json()
                if results:
                    options = {f"{r['title']} (id={r['movieId']})": r["movieId"] for r in results}
                    selected = st.selectbox("Select movie", list(options.keys()))
                    movie_id = options[selected]
                else:
                    st.info("No movies found — try a different keyword.")
            except Exception as exc:
                st.error(f"Search failed: {exc}")

        user_id = None
        submit_label = "Find similar movies"

    get_recs = st.button(submit_label, type="primary")

# ---------------------------------------------------------------------------
# Fetch and display recommendations
# ---------------------------------------------------------------------------
with col_results:
    st.subheader("Recommendations")

    if get_recs:
        payload = {
            "n": n_recs,
            "model_type": model_choice if mode == "By User ID" else "content",
        }
        if mode == "By User ID":
            payload["user_id"] = int(user_id)
        elif movie_id is not None:
            payload["movie_id"] = int(movie_id)
        else:
            st.warning("Please select a movie first.")
            st.stop()

        with st.spinner("Fetching recommendations …"):
            try:
                response = requests.post(
                    f"{API_BASE}/recommend", json=payload, timeout=10
                )
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to the API. Make sure it is running on port 8000.")
                st.stop()

        if response.status_code == 200:
            data = response.json()
            st.info(f"**{data['explanation']}**  _(model: {data['model_used']})_")

            recs = data["recommendations"]
            for i, rec in enumerate(recs, 1):
                with st.container():
                    cols = st.columns([0.4, 3, 2, 1])
                    cols[0].markdown(f"**#{i}**")
                    cols[1].markdown(f"**{rec['title']}**")
                    genres = rec.get("genres", "")
                    cols[2].caption(genres.replace("|", "  ·  "))
                    cols[3].markdown(f"`{rec['score']:.3f}`")

        elif response.status_code == 404:
            st.warning("No recommendations found for this input.")
        elif response.status_code == 503:
            st.error("Models not loaded. Run `python -m src.train` first.")
        else:
            st.error(f"API error {response.status_code}: {response.text}")

# ---------------------------------------------------------------------------
# Model metrics expander
# ---------------------------------------------------------------------------
with st.expander("Model Evaluation Metrics"):
    try:
        metrics = requests.get(f"{API_BASE}/metrics", timeout=3).json()
        if "message" in metrics:
            st.info(metrics["message"])
        else:
            for model_name, vals in metrics.items():
                st.markdown(f"**{model_name.replace('_', ' ').title()}**")
                if isinstance(vals, dict):
                    nice = {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in vals.items()}
                    st.json(nice)
    except Exception:
        st.info("Start the API to see metrics.")
