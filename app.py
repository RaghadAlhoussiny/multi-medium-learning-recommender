"""
Multi-Medium Learning Content Recommender
CM3070 Final Project - Raghad Mahmoud

Streamlit interface over the hybrid recommendation model built in notebooks 01 to 08.
Run with:  streamlit run app.py
"""

import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Learning Recommender", page_icon="📚", layout="wide")


# ----------------------------------------------------------------------
# Loading. Streamlit caches these so they run once rather than on every click.
# ----------------------------------------------------------------------

@st.cache_resource
def load_model():
    """Load the pre-trained Sentence-BERT model used to encode the user's query."""
    return SentenceTransformer('all-MiniLM-L6-v2')


@st.cache_data
def load_data():
    """Load the catalogues, embeddings and hybrid weights saved by the notebooks."""
    courses = pd.read_csv('courses_final.csv')
    books = pd.read_csv('books_final.csv')
    course_emb = np.load('course_embeddings.npy')
    book_emb = np.load('book_embeddings.npy')
    with open('hybrid_config.pkl', 'rb') as f:
        config = pickle.load(f)
    return courses, books, course_emb, book_emb, config


@st.cache_resource
def load_factors():
    """Latent item factors exported in notebook 08, already in catalogue order."""
    return np.load('course_item_factors.npy'), np.load('book_item_factors.npy')


sbert = load_model()
courses, books, course_emb, book_emb, config = load_data()
course_factors, book_factors = load_factors()

W_SEM = config['w_semantic']
W_POP = config['w_popularity']
W_CF = config['w_collaborative']
THRESHOLD = config['book_threshold']

# One topic value shared across every page
if "topic" not in st.session_state:
    st.session_state.topic = ""


def set_topic(chosen_topic):
    """Callback for the popular topic buttons. A callback runs before the
    script reruns, so writing to session state here is allowed."""
    st.session_state.topic = chosen_topic


# ----------------------------------------------------------------------
# The hybrid model, as built in notebook 06
# ----------------------------------------------------------------------

def recommend(query, catalogue, liked_rows=None, top_k=5):
    """Rank a catalogue by the weighted combination of the three signals."""
    if catalogue == 'courses':
        table, emb, factors = courses, course_emb, course_factors
    else:
        table, emb, factors = books, book_emb, book_factors

    # Semantic signal from the typed query
    q = sbert.encode([query], show_progress_bar=False)
    semantic = np.clip(cosine_similarity(q, emb).flatten(), 0, 1)

    popularity = table['popularity_norm'].values

    # Collaborative signal only when the user named items they valued
    if liked_rows:
        profile = factors[liked_rows].mean(axis=0).reshape(1, -1)
        collab = np.clip(cosine_similarity(profile, factors).flatten(), 0, 1)
        score = W_CF * collab + W_SEM * semantic + W_POP * popularity
    else:
        # No history, so the collaborative weight is shared out
        total = W_SEM + W_POP
        score = (W_SEM * semantic + W_POP * popularity) / total
        collab = np.zeros(len(table))

    result = table.copy()
    result['score'] = score
    result['semantic'] = semantic
    result['collab'] = collab
    if liked_rows:
        result = result.drop(result.index[liked_rows])
    return result.nlargest(top_k, 'score')


# ----------------------------------------------------------------------
# Navigation
# ----------------------------------------------------------------------

page = st.sidebar.radio(
    "Page",
    ["Recommender", "Cross-Medium Explorer", "Model Comparison"]
)
st.sidebar.markdown("---")


# ======================================================================
# PAGE 1 - Recommender
# The main system. A topic goes in, courses and books come out.
# ======================================================================

if page == "Recommender":

    st.title("📚 Multi-Medium Learning Recommender")
    st.caption("Recommends Coursera courses and Goodreads books for the same topic, "
               "using a hybrid of semantic, collaborative and popularity signals.")

    with st.sidebar:
        st.header("Tell me what you need")

        topic = st.text_input("What do you want to learn?", key="topic")

        st.markdown("**Popular topics**")
        for t in ["data science", "starting a business", "personal finance",
                  "python programming", "leadership"]:
            st.button(t, use_container_width=True, on_click=set_topic, args=(t,))

        level = st.radio("Your level", ["New to this topic", "I have some experience"])

        st.markdown("**Show me**")
        want_courses = st.checkbox("Courses", value=True)
        want_books = st.checkbox("Books", value=True)

        liked = st.multiselect(
            "Courses you have already enjoyed (optional)",
            options=courses['name'].tolist(),
            help="Adding these switches on the collaborative signal. "
                 "Two or three related courses work better than one."
        )

        n_results = st.slider("Number of results", 3, 10, 5)
        st.button("⚡ Get recommendations", type="primary", use_container_width=True)

    if not topic:
        # Landing state. Every user of this app arrives with no history,
        # so the first screen explains the system before anything is searched.
        st.markdown("### Start by telling me what you want to learn")
        st.write("Type a subject in the box on the left, or pick one of the popular "
                 "topics. The system searches courses and books at the same time and "
                 "returns the closest matches from both.")

        a, b, c = st.columns(3)
        with a:
            with st.container(border=True):
                st.markdown("**336 courses**")
                st.caption("Coursera, filtered to technical and professional subjects")
        with b:
            with st.container(border=True):
                st.markdown("**593 books**")
                st.caption("Goodreads, filtered to educational non-fiction")
        with c:
            with st.container(border=True):
                st.markdown("**One shared space**")
                st.caption("Both encoded by Sentence-BERT, so they can be compared")

        st.info("No account and no history needed. Every user arrives with no ratings, "
                "so the system is built around the cold-start case.")

    else:
        # The level answer is appended to the query, which shifts the embedding
        query = f"{topic} for complete beginners, introduction" if level.startswith("New") else topic
        liked_rows = [courses.index.get_loc(i) for i in
                      courses[courses['name'].isin(liked)].index] if liked else None

        st.markdown(f"### Results for *“{topic}”*")
        if liked_rows:
            st.caption(f"Personalised using {len(liked_rows)} course(s) you selected. "
                       f"Weights: collaborative {W_CF}, semantic {W_SEM}, popularity {W_POP}.")
        else:
            st.caption("No history given, so the collaborative weight is redistributed "
                       "across the other two signals.")

        col1, col2 = st.columns(2)

        if want_courses:
            with col1:
                st.subheader("🎓 Courses")
                for _, r in recommend(query, 'courses', liked_rows, n_results).iterrows():
                    with st.container(border=True):
                        st.markdown(f"**{r['name']}**")
                        st.caption(r['institution'])
                        st.progress(min(float(r['score']), 1.0),
                                    text=f"Match {r['score']:.2f}")
                        st.markdown(f"[View on Coursera]({r['course_url']})")

        if want_books:
            with col2:
                st.subheader("📖 Books")
                recs = recommend(query, 'books', None, n_results)
                recs = recs[recs['semantic'] >= THRESHOLD]
                if len(recs) == 0:
                    st.info("No book in the catalogue is closely related to this topic. "
                            "The library covers business, economics, psychology and popular "
                            "science rather than technical manuals.")
                for _, r in recs.iterrows():
                    with st.container(border=True):
                        c1, c2 = st.columns([1, 3])
                        with c1:
                            if isinstance(r.get('image_url'), str):
                                st.image(r['image_url'], width=70)
                        with c2:
                            st.markdown(f"**{r['title'][:70]}**")
                            st.caption(r['authors'])
                            st.progress(min(float(r['score']), 1.0),
                                        text=f"Match {r['score']:.2f}")


# ======================================================================
# PAGE 2 - Cross-Medium Explorer
# Functional requirement (d): match a course directly to books. The
# recommender page works from a typed topic, so it cannot do this.
# ======================================================================

elif page == "Cross-Medium Explorer":

    st.title("🔗 Cross-Medium Explorer")
    st.caption("Which book goes with this course? Courses and books are encoded by the "
               "same model into one 384-dimensional space, so they can be compared "
               "directly even though the two datasets share no users, no identifiers "
               "and no vocabulary.")

    course_names = courses['name'].tolist()
    start_at = course_names.index('Financial Markets') if 'Financial Markets' in course_names else 0

    with st.sidebar:
        st.header("Pick a course")
        chosen = st.selectbox("Course", course_names, index=start_at)
        how_many = st.slider("Books to show", 3, 10, 5)

    row_index = courses.index.get_loc(courses[courses['name'] == chosen].index[0])

    # Compare this single course against every book in the shared space
    course_vector = course_emb[row_index].reshape(1, -1)
    similarity = cosine_similarity(course_vector, book_emb).flatten()

    with st.container(border=True):
        st.markdown(f"**Selected course:** {chosen}")
        st.caption(courses.iloc[row_index]['institution'])

    best = similarity.max()
    matches = books.copy()
    matches['similarity'] = similarity
    matches = matches.nlargest(how_many, 'similarity')

    st.subheader("Closest books in the shared space")

    if best < THRESHOLD:
        st.warning(f"The closest book scores only {best:.2f}, below my confidence "
                   f"threshold of {THRESHOLD}. The system reports no match rather than "
                   f"pairing this course with an unrelated book. This happens most often "
                   f"for programming and technical subjects, because the book catalogue "
                   f"holds popular science and business writing rather than manuals.")
    else:
        # Only show books that clear the confidence threshold
        matches = matches[matches['similarity'] >= THRESHOLD]
        st.caption(f"{len(matches)} book(s) above the {THRESHOLD} confidence threshold.")

        for _, r in matches.iterrows():
            strength = "strong" if r['similarity'] >= 0.45 else "moderate"
            with st.container(border=True):
                c1, c2 = st.columns([1, 4])
                with c1:
                    if isinstance(r.get('image_url'), str):
                        st.image(r['image_url'], width=70)
                with c2:
                    st.markdown(f"**{r['title'][:70]}**")
                    st.caption(r['authors'])
                    st.progress(min(float(r['similarity']), 1.0),
                                text=f"Similarity {r['similarity']:.2f}  ({strength})")

    st.markdown("---")
    st.caption("Across the whole catalogue the mean best cross-medium match is 0.407. "
               "114 courses find a strong partner above 0.45, and 63 find nothing above "
               "0.30, because the book catalogue is weighted towards business and "
               "psychology rather than technical subjects.")


# ======================================================================
# PAGE 3 - Model Comparison
# Functional requirement (f): show what each signal contributes to the
# ranking, so a recommendation can be understood rather than just accepted.
# ======================================================================

elif page == "Model Comparison":

    st.title("⚖️ Model Comparison")
    st.caption("The same query ranked three different ways, so the contribution of each "
               "signal in the hybrid can be seen directly.")

    with st.sidebar:
        st.header("Query")
        query = st.text_input("Topic", key="topic")
        how_many = st.slider("Results per model", 3, 10, 5)

    if not query:
        st.info("Type a topic in the sidebar to compare the three models.")
        st.stop()

    q = sbert.encode([query], show_progress_bar=False)
    semantic = np.clip(cosine_similarity(q, course_emb).flatten(), 0, 1)
    popularity = courses['popularity_norm'].values
    blended = (W_SEM * semantic + W_POP * popularity) / (W_SEM + W_POP)

    def ranked(scores):
        """Attach a score column and return the highest scoring rows."""
        out = courses.copy()
        out['score'] = scores
        return out.nlargest(how_many, 'score')

    st.markdown(f"### Ranked for *“{query}”*")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("Popularity")
        st.caption("Ignores the query entirely.")
        for _, r in ranked(popularity).iterrows():
            with st.container(border=True):
                st.markdown(f"**{r['name']}**")
                st.caption(f"{r['institution']} · {r['score']:.2f}")

    with c2:
        st.subheader("Semantic")
        st.caption("Meaning only, ignores how well regarded an item is.")
        for _, r in ranked(semantic).iterrows():
            with st.container(border=True):
                st.markdown(f"**{r['name']}**")
                st.caption(f"{r['institution']} · {r['score']:.2f}")

    with c3:
        st.subheader("Hybrid")
        st.caption(f"Semantic {W_SEM/(W_SEM+W_POP):.2f}, popularity {W_POP/(W_SEM+W_POP):.2f}.")
        for _, r in ranked(blended).iterrows():
            with st.container(border=True):
                st.markdown(f"**{r['name']}**")
                st.caption(f"{r['institution']} · {r['score']:.2f}")

    st.markdown("---")

    # ---- Project evaluation summary -------------------------------------
    # Not a user feature. Measured results from notebook 07, included so the
    # system's performance can be inspected alongside its output.
    with st.expander("Project evaluation results (notebook 07)"):
        st.caption("Fixed results measured across 500 held-out test cases. "
                   "They do not change with the query above.")
        try:
            st.markdown("**Courses**")
            st.dataframe(pd.read_csv('evaluation_courses.csv'), use_container_width=True)
            st.markdown("**Books**")
            st.dataframe(pd.read_csv('evaluation_books.csv'), use_container_width=True)
        except FileNotFoundError:
            st.info("Evaluation files not found in this folder.")

        st.write("The popularity baseline finishes level with the hybrid on NDCG@10 "
                 "(0.1745 against 0.1742). Held-out review data is itself concentrated "
                 "on already-popular items, so a model that recommends popular things "
                 "scores well by construction. The comparison above shows what that tie "
                 "looks like to a learner with a specific topic in mind.")