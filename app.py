import streamlit as st
import joblib
import polars as pl
from scipy.sparse import hstack

# ---- Load model + data ----
@st.cache_resource
def load_model():
    nn_model = joblib.load("model/nn_model.pkl")
    title_vectorizer = joblib.load("vectorizers/title_vectorizer.pkl")
    desc_vectorizer = joblib.load("vectorizers/desc_vectorizer.pkl")
    books = pl.read_parquet("data/books.parquet")
    return nn_model, title_vectorizer, desc_vectorizer, books

nn_model, title_vectorizer, desc_vectorizer, books = load_model()

# ---- UI ----
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Alice&display=swap" rel="stylesheet">
    <style>
    h1, h2, h3 {
        font-family: 'Alice', sans-serif;
    }
    p {
        font-family: 'Alice', sans-serif;
    }
    .rating {
        font-family: 'Alice', sans-serif;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1>Nastenka's Library.</h1>", unsafe_allow_html=True)
st.write("I read. You read. Perhaps the next story waits for you here.")

new_title = st.text_input("Book Title", "")
new_desc  = st.text_area("Book Description (brief notes about the book's content would work best.)", "")

if st.button("Find Your Next Night") and new_title and new_desc:
    # Vectorize new input
    new_title_vec = title_vectorizer.transform([new_title]) * 0.5
    new_desc_vec  = desc_vectorizer.transform([new_desc])  * 3.0
    new_vec = hstack([new_title_vec, new_desc_vec])
    
    # Run nearest neighbors
    distances, indices = nn_model.kneighbors(new_vec, n_neighbors=5)
    recs = books[indices[0].tolist(), :]  # grab rows
    
    def get_cover(isbn, fallback):
        if isbn:
            return f"https://covers.openlibrary.org/b/isbn/{isbn}-L.jpg"
        return fallback


    st.subheader("Recommended for you:")

    custom_placeholder = "https://raw.githubusercontent.com/amirsohil/randomimages/refs/heads/main/placeholder.png"

    for row in recs.iter_rows(named=True):
        st.markdown(f"### {row['title_without_series']}")

        # Try OpenLibrary first, then fallback to Goodreads image_url, then placeholder
        img_url = get_cover(row.get("isbn", None), row.get("image_url", ""))  
        if not img_url or "nophoto" in img_url:
            img_url = custom_placeholder

        st.image(img_url, width=150)

        if row.get("description"):
            st.markdown(
                f'<p style="text-align: justify; font-family: Alice, sans-serif;">{row["description"]}</p>',
                unsafe_allow_html=True
                )

        st.markdown(f"Average Rating: ⭐ **{row['average_rating']}**")
        st.markdown("---")


