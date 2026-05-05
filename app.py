import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import sigmoid_kernel

st.set_page_config(page_title="Movie Matcher", page_icon="🎬")
st.title("🎬 TMDB Movie Recommendation System")

@st.cache_resource
def load_all():
    # Load the data we exported from Colab
    data = pd.read_csv("movie_data.csv")
    df = pd.read_csv("movie_dataframe.csv")
    tfidf_matrix = joblib.load("tfidf_matrix.pkl")
    return data, df, tfidf_matrix

data, dataframe, tfv_matrix = load_all()

# Sidebar for selection
st.sidebar.header("User Input")
movie_list = data['original_title'].values
selected_movie = st.sidebar.selectbox("Type or select a movie:", movie_list)

if st.button('Show Recommendations'):
    # Calculate similarity on the fly (Fast for 5000 movies)
    indices = pd.Series(data.index, index=data['original_title'])
    idx = indices[selected_movie]
    
    # Calculate sigmoid kernel for just this movie
    sig_scores = sigmoid_kernel(tfv_matrix[idx], tfv_matrix).flatten()
    
    # Sort and get top 10
    score_series = pd.Series(sig_scores).sort_values(ascending=False)
    top_10_indices = score_series.iloc[1:11].index
    
    recommendations = dataframe['original_title'].iloc[top_10_indices]
    
    st.subheader(f"Recommendations for {selected_movie}:")
    for i, m in enumerate(recommendations, 1):
        st.write(f"{i}. **{m}**")
