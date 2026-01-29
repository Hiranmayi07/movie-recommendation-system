import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Robust paths (works regardless of working directory)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

RATINGS_FILE = os.path.join(DATA_DIR, "ratings.csv")
MOVIES_FILE = os.path.join(DATA_DIR, "movies.csv")

# -----------------------------
# Load datasets
# -----------------------------
ratings = pd.read_csv(RATINGS_FILE, names=['user_id','movie_id','rating','timestamp'], skiprows=1)
movies = pd.read_csv(MOVIES_FILE, usecols=[0,1], names=['movie_id','title'], skiprows=1)

# -----------------------------
# Create user-item matrix
# -----------------------------
user_item_matrix = ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)

# -----------------------------
# Precompute user-user similarity
# -----------------------------
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# -----------------------------
# Recommendation function
# -----------------------------
def recommend_movies(target_user_id, user_item_matrix, user_similarity_df, movies, top_n=5):
    sim_scores = user_similarity_df[target_user_id]
    user_ratings = user_item_matrix.T
    pred_ratings = (user_ratings * sim_scores).sum(axis=1) / sim_scores.sum()
    # Remove already rated movies
    rated = user_item_matrix.loc[target_user_id]
    pred_ratings[rated > 0] = 0
    # Top-N recommendations
    top_movies = pred_ratings.sort_values(ascending=False).head(top_n)
    top_movies = pd.DataFrame({'movie_id': top_movies.index, 'predicted_rating': top_movies.values})
    top_movies = top_movies.merge(movies, on='movie_id')
    return top_movies[['title', 'predicted_rating']]

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🎬 Movie Recommendation System")

user_ids = user_item_matrix.index.tolist()
selected_user = st.selectbox("Select User ID:", user_ids)

top_n = st.slider("Number of Recommendations:", 1, 10, 5)

if st.button("Get Recommendations"):
    recommendations = recommend_movies(selected_user, user_item_matrix, user_similarity_df, movies, top_n)
    st.write("### Recommended Movies:")
    st.table(recommendations)
