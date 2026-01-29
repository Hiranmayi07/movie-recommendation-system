import pandas as pd
import numpy as np
import os

# Optional: Check current working directory
print("Current working directory:", os.getcwd())

# ==============================
# STEP 1: LOAD DATA
# ==============================

# ---- Load Ratings ----
ratings = pd.read_csv('ratings.csv')  
# Expected columns: userId, movieId, rating, timestamp

# ---- Load Movies ----
movies = pd.read_csv('movies.csv', usecols=['movieId', 'title'])

# ---- Rename columns for consistency ----
ratings.rename(columns={
    'userId': 'user_id',
    'movieId': 'movie_id'
}, inplace=True)

movies.rename(columns={
    'movieId': 'movie_id'
}, inplace=True)

# ---- Quick Look ----
print("\nFirst 5 rows of Ratings:")
print(ratings.head())

print("\nFirst 5 rows of Movies:")
print(movies.head())

# ==============================
# STEP 2: USER–ITEM MATRIX
# ==============================

user_item_matrix = ratings.pivot_table(
    index='user_id',
    columns='movie_id',
    values='rating'
)

print("\nUser-Item Matrix (first 5 users):")
print(user_item_matrix.head())

# ---- Sparsity Calculation ----
num_users, num_movies = user_item_matrix.shape
num_ratings = ratings.shape[0]

sparsity = 100 * (1 - (num_ratings / (num_users * num_movies)))
print(f"\nSparsity of User-Item Matrix: {sparsity:.2f}%")

# ==============================
# STEP 3: PREPROCESSING & NORMALIZATION
# ==============================

# 1. Fill missing ratings with 0 (for similarity computation)
user_item_matrix_filled = user_item_matrix.fillna(0)

print("\nUser-Item Matrix after filling NaNs with 0:")
print(user_item_matrix_filled.head())

# 2. User mean normalization
user_mean = user_item_matrix.mean(axis=1)

user_item_matrix_normalized = user_item_matrix.sub(user_mean, axis=0).fillna(0)

print("\nUser-Item Matrix after user-mean normalization:")
print(user_item_matrix_normalized.head())
from sklearn.metrics.pairwise import cosine_similarity

# ==============================
# STEP 4: USER-USER SIMILARITY
# ==============================

# Compute cosine similarity
user_similarity = cosine_similarity(user_item_matrix_normalized)

# Convert to DataFrame for readability
user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_item_matrix_normalized.index,
    columns=user_item_matrix_normalized.index
)

print("\nUser-User Similarity Matrix (first 5 users):")
print(user_similarity_df.head())
# ==============================
# STEP 5: RECOMMENDATION FUNCTION
# ==============================

def recommend_movies(
    target_user_id,
    user_item_matrix,
    user_similarity_df,
    movies,
    top_k=5,
    top_n=5
):
    # Get similarity scores for target user
    sim_scores = user_similarity_df[target_user_id]

    # Remove self-similarity
    sim_scores = sim_scores.drop(target_user_id)

    # Select top-K similar users
    top_users = sim_scores.sort_values(ascending=False).head(top_k)

    # Movies already rated by target user
    target_user_ratings = user_item_matrix.loc[target_user_id]

    recommendations = {}

    for movie_id in user_item_matrix.columns:
        # Skip movies already rated
        if not pd.isna(target_user_ratings[movie_id]):
            continue

        weighted_sum = 0
        sim_sum = 0

        for similar_user, similarity in top_users.items():
            rating = user_item_matrix.loc[similar_user, movie_id]

            if not pd.isna(rating):
                weighted_sum += similarity * rating
                sim_sum += abs(similarity)

        if sim_sum > 0:
            recommendations[movie_id] = weighted_sum / sim_sum

    # Sort recommendations
    recommended_movies = sorted(
        recommendations.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    # Convert to DataFrame
    rec_df = pd.DataFrame(recommended_movies, columns=['movie_id', 'predicted_rating'])

    # Merge with movie titles
    rec_df = rec_df.merge(movies, on='movie_id')

    return rec_df
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ==============================
# STEP 6: TRAIN–TEST SPLIT
# ==============================

train_data, test_data = train_test_split(
    ratings,
    test_size=0.2,
    random_state=42
)

# Create train user-item matrix
train_matrix = train_data.pivot_table(
    index='user_id',
    columns='movie_id',
    values='rating'
)

# Normalize train matrix
train_user_mean = train_matrix.mean(axis=1)
train_matrix_norm = train_matrix.sub(train_user_mean, axis=0).fillna(0)

# Compute similarity on train data
train_similarity = cosine_similarity(train_matrix_norm)

train_similarity_df = pd.DataFrame(
    train_similarity,
    index=train_matrix_norm.index,
    columns=train_matrix_norm.index
)

# ==============================
# STEP 6: PREDICTION FUNCTION
# ==============================

def predict_rating(user_id, movie_id):
    if user_id not in train_matrix.index or movie_id not in train_matrix.columns:
        return np.nan

    sim_scores = train_similarity_df[user_id].drop(user_id)
    movie_ratings = train_matrix[movie_id]

    mask = movie_ratings.notna()
    sim_scores = sim_scores[mask]
    movie_ratings = movie_ratings[mask]

    if sim_scores.sum() == 0:
        return np.nan

    return np.dot(sim_scores, movie_ratings) / np.sum(np.abs(sim_scores))


# ==============================
# STEP 6: RMSE COMPUTATION
# ==============================

y_true = []
y_pred = []

for _, row in test_data.iterrows():
    pred = predict_rating(row['user_id'], row['movie_id'])
    if not np.isnan(pred):
        y_true.append(row['rating'])
        y_pred.append(pred)

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"\nRMSE of Recommendation System: {rmse:.4f}")
target_user = user_item_matrix.index[0]

recs = recommend_movies(
    target_user_id=target_user,
    user_item_matrix=user_item_matrix,
    user_similarity_df=user_similarity_df,
    movies=movies,
    top_k=5,
    top_n=5
)

print("\nRecommended Movies:")
print(recs)
