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

