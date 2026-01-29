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

