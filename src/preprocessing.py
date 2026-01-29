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
