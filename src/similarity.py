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
