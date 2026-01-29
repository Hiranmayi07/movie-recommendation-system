# 🎬 Movie Recommendation System

A user-based collaborative filtering recommendation system built using Python.

## 🔹 Features
- User–User Collaborative Filtering
- Cosine Similarity with user-mean normalization
- Handles sparse data (98% sparsity)
- Generates Top-N personalized movie recommendations
- Evaluated using RMSE with train-test split

## 🔹 Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn

## 🔹 Dataset
- MovieLens dataset

## 🔹 How it works
1. Build user–item rating matrix
2. Normalize ratings to remove user bias
3. Compute user similarity using cosine similarity
4. Predict unseen movie ratings
5. Evaluate model using RMSE

## 🔹 Results
- RMSE ≈ 0.9 – 1.1 (varies by split)

## 🔹 Future Improvements
- Item-based collaborative filtering
- Cold-start handling
- Hybrid recommender system
## 🚀 Live Demo (Streamlit App)

You can try the interactive movie recommendation system locally using Streamlit:

### Requirements
Make sure you have Python 3.13+ and required libraries installed:

```bash
pip install -r requirements.txt
