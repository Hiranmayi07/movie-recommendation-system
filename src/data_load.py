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
