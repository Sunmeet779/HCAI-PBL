import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
import json
import os
from pathlib import Path

# Get the current script directory
script_dir = Path(__file__).parent.resolve()

# Create directories
data_dir = script_dir / 'data'
model_dir = script_dir / 'model'
model_dir.mkdir(exist_ok=True)

# Load data
print("Loading data...")
try:
    ratings = pd.read_csv(data_dir / 'ratings.csv')
    movies = pd.read_csv(data_dir / 'movies.csv')
    print(f"Found {len(ratings)} ratings and {len(movies)} movies")
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    print(f"Looking in: {data_dir}")
    print("Please download the dataset from https://files.grouplens.org/datasets/movielens/ml-latest-small.zip")
    print("And extract it into the 'data' directory")
    exit(1)

# Extract year from title
print("Processing movie titles...")
movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')
movies['year'] = pd.to_numeric(movies['year'], errors='coerce').fillna(1995).astype(int)
movies['title'] = movies['title'].str.replace(r'\(\d{4}\)', '', regex=True).str.strip()

# Create user-item matrix
print("Creating user-item matrix...")
user_item_matrix = ratings.pivot_table(
    index='userId', 
    columns='movieId', 
    values='rating',
    fill_value=0
)

matrix = user_item_matrix.values
user_ratings_mean = np.mean(matrix, axis=1)
matrix_normalized = matrix - user_ratings_mean.reshape(-1, 1)

# Perform Singular Value Decomposition
print("Performing SVD (this may take a few minutes)...")
k = 50  # Number of latent factors
U, sigma, Vt = svds(matrix_normalized.astype('float64'), k=k)

# Ensure singular values are in descending order
sigma = np.diag(sigma[::-1])
U = U[:, ::-1]
Vt = Vt[::-1, :]

# Save factors
print("Saving model files...")
np.save(model_dir / 'user_factors.npy', U)
np.save(model_dir / 'movie_factors.npy', Vt)

# Save movie metadata with correct indices
movie_indices = user_item_matrix.columns.tolist()
movie_metadata = movies[movies['movieId'].isin(movie_indices)].set_index('movieId').loc[movie_indices].reset_index()
movie_metadata.to_json(model_dir / 'movie_metadata.json', orient='records')

print("Model training complete!")
print(f"User factors shape: {U.shape}")
print(f"Movie factors shape: {Vt.shape}")
print(f"Movies metadata count: {len(movie_metadata)}")
print(f"Files saved in: {model_dir}")