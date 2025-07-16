import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
import os

class MatrixFactorizationRecommender:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        ratings_path = os.path.join(base_dir, 'data', 'ratings.csv')
        self.ratings_df = pd.read_csv(ratings_path)
        self.movie_ids = None
        self.V = None

    def train(self, k=20, lambda_reg=0.1):
        R_df = self.ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
        self.movie_ids = R_df.columns
        R = R_df.to_numpy()
        user_ratings_mean = np.mean(R, axis=1).reshape(-1, 1)
        R_demeaned = R - user_ratings_mean
        U, sigma, Vt = svds(R_demeaned, k=k)
        self.V = Vt.T  # shape: movieId x k
        self.movie_index_map = {mid: idx for idx, mid in enumerate(self.movie_ids)}
        return self

    def recommend_for_new_user(self, movie_ids, ratings):
        indices = [self.movie_index_map[mid] for mid in movie_ids if mid in self.movie_index_map]
        if not indices:
            return pd.DataFrame(columns=['movieId', 'score'])

        V_sub = self.V[indices]
        ratings = np.array(ratings).reshape(-1, 1)

        U_new = np.linalg.solve(
            V_sub.T @ V_sub + 0.1 * np.eye(V_sub.shape[1]),
            V_sub.T @ ratings
        )

        scores = self.V @ U_new
        scores = scores.flatten()

        scored_movies = pd.DataFrame({
            'movieId': self.movie_ids,
            'score': scores
        })

        rated_set = set(movie_ids)
        recommendations = scored_movies[~scored_movies['movieId'].isin(rated_set)]
        recommendations = recommendations.sort_values(by='score', ascending=False).head(5)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        movies_df = pd.read_csv(os.path.join(base_dir, 'data', 'movies.csv'))
        return recommendations.merge(movies_df, on='movieId')
