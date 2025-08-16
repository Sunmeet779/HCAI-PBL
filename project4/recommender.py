import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
import os

class MatrixFactorizationRecommender:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        ratings_path = os.path.join(base_dir, 'data', 'ratings.csv')
        movies_path = os.path.join(base_dir, 'data', 'movies.csv')
        
        self.ratings_df = pd.read_csv(ratings_path)
        self.movies_df = pd.read_csv(movies_path)
        
        # Extract genres
        self.movies_df['genres'] = self.movies_df['genres'].str.split('|')
        self.genre_list = sorted(list(set([g for sublist in self.movies_df['genres'] for g in sublist])))
        
        # Create genre matrix
        self.genre_matrix = pd.DataFrame(0, index=self.movies_df['movieId'], 
                                        columns=self.genre_list)
        for _, row in self.movies_df.iterrows():
            for genre in row['genres']:
                self.genre_matrix.loc[row['movieId'], genre] = 1
                
        self.movie_ids = None
        self.V = None
        self.user_means = None

    def train(self, k=20, lambda_reg=0.1):
        R_df = self.ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
        self.movie_ids = R_df.columns
        R = R_df.to_numpy()
        self.user_means = np.mean(R, axis=1).reshape(-1, 1)
        R_demeaned = R - self.user_means
        U, sigma, Vt = svds(R_demeaned, k=k)
        self.V = Vt.T  # shape: movieId x k
        self.movie_index_map = {mid: idx for idx, mid in enumerate(self.movie_ids)}
        return self

    def predict_impact(self, movie_id, rating, current_ratings={}):
        """Predict how rating a movie would affect recommendations"""
        if not current_ratings:
            test_ratings = {movie_id: rating}
        else:
            test_ratings = current_ratings.copy()
            test_ratings[movie_id] = rating
            
        return self.recommend_for_new_user(test_ratings.keys(), test_ratings.values())

    def recommend_for_new_user(self, movie_ids, ratings):
        indices = [self.movie_index_map[mid] for mid in movie_ids if mid in self.movie_index_map]
        if not indices:
            return pd.DataFrame(columns=['movieId', 'score', 'title', 'genres'])

        V_sub = self.V[indices]
        ratings = np.array(list(ratings)).reshape(-1, 1)
        
        # Solve for new user factors
        U_new = np.linalg.solve(
            V_sub.T @ V_sub + 0.1 * np.eye(V_sub.shape[1]),
            V_sub.T @ ratings
        )

        # Calculate scores for all movies
        scores = self.V @ U_new
        scores = scores.flatten()
        
        # Add user mean back
        user_mean = np.mean(ratings) if len(ratings) > 0 else 3.0
        scores = scores + user_mean
        scores = np.clip(scores, 0.5, 5.0)  # Keep within rating bounds

        # Create results dataframe
        scored_movies = pd.DataFrame({
            'movieId': self.movie_ids,
            'score': scores
        }).merge(self.movies_df, on='movieId')

        # Exclude already rated movies
        rated_set = set(movie_ids)
        recommendations = scored_movies[~scored_movies['movieId'].isin(rated_set)]
        
        # Sort by score and return top 5
        return recommendations.sort_values(by='score', ascending=False).head(5)

    def get_similar_movies(self, movie_id, n=5):
        """Get movies similar in genre and latent factors"""
        if movie_id not in self.movie_index_map:
            return pd.DataFrame(columns=['movieId', 'title', 'genres'])
            
        idx = self.movie_index_map[movie_id]
        movie_vec = self.V[idx].reshape(1, -1)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(movie_vec, self.V).flatten()
        
        # Get top similar movies (excluding itself)
        similar_indices = np.argsort(similarities)[-n-1:-1][::-1]
        similar_movies = pd.DataFrame({
            'movieId': [self.movie_ids[i] for i in similar_indices],
            'similarity': [similarities[i] for i in similar_indices]
        }).merge(self.movies_df, on='movieId')
        
        return similar_movies