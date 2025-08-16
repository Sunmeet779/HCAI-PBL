from django.shortcuts import render, redirect
from django.http import JsonResponse
from .recommender import MatrixFactorizationRecommender
from .generate_pdf import generate_method_study_pdf
import pandas as pd
import os
from django.template.defaulttags import register

rec = MatrixFactorizationRecommender().train()

@register.filter
def split(value, arg):
    return value.split(arg)

def load_sample_movies(n=10):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    movies_path = os.path.join(base_dir, 'data', 'movies.csv')
    movies_df = pd.read_csv(movies_path)
    movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)')
    movies_df['year'] = pd.to_numeric(movies_df['year'], errors='coerce').fillna(1995).astype(int)
    movies_df['title'] = movies_df['title'].str.replace(r'\(\d{4}\)', '', regex=True).str.strip()
    movies_df['genres'] = movies_df['genres'].str.split('|')
    sample = movies_df.sample(n).to_dict('records')
    return sample

def index(request):
    return render(request, 'project4/index.html')

def study_interface(request):
    movies = load_sample_movies()
    request.session['movies'] = movies
    return render(request, 'project4/study_interface.html', {'movies': movies})

def predict_impact(request):
    movie_id = int(request.GET.get('movie_id'))
    rating = float(request.GET.get('rating'))
    current_ratings = {}
    if 'current_ratings' in request.session:
        current_ratings = {str(k): v for k, v in request.session['current_ratings'].items()}
    current_ratings[str(movie_id)] = rating
    movie_ids = [int(mid) for mid in current_ratings.keys()]
    ratings = [current_ratings[mid] for mid in current_ratings.keys()]
    recommendations = rec.recommend_for_new_user(movie_ids, ratings)
    rec_list = []
    for _, row in recommendations.iterrows():
        rec_list.append({
            'movieId': row['movieId'],
            'title': row['title'],
            'genres': ', '.join(row['genres']) if isinstance(row['genres'], list) else row['genres'],
            'score': float(row['score'])
        })
    return JsonResponse({
        'status': 'success',
        'recommendations': rec_list,
        'rated_movie': {
            'movieId': movie_id,
            'rating': rating
        }
    })

def submit_ratings(request):
    movies = request.session.get('movies', [])
    rated_movie_ids = []
    user_ratings = []
    current_ratings = {}
    for movie in movies:
        rating_str = request.POST.get(f'rating_{movie["movieId"]}')
        if rating_str:
            try:
                rating = float(rating_str)
                if 0.5 <= rating <= 5.0:
                    rated_movie_ids.append(int(movie['movieId']))
                    user_ratings.append(rating)
                    current_ratings[str(movie['movieId'])] = rating
            except ValueError:
                pass
    request.session['current_ratings'] = current_ratings
    recommendations = rec.recommend_for_new_user(rated_movie_ids, user_ratings)
    recs_with_genres = []
    for _, row in recommendations.iterrows():
        recs_with_genres.append({
            'movieId': row['movieId'],
            'title': row['title'],
            'genres': row['genres'],
            'score': float(row['score'])
        })
    return render(request, 'project4/study_interface.html', {
        'movies': movies,
        'recommendations': recs_with_genres,
    })

def method_and_study(request):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    static_subdir = os.path.join(base_dir, 'static', 'project4')
    os.makedirs(static_subdir, exist_ok=True)
    pdf_path = os.path.join(static_subdir, 'method_and_study.pdf')
    if not os.path.exists(pdf_path):
        generate_method_study_pdf(pdf_path)
    return redirect('/static/project4/method_and_study.pdf')