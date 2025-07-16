from django.shortcuts import render, redirect
from .recommender import MatrixFactorizationRecommender
from .generate_pdf import generate_method_study_pdf
import pandas as pd
import os

# Train model once
rec = MatrixFactorizationRecommender().train()

def load_sample_movies(n=10):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    movies_path = os.path.join(base_dir, 'data', 'movies.csv')
    return pd.read_csv(movies_path).sample(n).to_dict('records')

def index(request):
    return render(request, 'project4/index.html')

def study_interface(request):
    movies = load_sample_movies()
    request.session['movies'] = movies
    return render(request, 'project4/study_interface.html', {'movies': movies})

def submit_ratings(request):
    movies = request.session.get('movies')
    if not movies:
        return redirect('project4:study_interface')

    rated_movie_ids = []
    user_ratings = []

    for movie in movies:
        rating_str = request.POST.get(f'rating_{movie["movieId"]}')
        if rating_str:
            try:
                rating = float(rating_str)
                if 0.5 <= rating <= 5.0:
                    rated_movie_ids.append(movie['movieId'])
                    user_ratings.append(rating)
            except ValueError:
                pass

    recommendations = rec.recommend_for_new_user(rated_movie_ids, user_ratings)
    return render(request, 'project4/study_interface.html', {
        'movies': movies,
        'recommendations': recommendations.to_dict('records'),
    })

def method_and_study(request):
    from .generate_pdf import generate_method_study_pdf
    base_dir = os.path.dirname(os.path.abspath(__file__))
    static_subdir = os.path.join(base_dir, 'static', 'project4')
    os.makedirs(static_subdir, exist_ok=True)

    pdf_path = os.path.join(static_subdir, 'method_and_study.pdf')
    if not os.path.exists(pdf_path):
        generate_method_study_pdf(pdf_path)

    return redirect('/static/project4/method_and_study.pdf')

