# project2/urls.py
from django.urls import path
from . import views

app_name = 'project2'

urlpatterns = [
    # Main View
    path('', views.index, name='index'),
    
    # Model Training
    path('train/<str:model_type>/', views.train_model, name='train_model'),
    path('train-baseline/<str:model_type>/', views.train_baseline_model, name='train_baseline_model'),
    path('load-pretrained/', views.load_pretrained_model, name='load_pretrained'),
    path('list-models/', views.list_models_api, name='list_models'),
    # Active Learning Endpoints
    path('active-learning/start/', views.start_active_learning, name='start_active_learning'),
    path('active-learning/next-batch/', views.get_next_batch, name='get_next_batch'),
    path('active-learning/submit-labels/', views.submit_labels, name='submit_labels'),
    path('active-learning/progress/', views.learning_progress, name='learning_progress'),
]