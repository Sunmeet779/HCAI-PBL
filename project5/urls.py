from django.urls import path
from . import views

app_name = 'project5'

urlpatterns = [
    path('', views.index, name='index'),
    path('start/', views.start_training, name='start_training'),
    path('feedback/<int:session_id>/', views.collect_feedback, name='collect_feedback'),
    path('retrain/<int:session_id>/', views.retrain_policy, name='retrain_policy'),
    path('reset/', views.reset_training, name='reset_training'),
]