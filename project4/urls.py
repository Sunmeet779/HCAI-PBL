from django.urls import path
from . import views

app_name = "project4"

urlpatterns = [
    path("", views.index, name="index"),
    path("study/", views.study_interface, name="study_interface"),
    path("submit_ratings/", views.submit_ratings, name="submit_ratings"),
    path("predict_impact/", views.predict_impact, name="predict_impact"),
    path("method_and_study/", views.method_and_study, name="method_and_study"),
]
