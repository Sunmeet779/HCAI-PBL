from django.urls import path
from . import views

app_name = 'project3'

urlpatterns = [
    path('', views.index, name='index'),
    path('simple-tree/', views.simple_tree_view, name='simple_tree'),
    path('sparse-tree/', views.sparse_tree_view, name='sparse_tree'),
    path('logistic-regression/', views.logistic_regression_view, name='logistic_regression'),
    path('counterfactuals/', views.counterfactual_explanations, name='counterfactual'),
]