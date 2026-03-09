from django.urls import path
from predictor import views

urlpatterns = [
    path("data_exploration", views.data_exploration_view, name="data_exploration"),
    # Adding stubs so template reverse url doesn't fail
    path("regression_analysis", views.regression_analysis, name="regression_analysis"),
    path("classification_analysis", views.data_exploration_view, name="classification_analysis"),
    path("clustering_analysis", views.data_exploration_view, name="clustering_analysis"),
]
