from django.urls import path
from . import views

urlpatterns = [
    path('', views.oddsComparison, name='odds'),
]
