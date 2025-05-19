from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('start_diagnosing/', views.start_diagnosing, name='start_diagnosing'),
]
