# documents/urls.py

from django.urls import path
from .views import DocumentUploadView, DocumentListView 

urlpatterns = [
    path('upload/', DocumentUploadView.as_view()),
    path('list/', DocumentListView.as_view()),
]