# myapp/urls.py
from django.urls import path
from . import views
from .views import DocumentUploadView, DocumentListView, DocumentDetailView, AskView

urlpatterns = [
    path('', views.home, name='home'),
    path('document', DocumentUploadView.as_view(), name='document-upload'),
    path('document/list', DocumentListView.as_view(), name='document-list'),
    path('document/<int:doc_id>', DocumentDetailView.as_view(), name='document-detail'),
    path('ask', AskView.as_view(), name='ask'),
]