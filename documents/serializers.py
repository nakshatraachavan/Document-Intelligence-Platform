# documents/serializers.py

from rest_framework import serializers
from .models import Document

class DocumentUploadSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = ['id', 'title', 'file', 'type', 'size', 'pages', 'status', 'created_at']
        read_only_fields = ['id', 'pages', 'status', 'created_at']
