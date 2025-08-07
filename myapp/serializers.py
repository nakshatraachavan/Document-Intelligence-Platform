from rest_framework import serializers
from .models import Document, DocumentChunk

class DocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = '__all__'

class ChunkSerializer(serializers.ModelSerializer):
    class Meta:
        model = DocumentChunk
        fields = '__all__'
