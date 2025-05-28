# documents/views.py

import os
from rest_framework.views import APIView
from rest_framework.generics import ListAPIView # <--- ADD THIS IMPORT
from rest_framework.response import Response
from rest_framework import status
from .models import Document
from .serializers import DocumentUploadSerializer
from .rag_engine import process_document



class DocumentListView(ListAPIView):
    queryset = Document.objects.all()
    serializer_class = DocumentUploadSerializer

class DocumentUploadView(APIView):
    def post(self, request, format=None):
        uploaded_file = request.FILES.get('file')

        if not uploaded_file:
            return Response({'error': 'No file uploaded.'}, status=status.HTTP_400_BAD_REQUEST)

        file_type = uploaded_file.name.split('.')[-1].lower()
        file_size = uploaded_file.size
        title = uploaded_file.name

        document = Document.objects.create(
            title=title,
            file=uploaded_file,
            type=file_type,
            size=file_size,
            pages=0,  # Temporary, update after processing
            status='processing'
        )
        process_document(document)

        # TODO: Trigger async document processing
        # e.g., process_document(document)

        serializer = DocumentUploadSerializer(document)
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    