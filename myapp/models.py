from django.db import models

class Document(models.Model):
    # Auto-incrementing ID field (Django creates this by default)
    name = models.CharField(max_length=255)
    file = models.FileField(upload_to='uploads/')
    file_type = models.CharField(max_length=50)
    size = models.BigIntegerField()
    pages = models.IntegerField(default=1)
    chroma_id = models.CharField(max_length=255, null=True, blank=True)  # ChromaDB Document ID
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

class DocumentChunk(models.Model):
    document = models.ForeignKey(Document, on_delete=models.CASCADE)
    chunk_index = models.IntegerField()
    page_number = models.IntegerField()
    text = models.TextField()
    embedding_id = models.CharField(max_length=255)  # ChromaDB DocumentChunk ID
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('document', 'chunk_index')

    def __str__(self):
        return f"Chunk {self.chunk_index} of {self.document.name}"