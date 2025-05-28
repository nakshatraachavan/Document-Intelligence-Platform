from django.db import models

# Create your models here.
class Document(models.Model):
    title = models.CharField(max_length=255)
    file = models.FileField(upload_to='documents/')
    type = models.CharField(max_length=10)
    size = models.IntegerField()
    pages = models.IntegerField()
    status = models.CharField(max_length=20, default='processing')
    created_at = models.DateTimeField(auto_now_add=True)

class DocumentChunk(models.Model):
    document = models.ForeignKey(Document, on_delete=models.CASCADE)
    chunk_index = models.IntegerField()
    text = models.TextField()
    page_number = models.IntegerField()
    embedding_id = models.CharField(max_length=255)
