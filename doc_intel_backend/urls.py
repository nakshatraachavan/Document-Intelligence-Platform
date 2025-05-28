# doc_intel_backend/doc_intel_backend/urls.py

from django.contrib import admin
from django.urls import path, include # Make sure 'include' is here
from django.conf import settings
from django.conf.urls.static import static

# 1. Initialize urlpatterns as a list first
urlpatterns = [
    # 2. Add your base project URLs here
    path('admin/', admin.site.urls),
    path('api/', include('documents.urls')), # This assumes you have an app named 'documents'
]

# 3. Conditionally add media/static serving for development (if DEBUG is True)
# This should always come AFTER urlpatterns is defined.
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    # It's also good practice to serve static files this way in debug mode
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)