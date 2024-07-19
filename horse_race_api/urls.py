from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('prediction/', include('prediction.urls')),
    path('odds/', include('odds.urls')),
]
