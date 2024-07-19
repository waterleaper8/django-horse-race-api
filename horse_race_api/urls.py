from django.contrib import admin
from django.urls import path, include

urlpatterns = [
<<<<<<< HEAD
    path('api/horse/admin/', admin.site.urls),
    path('api/horse/prediction/',
        include('prediction.urls')),
    path('api/horse/odds/',
        include('odds.urls')),
=======
    path('admin/', admin.site.urls),
    path('prediction/', include('prediction.urls')),
    path('odds/', include('odds.urls')),
>>>>>>> a063277 (update)
]
