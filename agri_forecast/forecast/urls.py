from django.contrib import admin
from django.urls import path
from forecast.views import home_view, forecast_view, dataset_view, download_dataset

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home_view, name='home'),  # Root URL
    path('home/', home_view, name='home'),
    path('forecast/', forecast_view, name='forecast'),
    path('dataset/', dataset_view, name='dataset'),  # if dataset view is needed
    path('download/', download_dataset, name='download_dataset'),  # if download view is needed
]
