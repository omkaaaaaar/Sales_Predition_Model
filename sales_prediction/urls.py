from django.urls import path
from . import views

urlpatterns = [
    # Route for uploading a file and predicting sales
    path('', views.upload_file, name='upload_file'),

    # Route for displaying the prediction results
    path('results/', views.results, name='results'),

    path('download_predictions/',views.download_predictions,name='download_predictions'),

     path('download_sample_csv/', views.download_sample_csv, name='download_sample_csv'),

    # Route for uploading a file, grouping by Item_Type, and showing the grouped results
    path('upload-group/', views.upload_file_group_by, name='upload_file_group_by'),
]
