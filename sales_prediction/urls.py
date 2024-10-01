from django.urls import path
from . import views

urlpatterns = [

     path('', views.index, name='index'),  # Ensure this pattern exists,

    # Route for uploading a file and predicting sales
    path('upload/', views.upload_file, name='upload_file'),

    # Route for displaying the prediction results
    path('results/', views.results, name='results'),

    path('download_predictions/',views.download_predictions,name='download_predictions'),

     path('download_sample_csv/', views.download_sample_csv, name='download_sample_csv'),

    # Route for uploading a file, grouping by Item_Type, and showing the grouped results
    path('upload-group/', views.upload_file_group_by, name='upload_file_group_by'),

    path('signup/', views.signup_view, name='signup'),  # Use 'signup' here
    path('login/', views.login_view, name='login'),  # Ensure 'login' matches

    path('about/',views.about,name='about'),

    path('contact_us/',views.contact,name='contact_us'),
    path('sample/', views.sample, name='sample'),
      # URL for the contact form
    \
    
]
