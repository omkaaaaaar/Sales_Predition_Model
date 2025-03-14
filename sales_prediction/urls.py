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
    path('admin/', views.admin_view, name='admin'),
    path('forgot-password/', views.forgot_password_view, name='forgot_password'),
    path('blogs/', views.blog_list, name='blog_list'),  # Main page displaying all blog posts
    path('blog/<int:post_id>/', views.blog_detail, name='blog_detail'),  # AJAX or same-page content 
    
    
    path('prediction/', views.predict_sales_from_csv, name='predict_sales'),
    path('prediction_grouped/', views.csv_grouped, name='predict_sales_grouped'),
    path("grouped-predictions/", views.csv_grouped_view, name="grouped_predictions"),
    path('filter_predictions/', views.filter_predictions, name='filter_predictions'),
    path("visualize/", views.visualize_filtered_data, name="visualize_filtered_data"),

]

