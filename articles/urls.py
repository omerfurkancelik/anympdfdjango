from django.urls import path
from . import views

urlpatterns = [
    # General
    path('', views.home, name='home'),
    
    # User section
    path('upload/', views.upload_article, name='upload_article'),
    path('upload/success/<str:tracking_code>/', views.upload_success, name='upload_success'),
    path('track/', views.track_article, name='track_article'),
    path('article/chat/<str:tracking_code>/', views.article_chat, name='article_chat'),
    
    # Editor section
    path('editor/dashboard/', views.editor_dashboard, name='editor_dashboard'),
    path('editor/review/<int:article_id>/', views.editor_review, name='editor_review'),
    path('editor/chat/<int:article_id>/', views.editor_chat, name='editor_chat'),
    path('editor/article/<int:article_id>/assign/', views.assign_referee, name='assign_referee'),
    path('editor/article/<int:article_id>/delete/', views.delete_article, name='delete_article'),
    
    # Referee section
    path('referees/', views.referee_list, name='referee_list'),
    path('referee/<int:referee_id>/dashboard/', views.referee_dashboard, name='referee_dashboard'),
    path('referee/<int:referee_id>/review/<int:article_id>/', views.referee_review, name='referee_review'),
    path('referee/dashboard/', views.referee_dashboard, name='referee_dashboard_default'),
    path('referees/add/', views.add_referee, name='add_referee'),

    

    
    # File download
    path('download/<int:article_id>/', views.download_article, name='download_article'),
    
]