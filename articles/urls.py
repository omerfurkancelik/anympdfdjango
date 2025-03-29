from django.urls import path
from . import views  # Bu satır önemli, tüm views'leri import ediyoruz
import os
from django.contrib import admin

urlpatterns = [
    # General
    path('', views.home, name='home'),
    path('admin/', admin.site.urls),
    
    # User section
    path('upload/', views.upload_article, name='upload_article'),
    path('upload/success/<str:tracking_code>/', views.upload_success, name='upload_success'),
    path('track/', views.track_article, name='track_article'),
    path('article/chat/<str:tracking_code>/', views.article_chat, name='article_chat'),
    path('track/', views.track_article, name='track_article'),
    path('article/chat/<str:tracking_code>/', views.article_chat, name='article_chat'),
    
    # Editor section
    path('editor/dashboard/', views.editor_dashboard, name='editor_dashboard'),
    path('editor/review/<int:article_id>/', views.editor_review, name='editor_review'),
    path('editor/chat/<int:article_id>/', views.editor_chat, name='editor_chat'),
    path('editor/article/<int:article_id>/assign/', views.assign_referee, name='assign_referee'),
    path('editor/article/<int:article_id>/delete/', views.delete_article, name='delete_article'),
    path('editor/article/<int:article_id>/metadata/', views.process_article_metadata, name='process_article_metadata'),
    path('editor/article/<int:article_id>/anonymize/', views.anonymize_article, name='anonymize_article'),
    path('editor/article/<int:article_id>/download-anonymized/', views.download_anonymized_article, name='download_anonymized_article'),
    path('editor/article/<int:article_id>/suggest-referees/', views.suggest_referees, name='suggest_referees'),
    path('editor/reset-database/', views.reset_database, name='reset_database'),
    path('editor/reset-database-orm/', views.reset_database_orm, name='reset_database_orm'),
    
    
    # Referee section
    path('referees/', views.referee_list, name='referee_list'),
    path('referee/<int:referee_id>/dashboard/', views.referee_dashboard, name='referee_dashboard'),
    path('referee/<int:referee_id>/review/<int:article_id>/', views.referee_review, name='referee_review'),
    path('referee/dashboard/', views.referee_dashboard, name='referee_dashboard_default'),
    path('referees/add/', views.add_referee, name='add_referee'),
    path('referee/article/<int:article_id>/<int:referee_id>/restore-info/', views.restore_article_info, name='restore_article_info'),
    path('referee/article/<int:article_id>/<int:referee_id>/quick-action/', views.referee_quick_action, name='referee_quick_action'),

    
    # File download
    path('download/<int:article_id>/', views.download_article, name='download_article'),
    
]


