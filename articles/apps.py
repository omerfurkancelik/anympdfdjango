from django.apps import AppConfig
import os


class ArticlesConfig(AppConfig):
    name = 'articles'
    
    def ready(self):
        # Ensure media directories exist when app starts
        from django.conf import settings
        
        media_root = settings.MEDIA_ROOT
        
        # Make sure the media root exists
        os.makedirs(media_root, exist_ok=True)
        
        # Make sure the articles directory exists
        articles_dir = os.path.join(media_root, 'articles')
        os.makedirs(articles_dir, exist_ok=True)
        
        # Make sure the anonymized_articles directory exists
        anon_dir = os.path.join(media_root, 'anonymized_articles')
        os.makedirs(anon_dir, exist_ok=True)
