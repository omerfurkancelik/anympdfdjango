import os
from django.db import models
from django.utils import timezone
from datetime import datetime
from django.contrib.auth.models import User
import uuid,json

class Article(models.Model):
    STATUS_CHOICES = (
        ('submitted', 'Submitted'),
        ('under_review', 'Under Review'),
        ('revision_required', 'Revision Required'),
        ('accepted', 'Accepted'),
        ('rejected', 'Rejected'),
    )
    
    title = models.CharField(max_length=255, null=True, blank=True)
    tracking_code = models.CharField(max_length=20, unique=True, editable=False)
    email = models.EmailField()
    file = models.FileField(upload_to='articles/')
    anonymized_file = models.FileField(upload_to='anonymized_articles/', null=True, blank=True)
    status = models.CharField(max_length=50, choices=STATUS_CHOICES, default='submitted')
    submission_date = models.DateTimeField(auto_now_add=True)
    last_updated = models.DateTimeField(auto_now=True)
    referee = models.ForeignKey('Referee', on_delete=models.SET_NULL, null=True, blank=True, related_name='assigned_articles')
    
    # Metadata extraction fields
    extracted_authors = models.TextField(blank=True, null=True, help_text="Automatically extracted author names")
    extracted_institutions = models.TextField(blank=True, null=True, help_text="Automatically extracted institutions")
    extracted_keywords = models.TextField(blank=True, null=True, help_text="Automatically extracted keywords")
    extracted_emails = models.TextField(blank=True, null=True, help_text="Automatically extracted email addresses")
    
    # Anonymization fields
    anonymization_map = models.TextField(blank=True, null=True, help_text="JSON mapping of original text to encrypted text")
    anonymization_key = models.CharField(max_length=128, blank=True, null=True, help_text="Encryption key for anonymization")
    is_anonymized = models.BooleanField(default=False, help_text="Whether the article has been anonymized")
    
    def save(self, *args, **kwargs):
        if not self.tracking_code:
            self.tracking_code = self.generate_tracking_code()
        super().save(*args, **kwargs)
    
    def generate_tracking_code(self):
        code = str(uuid.uuid4()).upper()[:8]
        return f"ART-{code}"
    
    def get_extracted_authors_list(self):
        """Return extracted authors as a list"""
        if self.extracted_authors:
            return self.extracted_authors.split('|')
        return []
    
    def get_extracted_institutions_list(self):
        """Return extracted institutions as a list"""
        if self.extracted_institutions:
            return self.extracted_institutions.split('|')
        return []
    
    def get_extracted_keywords_list(self):
        """Return extracted keywords as a list"""
        if self.extracted_keywords:
            return self.extracted_keywords.split('|')
        return []
    
    def get_extracted_emails_list(self):
        """Return extracted emails as a list"""
        if self.extracted_emails:
            return self.extracted_emails.split('|')
        return []
    
    def get_status_display(self):
        """
        Returns the display name for the current status
        """
        for status_code, status_name in self.STATUS_CHOICES:
            if self.status == status_code:
                return status_name
        return self.status  # Fallback to the code if no match
    
    def get_anonymization_map(self):
        """Return anonymization map as a dictionary"""
        if self.anonymization_map:
            return json.loads(self.anonymization_map)
        return {}
    
    def set_anonymization_map(self, mapping):
        """Set anonymization map from a dictionary"""
        self.anonymization_map = json.dumps(mapping)
    
    def __str__(self):
        return f"{self.tracking_code} - {self.email}"

class Editor(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    articles = models.ManyToManyField(Article, blank=True, related_name='editors')
    
    def __str__(self):
        return self.user.username
    
    
class Referee(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    specialization = models.CharField(max_length=255, blank=True, help_text="Referee's area of expertise (e.g., Machine Learning, Data Science)")
    
    def __str__(self):
        return self.user.username
    
class ArticleFeedback(models.Model):
    article = models.ForeignKey(Article, on_delete=models.CASCADE)
    referee = models.ForeignKey(Referee, on_delete=models.CASCADE)
    comments = models.TextField()
    recommendation = models.CharField(max_length=50, choices=(
        ('accept', 'Accept'),
        ('revise', 'Revise and Resubmit'),
        ('reject', 'Reject')
    ))
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Feedback on {self.article.tracking_code} by {self.referee.user.username}"
    
    def get_recommendation_display(self):
        """
        Returns the display name for the current recommendation
        """
        for code, name in (
        ('accept', 'Accept'),
        ('revise', 'Revise and Resubmit'),
        ('reject', 'Reject')
    ):
            if self.recommendation == code:
                return name
        return self.recommendation  # Fallback to the code if no match

class ChatMessage(models.Model):
    article = models.ForeignKey(Article, on_delete=models.CASCADE, related_name='messages')
    sender_email = models.EmailField(null=True, blank=True)  # For users without accounts
    sender_user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)  # For editors/referees
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    
    @property
    def sender_name(self):
        if self.sender_user and self.sender_user.is_authenticated:
            if hasattr(self.sender_user, 'editor'):
                return f"Editor: {self.sender_user.username}"
            elif hasattr(self.sender_user, 'referee'):
                return f"Referee: {self.sender_user.username}"
            return self.sender_user.username
        return "Author" if self.sender_email else "System"
    
    def __str__(self):
        return f"Message on {self.article.tracking_code} at {self.timestamp}"



class ActivityLog(models.Model):
    """Model to track all activities related to articles"""
    article = models.ForeignKey(Article, on_delete=models.CASCADE, related_name='logs')
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    email = models.EmailField(blank=True, null=True)  # For anonymous users
    action = models.CharField(max_length=255)
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-timestamp']
    
    def __str__(self):
        if self.user:
            actor = self.user.username
        else:
            actor = self.email or "System"
        return f"{actor} - {self.action} - {self.article.tracking_code}"