from django.db import models
from django.contrib.auth.models import User
import uuid

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
    status = models.CharField(max_length=50, choices=STATUS_CHOICES, default='submitted')
    submission_date = models.DateTimeField(auto_now_add=True)
    last_updated = models.DateTimeField(auto_now=True)
    
    def save(self, *args, **kwargs):
        if not self.tracking_code:
            self.tracking_code = self.generate_tracking_code()
        super().save(*args, **kwargs)
    
    def generate_tracking_code(self):
        code = str(uuid.uuid4()).upper()[:8]
        return f"ART-{code}"
    
    def __str__(self):
        return f"{self.tracking_code} - {self.email}"
    
    
class Editor(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    articles = models.ManyToManyField(Article, blank=True, related_name='editors')
    
    def __str__(self):
        return self.user.username
    
    
class Referee(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    articles = models.ManyToManyField(Article, blank=True, related_name='referees')
    
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

class ChatMessage(models.Model):
    article = models.ForeignKey(Article, on_delete=models.CASCADE, related_name='messages')
    sender_email = models.EmailField(null=True, blank=True)  # For users without accounts
    sender_user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)  # For editors/referees
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    
    @property
    def sender_name(self):
        if self.sender_user:
            if hasattr(self.sender_user, 'editor'):
                return f"Editor: {self.sender_user.username}"
            elif hasattr(self.sender_user, 'referee'):
                return f"Referee: {self.sender_user.username}"
            return self.sender_user.username
        return "Author"
    
    def __str__(self):
        return f"Message on {self.article.tracking_code} at {self.timestamp}"