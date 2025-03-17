from django.shortcuts import render, redirect, get_object_or_404, HttpResponse
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib import messages
from django.core.mail import send_mail
from django.conf import settings
from django.urls import reverse
from django.contrib.auth.models import User
from django.http import HttpResponseForbidden

from .models import Article, Editor, Referee, ChatMessage, ArticleFeedback
from .forms import ArticleUploadForm, ArticleTrackingForm, ChatMessageForm, ArticleFeedbackForm

def is_editor(user):
    return hasattr(user, 'editor')

def is_referee(user):
    return hasattr(user, 'referee')

def home(request):
    return render(request, 'articles/home.html')

# User section views
def upload_article(request):
    if request.method == 'POST':
        form = ArticleUploadForm(request.POST, request.FILES)
        
        if form.is_valid():
            article = form.save()
            
            # Send email with tracking code
            subject = 'Your Article Submission Tracking Code'
            message = f"""
            Thank you for submitting your article to our system.
            
            Your tracking code is: {article.tracking_code}
            
            You can use this code to track the status of your submission at {request.build_absolute_uri(reverse('track_article'))}
            
            Best regards,
            The Editorial Team
            """
            send_mail(subject, message, settings.EMAIL_HOST_USER, [article.email])
            
            messages.success(request, f'Your article has been successfully uploaded. A tracking code has been sent to {article.email}')
            return redirect('upload_success', tracking_code=article.tracking_code)
    else:
        form = ArticleUploadForm()
    
    return render(request, 'articles/user/upload.html', {'form': form})

def upload_success(request, tracking_code):
    article = get_object_or_404(Article, tracking_code=tracking_code)
    return render(request, 'articles/user/upload_success.html', {'article': article})

def track_article(request):
    article = None
    if request.method == 'POST':
        form = ArticleTrackingForm(request.POST)
        if form.is_valid():
            tracking_code = form.cleaned_data['tracking_code']
            email = form.cleaned_data['email']
            
            try:
                article = Article.objects.get(tracking_code=tracking_code, email=email)
            except Article.DoesNotExist:
                messages.error(request, 'No article found with the provided tracking code and email.')
    else:
        form = ArticleTrackingForm()
        
    return render(request, 'articles/user/track.html', {'form': form, 'article': article})

def article_chat(request, tracking_code):
    article = get_object_or_404(Article, tracking_code=tracking_code)
    
    # Verify access using query parameters
    email = request.GET.get('email')
    if email != article.email:
        return HttpResponseForbidden("You don't have permission to access this chat.")
    
    messages_list = article.messages.order_by('timestamp')
    
    if request.method == 'POST':
        form = ChatMessageForm(request.POST)
        if form.is_valid():
            message = form.save(commit=False)
            message.article = article
            message.sender_email = article.email
            message.save()
            return redirect(f"{reverse('article_chat', args=[tracking_code])}?email={email}")
    else:
        form = ChatMessageForm()
    
    return render(request, 'articles/user/chat.html', {
        'article': article,
        'chat_messages': messages_list,
        'form': form,
        'email': email
    })


def editor_dashboard(request):
    articles = Article.objects.all().order_by('-submission_date')
    return render(request, 'articles/editor/dashboard.html', {'articles': articles})


def assign_referee(request, article_id):
    
    article = get_object_or_404(Article, id=article_id)
    referees = Referee.objects.all()
    
    if request.method == 'POST':
        referee_id = request.POST.get('referee')
        if referee_id:
            referee = get_object_or_404(Referee, id=referee_id)
            article.referee = referee
            article.status = 'under_review'
            article.save()
            
            # Optional: Send notification to the referee
            # send_referee_notification(referee, article)
            
            messages.success(request, f"Article {article.tracking_code} has been assigned to {referee.user.username}")
            return redirect('editor_dashboard')
        else:
            messages.error(request, "Please select a referee")
    
    return render(request, 'articles/editor/assign_referee.html', {
        'article': article,
        'referees': referees
    })




def editor_review(request, article_id):
    
    article = get_object_or_404(Article, id=article_id)
    all_referees = Referee.objects.all()
    
    # Get the currently assigned referee (if any)
    assigned_referee = article.referee
    
    # Get feedback for this article
    feedback = ArticleFeedback.objects.filter(article=article)
    
    if request.method == 'POST':
        action = request.POST.get('action')
        
        if action == 'update_status':
            new_status = request.POST.get('status')
            if new_status in [status[0] for status in Article.STATUS_CHOICES]:
                article.status = new_status
                article.save()
                messages.success(request, f"Article status updated to {article.get_status_display()}")
                return redirect('editor_review', article_id=article.id)
        
        # The referee assignment is now handled in the assign_referee view
        
    return render(request, 'articles/editor/review.html', {
        'article': article,
        'all_referees': all_referees,
        'assigned_referee': assigned_referee,
        'feedback': feedback
    })

def editor_chat(request, article_id):
    article = get_object_or_404(Article, id=article_id)
    messages_list = article.messages.order_by('timestamp')
    
    if request.method == 'POST':
        form = ChatMessageForm(request.POST)
        if form.is_valid():
            message = form.save(commit=False)
            message.article = article
            message.sender_user = User.objects.get(email="editor1@test.com")
            message.save()
            
            # Notify author about new message
            subject = f'New Message on Article: {article.tracking_code}'
            message = f"""
            Dear Author,
            
            You have received a new message regarding your article with tracking code {article.tracking_code}.
            
            Please check your messages at {request.build_absolute_uri(reverse('track_article'))}
            
            Best regards,
            The Editorial Team
            """
            #send_mail(subject, message, settings.EMAIL_HOST_USER, [article.email])
            
            return redirect('editor_chat', article_id=article.id)
    else:
        form = ChatMessageForm()
    
    return render(request, 'articles/editor/chat.html', {
        'article': article,
        'chat_messages': messages_list,
        'form': form
    })

# Referee section views
def referee_dashboard(request):
    referee = User.objects.get(email="ref1@test.com").referee
    
    # Instead of referee.articles.all(), use the related_name from Article model
    articles = Article.objects.filter(referee=referee)
    
    if request.method == 'POST':
        article_id = request.POST.get('article_id')
        article = get_object_or_404(Article, pk=article_id)
        form = ArticleFeedbackForm(request.POST)
        if form.is_valid():
            feedback = form.save(commit=False)
            feedback.article = article
            feedback.referee = referee
            feedback.save()

    else:
        form = ArticleFeedbackForm()

    # Group articles by status for the dashboard stats
    articles_by_status = {
        'submitted': articles.filter(status='submitted'),
        'under_review': articles.filter(status='under_review'),
        'revision_required': articles.filter(status='revision_required'),
        'accepted': articles.filter(status='accepted')
    }

    context = {
        'referee': referee, 
        'form': form,
        'articles': articles_by_status,
        'all_articles': articles  # Add the full queryset for the table
    }
    return render(request, 'articles/referee/dashboard.html', context)

def referee_review(request, article_id):
    referee = User.objects.get(email="ref1@test.com").referee
    
    # Instead of filtering with referees=referee, filter with referee=referee
    article = get_object_or_404(Article, id=article_id, referee=referee)
    
    try:
        feedback = ArticleFeedback.objects.get(article=article, referee=referee)
    except ArticleFeedback.DoesNotExist:
        feedback = None
    
    if request.method == 'POST':
        form = ArticleFeedbackForm(request.POST, instance=feedback)
        if form.is_valid():
            feedback = form.save(commit=False)
            feedback.article = article
            feedback.referee = referee
            feedback.save()
            
            # Notify editor about new feedback
            # Since editors is a ManyToMany field, we don't need to change this part
            editors = article.editors.all()
            for editor in editors:
                subject = f'New Referee Feedback on Article: {article.tracking_code}'
                message = f"""
                Dear Editor,
                
                A referee has submitted feedback for article with tracking code {article.tracking_code}.
                
                Please log in to review the feedback.
                
                Best regards,
                The System
                """
                #send_mail(subject, message, settings.EMAIL_HOST_USER, [editor.user.email])
            
            messages.success(request, 'Your feedback has been submitted')
            return redirect('referee_dashboard')
    else:
        form = ArticleFeedbackForm(instance=feedback)
    
    return render(request, 'articles/referee/review.html', {
        'article': article,
        'form': form,
        'feedback': feedback
    })

def download_article(request, article_id):
    article = get_object_or_404(Article, id=article_id)
    # Add logic to serve the file securely
    # This is simplified and should be implemented properly in production
    response = HttpResponse(article.file, content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="{article.tracking_code}.pdf"'
    return response
