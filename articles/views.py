from django.shortcuts import render, redirect, get_object_or_404, HttpResponse
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib import messages
from django.core.mail import send_mail
from django.db import connection
from django.conf import settings
from django.urls import reverse
from django.contrib.auth.models import User
from django.http import HttpResponseForbidden
from .models import Article, Editor, Referee, ChatMessage, ArticleFeedback, ActivityLog
from .forms import ArticleUploadForm, ArticleTrackingForm, ChatMessageForm, ArticleFeedbackForm, AddRefereeForm, ArticleRevisionForm
from .utils import decrypt_anonymization_map

import os
import shutil
import datetime
from django.core.files.base import ContentFile
from .utils import (extract_text_from_pdf, extract_authors, extract_institutions, 
                   extract_keywords, create_anonymization_map, anonymize_pdf, anonymize_pdf_legacy,
                   match_referees_by_keywords)
import json


def log_activity(article, action, user=None, email=None):
    """Helper function to log activities related to articles"""
    # Kullanıcı anonim ise (AnonymousUser), None olarak ayarla
    if user and not user.is_authenticated:
        user = None
        
    ActivityLog.objects.create(
        article=article,
        user=user,
        email=email,
        action=action
    )


def home(request):
    """Home page view"""
    return render(request, 'articles/home.html')


def system_logs(request):
    """View to display system logs"""
    try:
        logs = ActivityLog.objects.all().select_related('article', 'user').order_by('-timestamp')
    except Exception as e:
        # Tablo yoksa veya başka bir hata olursa boş bir queryset kullan
        logs = []
        messages.error(request, f"Log kayıtları yüklenirken bir hata oluştu: {str(e)}")
    return render(request, 'articles/logs.html', {'logs': logs})


# User section views
def upload_article(request):
    """Upload a new article"""
    if request.method == 'POST':
        form = ArticleUploadForm(request.POST, request.FILES)
        
        if form.is_valid():
            article = form.save()
            
            # Log the article submission
            log_activity(article, "Article submitted", email=article.email)
            
            # Send email with tracking code
            subject = 'Your Article Submission Tracking Code'
            message = f"""
            Thank you for submitting your article to our system.
            
            Your tracking code is: {article.tracking_code}
            
            You can use this code to track the status of your submission at {request.build_absolute_uri(reverse('track_article'))}
            
            Best regards,
            The Editorial Team
            """
            #send_mail(subject, message, settings.EMAIL_HOST_USER, [article.email])
            
            messages.success(request, f'Your article has been successfully uploaded. A tracking code has been sent to {article.email}')
            return redirect('upload_success', tracking_code=article.tracking_code)
    else:
        form = ArticleUploadForm()
    
    return render(request, 'articles/user/upload.html', {'form': form})


def upload_success(request, tracking_code):
    """Display success page after upload"""
    article = get_object_or_404(Article, tracking_code=tracking_code)
    return render(request, 'articles/user/upload_success.html', {'article': article})


def track_article(request):
    """Track an article using tracking code and email"""
    article = None
    feedback = None
    
    # Check if this is a revision submission
    if request.method == 'POST' and 'submit_revision' in request.POST:
        # This is a revision submission
        tracking_code = request.POST.get('tracking_code')
        email = request.POST.get('email')
        
        try:
            article = Article.objects.get(tracking_code=tracking_code, email=email)
            
            if article.status == 'revision_required':
                revision_form = ArticleRevisionForm(request.POST, request.FILES)
                if revision_form.is_valid() and 'revised_article' in request.FILES:
                    # Save the revised file
                    article.file = request.FILES['revised_article']
                    article.status = 'submitted'  # Reset to submitted status for re-review
                    article.save()
                    
                    # Log the revision submission
                    log_activity(article, "Revised version submitted", email=article.email)
                    
                    # Create a message to notify about the revision
                    revision_comments = revision_form.cleaned_data.get('revision_comments', '')
                    message_content = "Revised version uploaded by author."
                    if revision_comments:
                        message_content += f" Comments: {revision_comments}"
                        
                    ChatMessage.objects.create(
                        article=article,
                        sender_email=email,
                        content=message_content
                    )
                    
                    messages.success(request, "Your revised article has been successfully uploaded.")
                    return redirect('track_article')
                else:
                    if 'revised_article' not in request.FILES:
                        messages.error(request, "Please select a file to upload.")
            else:
                messages.error(request, "This article does not require revision.")
                
            # Get the latest feedback for this article
            feedback = ArticleFeedback.objects.filter(article=article).order_by('-created_at').first()
            form = ArticleTrackingForm(initial={'tracking_code': tracking_code, 'email': email})
            revision_form = ArticleRevisionForm()
            
        except Article.DoesNotExist:
            messages.error(request, 'No article found with the provided tracking code and email.')
            form = ArticleTrackingForm()
            revision_form = ArticleRevisionForm()
    
    # Normal article tracking
    elif request.method == 'POST':
        form = ArticleTrackingForm(request.POST)
        if form.is_valid():
            tracking_code = form.cleaned_data['tracking_code']
            email = form.cleaned_data['email']
            
            try:
                article = Article.objects.get(tracking_code=tracking_code, email=email)
                # Get the latest feedback for this article
                feedback = ArticleFeedback.objects.filter(article=article).order_by('-created_at').first()
                revision_form = ArticleRevisionForm()
            except Article.DoesNotExist:
                messages.error(request, 'No article found with the provided tracking code and email.')
                revision_form = ArticleRevisionForm()
        else:
            revision_form = ArticleRevisionForm()
    else:
        form = ArticleTrackingForm()
        revision_form = ArticleRevisionForm()
        
    return render(request, 'articles/user/track.html', {
        'form': form, 
        'article': article,
        'feedback': feedback,
        'revision_form': revision_form
    })


def article_chat(request, tracking_code):
    """Chat with editors about an article"""
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
    """Editor dashboard view"""
    articles = Article.objects.all().order_by('-submission_date')
    return render(request, 'articles/editor/dashboard.html', {'articles': articles})


def referee_quick_action(request, article_id, referee_id):
    """
    Quick action for referee to accept or request revision for an article
    """
    referee = get_object_or_404(Referee, id=referee_id)
    article = get_object_or_404(Article, id=article_id, referee=referee)
    
    if request.method == 'POST':
        action = request.POST.get('action')
        
        if action == 'accept':
            # Change status to accepted
            article.status = 'accepted'
            article.save()
            
            # Create feedback record if it doesn't exist yet
            feedback, created = ArticleFeedback.objects.get_or_create(
                article=article, 
                referee=referee,
                defaults={
                    'comments': 'Article accepted via quick action.',
                    'recommendation': 'accept'
                }
            )
            
            # If feedback already existed, update it
            if not created:
                feedback.recommendation = 'accept'
                feedback.save()
            
            # Create system message
            # ChatMessage.objects.create(
            #     article=article,
            #     sender_user=request.user,
            #     content=f"Article accepted by referee: {referee.user.username}"
            # )
            
            messages.success(request, f"Article {article.tracking_code} has been accepted.")
        
        elif action == 'revise':
            # Change status to revision required
            article.status = 'revision_required'
            article.save()
            
            # Create feedback record if it doesn't exist yet
            feedback, created = ArticleFeedback.objects.get_or_create(
                article=article, 
                referee=referee,
                defaults={
                    'comments': 'Revision requested via quick action. Please see editor for details.',
                    'recommendation': 'revise'
                }
            )
            
            # If feedback already existed, update it
            if not created:
                feedback.recommendation = 'revise'
                feedback.save()
            
            # Create system message
            ChatMessage.objects.create(
                article=article,
                sender_user=request.user,
                content=f"Revision requested by referee: {referee.user.username}"
            )
            
            messages.success(request, f"Revision has been requested for article {article.tracking_code}.")
        
        else:
            messages.error(request, "Invalid action specified.")
    
    return redirect('referee_dashboard', referee_id=referee.id)


def assign_referee(request, article_id):
    """
    Assign a referee to an article with checks for anonymization
    """
    article = get_object_or_404(Article, id=article_id)
    referees = Referee.objects.all()
    
    # For suggested referees, get top matches based on keywords
    suggested_referees = []
    if article.extracted_keywords:
        article_keywords = article.get_extracted_keywords_list()
        suggested_referees = match_referees_by_keywords(article_keywords, referees)
    
    if request.method == 'POST':
        referee_id = request.POST.get('referee')
        if referee_id:
            referee = get_object_or_404(Referee, id=referee_id)
            
            # Check if article is anonymized - if not, show a warning
            if not article.is_anonymized:
                messages.warning(request, "Note: This article has not been anonymized. The referee will see all author information.")
            
            article.referee = referee
            article.status = 'under_review'
            article.save()
            
            # Oturumun açık olup olmadığını kontrol et
            user = request.user if request.user.is_authenticated else None
            
            # Log the referee assignment
            log_activity(
                article, 
                f"Assigned to referee: {referee.user.username}", 
                user=user
            )
            
            # Create a system message to log the assignment
            """ChatMessage.objects.create(
                article=article,
                sender_user=request.user,
                content=f"Article assigned to referee: {referee.user.username}"
            )"""
            
            messages.success(request, f"Article {article.tracking_code} has been assigned to {referee.user.username}")
            return redirect('editor_dashboard')
        else:
            messages.error(request, "Please select a referee")
    
    # Check if there's a suggested referee in the query params
    suggested_referee_id = request.GET.get('referee')
    if suggested_referee_id:
        try:
            # Pre-select this referee in the form
            suggested_referee = Referee.objects.get(id=suggested_referee_id)
            context = {
                'article': article,
                'referees': referees,
                'preselected_referee': suggested_referee,
                'suggested_referees': suggested_referees
            }
        except Referee.DoesNotExist:
            context = {
                'article': article,
                'referees': referees,
                'suggested_referees': suggested_referees
            }
    else:
        context = {
            'article': article,
            'referees': referees,
            'suggested_referees': suggested_referees
        }
    
    return render(request, 'articles/editor/assign_referee.html', context)


def restore_article_info(request, article_id):
    """Restore anonymized author information"""
    article = get_object_or_404(Article, id=article_id)
    
    # Get anonymization key from the article
    if not article.anonymization_key:
        messages.error(request, "This article has not been anonymized.")
        return redirect('editor_review', article_id=article.id)
    
    # Get anonymization map
    anon_map = article.get_anonymization_map()
    
    try:
        # Decrypt the anonymization map
        reverse_map, cipher = decrypt_anonymization_map(anon_map, article.anonymization_key)
        
        # Organize the information for display
        authors = []
        institutions = []
        
        for encrypted, original in reverse_map.items():
            # Try to decrypt each entry
            try:
                decrypted = cipher.decrypt(encrypted)
                if decrypted == original:  # This is an author or institution
                    if any(term in original.lower() for term in ['university', 'institute', 'college', 'department']):
                        institutions.append((encrypted, original))
                    else:
                        authors.append((encrypted, original))
            except:
                # Skip if decryption fails
                continue
        
        return render(request, 'articles/editor/restore_info.html', {
            'article': article,
            'authors': authors,
            'institutions': institutions
        })
    
    except Exception as e:
        messages.error(request, f"Error decrypting information: {e}")
        return redirect('editor_review', article_id=article.id)


def editor_review(request, article_id):
    """Editor review article page"""
    article = get_object_or_404(Article, id=article_id)
    all_referees = Referee.objects.all()
    
    # Get the currently assigned referee (if any)
    assigned_referee = article.referee
    
    # Get feedback for this article
    feedback = ArticleFeedback.objects.filter(article=article).order_by('-created_at')
    
    # Get revision history messages
    revision_messages = ChatMessage.objects.filter(
        article=article, 
        content__contains="Revised version uploaded"
    ).order_by('-timestamp')
    
    if request.method == 'POST':
        action = request.POST.get('action')
        
        if action == 'update_status':
            new_status = request.POST.get('status')
            if new_status in [status[0] for status in Article.STATUS_CHOICES]:
                old_status = article.status
                article.status = new_status
                article.save()
                
                # Oturumun açık olup olmadığını kontrol et
                user = request.user if request.user.is_authenticated else None
                
                # Log the status change
                log_activity(
                    article, 
                    f"Status changed from '{old_status}' to '{new_status}'", 
                    user=user
                )
                
                # Create a system message to log the status change
                ChatMessage.objects.create(
                    article=article,
                    sender_user=user,  # Burada da aynı kontrolü uygula
                    content=f"Article status updated to {article.get_status_display()}"
                )
                
                messages.success(request, f"Article status updated to {article.get_status_display()}")
                return redirect('editor_review', article_id=article.id)
    
    return render(request, 'articles/editor/review.html', {
        'article': article,
        'all_referees': all_referees,
        'assigned_referee': assigned_referee,
        'feedback': feedback,
        'revision_messages': revision_messages
    })


def editor_chat(request, article_id):
    """Editor chat with article author"""
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
def referee_list(request):
    """
    Display a list of all referees in the system.
    """
    referees = Referee.objects.all()
    return render(request, 'articles/referee/list.html', {'referees': referees})


# Update the referee_dashboard view to accept a referee_id parameter
def referee_dashboard(request, referee_id=None):
    """
    Display the dashboard for a specific referee.
    If referee_id is not provided, it falls back to ref1@test.com.
    """
    try:
        if referee_id:
            referee = get_object_or_404(Referee, id=referee_id)
        else:
            # Fallback to the default referee if no specific one is selected
            referee = User.objects.get(email="ref1@test.com").referee
    except (Referee.DoesNotExist, User.DoesNotExist):
        messages.error(request, "Referee not found.")
        return redirect('referee_list')
    
    # Get articles assigned to this referee
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


def process_article_metadata(request, article_id):
    """Extract metadata from an article using NLP and anonymize using AES encryption"""
    article = get_object_or_404(Article, id=article_id)
    
    if request.method == 'POST':
        action = request.POST.get('action', 'extract')
        
        if action == 'extract':
            # Extract text from PDF
            pdf_path = article.file.path
            text = extract_text_from_pdf(pdf_path)
            
            # Extract metadata using NLP
            authors = extract_authors(text)
            institutions = extract_institutions(text)
            keywords = extract_keywords(text)
            
            # Save extracted metadata
            article.extracted_authors = '|'.join(authors)
            article.extracted_institutions = '|'.join(institutions)
            article.extracted_keywords = '|'.join(keywords)
            article.save()
            
            messages.success(request, "Article metadata extracted successfully.")
            return redirect('process_article_metadata', article_id=article.id)
            
        elif action == 'anonymize':
            # Get authors and institutions to anonymize
            authors_to_anonymize = request.POST.getlist('authors')
            institutions_to_anonymize = request.POST.getlist('institutions')
            
            # Create anonymization map using AES encryption
            anon_map, encryption_key = create_anonymization_map(
                authors_to_anonymize, 
                institutions_to_anonymize
            )
            
            # Save anonymization map and key to the article
            article.set_anonymization_map(anon_map)
            article.anonymization_key = encryption_key
            
            # Create anonymized PDF
            if anon_map:
                pdf_path = article.file.path
                anonymized_path = anonymize_pdf(pdf_path, anon_map)
                
                if anonymized_path:
                    # Save anonymized file to the article
                    with open(anonymized_path, 'rb') as f:
                        article.anonymized_file.save(
                            f"anonymized_{os.path.basename(article.file.name)}", 
                            ContentFile(f.read())
                        )
                    
                    # Clean up temporary file
                    os.remove(anonymized_path)
                    
                    article.is_anonymized = True
                    article.save()
                    
                    messages.success(request, "Article anonymized successfully with AES encryption.")
                else:
                    messages.error(request, "Failed to anonymize article.")
            else:
                messages.warning(request, "No items selected for anonymization.")
            
            return redirect('editor_review', article_id=article.id)
    
    # Prepare data for template
    context = {
        'article': article,
        'authors': article.get_extracted_authors_list(),
        'institutions': article.get_extracted_institutions_list(),
        'keywords': article.get_extracted_keywords_list()
    }
    
    return render(request, 'articles/editor/process_metadata.html', context)


# Update the referee_review view to work with the new dashboard
def referee_review(request, article_id, referee_id=None):
    """
    View for referee to review an article and submit feedback
    """
    try:
        if referee_id:
            referee = get_object_or_404(Referee, id=referee_id)
        else:
            # Fallback to the default referee if no specific one is selected
            referee = User.objects.get(email="ref1@test.com").referee
    except (Referee.DoesNotExist, User.DoesNotExist):
        messages.error(request, "Referee not found.")
        return redirect('referee_list')
    
    # Get the article that belongs to this referee
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
            
            # Update article status based on recommendation
            recommendation = form.cleaned_data.get('recommendation')
            if recommendation == 'revise':
                article.status = 'revision_required'
                article.save()
                # Add a system message about the status change
                # ChatMessage.objects.create(
                #     article=article,
                #     sender_user=request.user,
                #     content=f"Status changed to Revision Required based on referee recommendation."
                # )
            elif recommendation == 'accept':
                article.status = 'accepted'
                article.save()
                # ChatMessage.objects.create(
                #     article=article,
                #     sender_user=request.user,
                #     content=f"Status changed to Accepted based on referee recommendation."
                # )
            elif recommendation == 'reject':
                article.status = 'rejected'
                article.save()
                # ChatMessage.objects.create(
                #     article=article,
                #     sender_user=request.user,
                #     content=f"Status changed to Rejected based on referee recommendation."
                # )
            
            # Log the feedback submission
            log_activity(
                article, 
                f"Referee feedback submitted with recommendation: {recommendation}", 
                user=referee.user
            )
            
            # Notify editor about new feedback
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
            
            messages.success(request, 'Your feedback has been submitted and the article status has been updated.')
            return redirect('referee_dashboard', referee_id=referee.id)
    else:
        form = ArticleFeedbackForm(instance=feedback)
    
    return render(request, 'articles/referee/review.html', {
        'article': article,
        'form': form,
        'feedback': feedback,
        'referee': referee
    })
        

def download_article(request, article_id):
    """Download article file"""
    article = get_object_or_404(Article, id=article_id)
    # Add logic to serve the file securely
    # This is simplified and should be implemented properly in production
    response = HttpResponse(article.file, content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="{article.tracking_code}.pdf"'
    return response


def add_referee(request):
    """Add a new referee"""
    if request.method == 'POST':
        form = AddRefereeForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            email = form.cleaned_data['email']
            password = form.cleaned_data['password']
            specialization = form.cleaned_data['specialization']
            
            # Check if username or email already exists
            if User.objects.filter(username=username).exists():
                messages.error(request, f"Username '{username}' is already taken.")
                return render(request, 'articles/referee/add.html', {'form': form})
                
            if User.objects.filter(email=email).exists():
                messages.error(request, f"Email '{email}' is already registered.")
                return render(request, 'articles/referee/add.html', {'form': form})
            
            # Create the user and referee
            user = User.objects.create_user(username=username, email=email, password=password)
            referee = Referee.objects.create(user=user, specialization=specialization)
            
            messages.success(request, f"Referee '{username}' has been successfully added.")
            return redirect('referee_list')
    else:
        form = AddRefereeForm()
    
    return render(request, 'articles/referee/add.html', {'form': form})


def delete_article(request, article_id):
    """Delete an article"""
    article = get_object_or_404(Article, id=article_id)
    
    if request.method == 'POST':
        # Delete the article file from storage
        if article.file:
            if os.path.isfile(article.file.path):
                os.remove(article.file.path)
        
        # Store tracking code for confirmation message
        tracking_code = article.tracking_code
        
        # Delete the article from the database
        article.delete()
        
        messages.success(request, f"Article {tracking_code} has been successfully deleted.")
        return redirect('editor_dashboard')
    
    return render(request, 'articles/editor/delete_confirm.html', {'article': article})


def reset_database(request):
    """
    Remove all data from the database and media files, then recreate initial admin account.
    CAUTION: This will delete all data!
    """
    if request.method == 'POST':
        # First, confirm the action with a security check
        confirmation = request.POST.get('confirmation')
        if confirmation != 'RESET':
            messages.error(request, "Incorrect confirmation code. Database reset aborted.")
            return redirect('editor_dashboard')
        
        try:
            # 1. Delete all media files
            media_root = settings.MEDIA_ROOT
            
            # Delete article files
            articles_dir = os.path.join(media_root, 'articles')
            if os.path.exists(articles_dir):
                shutil.rmtree(articles_dir)
                os.makedirs(articles_dir)  # Recreate empty directory
                
            # Delete anonymized article files
            anon_articles_dir = os.path.join(media_root, 'anonymized_articles')
            if os.path.exists(anon_articles_dir):
                shutil.rmtree(anon_articles_dir)
                os.makedirs(anon_articles_dir)  # Recreate empty directory
            
            # 2. Clear database tables (retain structure)
            with connection.cursor() as cursor:
                # Get list of all tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name!='sqlite_sequence' AND name!='auth_permission' AND name!='auth_group' AND name!='django_content_type' AND name!='django_migrations';")
                tables = [row[0] for row in cursor.fetchall()]
                
                # Delete data from each table
                for table in tables:
                    cursor.execute(f"DELETE FROM {table};")
            
            # 3. Recreate default editor account
            user = User.objects.create_user(
                username='editor1',
                email='editor1@test.com',
                password='password123'
            )
            Editor.objects.create(user=user)
            
            messages.success(request, "Database has been reset successfully. A new editor account has been created.")
            
            # Log out the user since they're likely deleted
            from django.contrib.auth import logout
            logout(request)
            
            return redirect('home')
            
        except Exception as e:
            messages.error(request, f"An error occurred during database reset: {str(e)}")
            return redirect('editor_dashboard')
    
    return render(request, 'articles/editor/reset_database.html')


# Alternative implementation using Django's ORM for database reset
def reset_database_orm(request):
    """
    Remove all data from the database and media files, then recreate initial admin account.
    This version uses Django ORM with proper handling of foreign key constraints.
    """
    if request.method == 'POST':
        # First, confirm the action with a security check
        confirmation = request.POST.get('confirmation')
        if confirmation != 'RESET':
            messages.error(request, "Incorrect confirmation code. Database reset aborted.")
            return redirect('editor_dashboard')
        
        try:
            # Save the current user's ID
            current_user_id = request.user.id
            is_current_user_editor = hasattr(request.user, 'editor')
            
            # 1. Delete all media files
            media_root = settings.MEDIA_ROOT
            
            # Delete article files
            articles_dir = os.path.join(media_root, 'articles')
            if os.path.exists(articles_dir):
                shutil.rmtree(articles_dir)
                os.makedirs(articles_dir)  # Recreate empty directory
                
            # Delete anonymized article files
            anon_articles_dir = os.path.join(media_root, 'anonymized_articles')
            if os.path.exists(anon_articles_dir):
                shutil.rmtree(anon_articles_dir)
                os.makedirs(anon_articles_dir)  # Recreate empty directory
            
            # 2. Clear database using Django ORM with proper order
            # Turn off foreign key constraints temporarily
            from django.db import connection
            with connection.cursor() as cursor:
                if connection.vendor == 'sqlite':
                    cursor.execute("PRAGMA foreign_keys = OFF;")
                elif connection.vendor == 'postgresql':
                    cursor.execute("SET CONSTRAINTS ALL DEFERRED;")
                elif connection.vendor == 'mysql':
                    cursor.execute("SET FOREIGN_KEY_CHECKS = 0;")
                    
            print("bura girdi canim")
            
            # Delete all data in correct order
            ChatMessage.objects.all().delete()
            ArticleFeedback.objects.all().delete()
            Referee.objects.all().delete()
            Editor.objects.all().delete()
            Article.objects.all().delete()
            
            # Delete all non-superuser users
            User.objects.filter(is_superuser=False).delete()
            
            # Turn foreign key constraints back on
            with connection.cursor() as cursor:
                if connection.vendor == 'sqlite':
                    cursor.execute("PRAGMA foreign_keys = ON;")
                elif connection.vendor == 'postgresql':
                    cursor.execute("SET CONSTRAINTS ALL IMMEDIATE;")
                elif connection.vendor == 'mysql':
                    cursor.execute("SET FOREIGN_KEY_CHECKS = 1;")
            
            # 3. Recreate default editor account
            try:
                user = User.objects.create_user(
                    username='editor1',
                    email='editor1@test.com',
                    password='passnord123',
                )
            except Exception as e:
                print(e)
            
            Editor.objects.create(user=user)
            
            messages.success(request, "Database has been reset successfully. A new editor account has been created.")
            
            # Check if the current user was deleted
            if is_current_user_editor:
                # The current user was an editor, they've been deleted
                from django.contrib.auth import logout
                logout(request)
                return redirect('home')
            else:
                # The user might still exist if they're a superuser
                try:
                    User.objects.get(id=current_user_id)
                    return redirect('editor_dashboard')
                except User.DoesNotExist:
                    from django.contrib.auth import logout
                    logout(request)
                    return redirect('home')
            
        except Exception as e:
            messages.error(request, f"An error occurred during database reset: {str(e)}")
            return redirect('editor_dashboard')
    
    return render(request, 'articles/editor/reset_database.html')


def update_article_file(request, tracking_code):
    """Update an existing article file"""
    article = get_object_or_404(Article, tracking_code=tracking_code)
    
    if request.method == 'POST':
        new_file = request.FILES.get('new_file')
        
        if not new_file:
            messages.error(request, "No file was uploaded. Please select a file.")
            return redirect('track_article')
            
        # Check if file is PDF
        if not new_file.name.endswith('.pdf'):
            messages.error(request, "Only PDF files are accepted.")
            return redirect('track_article')
            
        # Check file size (10MB limit)
        if new_file.size > 10 * 1024 * 1024:  # 10MB in bytes
            messages.error(request, "File size exceeds the 10MB limit.")
            return redirect('track_article')
            
        try:
            # Delete the old file if it exists
            if article.file:
                old_path = article.file.path
                if os.path.isfile(old_path):
                    os.remove(old_path)
                    
            # Save the new file
            article.file = new_file
            article.status = 'submitted'  # Reset status to submitted
            article.save()
            
            # Add a system message to chat if it exists
            try:
                ChatMessage.objects.create(
                    article=article,
                    sender_type='SYSTEM',
                    message=f"Author has updated the article file on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                )
            except:
                pass  # Skip if chat functionality is not available
                
            messages.success(request, "Your article has been successfully updated and is now under review.")
        except Exception as e:
            messages.error(request, f"An error occurred while updating your article: {str(e)}")
            
        return redirect('track_article')
        
    # If not POST, redirect to track article page
    return redirect('track_article')


def anonymize_article(request, article_id):
    """Anonymize an article by replacing author and institution information"""
    article = get_object_or_404(Article, id=article_id)
    
    if request.method == 'POST':
        # 1) Formdan seçilen yazarlar/kurumlar
        authors_to_anonymize = request.POST.getlist('authors')
        institutions_to_anonymize = request.POST.getlist('institutions')
        
        # 2) Anonimleştirme haritası (örn. "MOHAMMAD ASIF" => "Author-1")
        anon_map = {}
        for idx, author in enumerate(authors_to_anonymize):
            anon_map[author] = f"Author-{idx+1}"
        for idx, institution in enumerate(institutions_to_anonymize):
            anon_map[institution] = f"Institution-{idx+1}"
        
        # 3) Map'i kaydedebiliriz (projede set_anonymization_map varsa kullanabilirsiniz)
        article.set_anonymization_map(anon_map)  # varsayıyoruz ki modelde böyle bir fonksiyon var
        
        # 4) PDF Anonimleştirme
        if anon_map:
            pdf_path = article.file.path
            anonymized_path = anonymize_pdf(pdf_path, anon_map)
            
            if anonymized_path:
                # Anonimleştirilmiş PDF'yi Article modeline kaydet
                with open(anonymized_path, 'rb') as f:
                    article.anonymized_file.save(
                        f"anonymized_{os.path.basename(article.file.name)}",
                        ContentFile(f.read())
                    )
                os.remove(anonymized_path)  # temp dosyayı sil
                
                article.is_anonymized = True
                article.save()
                messages.success(request, "Article anonymized successfully.")
            else:
                messages.error(request, "Failed to anonymize article.")
        else:
            messages.warning(request, "No items selected for anonymization.")
        
        return redirect('editor_review', article_id=article.id)
    
    # GET isteğinde, kullanıcıya hangi yazar/kurum anonimleştirilsin diye seçenek sunuyoruz
    return render(request, 'articles/editor/anonymize.html', {
        'article': article,
        'authors': article.get_extracted_authors_list(),       # '|'-separated -> list
        'institutions': article.get_extracted_institutions_list()  # '|'-separated -> list
    })


def download_anonymized_article(request, article_id):
    """Download the anonymized version of an article"""
    article = get_object_or_404(Article, id=article_id)
    
    if not article.anonymized_file:
        messages.error(request, "No anonymized version available for this article.")
        return redirect('editor_review', article_id=article.id)
    
    # Serve the anonymized file
    response = HttpResponse(article.anonymized_file, content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="anonymized_{article.tracking_code}.pdf"'
    return response


def suggest_referees(request, article_id):
    """Suggest appropriate referees based on article keywords"""
    article = get_object_or_404(Article, id=article_id)
    
    if not article.extracted_keywords:
        messages.warning(request, "No keywords extracted. Process article metadata first.")
        return redirect('process_article_metadata', article_id=article.id)
    
    # Get all referees
    referees = Referee.objects.all()
    
    # Match referees by keywords
    article_keywords = article.get_extracted_keywords_list()
    matched_referees_raw = match_referees_by_keywords(article_keywords, referees)
    
    # Convert scores to percentages
    matched_referees = [(referee, score * 100) for referee, score in matched_referees_raw]
    
    print(matched_referees)
    
    return render(request, 'articles/editor/suggest_referees.html', {
        'article': article,
        'matched_referees': matched_referees,
        'keywords': article_keywords
    })


def restore_article_info(request, article_id, referee_id):
    """Restore original author information after review (for referee)"""
    article = get_object_or_404(Article, id=article_id)
    referee = get_object_or_404(Referee, id=referee_id)
    
    # Verify this referee is assigned to this article
    if article.referee != referee:
        messages.error(request, "You are not assigned to review this article.")
        return redirect('referee_dashboard', referee_id=referee.id)
    
    if not article.is_anonymized or not article.anonymization_map:
        messages.warning(request, "This article has not been anonymized.")
        return redirect('referee_review', article_id=article.id, referee_id=referee.id)
    
    # Get anonymization map
    anon_map = article.get_anonymization_map()
    
    # Create a reverse mapping (anonymized -> original)
    reverse_map = {v: k for k, v in anon_map.items()}
    
    return render(request, 'articles/referee/restore_info.html', {
        'article': article,
        'anonymization_map': anon_map,
        'reverse_map': reverse_map
    })