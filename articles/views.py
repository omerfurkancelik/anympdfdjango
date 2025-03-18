from django.shortcuts import render, redirect, get_object_or_404, HttpResponse
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib import messages
from django.core.mail import send_mail
from django.db import connection
from django.conf import settings
from django.urls import reverse
from django.contrib.auth.models import User
from django.http import HttpResponseForbidden
from .models import Article, Editor, Referee, ChatMessage, ArticleFeedback
from .forms import ArticleUploadForm, ArticleTrackingForm, ChatMessageForm, ArticleFeedbackForm, AddRefereeForm

import os
import shutil

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
            #send_mail(subject, message, settings.EMAIL_HOST_USER, [article.email])
            
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

# Update the referee_review view to work with the new dashboard
def referee_review(request, article_id, referee_id=None):
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
            
            messages.success(request, 'Your feedback has been submitted')
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
    article = get_object_or_404(Article, id=article_id)
    # Add logic to serve the file securely
    # This is simplified and should be implemented properly in production
    response = HttpResponse(article.file, content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="{article.tracking_code}.pdf"'
    return response



def add_referee(request):

        
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































































from .utils import (extract_text_from_pdf, extract_authors, extract_institutions, 
                   extract_keywords, create_anonymization_map, anonymize_pdf,
                   match_referees_by_keywords)
from django.core.files.base import ContentFile
import os
import json

def process_article_metadata(request, article_id):
    """Extract metadata from an article using NLP"""

    article = get_object_or_404(Article, id=article_id)
    
    if request.method == 'POST':
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
        return redirect('editor_review', article_id=article.id)
    
    return render(request, 'articles/editor/process_metadata.html', {'article': article})

def anonymize_article(request, article_id):
    """Anonymize an article by replacing author and institution information"""

    
    article = get_object_or_404(Article, id=article_id)
    
    if request.method == 'POST':
        # Get selected items to anonymize
        authors_to_anonymize = request.POST.getlist('authors')
        institutions_to_anonymize = request.POST.getlist('institutions')
        
        # Create anonymization map
        anon_map = {}
        for idx, author in enumerate(authors_to_anonymize):
            anon_map[author] = f"Author-{idx+1}"
        
        for idx, institution in enumerate(institutions_to_anonymize):
            anon_map[institution] = f"Institution-{idx+1}"
        
        # Save anonymization map to the article
        article.set_anonymization_map(anon_map)
        
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
                
                messages.success(request, "Article anonymized successfully.")
            else:
                messages.error(request, "Failed to anonymize article.")
        else:
            messages.warning(request, "No items selected for anonymization.")
        
        return redirect('editor_review', article_id=article.id)
    
    return render(request, 'articles/editor/anonymize.html', {
        'article': article,
        'authors': article.get_extracted_authors_list(),
        'institutions': article.get_extracted_institutions_list()
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
    matched_referees = match_referees_by_keywords(article_keywords, referees)
    
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