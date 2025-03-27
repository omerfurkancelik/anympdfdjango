from django import forms
from .models import Article, ChatMessage, ArticleFeedback

class ArticleUploadForm(forms.ModelForm):
    class Meta:
        model = Article
        fields = ['email', 'title', 'file']
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['email'].widget.attrs.update({'class': 'form-input px-4 py-3 rounded-md', 'placeholder': 'Your email address'})
        self.fields['title'].widget.attrs.update({'class': 'form-input px-4 py-3 rounded-md', 'placeholder': 'Title of article'})
        self.fields['file'].widget.attrs.update({'class': 'form-input px-4 py-3 rounded-md', 'accept': '.pdf'})
        
        
class ArticleRevisionForm(forms.Form):
    revised_article = forms.FileField(
        widget=forms.FileInput(attrs={
            'class': 'form-input px-4 py-3 rounded-md',
            'accept': '.pdf'
        }),
        help_text="Upload your revised article (PDF format only)"
    )
    revision_comments = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-textarea px-4 py-3 rounded-md w-full',
            'rows': 3,
            'placeholder': 'Describe the changes made in this revision'
        }),
        required=False
    )

class ArticleTrackingForm(forms.Form):
    tracking_code = forms.CharField(max_length=20, required=True, widget=forms.TextInput(attrs={
        'class': 'form-input px-4 py-3 rounded-md',
        'placeholder': 'Enter your tracking code (e.g., ART-12345678)'
    }))
    email = forms.EmailField(required=True, widget=forms.EmailInput(attrs={
        'class': 'form-input px-4 py-3 rounded-md',
        'placeholder': 'Enter your email address'
    }))

class ChatMessageForm(forms.ModelForm):
    class Meta:
        model = ChatMessage
        fields = ['content']
        widgets = {
            'content': forms.Textarea(attrs={
                'class': 'form-textarea px-4 py-3 rounded-md w-full',
                'placeholder': 'Type your message here...',
                'rows': 3
            })
        }

class ArticleFeedbackForm(forms.ModelForm):
    class Meta:
        model = ArticleFeedback
        fields = ['comments', 'recommendation']
        widgets = {
            'comments': forms.Textarea(attrs={
                'class': 'form-textarea px-4 py-3 rounded-md w-full',
                'rows': 6,
                'placeholder': 'Provide your detailed feedback on the article'
            }),
            'recommendation': forms.Select(attrs={
                'class': 'form-select px-4 py-3 rounded-md',
                'onchange': 'showRecommendationGuidance(this.value)'
            })
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['recommendation'].required = True
        self.fields['recommendation'].label = "Recommendation"
        self.fields['recommendation'].help_text = "Select your recommendation for this article"
        self.fields['comments'].required = True
        self.fields['comments'].label = "Review Comments"
        self.fields['comments'].help_text = "Provide constructive feedback to the authors"
        
        
class AddRefereeForm(forms.Form):
    username = forms.CharField(max_length=150, required=True, widget=forms.TextInput(attrs={
        'class': 'form-input px-4 py-3 rounded-md',
        'placeholder': 'Username'
    }))
    email = forms.EmailField(required=True, widget=forms.EmailInput(attrs={
        'class': 'form-input px-4 py-3 rounded-md',
        'placeholder': 'Email address'
    }))
    password = forms.CharField(widget=forms.PasswordInput(attrs={
        'class': 'form-input px-4 py-3 rounded-md',
        'placeholder': 'Password'
    }))
    specialization = forms.CharField(max_length=255, required=False, widget=forms.TextInput(attrs={
        'class': 'form-input px-4 py-3 rounded-md',
        'placeholder': 'Area of expertise (e.g., Machine Learning, Data Science)'
    }))