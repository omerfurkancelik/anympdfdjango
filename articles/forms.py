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
                'rows': 6
            }),
            'recommendation': forms.Select(attrs={
                'class': 'form-select px-4 py-3 rounded-md'
            })
        }