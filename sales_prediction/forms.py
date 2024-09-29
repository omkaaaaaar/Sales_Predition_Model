from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import authenticate
from .models import CustomUser

# Signup form for CustomUser model
class CustomUserCreationForm(UserCreationForm):
    class Meta:
        model = CustomUser
        fields = ['fullname', 'email', 'username', 'phone', 'role', 'password1', 'password2']

    fullname = forms.CharField(max_length=255, required=True, widget=forms.TextInput(attrs={
        'placeholder': 'Enter full name',
        'id': 'fullname'
    }))
    
    email = forms.EmailField(required=True, widget=forms.EmailInput(attrs={
        'placeholder': 'Enter email address',
        'id': 'signupEmail'
    }))
    
    username = forms.CharField(max_length=150, required=True, widget=forms.TextInput(attrs={
        'placeholder': 'Enter username',
        'id': 'signupUsername'
    }))
    
    phone = forms.CharField(max_length=15, required=False, widget=forms.TextInput(attrs={
        'placeholder': 'Enter phone number',
        'id': 'phone'
    }))
    
    role = forms.ChoiceField(choices=CustomUser.ROLE_CHOICES, widget=forms.Select(attrs={
        'id': 'role'
    }))
    
    password1 = forms.CharField(label="Password", widget=forms.PasswordInput(attrs={
        'placeholder': 'Enter password',
        'id': 'signupPassword'
    }))
    
    password2 = forms.CharField(label="Confirm Password", widget=forms.PasswordInput(attrs={
        'placeholder': 'Confirm password',
        'id': 'confirmPassword'
    }))


# Login form
class LoginForm(forms.Form):
    email_or_username = forms.CharField(label="Email Address or Username", widget=forms.TextInput(attrs={
        'placeholder': 'Enter email or username',
        'id': 'emailOrUsername'
    }))
    password = forms.CharField(widget=forms.PasswordInput(attrs={
        'placeholder': 'Enter password',
        'id': 'loginPassword'
    }))
    
    def clean(self):
        email_or_username = self.cleaned_data.get('email_or_username')
        password = self.cleaned_data.get('password')

        user = authenticate(username=email_or_username, password=password)

        if not user:
            raise forms.ValidationError("Invalid login credentials")
        
        return self.cleaned_data