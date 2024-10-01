# Signup Form
from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import CustomUser  # Import the custom user model

class SignupForm(UserCreationForm):
    fullname = forms.CharField(max_length=100, required=True, label='Full Name')
    email = forms.EmailField(max_length=254, required=True)
    phone = forms.CharField(max_length=15, required=True, label='Phone Number')
    
    ROLE_CHOICES = [
        ('Employee', 'Employee'),
        ('admin', 'Admin'),
        ('Developer','Developer'),
    ]
    role = forms.ChoiceField(choices=ROLE_CHOICES, required=True)

    class Meta:
        model = CustomUser  # Use the custom user model
        fields = ('fullname', 'email', 'username', 'password1', 'password2', 'phone', 'role')

    def save(self, commit=True):
        user = super(SignupForm, self).save(commit=False)
        user.email = self.cleaned_data['email']
        user.fullname = self.cleaned_data['fullname']
        user.phone = self.cleaned_data['phone']
        user.role = self.cleaned_data['role']
        if commit:
            user.save()
        return user



# Login Form
class LoginForm(forms.Form):
    email_or_username = forms.CharField(max_length=254, label='Email or Username', required=True)
    password = forms.CharField(widget=forms.PasswordInput, label='Password', required=True)

    def clean(self):
        cleaned_data = super().clean()
        email_or_username = cleaned_data.get('email_or_username')
        password = cleaned_data.get('password')

        # Perform any custom validation or data cleaning here
        if not email_or_username or not password:
            raise forms.ValidationError("Please provide both email/username and password.")
        
        return cleaned_data
