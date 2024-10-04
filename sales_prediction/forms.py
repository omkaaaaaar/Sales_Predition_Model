# Signup Form
from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import CustomUser, Forgot_User # Import the custom user model
from django import forms
from .models import Contact

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
    

class ContactForm(forms.ModelForm):
    class Meta:
        model = Contact
        fields = ['full_name', 'email', 'message']  # Specify the fields to include in the form

        # Add custom widgets and field attributes if necessary
        widgets = {
            'full_name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Your Full Name'}),
            'email': forms.EmailInput(attrs={'class': 'form-control', 'placeholder': 'Your Email'}),
            'message': forms.Textarea(attrs={'class': 'form-control', 'placeholder': 'Your Message', 'rows': 5}),
        }

        labels = {
            'full_name': 'Full Name',
            'email': 'Email Address',
            'message': 'Message',
        }
        



class ResetPasswordForm(forms.ModelForm):
    new_password1 = forms.CharField(widget=forms.PasswordInput, label="New Password")
    new_password2 = forms.CharField(widget=forms.PasswordInput, label="Confirm New Password")

    class Meta:
        model = Forgot_User  # Use Forgot_User model instead of CustomUser
        fields = ['email', 'username', 'phone', 'new_password1', 'new_password2']  # Include relevant fields

    def clean(self):
        cleaned_data = super().clean()
        new_password1 = cleaned_data.get("new_password1")
        new_password2 = cleaned_data.get("new_password2")

        if new_password1 and new_password2 and new_password1 != new_password2:
            raise forms.ValidationError("Passwords do not match")

        return cleaned_data

    def save(self, commit=True):
        """
        Override save method to handle password update and save other user details.
        """
        user = super().save(commit=False)  # Save the user instance without committing yet
        new_password1 = self.cleaned_data.get("new_password1")

        # Use the model method to set and save the new password
        user.set_passwords(new_password1, new_password1)

        if commit:
            user.save()  # Commit the user instance to the database

        return user
