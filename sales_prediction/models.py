from django.contrib.auth.models import AbstractUser, Group, Permission
from django.db import models
from django.core.exceptions import ValidationError
from django.utils import timezone

class CustomUser(AbstractUser):
    fullname = models.CharField(max_length=100, blank=True, null=True)
    phone = models.CharField(max_length=15, blank=True, null=True)
    
    ROLE_CHOICES = [
        ('Employee', 'Employee'),
        ('admin', 'Admin'),
        ('Developer','Developer')
    ]
    role = models.CharField(max_length=10, choices=ROLE_CHOICES, default='user')

    def __str__(self):
        return self.username
    

class Contact(models.Model):
    full_name = models.CharField(max_length=100)
    email = models.EmailField()
    message = models.TextField()
    submitted_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return self.full_name
    

class Forgot_User(AbstractUser):
    # Add any additional fields specific to Forgot_User here
    email = models.EmailField(('email address'), unique=True)
    phone = models.CharField(max_length=15, blank=True, null=True)
    password_change_time = models.DateTimeField(auto_now=True)  # Track when password is changed

    class Meta:
        permissions = (('can_do_something_else', 'Can do something else'),)  # Example permission
        unique_together = ('username', 'email')  # Example unique constraint

    # Custom related names for groups and user_permissions
    groups = models.ManyToManyField(Group, related_name='forgotuser_set', blank=True)
    user_permissions = models.ManyToManyField(Permission, related_name='forgotuser_set', blank=True)

    def __str__(self):
        return self.username

    def set_passwords(self, new_password1, new_password2):
        """
        A method to update the user's password with validation.
        """
        if new_password1 and new_password2 and new_password1 != new_password2:
            raise ValidationError("Passwords do not match")

        self.set_password(new_password1)
        self.save()



