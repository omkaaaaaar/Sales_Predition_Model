from django.contrib.auth.models import AbstractUser
from django.db import models
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