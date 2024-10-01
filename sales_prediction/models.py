from django.contrib.auth.models import AbstractUser
from django.db import models

class CustomUser(AbstractUser):
    fullname = models.CharField(max_length=100, blank=True, null=True)
    phone = models.CharField(max_length=15, blank=True, null=True)
    
    ROLE_CHOICES = [
        ('user', 'User'),
        ('admin', 'Admin'),
        ('Developer','Developer')
    ]
    role = models.CharField(max_length=10, choices=ROLE_CHOICES, default='user')

    def __str__(self):
        return self.username
