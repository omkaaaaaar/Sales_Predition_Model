from django.db import models
from django.contrib.auth.models import AbstractUser, Group, Permission

# Custom User model inheriting from AbstractUser
class CustomUser(AbstractUser):
    fullname = models.CharField(max_length=255)
    phone = models.CharField(max_length=15, blank=True, null=True)
    
    ROLE_CHOICES = (
        ('Developer', 'Developer'),
        ('Admin', 'Admin'),
        ('Employee', 'Employee'),
    )
    role = models.CharField(max_length=10, choices=ROLE_CHOICES, default='Employee')

    groups = models.ManyToManyField(Group, related_name='customuser_set', blank=True)
    user_permissions = models.ManyToManyField(Permission, related_name='customuser_permissions', blank=True)

    def __str__(self):
        return self.username
