from django.contrib import admin
from .models import CustomUser, Contact  # Import your models

# Register your models here.
# Register CustomUser with the admin site
class CustomUserAdmin(admin.ModelAdmin):
    list_display = ('username', 'fullname', 'email', 'phone', 'role', 'is_staff','password')  # Display these fields in the admin list view
    search_fields = ('username', 'email', 'fullname')  # Enable search on these fields
    list_filter = ('role', 'is_staff')  # Allow filtering by role and staff status
    ordering = ('-date_joined',)  # Order users by their join date, newest first

admin.site.register(CustomUser, CustomUserAdmin)  # Register the CustomUser model

# Register Contact model
class ContactAdmin(admin.ModelAdmin):
    list_display = ('full_name', 'email', 'submitted_at','message')  # Display these fields in the admin list view
    search_fields = ('full_name', 'email')  # Enable search on these fields
    list_filter = ('submitted_at',)  # Allow filtering by submission date
    ordering = ('-submitted_at',)  # Order contacts by submission date, newest first

admin.site.register(Contact, ContactAdmin)  # Register the Contact model