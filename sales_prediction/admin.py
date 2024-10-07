from django.contrib import admin
from .models import CustomUser,Contact,Forgot_User,BlogPost  # Import your models

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

# Customize the admin form for Forgot_User
class ForgotUserAdmin(admin.ModelAdmin):
    model = Forgot_User

    # Define which fields should be displayed in the list view
    list_display = ('username', 'email', 'phone', 'is_staff', 'password_change_time')
    search_fields = ('username', 'email', 'phone')
    readonly_fields = ('password_change_time',)  # Make password_change_time read-only

    # Specify the fieldsets to display fields in sections
    fieldsets = (
        (None, {'fields': ('username', 'password')}),
        (('Personal info'), {'fields': ('email', 'phone')}),
        (('Permissions'), {'fields': ('is_active', 'is_staff', 'is_superuser', 'groups', 'user_permissions')}),
        (('Important dates'), {'fields': ('last_login', 'password_change_time')}),
    )

    # Fields to be displayed when creating a new user
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('username', 'email', 'phone', 'password1', 'password2')}
        ),
    )

    ordering = ('username',)

# Register your custom user model with the admin
admin.site.register(Forgot_User, ForgotUserAdmin)

class BlogPostAdmin(admin.ModelAdmin):
    list_display = ('title', 'author', 'date_posted', 'status')  # Fields to display in the admin list view
    list_filter = ('status', 'date_posted')  # Filters to apply in the list view
    search_fields = ('title', 'content', 'tags')  # Fields to search through
    ordering = ('-date_posted',)  # Default ordering for the list view

    # Fields to show in the detail view
    fieldsets = (
        (None, {
            'fields': ('title', 'content', 'image_url', 'author', 'status', 'tags')
        }),
    )

    # Optional: Add inline editing for related models (if needed)

admin.site.register(BlogPost, BlogPostAdmin)  # Register the BlogPost model with the admin interface