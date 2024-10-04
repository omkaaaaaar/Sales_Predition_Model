import pandas as pd
import joblib
from django.shortcuts import render,HttpResponse
from django.http import HttpResponse, Http404
from django.contrib import messages
from django.core.files.storage import FileSystemStorage
from .prediction_model import predict_sales
import os 
import plotly.utils
import json
import plotly.graph_objects as go
import json
import numpy as np
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from .forms import SignupForm, LoginForm , ContactForm , ResetPasswordForm
from django.contrib.auth import login, authenticate
from django.contrib.auth import get_user_model
from django.contrib.auth.hashers import make_password
from django.utils import timezone
from django.contrib.auth.decorators import user_passes_test
from .models import Contact,Forgot_User  # Import your Contact model

CustomUser = get_user_model()
User = get_user_model()  # Reference to your CustomUser model


# Get the absolute path of the pickle file
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Get the base directory of your Django project
model_path = os.path.join(base_dir, 'sales_prediction', 'pickle_models', 'sales_prediction_model.pkl')
label_encoders_path = os.path.join(base_dir, 'sales_prediction', 'pickle_models', 'label_encoders.pkl')

# Load the trained model and label encoders at the module level
model = joblib.load(model_path)
label_encoders = joblib.load(label_encoders_path)

# Assuming label_encoders['Item_Type'] is the label encoder for the Item_Type column
item_type_encoder = label_encoders['Item_Type']

def upload_file(request):
    if request.method == 'POST' and request.FILES['file']:
        upload = request.FILES['file']
        fs = FileSystemStorage()
        file_name = fs.save(upload.name, upload)
        file_url = fs.url(file_name)

        try:
            # Load the CSV into a pandas DataFrame
            data = pd.read_csv(fs.path(file_name))
        except Exception as e:
            return render(request, 'upload.html', {'error': 'Error reading CSV file: {}'.format(str(e))})

        # List of required columns in the uploaded CSV
        required_columns = [
            'Item_Identifier', 'Item_Weight', 'Item_Fat_Content', 'Item_Visibility',
            'Item_Type', 'Item_MRP', 'Outlet_Identifier', 'Outlet_Establishment_Year',
            'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'Sales'
        ]

        # Check if all required columns exist in the uploaded CSV
        if not all(column in data.columns for column in required_columns):
            missing_columns = [col for col in required_columns if col not in data.columns]
            error_message = "The uploaded CSV is missing the following required columns: " + ", ".join(missing_columns)
            return render(request, 'upload.html', {'error': error_message})

        try:
            # Predict sales using the trained model
            predictions = predict_sales(data)
        except Exception as e:
            return render(request, 'upload.html', {'error': 'Error during prediction: {}'.format(str(e))})

        # Add predictions to the DataFrame
        data['Predicted_Sales'] = predictions

        # Decode the Item_Type column to get original labels (like "Snacks", "Dairy", etc.)
        try:
            # Check if 'Item_Type' is numeric and if the values are within valid encoder classes
            valid_labels = np.isin(data['Item_Type'], range(len(item_type_encoder.classes_)))

            # Temporarily assign an invalid value (-1) for unseen labels
            data.loc[~valid_labels, 'Item_Type'] = -1

            # Cast 'Item_Type' to integers and inverse transform valid values
            data['Item_Type'] = data['Item_Type'].astype(int)
            data.loc[valid_labels, 'Item_Type'] = item_type_encoder.inverse_transform(data.loc[valid_labels, 'Item_Type'])

            # Replace any invalid labels (-1) with a placeholder such as 'Unknown'
            data['Item_Type'].replace(-1, 'Cloths', inplace=True)
        except ValueError as e:
            return render(request, 'upload.html', {'error': f"Error decoding 'Item_Type': {str(e)}"})

        # Prepare results for display without grouping by any column
        prediction_results = []
        for idx, row in data.iterrows():
            predicted_sales = row['Predicted_Sales']
            actual_sales = row['Sales'] if 'Sales' in row else None  # If 'Sales' is present in the CSV
            sales_difference = predicted_sales - actual_sales if actual_sales else None
            status = "increasing" if actual_sales and predicted_sales > actual_sales else "decreasing"

            prediction_results.append({
                'Item_Identifier': row['Item_Identifier'],
                'Item_Type': row['Item_Type'],
                'Item_MRP': row['Item_MRP'],
                'Predicted_Sales': predicted_sales,
                'Actual_Sales': actual_sales,
                'Status': status if actual_sales else None
            })

        # Pass this information to the template
        return render(request, 'results.html', {'predictions': prediction_results})

    return render(request, 'upload.html')




#Grouped by Code

def upload_file_group_by(request):
    if request.method == 'POST' and request.FILES['file']:
        upload = request.FILES['file']
        fs = FileSystemStorage()
        file_name = fs.save(upload.name, upload)
        file_url = fs.url(file_name)

        try:
            # Load the CSV into a pandas DataFrame
            data = pd.read_csv(fs.path(file_name))
        except Exception as e:
            return render(request, 'upload.html', {'error': 'Error reading CSV file: {}'.format(str(e))})

        # List of required columns in the uploaded CSV
        required_columns = [
            'Item_Identifier', 'Item_Weight', 'Item_Fat_Content', 'Item_Visibility',
            'Item_Type', 'Item_MRP', 'Outlet_Identifier', 'Outlet_Establishment_Year',
            'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'Sales'
        ]

        # Check if all required columns exist in the uploaded CSV
        if not all(column in data.columns for column in required_columns):
            missing_columns = [col for col in required_columns if col not in data.columns]
            error_message = "The uploaded CSV is missing the following required columns: " + ", ".join(missing_columns)
            return render(request, 'upload.html', {'error': error_message})

        try:
            # Predict sales using the trained model
            predictions = predict_sales(data)
        except Exception as e:
            return render(request, 'upload.html', {'error': 'Error during prediction: {}'.format(str(e))})

        # Add predictions to the DataFrame
        data['Predicted_Sales'] = predictions

        # Group the data by 'Item_Type'
        grouped_data = data.groupby('Item_Type').agg({
            'Item_MRP': 'mean',
            'Sales': 'sum',
            'Predicted_Sales': 'sum'
        }).reset_index()

        # Decode the Item_Type column to get original labels (like "Snacks", "Dairy", etc.)
        item_type_encoder = label_encoders['Item_Type']  # Load the label encoder for Item_Type

        try:
            # Validate that the 'Item_Type' column contains valid numeric labels
            valid_labels = np.isin(grouped_data['Item_Type'], range(len(item_type_encoder.classes_)))

            # Temporarily assign an invalid value (-1) for unseen labels
            grouped_data.loc[~valid_labels, 'Item_Type'] = -1

            # Cast 'Item_Type' to integers and inverse transform valid values
            grouped_data['Item_Type'] = grouped_data['Item_Type'].astype(int)
            grouped_data.loc[valid_labels, 'Item_Type'] = item_type_encoder.inverse_transform(
                grouped_data.loc[valid_labels, 'Item_Type'])

            # Replace any invalid labels (-1) with a placeholder such as 'Unknown'
            grouped_data['Item_Type'].replace(-1, 'Unknown', inplace=True)
        except ValueError as e:
            return render(request, 'upload.html', {'error': f"Error decoding 'Item_Type': {str(e)}"})

        # Plotly graph using bar chart with custom colors for each trace
        fig = go.Figure()

        # Add tooltips to the bar chart
        fig.add_trace(go.Bar(
            x=grouped_data['Item_Type'],
            y=grouped_data['Sales'],
            name='Sales',
            marker_color='#1f77b4',  # Blue color for Sales
            hoverinfo='text',
            text=[f"Item MRP: {mrp}<br>Actual Sales: {sales}" for mrp, sales in zip(grouped_data['Item_MRP'], grouped_data['Sales'])]
        ))

        fig.add_trace(go.Bar(
            x=grouped_data['Item_Type'],
            y=grouped_data['Predicted_Sales'],
            name='Predicted Sales',
            marker_color='#2ca02c',  # Green color for Predicted Sales
            hoverinfo='text',
            text=[f"Item MRP: {mrp}<br>Predicted Sales: {predicted_sales}" for mrp, predicted_sales in zip(grouped_data['Item_MRP'], grouped_data['Predicted_Sales'])]
        ))

        # Update layout for better visualization
        fig.update_layout(
            barmode='group',
            width=1200,  # Increase width
            height=970,  # Adjust height
            title='Comparison of Actual vs Predicted Sales',
            title_x=0.5,  # Center the title
            title_y=0.9,
            font=dict(size=20),  # Font size for readability
            title_font=dict(size=24),  # Title font size
            xaxis_tickangle=-45,  # Rotate x-axis labels
            yaxis=dict(
                tick0=0,
                dtick=0.1e6  # Set y-axis tick interval to 0.1M
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )  # Position legend at the top
        )

        # Convert plot to JSON
        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        # Prepare results for the table
        prediction_results = []
        for idx, row in grouped_data.iterrows():
            status = "increasing" if row['Predicted_Sales'] > row['Sales'] else "decreasing"
            prediction_results.append({
                'Item_Type': row['Item_Type'],
                'Item_MRP': row['Item_MRP'],
                'Predicted_Sales': row['Predicted_Sales'],
                'Actual_Sales': row['Sales'],
                'Status': status
            })

        return render(request, 'index_results.html', {
            'predictions': prediction_results,
            'graph_json': graph_json
        })
    return render(request, 'upload.html')






def results(request):
    return render(request,'results.html')




def download_predictions(request):
    if request.method == 'POST':
        # Get the grouped data from the session and convert it back to a DataFrame
        grouped_data_json = request.session.get('grouped_data', None)

        if not grouped_data_json:
            return HttpResponse("No data available for download.", status=400)

        # Convert JSON back to DataFrame
        grouped_data = pd.read_json(grouped_data_json)

        # Prepare CSV for download
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="predictions.csv"'
        grouped_data.to_csv(response, index=False)
        return response




def download_sample_csv(request):
    # Define the path to the CSV file based on the provided location
    file_path = os.path.join('sales_prediction', 'new_data_file.csv')

    # Check if the file exists
    if os.path.exists(file_path):
        with open(file_path, 'rb') as csv_file:
            response = HttpResponse(csv_file.read(), content_type='text/csv')
            response['Content-Disposition'] = 'attachment; filename="new_data_file.csv"'
            return response
    else:
        raise Http404("Sample CSV file not found.")


def index(request):
    return render(request,'index.html')

def sample(request):
    return render(request,'sample.html')


def signup_view(request):
    if request.method == 'POST':
        form = SignupForm(request.POST)
        if form.is_valid():
            print("Form is valid, saving user...")
            form.save()
            return redirect('login')  # Redirect to login after signup
        else:
            print("Form errors:", form.errors)  # Print out form errors
    else:
        form = SignupForm()
    return render(request, 'signup.html', {'form': form})



def login_view(request):
    print("Login view accessed")
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            username_or_email = form.cleaned_data['email_or_username']
            password = form.cleaned_data['password']

            # Check if the input is an email or username
            try:
                if '@' in username_or_email:
                    user = User.objects.get(email=username_or_email)
                else:
                    user = User.objects.get(username=username_or_email)

                # Authenticate using the username (Django uses 'username' for login)
                user = authenticate(request, username=user.username, password=password)

                if user is not None:
                    login(request, user)
                    return redirect('upload_file')  # Redirect to the dashboard after successful login
                else:
                    form.add_error(None, 'Invalid credentials')
            except User.DoesNotExist:
                form.add_error(None, 'User does not exist')
    else:
        form = LoginForm()

    return render(request, 'login.html', {'form': form})


def about(request):
    return render(request,'about.html')


def contact(request):
    success_message = None  # Initialize message as None
    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            # Create an object from the form but don't save it yet
            contact = form.save(commit=False)
            # Set the current timestamp for submission
            contact.submitted_at = timezone.now()
            # Save the contact to the database
            contact.save()
            # Display success message
            success_message = "Form submitted successfully. Thank you for contacting us!"
            form = ContactForm()  # Reset the form after submission
        else:
            success_message = "There was an error submitting the form. Please try again."
    else:
        form = ContactForm()

    return render(request, 'contact_us.html', {'form': form, 'success_message': success_message})


# Decorator to check if the user is an admin
def is_admin(user):
    return user.is_authenticated and user.role == 'admin'

@user_passes_test(is_admin)
def admin_view(request):
    contacts = Contact.objects.all()  # Fetch all contact entries
    users = CustomUser.objects.all()  # Fetch all users

    return render(request, 'admin.html', {'contacts': contacts, 'users': users})

def forgot_password_view(request):
    if request.method == 'POST':
        form = ResetPasswordForm(request.POST)  # Use ResetPasswordForm for validation
        if form.is_valid():
            username = form.cleaned_data['username']
            email = form.cleaned_data['email']
            phone = form.cleaned_data['phone']

            # Check if the user exists with the provided username, email, and phone
            try:
                user = Forgot_User.objects.get(username=username, email=email, phone=phone)
                # Once verified, redirect to the password reset form
                request.session['reset_user_id'] = user.id  # Store user id in session
                return redirect('reset_password')  # Redirect to reset password form
            except Forgot_User.DoesNotExist:
                messages.error(request, "User with the provided details does not exist.")
    else:
        form = ResetPasswordForm()  # Render the reset password form

    return render(request, 'forgot_password.html', {'form': form})


def reset_password_view(request):
    if 'reset_user_id' not in request.session:
        return redirect('forgot_password')  # Redirect to forgot password if no user in session

    user_id = request.session['reset_user_id']
    try:
        user = Forgot_User.objects.get(id=user_id)
    except Forgot_User.DoesNotExist:
        messages.error(request, "User does not exist.")
        return redirect('forgot_password')

    if request.method == 'POST':
        form = ResetPasswordForm(request.POST)
        if form.is_valid():
            # Update the user's password
            new_password1 = form.cleaned_data['new_password1']
            new_password2 = form.cleaned_data['new_password2']
            
            # Ensure passwords match
            if new_password1 != new_password2:
                messages.error(request, "Passwords do not match.")
            else:
                user.set_password(new_password1)  # Set the new password
                user.save()
                del request.session['reset_user_id']  # Clear the session after password is changed
                messages.success(request, "Your password has been reset successfully.")
                return redirect('login')  # Redirect to login after successful password reset
    else:
        form = ResetPasswordForm()

    return render(request, 'sample.html', {'form': form}) #reset password page