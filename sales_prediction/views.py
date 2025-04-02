import pandas as pd
import joblib
from django.shortcuts import render,HttpResponse,get_object_or_404
from django.http import HttpResponse, Http404 , JsonResponse
from django.contrib import messages
from django.core.files.storage import FileSystemStorage
from .prediction_model import predict_sales
import os 
import plotly.utils
import json
import plotly.graph_objects as go
import numpy as np
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from .forms import SignupForm, LoginForm , ContactForm , ResetPasswordForm,BlogPostForm
from django.contrib.auth import login, authenticate
from django.contrib.auth import get_user_model
from django.utils import timezone
from django.contrib.auth.decorators import user_passes_test
from .models import Contact, BlogPost # Import your Contact model
import plotly.express as px
import re
from django.http import JsonResponse
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
import time 
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from django.core.cache import cache
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from tabulate import tabulate
from django.contrib.sessions.models import Session



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


def predict_sales_from_csv(request):
    available_columns = []  # Store column names

    if request.method == 'POST':
        csv_file = request.FILES.get('file')
        selected_columns = request.POST.get('columns', '').split(',')

        if not csv_file:
            return render(request, 'upload.html', {'error': 'No file uploaded'})

        # Save uploaded file
        fs = FileSystemStorage()
        filename = fs.save(csv_file.name, csv_file)
        file_path = fs.path(filename)

        # Load CSV data
        try:
            df = pd.read_csv(file_path)

            if df.empty:
                return render(request, 'upload.html', {'error': 'CSV file is empty.'})

            # Store available column names
            available_columns = df.columns.tolist()

            # Filter DataFrame if columns are selected
            if selected_columns and selected_columns[0]:  # Check if user entered anything
                df = df[selected_columns]

            csv_data = df.to_dict(orient='records')  # Convert DataFrame to list of dictionaries

            return render(request, 'results.html', {
                'csv_data': csv_data,
                'selected_columns': selected_columns
            })
        except Exception as e:
            return render(request, 'upload.html', {'error': f'Error reading CSV file: {str(e)}'})

    return render(request, 'upload.html', {'available_columns': available_columns})


def csv_grouped_main(request):
    available_columns = []  # Store column names

    if request.method == 'POST':
        csv_file = request.FILES.get('file')
        selected_columns = request.POST.get('columns', '').split(',')

        if not csv_file:
            return render(request, 'upload.html', {'error': 'No file uploaded'})

        # Save uploaded file
        fs = FileSystemStorage()
        filename = fs.save(csv_file.name, csv_file)
        file_path = fs.path(filename)

        # Load CSV data
        try:
            df = pd.read_csv(file_path)

            if df.empty:
                return render(request, 'upload.html', {'error': 'CSV file is empty.'})

            # Store available column names
            available_columns = df.columns.tolist()

            # Filter DataFrame if columns are selected
            if selected_columns and selected_columns[0]:  # Check if user entered anything
                df = df[selected_columns]

            csv_data = df.to_dict(orient='records')  # Convert DataFrame to list of dictionaries

            return render(request, 'results.html', {
                'csv_data': csv_data,
                'selected_columns': selected_columns
            })
        except Exception as e:
            return render(request, 'upload.html', {'error': f'Error reading CSV file: {str(e)}'})

    return render(request, 'upload.html', {'available_columns': available_columns})











######################################################################################

import joblib
import pandas as pd
import numpy as np
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def csv_grouped(request):
    available_columns = []  # Store available columns

    if request.method == "POST":
        csv_file = request.FILES.get("file")
        selected_features = request.POST.get("features", "").split(",")
        target_column = request.POST.get("target", "")

        if not csv_file:
            return render(request, "upload.html", {"error": "No file uploaded"})

        # Save uploaded file
        fs = FileSystemStorage()
        filename = fs.save(csv_file.name, csv_file)
        file_path = fs.path(filename)

        try:
            # Load CSV data
            df = pd.read_csv(file_path)

            if df.empty:
                return render(request, "upload.html", {"error": "CSV file is empty."})

            # Store column names for selection
            available_columns = df.columns.tolist()

            # Check if user selected features and target
            if not selected_features or not target_column:
                return render(
                    request,
                    "upload.html",
                    {"available_columns": available_columns, "error": "Select features and target."},
                )

            # Encode categorical columns
            label_encoders = {}
            for col in df.select_dtypes(include=["object"]).columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                label_encoders[col] = le

            # Prepare data for training
            X = df[selected_features]
            y = df[target_column]

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Predict sales
            df["Predicted Sales"] = model.predict(X)

            # Decode categorical columns
            for col in label_encoders:
                df[col] = label_encoders[col].inverse_transform(df[col])

            # Add trend column (increase/decrease)
            df["Trend"] = df.apply(
                lambda row: "🟢 Increasing" if row["Predicted Sales"] > row[target_column] else "🔴 Decreasing", axis=1
            )

            # Convert to list of dictionaries for rendering
            csv_data = df.to_dict(orient="records")

            print(csv_data)

            return render(
                request,
                "sample.html",
                {"csv_data": csv_data, "selected_features": selected_features, "target_column": target_column},
            )

        except Exception as e:
            return render(request, "upload.html", {"error": f"Error processing CSV: {str(e)}"})

    return render(request, "upload.html", {"available_columns": available_columns})


def csv_grouped_view(request):
    available_columns = []  # Store available columns

    if request.method == "POST":
        csv_file = request.FILES.get("file")
        selected_features = request.POST.get("features", "").split(",")
        target_column = request.POST.get("target", "")

        if not csv_file:
            return render(request, "upload.html", {"error": "No file uploaded"})

        # Save uploaded file
        fs = FileSystemStorage()
        filename = fs.save(csv_file.name, csv_file)
        file_path = fs.path(filename)

        try:
            # Load CSV data
            df = pd.read_csv(file_path)

            if df.empty:
                return render(request, "upload.html", {"error": "CSV file is empty."})

            # Store column names for selection
            available_columns = df.columns.tolist()
            request.session["available_columns"] = available_columns

            # Check if user selected features and target
            if not selected_features or not target_column:
                return render(
                    request,
                    "upload.html",
                    {"available_columns": available_columns, "error": "Select features and target."},
                )

            # Encode categorical columns
            label_encoders = {}
            for col in df.select_dtypes(include=["object"]).columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                label_encoders[col] = le

            # Prepare data for training
            X = df[selected_features]
            y = df[target_column]

            # Store selected features, target column, and preprocessed data in session
            request.session["selected_features"] = selected_features
            request.session["target_column"] = target_column

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Predict sales
            df["Predicted_Sales"] = model.predict(X)

            # Decode categorical columns
            for col in label_encoders:
                df[col] = label_encoders[col].inverse_transform(df[col])

            # Add trend column (increase/decrease)
            df["Trend"] = df.apply(
                lambda row: "🟢 Increasing" if row["Predicted_Sales"] > row[target_column] else "🔴 Decreasing",
                axis=1
            )

            # Dynamically detect grouping columns
            group_by_columns = [col for col in ["category", "item", "location", "month"] if col in df.columns]

            if not group_by_columns:
                return render(
                    request,
                    "upload.html",
                    {"error": "No valid columns found for grouping. Ensure your CSV has at least one of: category, item, location, month."},
                )

            # Group data dynamically
            grouped_data = df.groupby(group_by_columns).agg(
                total_sales_value=(target_column, 'sum'),
                Predicted_Sales=('Predicted_Sales', 'sum')
            ).reset_index()
            
            # Convert Predicted_Sales to integer (whole number)
            grouped_data["Predicted_Sales"] = grouped_data["Predicted_Sales"].astype(int)


            # Add Trend Column (Increase/Decrease) after grouping
            grouped_data["Trend"] = grouped_data.apply(
                lambda row: "🟢 Increasing" if row["total_sales_value"] > row["Predicted_Sales"] else "🔴 Decreasing",
                axis=1
            )

            # Add Difference column (Actual Sales - Predicted Sales) and ensure it's an integer
            # Calculate Difference and ensure it's always positive
            grouped_data["Difference"] = (grouped_data["total_sales_value"] - grouped_data["Predicted_Sales"]).astype(int)

            # Convert Difference to absolute value to remove negative signs
            grouped_data["Difference"] = grouped_data["Difference"].abs()
            # Convert to list of dictionaries for rendering
            grouped_data_list = grouped_data.to_dict(orient="records")
            print(grouped_data_list)
            # Store the processed grouped data in session
            request.session["grouped_data"] = grouped_data_list  
            print(grouped_data.head())  # Debugging: Check if Difference column exists

            return render(
                request,
                "grouped_predictions.html",
                {
                    "grouped_data": grouped_data_list,
                    "available_columns": available_columns,
                },
            )

        except Exception as e:
            return render(request, "upload.html", {"error": f"Error processing CSV: {str(e)}"})

    return render(request, "upload.html", {"available_columns": available_columns})


def filter_predictions(request):
    filter_value = request.GET.get("filter_value", "").lower()

    # ✅ Load the grouped data from session
    grouped_data = request.session.get("grouped_data", [])

    # ✅ Debugging: Print initial data size
    print(f"\n🔍 Initial Grouped Data Size: {len(grouped_data)}")

    # ✅ Apply filtering
    if filter_value:
        filtered_data = [
            row for row in grouped_data
            if any(filter_value in str(value).lower() for value in row.values())
        ]
    else:
        filtered_data = grouped_data

    # ✅ Debugging: Print filtered data preview
    print(f"\n🔍 Filter Value: {filter_value}")
    print("📊 Filtered Data Preview:\n", filtered_data[:5])

    # ✅ Store filtered data in session for verification
    request.session["debug_filtered_data"] = filtered_data
    request.session.modified = True

    # ✅ Convert filtered data to DataFrame
    df = pd.DataFrame(filtered_data)

    # ✅ Debugging: Check if filtered data is stored in session
    session_filtered_data = request.session.get("debug_filtered_data", [])
    print(f"\n🔎 Stored in Session (After Filtering): {len(session_filtered_data)} rows")

    # ✅ Check if filtered data is empty
    if df.empty:
        return JsonResponse({"error": f"No data found for filter: {filter_value}"}, status=400)

    # ✅ Generate visualization if required columns exist
    chart_html = ""
    if "category" in df.columns and "Predicted_Sales" in df.columns:
        fig = go.Figure(
            data=[
                go.Bar(
                    x=df["category"],
                    y=df["Predicted_Sales"],
                    marker=dict(color="blue"),
                    name="Predicted Sales",
                )
            ]
        )
        fig.update_layout(title="Filtered Predicted Sales by Category", xaxis_title="Category", yaxis_title="Predicted Sales")
        chart_html = fig.to_html(full_html=False)

    return render(
        request,
        "grouped_predictions.html",
        {"grouped_data": filtered_data, "chart_html": chart_html}
    )




def visualize_filtered_data(request):
    # Retrieve filtered data from session
    filtered_data = request.session.get("filtered_data", [])

    if not filtered_data:
        print("No filtered data found in session.")
        return JsonResponse({"error": "No filtered data available to visualize."})

    # Convert to DataFrame
    df = pd.DataFrame(filtered_data)

    if df.empty:
        print("DataFrame is empty after filtering.")
        return JsonResponse({"error": "No filtered data available to visualize."})

    print("Filtered Data for Visualization:", df.head())  # Debugging

    # Ensure required columns exist
    if "category" in df.columns and "Predicted_Sales" in df.columns:
        fig = go.Figure(
            data=[
                go.Bar(
                    x=df["category"],
                    y=df["Predicted_Sales"],
                    marker=dict(color="blue"),
                    name="Predicted Sales",
                )
            ]
        )
        fig.update_layout(title="Filtered Predicted Sales by Category", xaxis_title="Category", yaxis_title="Predicted Sales")
    else:
        return JsonResponse({"error": "Required columns are missing in the dataset."})

    # Convert Plotly figure to HTML
    chart_html = fig.to_html(full_html=False)

    return JsonResponse({"chart_html": chart_html})


# Load API key for Google Gemini

api_key = "AIzaSyBM2Xj-1r8RKRiEIooOFnL1c-yYeJvxYRU"

if not api_key:
    raise ValueError("Error: GEMINI_API_KEY is not set. Check your .env file!")

# Initialize Google Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key=api_key
)


def remove_markdown_bold(text):
    """ Remove bold formatting (**) from AI-generated text. """
    return re.sub(r"\*\*(.*?)\*\*", r"\1", text)  # Replace **text** with text

def clear_trend_explanation(request):
    """ Clears the AI-generated trend explanation from the session """
    if "trend_explanation" in request.session:
        del request.session["trend_explanation"]
    return JsonResponse({"message": "Trend analysis cleared successfully."})

def generate_trend_explanation(request):
    """ Generate an AI-powered trend analysis based on filtered sales data. """
    
    # Retrieve filtered data from session
    filtered_data = request.session.get("filtered_data", [])

    if not filtered_data:
        return JsonResponse({"error": "No filtered data available for analysis."})

    # Convert to DataFrame
    df = pd.DataFrame(filtered_data)

    if df.empty:
        return JsonResponse({"error": "Filtered data is empty."})

    # Extract relevant columns
    if "category" not in df.columns or "Predicted_Sales" not in df.columns:
        return JsonResponse({"error": "Required columns are missing."})

    # Generate summary statistics for AI model
    total_sales = df["Predicted_Sales"].sum()
    avg_sales = df["Predicted_Sales"].mean()
    top_categories = df.groupby("category")["Predicted_Sales"].sum().sort_values(ascending=False).head(3).to_dict()

    # Prepare structured prompt
    prompt = f"""
    You are a professional business analyst specializing in sales forecasting. Based on the following sales data, provide insights in the specified format:

    📊 Sales Overview  
    - Total Predicted Sales: ₹{total_sales}  
    - Average Predicted Sales per Category: ₹{avg_sales:.2f}  

    🏆 Top 3 Best-Performing Categories:  
    {top_categories}  

    📢 AI Sales Insights  

    1️⃣ Trend Overview: Explain the general sales trend.  
    2️⃣ Category Performance: Highlight which categories are growing or declining.  
    3️⃣ Business Strategy: Provide 2-3 strategic suggestions to improve sales.  

    Ensure your response follows this format strictly to maintain clarity.
    """

    try:
        # Send request to Google Gemini AI
        response = model.invoke(
            [
                SystemMessage("You are an expert business analyst providing structured insights on sales data."),
                HumanMessage(prompt)
            ]
        )
        ai_explanation = response.content.strip()
        
        # Remove Markdown bold formatting (**) before saving
        ai_explanation = remove_markdown_bold(ai_explanation)

    except Exception as e:
        ai_explanation = f"AI analysis failed: {str(e)}"

    # Store cleaned explanation in session
    request.session["trend_explanation"] = ai_explanation

    return JsonResponse({"trend_explanation": ai_explanation})






# Load API key from environment variable
api_key =  "AIzaSyDLOi_cx0pJ0NgTFsxE8XMd8HaveCXCuTA"

# Ensure API key is set
if not api_key:
    raise ValueError("❌ Error: GEMINI_API_KEY is not set. Check your .env file!")

# Initialize Google Gemini Model
try:
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key
    )
    print("✅ Gemini Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Failed to Load Model: {e}")


def format_markdown_to_html(text):
    """ Convert bullet points (*) and line breaks to HTML format. """
    text = remove_markdown_bold(text)  # Convert bold markdown to HTML <strong>

    # Convert bullet points (*) to <ul><li> format
    text = re.sub(r"\n\* (.+)", r"<li>\1</li>", text)  # Convert * points into <li> items
    text = re.sub(r"(</li>)\n(<li>)", r"\1\n\2", text)  # Ensure proper spacing between list items
    text = re.sub(r"(<li>.*?</li>)", r"<ul>\n\1\n</ul>", text)  # Wrap <li> items inside <ul>

    # Convert double newlines to paragraph breaks for better formatting
    text = re.sub(r"\n\n", r"<br><br>", text)

    return text


def Quick_AI_Sales_Summary(request):
    """
    Generates an AI-powered summary of dynamically filtered sales data.
    """
    if request.method == "POST":
        filtered_data = request.session.get("filtered_data", [])

        if not filtered_data:
            return JsonResponse({"error": "No sufficient data available for sales summary."})

        data_payload = json.dumps(filtered_data, indent=4)

        prompt = f"""
        You are an AI sales analyst.

        Given the following dynamically filtered sales data in JSON format:
        {data_payload}

        Provide:
        1. A summary of key sales trends.
        2. The best-performing and worst-performing items.
        3. Any noticeable patterns or insights.
        
        Format the summary in a simple, easy-to-read format.
        """

        # Retry mechanism for handling rate limits (429 errors)
        max_retries = 3
        delay = 2  # Start with 2 seconds delay

        for attempt in range(max_retries):
            try:
                response = model.invoke(prompt)

                if not response or not response.content:
                    return JsonResponse({"error": "AI model did not return a response."})

                sales_summary = format_markdown_to_html(response.content)

                return JsonResponse({"summary": sales_summary})

            except Exception as e:
                error_message = str(e)
                if "429 Resource has been exhausted" in error_message:
                    print(f"⚠️ Rate limit hit. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff (2s → 4s → 8s)
                else:
                    return JsonResponse({"error": f"Failed to generate summary: {error_message}"})

        return JsonResponse({"error": "AI model quota exceeded. Try again later."})

    return JsonResponse({"error": "Invalid request method."})



def ai_business_insights(request):
    # Retrieve filtered data from session
    session_filtered_data = request.session.get("debug_filtered_data", [])

    if not session_filtered_data:
        return JsonResponse({"error": "No data available for analysis"}, status=400)

    # Initialize data structures
    demand_forecast = []
    revenue_projection = 0
    category_contributions = {}
    region_sales = {}

    # Process data
    for entry in session_filtered_data:
        category = entry.get("category", "Unknown")
        item = entry.get("item", "Unknown")
        location = entry.get("location", "Unknown")
        total_sales = entry.get("total_sales_value", 0)
        predicted_sales = entry.get("Predicted_Sales", 0)
        trend = entry.get("Trend", "Unknown")

        # Demand Forecasting
        demand_forecast.append({
            "item": item,
            "location": location,
            "predicted_demand": predicted_sales,
            "trend": trend
        })

        # Revenue Projection
        revenue_projection += predicted_sales
        category_contributions[category] = category_contributions.get(category, 0) + predicted_sales

        # Region-wise Sales Insights
        region_sales[location] = region_sales.get(location, 0) + total_sales

    # Convert category contributions to percentages
    total_revenue = sum(category_contributions.values())
    category_percentages = [
        {"category": cat, "percentage": round((sales / total_revenue) * 100, 2)}
        for cat, sales in category_contributions.items()
    ]

    # Get top and low-performing cities
    sorted_regions = sorted(region_sales.items(), key=lambda x: x[1], reverse=True)
    top_cities = sorted_regions[:3]  # Top 3 selling regions
    low_performance_regions = sorted_regions[-3:]  # Bottom 3 selling regions

    response_data = {
        "demand_forecast": demand_forecast,
        "expected_revenue": revenue_projection,
        "top_contributing_categories": category_percentages,
        "top_cities": [{"location": loc, "total_sales": sales} for loc, sales in top_cities],
        "low_performance_regions": [{"location": loc, "total_sales": sales} for loc, sales in low_performance_regions],
    }

    return JsonResponse(response_data)




def customer_segmentation(request):
    try:
        # ✅ Get filtered data from session (fixing key name)
        filtered_data = request.session.get("debug_filtered_data", [])  # Use correct session key

        if not filtered_data:
            return JsonResponse({"error": "No filtered data found. Upload CSV first."}, status=400)

        df = pd.DataFrame(filtered_data)

        # ✅ Apply location filter from request GET parameters
        filter_value = request.GET.get("filter_value", "").strip()
        if filter_value:
            df["location"] = df["location"].str.strip().str.lower()  # Normalize location
            df = df[df["location"] == filter_value.lower()]

        # ✅ Print the first 5 rows after filtering to debug
        print("\n🔍 **Filtered Data (First 5 Rows) for Location:**", filter_value)
        print(df.head(5))

        if df.empty:
            return JsonResponse({"error": f"No data found for location: {filter_value}"}, status=400)

        # ✅ Data Cleaning
        df.columns = df.columns.str.strip()
        df.rename(columns=lambda x: x.replace("\n", "").strip(), inplace=True)
        df["item"] = df["item"].str.strip()
        df = df.drop_duplicates()

        # ✅ Ensure numeric conversion
        for col in ["total_sales_value", "Predicted_Sales"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # ✅ Calculate 'Difference' if missing
        if "Difference" not in df.columns:
            df["Difference"] = df["Predicted_Sales"] - df["total_sales_value"]

        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
        if not numeric_columns:
            return JsonResponse({"error": "No numeric columns found for clustering."}, status=400)

        X = df[numeric_columns].dropna()
        if len(X.drop_duplicates()) < 3:
            return JsonResponse({"error": "Not enough unique data points for clustering."}, status=400)

        # ✅ Standardize Features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # ✅ Apply K-Means Clustering
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df.loc[X.index, "Segment"] = kmeans.fit_predict(X_scaled)

        # ✅ Assign meaningful segment labels
        def assign_segment_label(row):
            if row["total_sales_value"] > 2e7 and row["Difference"] > 1e5:
                return "High-Value Customers"
            elif row["total_sales_value"] > 1e7:
                return "Moderate Buyers"
            elif row["total_sales_value"] < 5e6:
                return "Low-Value Customers"
            elif row["Trend"] == "🟢 Increasing":
                return "Growth Potential"
            else:
                return "At-Risk Customers"

        df["Segment_Label"] = df.apply(assign_segment_label, axis=1)

        # ✅ Assign sales trends
        df["Trend_Summary"] = df["Trend"].apply(
            lambda x: "📈 Increasing Sales" if "🟢" in x else ("📉 Decreasing Sales" if "🔴" in x else "➖ Stable Sales")
        )

        # ✅ Create structured summary
        summary_table = df.groupby(["Segment_Label", "category", "location"]).agg(
            Avg_Sales_Value=("total_sales_value", "mean"),
            Trend=("Trend_Summary", lambda x: x.value_counts().idxmax())  # Most common trend
        ).reset_index()

        # ✅ Print summary for debugging
        print("\n📊 **Filtered Customer Segmentation Summary for Location:**", filter_value)
        print(summary_table.head(5))  # Print only the first 5 rows

        # ✅ Convert to JSON
        segmented_data = df.to_dict(orient="records")

        # ✅ Store in session (limit to 1000 records)
        request.session["segmented_data"] = segmented_data[:1000]
        request.session.modified = True
        request.session.save()

        return JsonResponse({"segmentation": segmented_data[:100]}, status=200)

    except ValueError as ve:
        print(f"⚠ ValueError: {str(ve)}")
        return JsonResponse({"error": f"Clustering failed: {str(ve)}"}, status=400)

    except Exception as e:
        print(f"❌ Unexpected Error: {str(e)}")
        return JsonResponse({"error": f"Unexpected error: {str(e)}"}, status=500)



def confidence_score(request):
    try:
        # ✅ Print first 5 entries for debugging
        filtered_data = request.session.get("debug_filtered_data", [])
        print("Filtered Data from Session (First 5 Entries):", filtered_data[:5])  

        if not filtered_data:
            return JsonResponse({"error": "No filtered data available"}, status=400)

        # ✅ Ensure required keys exist
        for entry in filtered_data:
            if "total_sales_value" not in entry or "Predicted_Sales" not in entry:
                return JsonResponse({
                    "error": "Missing required keys in session data",
                    "expected_keys": ["total_sales_value", "Predicted_Sales"],
                    "received_data": entry  # Print problematic entry
                }, status=400)

        # Convert to NumPy arrays
        actual_sales = np.array([entry["total_sales_value"] for entry in filtered_data])
        predicted_sales = np.array([entry["Predicted_Sales"] for entry in filtered_data])

        # Avoid division by zero
        actual_sales = np.where(actual_sales == 0, 1, actual_sales)

        # Compute Mean Absolute Percentage Error (MAPE)
        percentage_errors = np.abs((actual_sales - predicted_sales) / actual_sales)
        mape = np.mean(percentage_errors)

        # Confidence Score Calculation
        confidence = max(0.5, min(0.99, (1 - mape)))  

        # Additional Insights
        total_actual_sales = np.sum(actual_sales)
        total_predicted_sales = np.sum(predicted_sales)
        individual_errors = [round(float(err) * 100, 2) for err in percentage_errors.tolist()]  

        # ✅ New Features
        highest_error = max(individual_errors)
        lowest_error = min(individual_errors)
        avg_error = round(np.mean(individual_errors), 2)

        # ✅ Category-wise Confidence
        category_wise_data = defaultdict(list)
        location_wise_data = defaultdict(list)
        trend_count = {"increasing": 0, "decreasing": 0}

        for entry, err in zip(filtered_data, individual_errors):
            category_wise_data[entry["category"]].append(err)
            location_wise_data[entry["location"]].append(err)
            if "Trend" in entry:
                if "🟢" in entry["Trend"]:
                    trend_count["increasing"] += 1
                elif "🔴" in entry["Trend"]:
                    trend_count["decreasing"] += 1

        # ✅ Compute average errors per category and location
        category_confidence = {
            cat: round((1 - np.mean(errs) / 100), 2)
            for cat, errs in category_wise_data.items()
        }
        location_confidence = {
            loc: round((1 - np.mean(errs) / 100), 2)
            for loc, errs in location_wise_data.items()
        }

        # ✅ Final Response
        confidence_data = {
            "confidence": round(float(confidence), 2),
            "mape": round(float(mape) * 100, 2),
            "total_actual_sales": int(total_actual_sales),
            "total_predicted_sales": int(total_predicted_sales),
            "error_per_prediction": individual_errors,
            "data_points_used": len(filtered_data),
            "highest_error": highest_error,
            "lowest_error": lowest_error,
            "average_error": avg_error,
            "category_wise_confidence": category_confidence,
            "location_wise_confidence": location_confidence,
            "trend_summary": trend_count
        }

        # Store in session
        request.session["confidence_data"] = confidence_data

        return JsonResponse(confidence_data)

    except Exception as e:
        return JsonResponse({"error": f"An error occurred: {str(e)}"}, status=500)
    
    

def ai_inventory_optimization(request):
    """AI-powered Inventory Optimization View"""
    
    # Retrieve session data
    filtered_data = request.session.get("debug_filtered_data", [])
    
    if not filtered_data:
        return JsonResponse({"error": "No data available for inventory analysis."}, status=400)

    # 📊 Process Data: Identify Fast-Moving and Slow-Moving Items
    fast_moving = []
    slow_moving = []
    
    for entry in filtered_data:
        item = entry.get("item")
        location = entry.get("location")
        predicted_sales = entry.get("Predicted_Sales")
        trend = entry.get("Trend")
        total_sales = entry.get("total_sales_value")

        if trend == "🟢 Increasing":
            fast_moving.append({
                "item": item,
                "location": location,
                "predicted_sales": predicted_sales,
                "trend": trend
            })
        else:
            slow_moving.append({
                "item": item,
                "location": location,
                "total_sales": total_sales,
                "predicted_sales": predicted_sales,
                "trend": trend
            })
    
    # 🔍 AI-generated recommendations
    recommendations = {
        "fast_moving": fast_moving,
        "slow_moving": slow_moving
    }

    return JsonResponse(recommendations, safe=False)




#################################################################################################








































































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

        # List_of_required_columns_in_the_uploaded_CSV
        required_columns = [
            'Item_Identifier', 'Item_Weight', 'Item_Visibility',
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
            data['Item_Type'].replace(-1, 'Unknown', inplace=True)
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
            'Item_Identifier', 'Item_Weight',  'Item_Visibility',
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
    User = get_user_model()  # Get the CustomUser model
    if request.method == 'POST':
        if 'reset_user_id' in request.session:
            # We're at the password reset stage
            form = ResetPasswordForm(request.POST)  # Use the same form for resetting the password
            if form.is_valid():
                new_password1 = form.cleaned_data['new_password1']
                new_password2 = form.cleaned_data['new_password2']
                
                # Ensure passwords match
                if new_password1 != new_password2:
                    messages.error(request, "Passwords do not match.")
                else:
                    user_id = request.session['reset_user_id']
                    try:
                        user = User.objects.get(id=user_id)
                        user.set_password(new_password1)  # Set the new password
                        user.save()
                        del request.session['reset_user_id']  # Clear the session
                        messages.success(request, "Your password has been reset successfully.")
                    except User.DoesNotExist:
                        messages.error(request, "User does not exist.")
                        del request.session['reset_user_id']  # Clear the session if something goes wrong

        else:
            # We're at the user verification stage
            form = ResetPasswordForm(request.POST)  # Use the form to validate user information
            if form.is_valid():
                username = form.cleaned_data['username']
                email = form.cleaned_data['email']
                phone = form.cleaned_data['phone']

                # Check if the user exists with the provided username, email, and phone
                try:
                    user = User.objects.get(username=username, email=email, phone=phone)
                    # Once verified, save user id in session and prepare for password reset
                    request.session['reset_user_id'] = user.id  # Store user id in session
                    messages.info(request, "User verified. Please enter a new password.")
                except User.DoesNotExist:
                    messages.error(request, "User with the provided details does not exist.")
    else:
        form = ResetPasswordForm()

    return render(request, 'forgot_password.html', {'form': form})



def blog_list(request):
    """View to display all published blog posts."""
    posts = BlogPost.objects.filter(status='published').order_by('-date_posted')
    return render(request, 'blog_list.html', {'posts': posts})

def blog_detail(request, post_id):
    """View to display the full content of a single blog post (fetches via AJAX or display on same page)."""
    post = get_object_or_404(BlogPost, id=post_id)
    return render(request, 'blog_detail.html', {'post': post})