import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import joblib
import os

# Function to load CSV and preprocess
def preprocess_and_train_model(csv_file_path):
    try:
        # Load the dataset
        data = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Broadly expected columns that are relevant for e-commerce
    general_columns = {
        'Item_MRP': 0.0,                   # Price of the item
        'Outlet_Identifier': 'Other',       # Store Identifier
        'Outlet_Establishment_Year': 0,     # Store establishment year
        'Outlet_Size': 'Other',             # Store size (Small/Medium/Large)
        'Outlet_Location_Type': 'Other',    # Location type (Urban/Rural)
        'Outlet_Type': 'Other',             # Type of outlet (Supermarket, Electronics Store, etc.)
        'Sales': 0.0                        # Target column
    }

    # Additional domain-specific columns (e.g., for groceries or electronics)
    domain_specific_columns = {
        'Item_Weight': 0.0,                 # For groceries/electronics
        'Item_Fat_Content': 'Other',        # Relevant only for groceries
        'Item_Visibility': 0.0,             # Could be interpreted as availability or shelf space
        'Item_Type': 'Other'                # Type of product (could be used for general categorization)
    }

    # Merge general columns and domain-specific columns into one dictionary
    expected_columns = {**general_columns, **domain_specific_columns}

    # Ensure all expected columns are present in the dataset
    for column, default_value in expected_columns.items():
        if column not in data.columns:
            data[column] = default_value  # Fill missing columns with default values

    # Separate the features and target variable
    if 'Sales' in data.columns:
        X = data.drop(columns=['Sales'])
        y = data['Sales']
    else:
        X = data
        y = None

    # Dynamically encode categorical columns
    label_encoders = {}

    # Identify categorical columns dynamically (all object type columns)
    categorical_columns = X.select_dtypes(include=['object']).columns

    for column in categorical_columns:
        le = LabelEncoder()
        X[column] = X[column].apply(lambda x: x if x in le.classes_ else 'Other')
        
        # Fit the LabelEncoder and include 'Other'
        le.fit(list(X[column]) + ['Other'])
        X[column] = le.transform(X[column])
        label_encoders[column] = le  # Save the label encoder for future use

    # Handle numeric features and NaN values
    X = X.fillna(0)  # Fill NaNs with 0 for numeric columns

    # Train-test split (if Sales column is present)
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the XGBoost model
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=6)
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")

        # Save the model and label encoders
        os.makedirs('pickle_models', exist_ok=True)
        joblib.dump(model, 'pickle_models/sales_prediction_model.pkl')
        joblib.dump(label_encoders, 'pickle_models/label_encoders.pkl')

        print("Model and label encoders saved successfully.")
    else:
        print("No target ('Sales') column found in the dataset. Only preprocessing completed.")

# Example usage: dynamically upload and preprocess any CSV file
csv_file = 'path_to_uploaded_sales_csv.csv'  # Replace with the path to the uploaded CSV file
preprocess_and_train_model(csv_file)
