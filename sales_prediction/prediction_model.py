# prediction_model.py
import pandas as pd
import joblib
import numpy as np
import os 

# Get the absolute path of the pickle file
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Get the base directory of your Django project
model_path = os.path.join(base_dir, 'sales_prediction', 'pickle_models', 'sales_prediction_model.pkl')
label_encoders_path = os.path.join(base_dir, 'sales_prediction', 'pickle_models', 'label_encoders.pkl')

# Load the trained model and label encoders
model = joblib.load(model_path)
label_encoders = joblib.load(label_encoders_path)

def predict_sales(data):
    # Ensure necessary columns are present
    required_columns = [
        'Item_Identifier', 'Item_Weight', 'Item_Fat_Content', 'Item_Visibility',
        'Item_Type', 'Item_MRP', 'Outlet_Identifier', 'Outlet_Establishment_Year',
        'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'
    ]

    # Handle unseen labels for categorical columns during prediction
    for column in ['Item_Identifier', 'Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']:
        le = label_encoders[column]
        
        # Ensure the column is a pandas Series before applying the transformation
        if isinstance(data[column], list):
            data[column] = pd.Series(data[column])

        # Replace unseen labels with a default (e.g., the most common category or 'Other')
        data[column] = data[column].apply(lambda x: x if x in le.classes_ else 'Other')
        
        # Fit the label encoder to include 'Other' if needed
        if 'Other' not in le.classes_:
            le.classes_ = np.append(le.classes_, 'Other')  # Append 'Other' to the label encoder classes

        data[column] = le.transform(data[column])

    # Prepare the input data for prediction
    X = data[['Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type', 'Item_MRP', 'Outlet_Identifier', 'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']]

    # Predict using the loaded model
    predictions = model.predict(X)
    
    return predictions
