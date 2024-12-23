def retrain_model(csv_file_path):
    import pandas as pd
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import mean_squared_error
    import joblib
    import os

    # Load the dataset
    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Display the most common labels in the 'Item_Type' column
    common_labels = df['Item_Type'].value_counts()
    print("Most common labels in the 'Item_Type' column:")
    print(common_labels)

    # Select features and target column
    if 'Sales' in df.columns:
        X = df.drop(columns=['Sales'])
        y = df['Sales']
    else:
        print("Sales column not found in the dataset.")
        return

    # Handle missing data and preprocess
    X = X.fillna(0)
    categorical_columns = X.select_dtypes(include=['object']).columns

    # Load existing label encoders or create new ones
    label_encoders_path = 'pickle_models/label_encoders.pkl'
    if os.path.exists(label_encoders_path):
        label_encoders = joblib.load(label_encoders_path)
        print("Loaded existing label encoders.")
    else:
        label_encoders = {}

    for column in categorical_columns:
        if column not in label_encoders:
            label_encoders[column] = LabelEncoder()
            label_encoders[column].fit(list(X[column].dropna().unique()) + ['Other'])

        X[column] = X[column].apply(lambda x: x if x in label_encoders[column].classes_ else 'Other')
        X[column] = label_encoders[column].transform(X[column])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load existing model or train a new one
    model_path = 'pickle_models/sales_prediction_model.pkl'
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("Loaded existing model for retraining.")
    else:
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=6)
        print("Training a new model.")

    # Retrain the model
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error after retraining: {mse}")

    # Save the updated model and encoders
    os.makedirs('pickle_models', exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(label_encoders, label_encoders_path)
    print("Model and label encoders updated successfully.")

# Example usage
csv_file = 'sales_prediction\ecommerce_data.csv'  # Path to the CSV file
retrain_model(csv_file)
