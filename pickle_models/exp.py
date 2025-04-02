import joblib

# Load the model correctly
model = joblib.load("pickle_models\sales_prediction_model_new.pkl")

# Verify model type
print(type(model)) 
