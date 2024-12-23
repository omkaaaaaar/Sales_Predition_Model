import os
import pickle
from sklearn.preprocessing import LabelEncoder 

def load_or_create_pickle(file_path, default_data):
    if os.path.exists(file_path):
        try:
            with open(file_path, 'rb') as file:
                return pickle.load(file)
        except (pickle.UnpicklingError, EOFError):
            print(f"Corrupted pickle file. Recreating: {file_path}")
    # Save default data to file
    with open(file_path, 'wb') as file:
        pickle.dump(default_data, file)
    return default_data

# Example usage
default_encoders = {
    "Item_Type": LabelEncoder().fit(["Dairy", "Snacks","Appliances","Clothing","Books","Electronics"]),
    "Outlet_Type": LabelEncoder().fit(["Supermarket", "Grocery Store"])
}
encoders = load_or_create_pickle('pickle_models/label_encoders.pkl', default_encoders)
