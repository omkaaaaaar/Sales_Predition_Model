def clean_data(data):
    # Fill missing values (e.g., if 'Item_Weight' has missing values)
    data['Item_Weight'].fillna(data['Item_Weight'].mean(), inplace=True)
    
    # Handle categorical variables (e.g., one-hot encoding 'Item_Fat_Content')
    data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'})

    # Replace unseen labels in 'Item_Type' with 'Other'
    known_item_types = ['Fruits and Vegetables', 'Frozen Foods', 'Dairy', 'Snack Foods', 'Meat', 'Breakfast', 'Baking Goods', 'Health and Hygiene']
    data['Item_Type'] = data['Item_Type'].apply(lambda x: x if x in known_item_types else 'Other')

    return data  # Ensure the return type is a DataFrame
