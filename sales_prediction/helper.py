from faker import Faker
import pandas as pd
import random

# Initialize the Faker library
fake = Faker()

# Reset Faker's unique generation to avoid running out of unique values
fake.unique.clear()

# Define categories for the Item_Type based on the provided labels
categories = [
    'Fruits and Vegetables', 'Snack Foods', 'Household', 'Frozen Foods', 'Dairy', 'Canned', 'Baking Goods',
    'Health and Hygiene', 'Soft Drinks', 'Meat', 'Breads', 'Hard Drinks', 'Others', 'Electronics', 'Starchy Foods',
    'Appliances', 'Clothing', 'Books', 'Breakfast', 'Seafood'
]

# Define outlet types
outlet_types = ['Supermarket Type1', 'Supermarket Type2', 'Fashion Store', 'Electronics Store']

# Define possible values for outlet size and location type
outlet_sizes = ['Small', 'Medium', 'Large']
outlet_locations = ['Tier 1', 'Tier 2', 'Tier 3']

# Define weights for different item types
item_weights = {
    'Fruits and Vegetables': lambda: round(random.uniform(0.1, 5.0), 2),
    'Snack Foods': lambda: round(random.uniform(0.2, 2.5), 2),
    'Household': lambda: round(random.uniform(0.5, 10.0), 2),
    'Frozen Foods': lambda: round(random.uniform(0.2, 8.0), 2),
    'Dairy': lambda: round(random.uniform(0.2, 4.0), 2),
    'Canned': lambda: round(random.uniform(0.3, 6.0), 2),
    'Baking Goods': lambda: round(random.uniform(0.5, 7.0), 2),
    'Health and Hygiene': lambda: round(random.uniform(0.1, 3.0), 2),
    'Soft Drinks': lambda: round(random.uniform(0.5, 3.0), 2),
    'Meat': lambda: round(random.uniform(0.5, 10.0), 2),
    'Breads': lambda: round(random.uniform(0.2, 1.5), 2),
    'Hard Drinks': lambda: round(random.uniform(0.5, 5.0), 2),
    'Others': lambda: round(random.uniform(0.1, 2.0), 2),
    'Electronics': lambda: round(random.uniform(0.5, 15.0), 2),
    'Starchy Foods': lambda: round(random.uniform(0.3, 3.0), 2),
    'Appliances': lambda: round(random.uniform(1.0, 20.0), 2),
    'Clothing': lambda: round(random.uniform(0.2, 2.0), 2),
    'Books': lambda: round(random.uniform(0.1, 1.0), 2),
    'Breakfast': lambda: round(random.uniform(0.2, 2.0), 2),
    'Seafood': lambda: round(random.uniform(0.5, 10.0), 2)
}

# Generate synthetic data
data = []
for _ in range(2000):  # Generate more data to reflect the variety of Item_Types
    item_type = random.choice(categories)
    data.append({
        'Item_Identifier': fake.unique.bothify(text='???##'),  # Random unique identifier
        'Item_Weight': item_weights[item_type](),
        'Item_Fat_Content': 'Other',  # Not applicable to non-food items
        'Item_Visibility': round(random.uniform(0.0, 0.2), 6),  # Random visibility value
        'Item_Type': item_type,
        'Item_MRP': round(random.uniform(50, 3000), 2),  # Random MRP between 50 and 3000
        'Outlet_Identifier': fake.unique.bothify(text='OUT####'),  # Updated pattern for more unique values
        'Outlet_Establishment_Year': random.randint(1980, 2022),  # Random establishment year
        'Outlet_Size': random.choice(outlet_sizes),
        'Outlet_Location_Type': random.choice(outlet_locations),
        'Outlet_Type': random.choice(outlet_types),
        'Sales': round(random.uniform(100, 20000), 2)  # Random sales between 100 and 20000
    })

# Create a DataFrame and save to CSV
df = pd.DataFrame(data)
df.to_csv('ecommerce_data_expanded.csv', index=False)

print("CSV file 'ecommerce_data_expanded.csv' has been created successfully.")
