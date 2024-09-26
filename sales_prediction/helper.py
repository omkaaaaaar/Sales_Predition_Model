from faker import Faker
import pandas as pd
import random

# Initialize the Faker library
fake = Faker()

# Define categories for the Item_Type
categories = ['Clothing', 'Electronics', 'Appliances', 'Books']

# Define outlet types
outlet_types = ['Supermarket Type1', 'Supermarket Type2', 'Fashion Store', 'Electronics Store']

# Define possible values for outlet size and location type
outlet_sizes = ['Small', 'Medium', 'Large']
outlet_locations = ['Tier 1', 'Tier 2', 'Tier 3']

# Define weights for different item types
item_weights = {
    'Clothing': lambda: round(random.uniform(0.2, 2.0), 2),
    'Electronics': lambda: round(random.uniform(0.5, 15.0), 2),
    'Appliances': lambda: round(random.uniform(1.0, 20.0), 2),
    'Books': lambda: round(random.uniform(0.1, 1.0), 2)
}

# Generate synthetic data
data = []
for _ in range(1000):
    item_type = random.choice(categories)
    data.append({
        'Item_Identifier': fake.unique.bothify(text='???##'),  # Random unique identifier
        'Item_Weight': item_weights[item_type](),
        'Item_Fat_Content': 'Other',  # Not applicable to non-food items
        'Item_Visibility': round(random.uniform(0.0, 0.2), 6),  # Random visibility value
        'Item_Type': item_type,
        'Item_MRP': round(random.uniform(100, 2000), 2),  # Random MRP between 100 and 2000
        'Outlet_Identifier': fake.unique.bothify(text='OUT###'),  # Outlet ID
        'Outlet_Establishment_Year': random.randint(1990, 2022),  # Random establishment year
        'Outlet_Size': random.choice(outlet_sizes),
        'Outlet_Location_Type': random.choice(outlet_locations),
        'Outlet_Type': random.choice(outlet_types),
        'Sales': round(random.uniform(500, 15000), 2)  # Random sales between 500 and 15000
    })

# Create a DataFrame and save to CSV
df = pd.DataFrame(data)
df.to_csv('ecommerce_data.csv', index=False)

print("CSV file 'ecommerce_data.csv' has been created successfully.")
