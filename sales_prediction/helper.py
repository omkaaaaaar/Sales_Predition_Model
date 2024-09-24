from faker import Faker
import pandas as pd
import random

fake = Faker()

# Define categories and outlet types
categories = ['Electronics', 'Clothing', 'Food', 'Appliances']
outlet_types = ['Supermarket', 'Electronics Store', 'Fashion Store', 'Online']

# Define default values for missing columns
item_weights = {
    'Food': random.uniform(0.5, 5.0),  # weight in kg for food items
    'Electronics': random.uniform(0.1, 10.0),  # weight in kg for electronics
    'Clothing': random.uniform(0.1, 2.0),  # weight in kg for clothing
    'Appliances': random.uniform(2.0, 15.0)  # weight in kg for appliances
}

# Define fat content for food items
fat_contents = ['Low Fat', 'Regular']

# Define outlet sizes
outlet_sizes = ['Small', 'Medium', 'Large']

# Define outlet locations
outlet_locations = ['Urban', 'Rural']

data = []
for _ in range(1000):
    item_type = random.choice(categories)
    data.append({
        'Item_Identifier': fake.uuid4(),
        'Item_Type': item_type,
        'Item_Weight': round(item_weights[item_type], 2),
        'Item_Fat_Content': random.choice(fat_contents) if item_type == 'Food' else 'Other',
        'Item_Visibility': round(random.uniform(0.0, 1.0), 6),  # visibility score between 0 and 1
        'Item_MRP': round(random.uniform(50, 500), 2),
        'Sales': random.randint(100, 10000),
        'Outlet_Identifier': fake.uuid4(),
        'Outlet_Type': random.choice(outlet_types),
        'Outlet_Establishment_Year': random.randint(1990, 2020),
        'Outlet_Size': random.choice(outlet_sizes),
        'Outlet_Location_Type': random.choice(outlet_locations),
    })

# Create a DataFrame and save to CSV
df = pd.DataFrame(data)
df.to_csv('ecommerce_data.csv', index=False)
