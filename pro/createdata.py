import pandas as pd
import numpy as np

# Define the number of rows in the dataset
num_rows = 10000

# Define lists of possible values for each field
vegetables = ['Bitter gourd', 'Brinjal', 'Cabbage', 'Cauliflower', 'Chilly', 'Cucumber', 'Garlic', 'Ginger', 'Okra', 'Onion', 'Peas', 'Pointed grourd', 'Potato', 'Pumpkin', 'Radish', 'Tomato']
seasons = ['Autumn', 'Monsoon', 'Spring', 'Summer', 'Winter']
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
states = ['Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal', 'Andaman and Nicobar Islands', 'Chandigarh', 'Dadra and Nagar Haveli', 'Daman and Diu', 'Lakshadweep', 'Delhi', 'Puducherry']
prices_per_kg = np.random.uniform(20, 200, num_rows)  # Random prices between 20 and 200 per kg
temps = np.random.uniform(10, 40, num_rows)  # Random temperatures between 10 and 40
disasters = np.random.randint(0, 2, num_rows)  # Random values 0 or 1 for disasters

# Create the DataFrame
data = {
    'Vegetable': np.random.choice(vegetables, num_rows),
    'Season': np.random.choice(seasons, num_rows),
    'Month': np.random.choice(months, num_rows),
    'Temp': temps,
    'Deasaster Happen in last 3month': disasters,
    'State': np.random.choice(states, num_rows),
    'Price per kg': prices_per_kg
}

df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('vegetable_data_big.csv', index=False)
