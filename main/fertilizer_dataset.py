import pandas as pd
import random

# Define possible values for each feature
soil_types = ['Sandy', 'Clay', 'Loam']
crop_types = ['Wheat', 'Rice', 'Maize']
weather_conditions = ['Sunny', 'Rainy', 'Cloudy']
fertilizers = ['NPK', 'DAP', 'Urea']

# Generate mock data
data = []
for _ in range(1000):
    soil = random.choice(soil_types)
    crop = random.choice(crop_types)
    weather = random.choice(weather_conditions)
    # Randomly assign a fertilizer type and ratio based on features
    if soil == 'Sandy' and weather == 'Sunny':
        fertilizer = 'NPK'
        ratio = '10-26-26'
    elif soil == 'Clay' and crop == 'Rice':
        fertilizer = 'DAP'
        ratio = '18-46-0'
    elif weather == 'Rainy':
        fertilizer = 'Urea'
        ratio = '46-0-0'
    else:
        fertilizer = random.choice(fertilizers)
        ratio = random.choice(['12-32-16', '10-26-26', '20-20-20'])
    data.append([soil, crop, weather, fertilizer, ratio])

# Create a DataFrame
df = pd.DataFrame(data, columns=['Soil Type', 'Crop Type', 'Weather Condition', 'Fertilizer', 'Ratio'])

# Save the dataset to a CSV file
df.to_csv('fertilizer_dataset.csv', index=False)
