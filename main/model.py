import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
df = pd.read_csv('fertilizer_dataset.csv')

# Encode categorical features
df_encoded = pd.get_dummies(df, columns=['Soil Type', 'Crop Type', 'Weather Condition'])

# Separate features and target variables
X = df_encoded.drop(columns=['Fertilizer', 'Ratio'])
y_fertilizer = df_encoded['Fertilizer']
y_ratio = df_encoded['Ratio']

# Split the data
X_train, X_test, y_train_fertilizer, y_test_fertilizer = train_test_split(X, y_fertilizer, test_size=0.2, random_state=42)
_, _, y_train_ratio, y_test_ratio = train_test_split(X, y_ratio, test_size=0.2, random_state=42)

# Train the model
model_fertilizer = RandomForestClassifier(n_estimators=100, random_state=42)
model_fertilizer.fit(X_train, y_train_fertilizer)

model_ratio = RandomForestClassifier(n_estimators=100, random_state=42)
model_ratio.fit(X_train, y_train_ratio)

# Save the models and column names
joblib.dump(model_fertilizer, 'fertilizer_model.pkl')
joblib.dump(model_ratio, 'ratio_model.pkl')
joblib.dump(X.columns, 'X_columns.pkl')
