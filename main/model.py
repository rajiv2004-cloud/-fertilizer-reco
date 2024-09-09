import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
df = pd.read_csv('fertilizer_dataset.csv')

# Convert categorical variables to numeric
df_encoded = pd.get_dummies(df, columns=['Soil Type', 'Crop Type', 'Weather Condition'])

# Split the data into features and labels
X = df_encoded.drop(columns=['Fertilizer', 'Ratio'])
y = df_encoded['Fertilizer']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model and feature columns
joblib.dump(model, 'fertilizer_model.pkl')
joblib.dump(X.columns, 'X_columns.pkl')