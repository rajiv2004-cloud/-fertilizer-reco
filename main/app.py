from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('fertilizer_model.pkl')

# Load the feature columns used during training
X_columns = joblib.load('X_columns.pkl')

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    soil_type = request.form['soil_type']
    crop_type = request.form['crop_type']
    weather_condition = request.form['weather_condition']
    
    # Convert input to DataFrame and one-hot encode
    input_df = pd.DataFrame([[soil_type, crop_type, weather_condition]], 
                            columns=['Soil Type', 'Crop Type', 'Weather Condition'])
    input_encoded = pd.get_dummies(input_df).reindex(columns=X_columns, fill_value=0)
    
    # Make prediction
    prediction = model.predict(input_encoded)
    
    return jsonify({'Recommended Fertilizer': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
