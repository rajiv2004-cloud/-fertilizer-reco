from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the models and column names
model_fertilizer = joblib.load('fertilizer_model.pkl')
model_ratio = joblib.load('ratio_model.pkl')
X_columns = joblib.load('X_columns.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    soil_type = request.form['soil_type']
    crop_type = request.form['crop_type']
    weather_condition = request.form['weather_condition']
    
    # Create a DataFrame for the input
    input_df = pd.DataFrame([[soil_type, crop_type, weather_condition]], 
                            columns=['Soil Type', 'Crop Type', 'Weather Condition'])
    
    # One-hot encode the input
    input_encoded = pd.get_dummies(input_df).reindex(columns=X_columns, fill_value=0)
    
    # Predict the fertilizer and ratio
    predicted_fertilizer = model_fertilizer.predict(input_encoded)[0]
    predicted_ratio = model_ratio.predict(input_encoded)[0]
    
    # Return the result as JSON
    return jsonify({
        'Recommended Fertilizer': predicted_fertilizer,
        'Recommended Ratio': predicted_ratio
    })

if __name__ == '__main__':
    app.run(debug=True)
