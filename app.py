from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

     # Load the model and scaler
with open("model.pkl", "rb") as f:
         model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
         scaler = pickle.load(f)

@app.route('/')
def home():
     return jsonify({"message": "Welcome to the Student Performance Prediction API"})

@app.route('/predict', methods=['POST'])
def predict():
     try:
             # Get JSON data from request
             data = request.get_json()

             # Expected features from Student_Performance.csv
             required_features = ['Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced']
             if not all(feature in data for feature in required_features):
                    return jsonify({"error": "Missing required features: Hours Studied, Previous Scores, Sleep Hours, Sample Question Papers Practiced"}), 400

             # Create feature array
             features = np.array([[data['Hours Studied'], 
                                 data['Previous Scores'], 
                                 data['Sleep Hours'], 
                                 data['Sample Question Papers Practiced']]])
            
             # Calculate derived features (as in the notebook)
             study_efficiency = features[0][0] / (features[0][2] + 1e-5)
             practice_intensity = features[0][3] / (features[0][0] + 1e-5)
             features = np.append(features, [[study_efficiency, practice_intensity]], axis=1)

             # Scale features
             features_scaled = scaler.transform(features)

             # Make prediction
             prediction = model.predict(features_scaled)[0]

             return jsonify({"Performance Index": round(float(prediction), 2)})

     except Exception as e:
           return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
         app.run(debug=True)