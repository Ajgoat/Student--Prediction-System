from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS
from flask import render_template

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

@app.route('/ui')
def ui():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received JSON:", data)
        # Expected features from Student_Performance.csv
        required_features = [
            'Hours Studied',
            'Previous Scores',
            'Sleep Hours',
            'Sample Question Papers Practiced'
        ]

        # Validate input
        missing_features = [f for f in required_features if f not in data]
        if missing_features:
            return jsonify({"error": f"Missing required features: {', '.join(missing_features)}"}), 400

        # Create feature array
        features = np.array([[data['Hours Studied'], 
                              data['Previous Scores'], 
                              data['Sleep Hours'], 
                              data['Sample Question Papers Practiced']]])

        # Calculate derived features
        sleep_hours = features[0][2] or 1e-5
        study_hours = features[0][0] or 1e-5
        study_efficiency = features[0][0] / sleep_hours
        practice_intensity = features[0][3] / study_hours

        # Append derived features
        features = np.append(features, [[study_efficiency, practice_intensity]], axis=1)

        # Scale features
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)[0]

        return jsonify({"Performance Index": round(float(prediction), 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
