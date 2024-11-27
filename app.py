from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

# Load your trained model
model_path = os.path.join(os.path.dirname(__file__), 'rf_h2h_model_updated.pkl')
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, this is the Qualia Bets Prediction Service!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Extracting features from the input data
        fighter1_features = [
            data['f1_avg_fight_duration'],
            data['f1_knockdown_rate'],
            data['f1_takedown_success_rate'],
            data['f1_strike_defense'],
            data['f1_striking_balance'],
            data['f1_finish_rate']
        ]

        fighter2_features = [
            data['f2_avg_fight_duration'],
            data['f2_knockdown_rate'],
            data['f2_takedown_success_rate'],
            data['f2_strike_defense'],
            data['f2_striking_balance'],
            data['f2_finish_rate']
        ]

        # Combine features into a single input for the model
        feature_array = np.array([fighter1_features + fighter2_features])

        # Make a prediction
        prediction = model.predict(feature_array)

        # Return the prediction result
        return jsonify({
            'predicted_winner': 'Fighter 1' if prediction[0] == 1 else 'Fighter 2'
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

