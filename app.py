from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

model = joblib.load('career_model.pkl')
scaler = joblib.load('scaler.pkl')

FEATURES = ['math_score', 'physics_score', 'chemistry_score', 'biology_score',
            'programming_interest', 'electronics_interest', 'mechanical_interest',
            'medical_interest', 'creativity_score', 'communication_score']

@app.route('/')
def home():
    return jsonify({
        "message": "Career Prediction API",
        "status": "active",
        "model_accuracy": "58.82%"
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if not all(f in data for f in FEATURES):
            missing = [f for f in FEATURES if f not in data]
            return jsonify({"error": f"Missing: {missing}"}), 400
        
        features = np.array([[data[f] for f in FEATURES]])
        features_scaled = scaler.transform(features)
        
        probs = model.predict_proba(features_scaled)[0]
        top3_idx = np.argsort(probs)[-3:][::-1]
        
        recommendations = []
        for idx in top3_idx:
            recommendations.append({
                "department": model.classes_[idx],
                "confidence": float(probs[idx]),
                "confidence_percent": f"{probs[idx]*100:.1f}%"
            })
        
        return jsonify({
            "success": True,
            "recommendations": recommendations
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
