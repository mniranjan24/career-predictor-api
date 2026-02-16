from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import json
import os

app = Flask(__name__)
CORS(app, origins="*")  # Allow all origins for Flutter

# Load model files
print("🔄 Loading model files...")
try:
    model = joblib.load('career_model_optimized.pkl')
    scaler = joblib.load('scaler_optimized.pkl')
    with open('features.json', 'r') as f:
        all_features = json.load(f)
    print(f"✅ Model loaded: Extra Trees")
    print(f"✅ Features: {len(all_features)}")
    print(f"✅ Departments: {list(model.classes_)}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

@app.route('/')
def home():
    return jsonify({
        "message": "Career Prediction API",
        "status": "active",
        "model": "Extra Trees Optimized",
        "accuracy": "93.64%",
        "departments": model.classes_.tolist(),
        "total_features": len(all_features)
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    # Support both GET (for testing) and POST (for Flutter)
    if request.method == 'GET':
        data = request.args.to_dict()
        # Convert to float
        for key in data:
            try:
                data[key] = float(data[key])
            except:
                pass
    else:
        data = request.get_json() or {}
    
    try:
        # Required fields
        required = ['math_score', 'physics_score', 'chemistry_score', 'biology_score',
                   'programming_interest', 'electronics_interest', 'mechanical_interest',
                   'medical_interest', 'creativity_score', 'communication_score']
        
        # Check missing
        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({
                "success": False,
                "error": f"Missing fields: {missing}",
                "required_fields": required
            }), 400
        
        # Build features
        base = {f: float(data[f]) for f in required}
        
        # Engineered features
        features = base.copy()
        features['science_avg'] = (base['physics_score'] + base['chemistry_score'] + base['biology_score']) / 3
        features['stem_score'] = (base['math_score'] + base['physics_score']) / 2
        features['bio_chem_score'] = (base['biology_score'] + base['chemistry_score']) / 2
        features['tech_interest'] = (base['programming_interest'] + base['electronics_interest']) / 2
        features['engineering_interest'] = (base['electronics_interest'] + base['mechanical_interest']) / 2
        features['medical_affinity'] = (base['medical_interest'] + features['bio_chem_score']) / 2
        features['math_bio_gap'] = base['math_score'] - base['biology_score']
        features['tech_medical_gap'] = features['tech_interest'] - base['medical_interest']
        features['science_comm_gap'] = features['science_avg'] - base['communication_score']
        features['soft_skills'] = (base['creativity_score'] + base['communication_score']) / 2
        features['is_tech'] = int((base['programming_interest'] >= 8) and (base['math_score'] >= 85))
        features['is_medical'] = int((base['medical_interest'] >= 8) and (base['biology_score'] >= 85))
        features['is_business'] = int((base['communication_score'] >= 8) and (base['creativity_score'] >= 7))
        features['is_engineering'] = int((base['physics_score'] >= 85) and (base['mechanical_interest'] >= 7))
        
        # Predict
        X = np.array([[features[f] for f in all_features]])
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]
        probs = model.predict_proba(X_scaled)[0]
        
        # Top 3 recommendations
        top3_idx = np.argsort(probs)[-3:][::-1]
        recommendations = [{
            "department": model.classes_[i],
            "confidence": round(float(probs[i]), 4),
            "confidence_percent": f"{probs[i]*100:.1f}%"
        } for i in top3_idx]
        
        return jsonify({
            "success": True,
            "prediction": prediction,
            "confidence_percent": recommendations[0]['confidence_percent'],
            "recommendations": recommendations
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)