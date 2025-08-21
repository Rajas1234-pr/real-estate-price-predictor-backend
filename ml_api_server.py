from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for Next.js frontend

# Global variables for model and scaler
model = None
scaler = None
model_loaded = False

# Load your trained model and scaler
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'model.pkl')
    scaler_path = os.path.join(script_dir, 'scaler.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    model_loaded = True
    print("✅ Model and scaler loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model/scaler: {e}")
    print("⚠️ Running with fallback predictions only")

# --- categorical encodings (must match training notebook) ---
status_map = {"Ready to move": 2, "Under Construction": 1, "Resale": 0}
furnished_map = {"Furnished": 2, "Semi-Furnished": 1, "Unfurnished": 0}
building_map = {"Apartment": 0, "Independent House": 1, "Villa": 2}

def fallback_prediction(data):
    base_price = 50000
    area = data.get("area", 500)
    bedrooms = data.get("bedrooms", 2)
    bathrooms = data.get("bathrooms", 1)
    balcony = data.get("balcony", 0)
    parking = data.get("parking", 0)

    price = area * base_price
    price += bedrooms * 500000
    price += bathrooms * 300000
    price += balcony * 200000
    price += parking * 400000

    lat = data.get("latitude", 19.0760)
    lng = data.get("longitude", 72.8777)

    if 18.9 <= lat <= 19.3 and 72.7 <= lng <= 73.0:
        price *= 1.2

    return price

@app.route("/predict", methods=["POST"])
def predict_price():
    try:
        data = request.json

        if not model_loaded or model is None or scaler is None:
            prediction = fallback_prediction(data)
            return jsonify({
                "price": float(prediction),
                "status": "success",
                "note": "Fallback prediction (model not loaded)"
            })

        # --- apply encoding ---
        status = status_map.get(data["status"], 0)
        furnished = furnished_map.get(data["furnished_status"], 0)
        building = building_map.get(data["type_of_building"], 0)

        # --- feature order must match training ---
        features = np.array([[
            data["area"],
            data["latitude"],
            data["longitude"],
            data["bedrooms"],
            data["bathrooms"],
            data["balcony"],
            status,
            data["parking"],
            furnished,
            building
        ]], dtype=float)

        # --- scale + predict ---
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]

        return jsonify({
            "price": float(prediction),
            "status": "success"
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        try:
            prediction = fallback_prediction(data)
            return jsonify({
                "price": float(prediction),
                "status": "success",
                "note": f"Fallback used due to error: {str(e)}"
            })
        except Exception as fallback_error:
            return jsonify({
                "status": "error",
                "error": f"Both model and fallback failed: {fallback_error}"
            }), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model_loaded
    })

# ✅ Added root route for Render health check
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "message": "Backend running!"
    })

if __name__ == "__main__":
    # ✅ Use dynamic port for Render
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Starting ML API server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
