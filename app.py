import pickle
import numpy as np
from flask import Flask, request, jsonify

# Load the trained model
try:
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    # ✅ Ensure the loaded object has the `predict` method
    if not hasattr(model, "predict"):
        raise ValueError("Loaded object is not a valid model. Ensure 'model.pkl' contains a trained classifier.")

    print("✅ Model loaded successfully!")

except FileNotFoundError:
    print("❌ Error: 'model.pkl' file not found. Train a model first.")
    model = None

except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ✅ Validate input JSON
        data = request.get_json()
        if not data or "features" not in data:
            return jsonify({"error": "Missing 'features' key in request"}), 400

        # ✅ Ensure features is a list before converting to numpy ar ray
        features = data["features"]
        if not isinstance(features, list):
            return jsonify({"error": "'features' must be a list of numbers"}), 400

        # ✅ Convert input to a NumPy array and ensure correct shape
        features = np.array(features, dtype=float).reshape(1, -1)

        # ✅ Validate feature count
        if features.shape[1] != 25:
            return jsonify({"error": f"Incorrect number of features. Expected {25}, but got {features.shape[1]}"}), 400

        # ✅ Perform prediction using the trained model
        prediction = model.predict(features)

        return jsonify({"prediction": prediction.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
