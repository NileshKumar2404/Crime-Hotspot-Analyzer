import pickle
import numpy as np
from flask import Flask, request, jsonify
from pymongo import MongoClient
from datetime import datetime, UTC


# Load the trained model
try:
    with open("./model1.pkl", "rb") as model_file:
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

# MongoDB setup
try:
    client = MongoClient("mongodb://localhost:27017/")
    db = client["crime_prediction_db"]
    predictions_collection = db["predictions"]
    print("✅ MongoDB connection successful!")
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
    predictions_collection = None


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print("🔹 Incoming data:", data)

        if not data or "features" not in data:
            return jsonify({"error": "Missing 'features' key in request"}), 400

        features = data["features"]
        print("🔹 Features length:", len(features))

        if not isinstance(data["features"], list) or len(data["features"]) != 13:
            return jsonify({"error": "Expected 13 features as a list"}), 400


        features = np.array(features, dtype=float).reshape(1, -1)
        print("🔹 Reshaped features:", features)

        prediction = model.predict(features)
        prediction_result = prediction.tolist()
        print("✅ Prediction result:", prediction_result)

        if predictions_collection is not None:
            predictions_collection.insert_one({
                "features": data["features"],
                "prediction": prediction_result,
                "timestamp": datetime.now(UTC)
            })
            print("✅ Saved to MongoDB")
            print(db.list_collection_names())
    

        return jsonify({"prediction": prediction_result})
    
    except Exception as e:
        print("❌ Exception:", str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/history", methods=["GET"])
def history():
    try:
        history = list(predictions_collection.find({}, {"_id": 0}).sort("timestamp", -1).limit(50))
        return jsonify(history)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)



# import pickle
# import numpy as np
# from flask import Flask, request, jsonify

# # Load the trained model
# try:
#     with open("./model1.pkl", "rb") as model_file:
#         model = pickle.load(model_file)

#     # ✅ Ensure the loaded object has the `predict` method
#     if not hasattr(model, "predict"):
#         raise ValueError("Loaded object is not a valid model. Ensure 'model.pkl' contains a trained classifier.")

#     print("✅ Model loaded successfully!")

# except FileNotFoundError:
#     print("❌ Error: 'model.pkl' file not found. Train a model first.")
#     model = None

# except Exception as e:
#     print(f"❌ Error loading model: {e}")
#     model = None

# # Flask app
# app = Flask(__name__)

# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         # ✅ Validate input JSON
#         data = request.get_json()
#         if not data or "features" not in data:
#             return jsonify({"error": "Missing 'features' key in request"}), 400

#         # ✅ Ensure features is a list before converting to numpy ar ray
#         features = data["features"]
#         if not isinstance(features, list):
#             return jsonify({"error": "'features' must be a list of numbers"}), 400

#         # ✅ Convert input to a NumPy array and ensure correct shape
#         features = np.array(features, dtype=float).reshape(1, -1)

#         # ✅ Validate feature count
#         if features.shape[1] != 13:
#             return jsonify({"error": f"Incorrect number of features. Expected {13}, but got {features.shape[1]}"}), 400

#         # ✅ Perform prediction using the trained model
#         prediction = model.predict(features)

#         return jsonify({"prediction": prediction.tolist()})

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)
