from flask import Flask, request, jsonify
import joblib
from score import score

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "./../Assignment2/mlruns/130972181796319339/8b0ee71a5e6e4d419ba0a55a546d32b3/artifacts/model/model.pkl"
model = joblib.load(MODEL_PATH)

@app.route('/')
def home():
    return "Flask app is running!"

@app.route('/score', methods=['POST'])
def score_endpoint():
    """Flask endpoint to score text."""
    data = request.get_json()
    text = data.get("text", "")
    threshold = float(data.get("threshold", 0.5))

    if not isinstance(text, str) or len(text) == 0:
        return jsonify({"error": "Invalid input text"}), 400

    prediction, propensity = score(text, model, 0.5)
    return jsonify({
        "prediction": prediction,
        "propensity": propensity
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
