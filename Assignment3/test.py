import os
import json
import joblib
import requests
from score import score
from flask import Flask
from multiprocessing import Process
import time
import subprocess


# Load the saved model
MODEL_PATH = "./../Assignment2/mlruns/130972181796319339/8b0ee71a5e6e4d419ba0a55a546d32b3/artifacts/model/model.pkl"
model = joblib.load(MODEL_PATH)

def test_smoke():
    # Smoke test: Ensure function runs without crashing
    text = "This is a test message."
    score(text, model, 0.5)
    

def test_format():
    # Format tests
    text = "This is a test message."
    prediction, propensity = score(text, model, 0.5)
    assert isinstance(prediction, bool)
    assert isinstance(propensity, float)

def test_prediction():
    text = "This is a test message."
    prediction, propensity = score(text, model, 0.5)
    assert prediction in [True, False]

def test_propensity():
    text = "This is a test message."
    prediction, propensity = score(text, model, 0.5)
    assert 0 <= propensity <= 1

def test_threshold_0():
    text = "This is a test message." 
    assert score(text, model, 0)[0] == True        # Always spam if threshold is 0

def test_threshold_1():
    text = "This is a test message."
    assert score(text, model, 1)[0] == False       # Always non-spam if threshold is 1

def test_spam_specific():
    text = "Dear Dave this is your final notice to collect your 4* Tenerife Holiday or #5000 CASH award! Call 09061743806 from landline. TCs SAE Box326 CW25WX 150ppm"
    pre, prop =  score(text, model, 0.5)
    print(pre, prop)

def test_non_spam_specific():
    text = "Hello, how are you?"
    assert score(text, model, 0.5)[0] == False  # Non-spam


def test_flask():
    """Tests the Flask API."""
    # Start the Flask app in a separate process
    flask_process = subprocess.Popen(["python", "app.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(10)

    try:
        response = requests.get("http://127.0.0.1:5000/")
        time.sleep(2)  # Allow time for Flask to start
        assert response.status_code == 200
        
        # Test endpoint
        response = requests.post(
            "http://127.0.0.1:5000/score", 
            data=json.dumps({"text": "You won a lottery!"}),
            headers={"Content-Type": "application/json"}
        )
        time.sleep(2)  # Allow time for Flask to process the request
        assert response.status_code == 200

        data = response.json()
        assert "prediction" in data
        assert "propensity" in data
        assert isinstance(data["prediction"], bool)
        assert 0 <= data["propensity"] <= 1

    finally:
        # Stop the Flask app
        flask_process.terminate()
