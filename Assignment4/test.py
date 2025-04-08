import os
import time
import requests
import subprocess

def test_docker():
    # Build the image
    os.system("docker build -t flask-score-app .")

    # Run the container
    os.system("docker run -d -p 5000:5000 --name flask_score_container flask-score-app")

    # Give container a few seconds to start
    time.sleep(5)

    # Send a request to /score
    try:
        sample_text = {"text": "This is a test message"}
        response = requests.post("http://localhost:5000/score", json=sample_text)
        assert response.status_code == 200
        assert "score" in response.json()
        assert isinstance(response["prediction"], bool)
        assert 0 <= response["propensity"] <= 1
        print("Test passed: /score endpoint working")
    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        # Stop and remove the container
        os.system("docker stop flask_score_container")
        os.system("docker rm flask_score_container")
