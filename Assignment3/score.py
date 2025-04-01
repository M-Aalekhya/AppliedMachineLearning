import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer


def score(text: str, model: BaseEstimator, threshold: float) -> tuple[bool, float]:
    """
    Scores a trained model on a given text.

    Parameters:
    - text (str): Input text to classify
    - model (sklearn.base.BaseEstimator): Trained model
    - threshold (float): Threshold for classifying spam (0-1)

    Returns:
    - prediction (bool): True for spam, False for non-spam
    - propensity (float): Probability of being spam
    """

    # Model pipeline already includes TfidfVectorizer
    # Predict probability
    propensity = model.predict_proba([text])[0][1]
    prediction = bool(propensity >= threshold)

    return prediction, float(propensity)
