"""FastAPI service exposing a Parkinson's prediction endpoint.

Loads a pre-trained model from disk and exposes a `/predict`
endpoint that accepts a JSON payload with a `features` list.
"""

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# Load pre-trained classifier from disk (expects `parkinsons_model.pkl`).
model = joblib.load("parkinsons_model.pkl")

app = FastAPI()


class Features(BaseModel):
    """Request schema for prediction endpoint.

    Attributes:
        features: list of floats in the same order as the model expects.
    """
    features: list[float]


@app.get("/")
def home():
    """Health-check endpoint for the API."""
    return {"message": "Parkinson's Detection API running"}


@app.post("/predict")
def predict(data: Features):
    """Predict class from a list of features.

    Converts incoming list into a numpy array, calls the model, and
    returns both a human-readable label and the integer class.
    """
    X = np.array(data.features).reshape(1, -1)
    prediction = model.predict(X)[0]

    result = "Parkinson's Detected" if prediction == 1 else "Healthy"

    return {
        "prediction": result,
        "class": int(prediction)
    }
