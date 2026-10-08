"""
Store Vision AI — ML Prediction

Loads the trained Random Forest model and
produces a risk probability.

The ML model is optional. Store Vision AI's
core workflow must continue to work if the
model file is unavailable.
"""

from pathlib import Path

import joblib
import pandas as pd

from .feature_adapter import (
    FEATURE_COLUMNS,
)


BASE_DIR = Path(__file__).resolve().parent

MODEL_FILE = (
    BASE_DIR / "store_vision_ml_model.joblib"
)


def load_model():
    """
    Load the trained Random Forest model.
    """

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            "Store Vision ML model has not "
            "been trained yet."
        )

    package = joblib.load(
        MODEL_FILE
    )

    if not isinstance(package, dict):
        raise ValueError(
            "Invalid Store Vision ML model package."
        )

    if "model" not in package:
        raise ValueError(
            "Model package does not contain 'model'."
        )

    return package["model"]


def predict_risk(features):
    """
    Predict risk for one Store Vision visit.
    """

    model = load_model()

    row = {}

    for column in FEATURE_COLUMNS:

        if not hasattr(
            features,
            column,
        ):
            raise ValueError(
                f"Missing feature: {column}"
            )

        row[column] = getattr(
            features,
            column,
        )

    dataframe = pd.DataFrame(
        [row],
        columns=FEATURE_COLUMNS,
    )

    probabilities = model.predict_proba(
        dataframe
    )[0]

    classes = list(
        model.classes_
    )

    if 1 in classes:

        positive_index = (
            classes.index(1)
        )

        probability = float(
            probabilities[
                positive_index
            ]
        )

    else:

        probability = 0.0

    if probability >= 0.75:
        level = "HIGH"

    elif probability >= 0.40:
        level = "MEDIUM"

    else:
        level = "LOW"

    return {
        "risk_probability": round(
            probability,
            4,
        ),
        "risk_level": level,
        "model_type":
            "RandomForestClassifier",
    }