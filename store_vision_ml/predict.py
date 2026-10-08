"""
Store Vision AI — ML Prediction

Loads the trained Random Forest model and
produces a risk probability.
"""

import os

import joblib
import pandas as pd

from feature_adapter import (
    FEATURE_COLUMNS
)


MODEL_FILE = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)
    ),
    "store_vision_ml_model.joblib"
)


def load_model():
    """
    Load the trained ML model.
    """

    if not os.path.exists(
        MODEL_FILE
    ):

        raise FileNotFoundError(
            "Store Vision ML model has "
            "not been trained yet."
        )

    package = joblib.load(
        MODEL_FILE
    )

    return package["model"]


def predict_risk(features):
    """
    Predict risk for one Store Vision visit.
    """

    model = load_model()

    row = {}

    for column in FEATURE_COLUMNS:

        row[column] = getattr(
            features,
            column
        )

    dataframe = pd.DataFrame(
        [row],
        columns=FEATURE_COLUMNS
    )

    probabilities = model.predict_proba(
        dataframe
    )[0]

    classes = list(
        model.classes_
    )

    if 1 in classes:

        confirmed_index = (
            classes.index(1)
        )

        probability = float(
            probabilities[
                confirmed_index
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
            4
        ),

        "risk_level": level,
    }