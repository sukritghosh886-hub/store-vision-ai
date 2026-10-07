import os
import joblib
import pandas as pd

from feature_adapter import (
    FEATURE_COLUMNS
)


MODEL_FILE = os.path.join(
    os.path.dirname(__file__),
    "store_vision_ml_model.joblib"
)


def load_model():

    if not os.path.exists(MODEL_FILE):

        raise FileNotFoundError(
            "ML model has not been trained yet."
        )

    package = joblib.load(
        MODEL_FILE
    )

    return package["model"]


def predict_risk(features):

    model = load_model()

    row = {
        column: getattr(
            features,
            column
        )
        for column in FEATURE_COLUMNS
    }

    dataframe = pd.DataFrame(
        [row],
        columns=FEATURE_COLUMNS
    )

    probability = float(
        model.predict_proba(
            dataframe
        )[0][1]
    )

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