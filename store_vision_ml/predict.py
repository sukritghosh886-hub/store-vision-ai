"""
Store Vision AI
Machine Learning Prediction Module

Connects CustomerSession data from the feature adapter
to the trained Random Forest model.
"""

from pathlib import Path

import joblib
import pandas as pd

from feature_adapter import CustomerSession, create_ml_features


BASE_DIR = Path(__file__).resolve().parent

MODEL_FILE = BASE_DIR / "store_vision_theft_model.joblib"


def load_model():

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "store_vision_theft_model.joblib was not found. "
            "Run generate_dataset.py and train_model.py first."
        )

    package = joblib.load(MODEL_FILE)

    return package["model"], package["features"]


def predict_session(session: CustomerSession) -> dict:
    """
    Predict theft-risk for one customer session.
    """

    model, features = load_model()

    feature_data = create_ml_features(session)

    sample = pd.DataFrame([feature_data])

    sample = sample[features]

    prediction = model.predict(sample)[0]

    probability = model.predict_proba(sample)[0][1]

    if prediction == 1:
        risk_level = "HIGH"
    else:
        risk_level = "LOW"

    return {
        "prediction": int(prediction),
        "risk_level": risk_level,
        "risk_probability": round(float(probability), 4),
    }


if __name__ == "__main__":

    session = CustomerSession(
        dwell_time=95,
        item_interactions=7,
        items_picked=4,
        items_returned=0,
        movement_speed=1.2,
        shelf_visits=8,
        exit_without_billing=1,
        billing_mismatch=1,
    )

    result = predict_session(session)

    print("===================================")
    print("STORE VISION AI")
    print("ML SESSION PREDICTION")
    print("===================================")

    print(f"Prediction       : {result['prediction']}")
    print(f"Risk level       : {result['risk_level']}")
    print(f"Risk probability : {result['risk_probability']}")