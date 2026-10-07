"""
Store Vision AI
Machine Learning Pipeline

Inference module for theft-risk prediction.
"""

from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

MODEL_FILE = BASE_DIR / "store_vision_theft_model.joblib"


def load_model():

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "store_vision_theft_model.joblib was not found. "
            "Run train_model.py first."
        )

    package = joblib.load(MODEL_FILE)

    return package["model"], package["features"]


def predict_theft_risk(
    dwell_time,
    item_interactions,
    items_picked,
    items_returned,
    movement_speed,
    shelf_visits,
    exit_without_billing,
    billing_mismatch,
):

    model, features = load_model()

    sample = pd.DataFrame(
        [
            {
                "dwell_time": dwell_time,
                "item_interactions": item_interactions,
                "items_picked": items_picked,
                "items_returned": items_returned,
                "movement_speed": movement_speed,
                "shelf_visits": shelf_visits,
                "exit_without_billing": exit_without_billing,
                "billing_mismatch": billing_mismatch,
            }
        ]
    )

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

    result = predict_theft_risk(
        dwell_time=95,
        item_interactions=7,
        items_picked=4,
        items_returned=0,
        movement_speed=1.2,
        shelf_visits=8,
        exit_without_billing=1,
        billing_mismatch=1,
    )

    print("===================================")
    print("STORE VISION AI")
    print("ML PREDICTION")
    print("===================================")

    print(f"Prediction      : {result['prediction']}")
    print(f"Risk level      : {result['risk_level']}")
    print(f"Risk probability: {result['risk_probability']}")