"""
Store Vision AI
Machine Learning Pipeline

Random Forest theft-risk classifier.

IMPORTANT:
The current model is trained on synthetic data.
It is NOT validated for real-world theft detection.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent

DATASET_FILE = BASE_DIR / "training_data.csv"
MODEL_FILE = BASE_DIR / "store_vision_theft_model.joblib"


FEATURES = [
    "dwell_time",
    "item_interactions",
    "items_picked",
    "items_returned",
    "movement_speed",
    "shelf_visits",
    "exit_without_billing",
    "billing_mismatch",
]

TARGET = "theft_risk"


def train_model():

    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            "training_data.csv was not found. "
            "Run generate_dataset.py first."
        )

    dataframe = pd.read_csv(DATASET_FILE)

    X = dataframe[FEATURES]
    y = dataframe[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("===================================")
    print("STORE VISION AI")
    print("RANDOM FOREST TRAINING")
    print("===================================")

    print(f"Training samples : {len(X_train)}")
    print(f"Testing samples  : {len(X_test)}")
    print(f"Accuracy         : {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    model_package = {
        "model": model,
        "features": FEATURES,
        "target": TARGET,
        "model_type": "RandomForestClassifier",
        "version": "1.0",
        "training_data": "synthetic",
    }

    joblib.dump(model_package, MODEL_FILE)

    print("-----------------------------------")
    print(f"Model saved to: {MODEL_FILE}")
    print("-----------------------------------")


if __name__ == "__main__":
    train_model()