"""
Store Vision AI Machine Learning Pipeline

Real data flow:

Supabase
   ↓
Store Vision visits
   ↓
Item + billing events
   ↓
Feature extraction
   ↓
Reviewed alert labels
   ↓
Random Forest
   ↓
Risk prediction
"""

import os
import sys
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.supabase_client import (
    create_supabase_client
)

from real_event_adapter import (
    build_features
)

from feature_adapter import (
    FEATURE_COLUMNS
)


MODEL_FILE = os.path.join(
    os.path.dirname(__file__),
    "store_vision_ml_model.joblib"
)


def load_reviewed_data():

    client = create_supabase_client()

    response = (
        client
        .table("alerts")
        .select(
            "visit_id,status"
        )
        .in_(
            "status",
            ["confirmed", "dismissed"]
        )
        .execute()
    )

    alerts = response.data or []

    records = []

    for alert in alerts:

        visit_id = alert.get("visit_id")
        status = alert.get("status")

        if visit_id is None:
            continue

        try:

            features = build_features(
                int(visit_id)
            )

            record = {
                column: getattr(
                    features,
                    column
                )
                for column in FEATURE_COLUMNS
            }

            record["target"] = (
                1
                if status == "confirmed"
                else 0
            )

            records.append(record)

        except Exception as error:

            print(
                f"Skipping visit {visit_id}: "
                f"{error}"
            )

    return pd.DataFrame(records)


def train():

    dataframe = load_reviewed_data()

    if dataframe.empty:

        raise RuntimeError(
            "No reviewed Store Vision events "
            "are available for training."
        )

    if dataframe["target"].nunique() < 2:

        raise RuntimeError(
            "Training requires both confirmed "
            "and dismissed reviewed alerts."
        )

    X = dataframe[
        FEATURE_COLUMNS
    ]

    y = dataframe["target"]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "dismissed",
                "confirmed"
            ],
            zero_division=0
        )
    )

    joblib.dump(
        {
            "model": model,
            "features": FEATURE_COLUMNS,
        },
        MODEL_FILE
    )

    print(
        f"Model saved: {MODEL_FILE}"
    )

    return model


if __name__ == "__main__":

    train()