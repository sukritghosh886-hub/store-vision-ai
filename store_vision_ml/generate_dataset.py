"""
Store Vision AI
Machine Learning Pipeline

Synthetic training-data generator for theft-risk classification.

IMPORTANT:
This dataset is synthetic and is intended only for
prototype/model-development purposes.
"""

import numpy as np
import pandas as pd


RANDOM_SEED = 42
SAMPLES = 2000
OUTPUT_FILE = "training_data.csv"


np.random.seed(RANDOM_SEED)


def generate_dataset():
    data = []

    for _ in range(SAMPLES):

        dwell_time = np.random.uniform(1, 120)

        item_interactions = np.random.randint(0, 10)

        items_picked = np.random.randint(0, 6)

        items_returned = np.random.randint(0, 6)

        movement_speed = np.random.uniform(0.1, 3.0)

        shelf_visits = np.random.randint(0, 12)

        exit_without_billing = np.random.randint(0, 2)

        billing_mismatch = np.random.randint(0, 2)

        risk_score = (
            exit_without_billing * 5
            + billing_mismatch * 4
            + max(items_picked - items_returned, 0) * 0.8
            + shelf_visits * 0.15
            + item_interactions * 0.1
            + (1 if dwell_time > 90 else 0) * 0.5
        )

        theft_risk = 1 if risk_score >= 5 else 0

        data.append(
            [
                dwell_time,
                item_interactions,
                items_picked,
                items_returned,
                movement_speed,
                shelf_visits,
                exit_without_billing,
                billing_mismatch,
                theft_risk,
            ]
        )

    columns = [
        "dwell_time",
        "item_interactions",
        "items_picked",
        "items_returned",
        "movement_speed",
        "shelf_visits",
        "exit_without_billing",
        "billing_mismatch",
        "theft_risk",
    ]

    dataframe = pd.DataFrame(data, columns=columns)

    dataframe.to_csv(OUTPUT_FILE, index=False)

    print("===================================")
    print("STORE VISION AI")
    print("ML DATASET GENERATOR")
    print("===================================")
    print(f"Samples generated: {len(dataframe)}")
    print(f"Output file: {OUTPUT_FILE}")
    print("\nClass distribution:")
    print(dataframe["theft_risk"].value_counts())


if __name__ == "__main__":
    generate_dataset()