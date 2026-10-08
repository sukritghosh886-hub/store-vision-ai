"""
Store Vision AI — Synthetic ML Dataset Generator

IMPORTANT:
This dataset is synthetic.

It is useful for software integration and prototype
development only. It does not prove real-world theft
detection performance.
"""

from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
SAMPLES = 2000

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "training_data.csv"


def generate_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)

    records = []

    for _ in range(SAMPLES):

        item_event_count = int(
            rng.integers(0, 20)
        )

        shelf_item_count = int(
            rng.integers(0, 12)
        )

        billed_item_count = int(
            rng.integers(0, 12)
        )

        unpaid_item_count = max(
            shelf_item_count - billed_item_count,
            0,
        )

        unique_detected_items = int(
            rng.integers(
                0,
                max(shelf_item_count, 1) + 1,
            )
        )

        unique_billed_items = int(
            rng.integers(
                0,
                max(billed_item_count, 1) + 1,
            )
        )

        billing_mismatch = int(
            unpaid_item_count > 0
        )

        if shelf_item_count > 0:
            billing_coverage_ratio = min(
                billed_item_count / shelf_item_count,
                1.0,
            )
        else:
            billing_coverage_ratio = 1.0

        mean_detection_confidence = float(
            rng.uniform(0.55, 0.99)
        )

        risk_score = (
            unpaid_item_count * 4.0
            + billing_mismatch * 2.0
            + max(
                unique_detected_items
                - unique_billed_items,
                0,
            ) * 0.8
            + max(
                0.75 - billing_coverage_ratio,
                0,
            ) * 4.0
            + item_event_count * 0.05
            + (
                0.5
                if mean_detection_confidence < 0.70
                else 0
            )
        )

        theft_risk = int(
            risk_score >= 4.0
        )

        records.append(
            {
                "item_event_count":
                    item_event_count,

                "shelf_item_count":
                    shelf_item_count,

                "billed_item_count":
                    billed_item_count,

                "unpaid_item_count":
                    unpaid_item_count,

                "unique_detected_items":
                    unique_detected_items,

                "unique_billed_items":
                    unique_billed_items,

                "billing_mismatch":
                    billing_mismatch,

                "billing_coverage_ratio":
                    round(
                        billing_coverage_ratio,
                        4,
                    ),

                "mean_detection_confidence":
                    round(
                        mean_detection_confidence,
                        4,
                    ),

                "theft_risk":
                    theft_risk,
            }
        )

    dataframe = pd.DataFrame(records)

    dataframe.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    return dataframe


def main():
    dataframe = generate_dataset()

    print("===================================")
    print("STORE VISION AI")
    print("ML DATASET GENERATOR")
    print("===================================")

    print(
        f"Samples generated: {len(dataframe)}"
    )

    print(
        f"Output file: {OUTPUT_FILE}"
    )

    print("\nClass distribution:")

    print(
        dataframe["theft_risk"]
        .value_counts()
        .sort_index()
    )


if __name__ == "__main__":
    main()