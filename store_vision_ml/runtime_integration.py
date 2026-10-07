"""
Store Vision AI — ML Runtime Integration

Connects real Store Vision events to the
machine-learning prediction layer.

Flow:

Store Vision events
        ↓
Feature extraction
        ↓
Trained Random Forest
        ↓
Risk score
        ↓
Human review
"""

import os
import sys

ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from real_event_adapter import build_features
from predict import predict_risk


def analyze_visit(visit_id):
    """
    Analyze one real Store Vision visit.
    """

    features = build_features(
        int(visit_id)
    )

    prediction = predict_risk(
        features
    )

    return {
        "visit_id": int(visit_id),

        "risk_probability":
            prediction["risk_probability"],

        "risk_level":
            prediction["risk_level"],

        "features": {
            "item_event_count":
                features.item_event_count,

            "shelf_item_count":
                features.shelf_item_count,

            "billed_item_count":
                features.billed_item_count,

            "unpaid_item_count":
                features.unpaid_item_count,

            "unique_detected_items":
                features.unique_detected_items,

            "unique_billed_items":
                features.unique_billed_items,

            "billing_mismatch":
                features.billing_mismatch,

            "billing_coverage_ratio":
                features.billing_coverage_ratio,

            "mean_detection_confidence":
                features.mean_detection_confidence,
        }
    }


def should_review(result):
    """
    Determine whether the visit should be
    sent for human review.
    """

    return (
        result["risk_level"]
        in ["MEDIUM", "HIGH"]
    )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Analyze a Store Vision AI visit "
            "using the ML pipeline."
        )
    )

    parser.add_argument(
        "visit_id",
        type=int
    )

    args = parser.parse_args()

    result = analyze_visit(
        args.visit_id
    )

    print(
        "\n================================"
    )

    print(
        "STORE VISION AI — ML ANALYSIS"
    )

    print(
        "================================"
    )

    print(
        f"Visit ID: "
        f"{result['visit_id']}"
    )

    print(
        f"Risk probability: "
        f"{result['risk_probability']:.2%}"
    )

    print(
        f"Risk level: "
        f"{result['risk_level']}"
    )

    print(
        f"Human review required: "
        f"{should_review(result)}"
    )

    print(
        "\nFeatures:"
    )

    for key, value in (
        result["features"].items()
    ):

        print(
            f"  {key}: {value}"
        )