"""
Store Vision AI — Runtime ML Integration

Connects real Store Vision visit data to
the optional Random Forest predictor.
"""

from .predict import predict_risk
from .real_event_adapter import build_features


def analyze_visit(
    visit_id,
) -> dict:
    """
    Build real application features and run
    the trained ML model.
    """

    features = build_features(
        visit_id
    )

    prediction = predict_risk(
        features
    )

    return {
        "visit_id":
            int(visit_id),

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
        },

        "prediction":
            prediction,
    }