"""
Store Vision AI — Feature Adapter

Defines the common feature schema used by
training and prediction.
"""

from dataclasses import asdict, dataclass


@dataclass
class StoreVisitFeatures:
    visit_id: int

    item_event_count: int
    shelf_item_count: int
    billed_item_count: int
    unpaid_item_count: int

    unique_detected_items: int
    unique_billed_items: int

    billing_mismatch: int
    billing_coverage_ratio: float
    mean_detection_confidence: float


FEATURE_COLUMNS = [
    "item_event_count",
    "shelf_item_count",
    "billed_item_count",
    "unpaid_item_count",
    "unique_detected_items",
    "unique_billed_items",
    "billing_mismatch",
    "billing_coverage_ratio",
    "mean_detection_confidence",
]


def features_to_dict(features):
    """
    Convert StoreVisitFeatures into a dictionary.
    """

    return asdict(features)


def model_features(features):
    """
    Convert features into the 2D format expected
    by scikit-learn.
    """

    data = features_to_dict(features)

    return [
        [
            data[column]
            for column in FEATURE_COLUMNS
        ]
    ]