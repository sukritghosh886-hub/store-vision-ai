from dataclasses import dataclass, asdict


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
    return asdict(features)


def model_features(features):
    data = features_to_dict(features)

    return [
        [
            data[column]
            for column in FEATURE_COLUMNS
        ]
    ]