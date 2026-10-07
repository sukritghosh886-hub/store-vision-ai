import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass

ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import store_events


@dataclass
class RealVisitFeatures:
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


def get_label(event):
    return (
        event.get("item_label")
        or event.get("item_name")
        or "unknown"
    )


def build_features(visit_id):

    item_events = (
        store_events.get_item_events(visit_id)
        or []
    )

    billing_events = (
        store_events.get_billing_events(visit_id)
        or []
    )

    unpaid_count = int(
        store_events.get_unpaid_count(visit_id)
    )

    shelf_events = [
        event
        for event in item_events
        if event.get("zone") == "shelf"
    ]

    detected = Counter(
        get_label(event)
        for event in shelf_events
    )

    billed = Counter()

    for event in billing_events:

        billed[get_label(event)] += int(
            event.get("quantity") or 1
        )

    detected_total = sum(
        detected.values()
    )

    billed_total = sum(
        billed.values()
    )

    if detected_total:

        coverage = min(
            billed_total / detected_total,
            1.0
        )

    else:

        coverage = 1.0

    confidence_values = [
        float(event["confidence"])
        for event in shelf_events
        if event.get("confidence") is not None
    ]

    if confidence_values:

        mean_confidence = (
            sum(confidence_values)
            / len(confidence_values)
        )

    else:

        mean_confidence = 0.0

    return RealVisitFeatures(
        visit_id=int(visit_id),
        item_event_count=len(item_events),
        shelf_item_count=detected_total,
        billed_item_count=billed_total,
        unpaid_item_count=unpaid_count,
        unique_detected_items=len(detected),
        unique_billed_items=len(billed),
        billing_mismatch=int(
            unpaid_count > 0
        ),
        billing_coverage_ratio=round(
            coverage,
            4
        ),
        mean_detection_confidence=round(
            mean_confidence,
            4
        ),
    )


def features_as_dict(visit_id):

    return asdict(
        build_features(visit_id)
    )