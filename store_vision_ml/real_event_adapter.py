"""
Store Vision AI — Real Event Adapter

Converts Store Vision application events
into the common ML feature schema.

Data source:
Store Vision event system / Supabase.

IMPORTANT:
These features represent application-generated
events. They are not proof of theft.
"""

from collections import Counter

from .feature_adapter import (
    StoreVisitFeatures,
)

from store_events import (
    get_billing_events,
    get_item_events,
    get_unpaid_count,
)


def get_label(event: dict) -> str:
    """
    Extract the product/item label from an event.
    """

    return str(
        event.get("item_label")
        or event.get("item_name")
        or event.get("label")
        or "unknown"
    )


def build_features(
    visit_id,
) -> StoreVisitFeatures:
    """
    Build ML features for one Store Vision visit.
    """

    visit_id = int(
        visit_id
    )

    item_events = (
        get_item_events(
            visit_id
        )
        or []
    )

    billing_events = (
        get_billing_events(
            visit_id
        )
        or []
    )

    unpaid_count = int(
        get_unpaid_count(
            visit_id
        )
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

        label = get_label(
            event
        )

        try:
            quantity = int(
                event.get("quantity")
                or 1
            )

        except (
            TypeError,
            ValueError,
        ):
            quantity = 1

        billed[label] += quantity

    detected_total = sum(
        detected.values()
    )

    billed_total = sum(
        billed.values()
    )

    if detected_total > 0:

        coverage = min(
            billed_total / detected_total,
            1.0,
        )

    else:

        coverage = 1.0

    confidence_values = []

    for event in shelf_events:

        confidence = event.get(
            "confidence"
        )

        if confidence is None:
            continue

        try:

            confidence_values.append(
                float(confidence)
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

    if confidence_values:

        mean_confidence = (
            sum(confidence_values)
            / len(confidence_values)
        )

    else:

        mean_confidence = 0.0

    return StoreVisitFeatures(

        visit_id=visit_id,

        item_event_count=len(
            item_events
        ),

        shelf_item_count=detected_total,

        billed_item_count=billed_total,

        unpaid_item_count=unpaid_count,

        unique_detected_items=len(
            detected
        ),

        unique_billed_items=len(
            billed
        ),

        billing_mismatch=int(
            unpaid_count > 0
        ),

        billing_coverage_ratio=round(
            coverage,
            4,
        ),

        mean_detection_confidence=round(
            mean_confidence,
            4,
        ),
    )


def features_as_dict(
    visit_id,
) -> dict:
    """
    Return visit features as a dictionary.
    """

    features = build_features(
        visit_id
    )

    return {
        "visit_id":
            features.visit_id,

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