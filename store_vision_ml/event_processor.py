"""
Store Vision AI — Event Processor

Compatibility integration layer for manually
constructed Store Vision feature events.

The active runtime pipeline uses StoreVisitFeatures.
"""

from .feature_adapter import (
    StoreVisitFeatures,
)

from .predict import (
    predict_risk,
)


class CustomerEventProcessor:

    def __init__(
        self,
        visit_id: int = 0,
    ):

        self.visit_id = int(
            visit_id
        )

        self.item_event_count = 0
        self.shelf_item_count = 0
        self.billed_item_count = 0
        self.unpaid_item_count = 0

        self.unique_detected_items = 0
        self.unique_billed_items = 0

        self.billing_mismatch = 0
        self.billing_coverage_ratio = 1.0
        self.mean_detection_confidence = 0.0

    def record_item_event(
        self,
        shelf: bool = True,
        confidence: float = 0.0,
    ):

        self.item_event_count += 1

        if shelf:
            self.shelf_item_count += 1

        if confidence > 0:
            current_count = (
                self.item_event_count
            )

            previous_total = (
                self.mean_detection_confidence
                * (current_count - 1)
            )

            self.mean_detection_confidence = (
                previous_total
                + float(confidence)
            ) / current_count

    def set_billed_items(
        self,
        count: int,
    ):

        self.billed_item_count = max(
            int(count),
            0,
        )

        self.unpaid_item_count = max(
            self.shelf_item_count
            - self.billed_item_count,
            0,
        )

        self.billing_mismatch = int(
            self.unpaid_item_count > 0
        )

        if self.shelf_item_count > 0:

            self.billing_coverage_ratio = min(
                self.billed_item_count
                / self.shelf_item_count,
                1.0,
            )

        else:

            self.billing_coverage_ratio = 1.0

    def set_unique_items(
        self,
        detected: int,
        billed: int,
    ):

        self.unique_detected_items = max(
            int(detected),
            0,
        )

        self.unique_billed_items = max(
            int(billed),
            0,
        )

    def create_features(
        self,
    ) -> StoreVisitFeatures:

        return StoreVisitFeatures(

            visit_id=self.visit_id,

            item_event_count=
                self.item_event_count,

            shelf_item_count=
                self.shelf_item_count,

            billed_item_count=
                self.billed_item_count,

            unpaid_item_count=
                self.unpaid_item_count,

            unique_detected_items=
                self.unique_detected_items,

            unique_billed_items=
                self.unique_billed_items,

            billing_mismatch=
                self.billing_mismatch,

            billing_coverage_ratio=round(
                self.billing_coverage_ratio,
                4,
            ),

            mean_detection_confidence=round(
                self.mean_detection_confidence,
                4,
            ),
        )

    def predict(self):

        features = self.create_features()

        return predict_risk(
            features
        )


if __name__ == "__main__":

    processor = CustomerEventProcessor(
        visit_id=1
    )

    processor.record_item_event(
        shelf=True,
        confidence=0.91,
    )

    processor.record_item_event(
        shelf=True,
        confidence=0.94,
    )

    processor.record_item_event(
        shelf=True,
        confidence=0.89,
    )

    processor.set_billed_items(
        2
    )

    processor.set_unique_items(
        detected=3,
        billed=2,
    )

    print(
        processor.create_features()
    )

    try:

        print(
            processor.predict()
        )

    except FileNotFoundError:

        print(
            "ML model not trained yet."
        )