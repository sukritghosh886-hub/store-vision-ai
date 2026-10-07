"""
Store Vision AI
ML Event Processor

Collects vision events for a customer session and converts
them into CustomerSession data for the ML model.

This is an integration layer.
It does not perform computer vision itself.
"""

from feature_adapter import CustomerSession
from predict import predict_session


class CustomerEventProcessor:
    """
    Collects events belonging to one customer session.
    """

    def __init__(self):

        self.dwell_time = 0.0
        self.item_interactions = 0
        self.items_picked = 0
        self.items_returned = 0
        self.movement_speed = 0.0
        self.shelf_visits = 0
        self.exit_without_billing = 0
        self.billing_mismatch = 0

    def set_dwell_time(self, seconds):
        self.dwell_time = float(seconds)

    def record_item_interaction(self):
        self.item_interactions += 1

    def record_item_picked(self):
        self.items_picked += 1

    def record_item_returned(self):
        self.items_returned += 1

    def record_shelf_visit(self):
        self.shelf_visits += 1

    def set_movement_speed(self, speed):
        self.movement_speed = float(speed)

    def set_exit_without_billing(self, value=True):
        self.exit_without_billing = int(value)

    def set_billing_mismatch(self, value=True):
        self.billing_mismatch = int(value)

    def create_session(self):

        return CustomerSession(
            dwell_time=self.dwell_time,
            item_interactions=self.item_interactions,
            items_picked=self.items_picked,
            items_returned=self.items_returned,
            movement_speed=self.movement_speed,
            shelf_visits=self.shelf_visits,
            exit_without_billing=self.exit_without_billing,
            billing_mismatch=self.billing_mismatch,
        )

    def predict(self):

        session = self.create_session()

        return predict_session(session)


if __name__ == "__main__":

    processor = CustomerEventProcessor()

    # Simulated events coming from Store Vision AI.

    processor.set_dwell_time(95)

    processor.record_item_interaction()
    processor.record_item_interaction()
    processor.record_item_interaction()
    processor.record_item_interaction()
    processor.record_item_interaction()

    processor.record_item_picked()
    processor.record_item_picked()
    processor.record_item_picked()

    processor.set_movement_speed(1.2)

    processor.record_shelf_visit()
    processor.record_shelf_visit()
    processor.record_shelf_visit()

    processor.set_exit_without_billing(True)

    processor.set_billing_mismatch(True)

    result = processor.predict()

    print("===================================")
    print("STORE VISION AI")
    print("EVENT PROCESSOR")
    print("===================================")

    print("ML prediction:")
    print(result)