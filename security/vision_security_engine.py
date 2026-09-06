from __future__ import annotations

from typing import Dict, List, Optional

from models.theft_detector import (
    TheftDetector
)

from models.retail_assistant import (
    RetailAssistant
)

from security.evidence_recorder import (
    EvidenceRecorder
)


class VisionSecurityEngine:

    def __init__(
        self,
        exit_zone,
        evidence_directory="evidence",
    ):

        self.theft_detector = (
            TheftDetector(
                exit_zone=exit_zone
            )
        )

        self.retail_assistant = (
            RetailAssistant()
        )

        self.evidence_recorder = (
            EvidenceRecorder(
                evidence_directory
            )
        )

    def process_frame(
        self,
        frame,
        people: List[dict],
        products: List[dict],
        paid_items: Optional[
            Dict[int, List[str]]
        ] = None,
    ):

        if paid_items is None:

            paid_items = {}

        # ==================================================
        # RETAIL ANALYTICS
        # ==================================================

        for product in products:

            self.retail_assistant.observe_product(
                product["class_name"]
            )

        # ==================================================
        # THEFT DETECTION
        # ==================================================

        alerts = (
            self.theft_detector.update(
                people=people,
                products=products,
                paid_items=paid_items,
            )
        )

        alert_results = []

        for alert in alerts:

            evidence_path = (
                self.evidence_recorder.save_frame(
                    frame=frame,

                    track_id=
                        alert.track_id,

                    item_name=
                        alert.item_name,
                )
            )

            alert_results.append({

                "track_id":
                    alert.track_id,

                "item_name":
                    alert.item_name,

                "reason":
                    alert.reason,

                "confidence":
                    round(
                        alert.confidence,
                        4
                    ),

                "evidence_path":
                    evidence_path,

                "status":
                    "pending_review",

                "created_at":
                    alert.created_at,
            })

        return {

            "theft_alerts":
                alert_results,

            "retail":
                self.retail_assistant.summary(),
        }