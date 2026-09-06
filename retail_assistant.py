"""
Store Vision AI
AI Retail Store Assistant

Rule-based analytics engine.

The engine converts Store Vision AI events into practical
store-layout and merchandising recommendations.

No external paid LLM API is required.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List


class RetailAssistant:

    def __init__(
        self,
        products: List[Dict[str, Any]],
        sales: List[Dict[str, Any]],
        visits: List[Dict[str, Any]],
        attraction_events: List[Dict[str, Any]],
    ):

        self.products = products or []

        self.sales = sales or []

        self.visits = visits or []

        self.events = attraction_events or []


    # ========================================================
    # ATTRACTION SCORE
    # ========================================================

    def attraction_scores(
        self,
    ) -> Dict[str, Dict[str, float]]:

        stats = defaultdict(
            lambda: {
                "approaches": 0,
                "inspects": 0,
                "picks": 0,
                "returns": 0,
                "carries": 0,
            }
        )

        for event in self.events:

            label = (
                event.get(
                    "detected_label"
                )
                or "unknown"
            )

            event_type = str(
                event.get(
                    "event_type",
                    ""
                )
            )

            if event_type in stats[label]:

                stats[label][
                    event_type
                ] += 1

        scores = {}

        for label, values in stats.items():

            score = (
                values["approaches"] * 1
                +
                values["inspects"] * 2
                +
                values["picks"] * 4
                +
                values["carries"] * 5
                -
                values["returns"] * 1
            )

            scores[label] = {

                **values,

                "attraction_score":
                    float(score),

            }

        return scores


    # ========================================================
    # SALES
    # ========================================================

    def sales_by_product(
        self,
    ) -> Dict[str, int]:

        result = defaultdict(int)

        for sale in self.sales:

            product_id = str(
                sale.get(
                    "product_id",
                    ""
                )
            )

            try:

                quantity = int(
                    sale.get(
                        "quantity",
                        0
                    )
                )

            except (
                TypeError,
                ValueError
            ):

                quantity = 0

            result[
                product_id
            ] += quantity

        return dict(result)


    # ========================================================
    # VISITOR COUNT
    # ========================================================

    def visitor_count(
        self,
    ) -> int:

        return len(
            self.visits
        )


    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    def generate_recommendations(
        self,
    ) -> List[Dict[str, Any]]:

        scores = (
            self.attraction_scores()
        )

        recommendations = []

        # ----------------------------------------------------
        # Highest attraction
        # ----------------------------------------------------

        if scores:

            ranking = sorted(
                scores.items(),
                key=lambda x:
                    x[1][
                        "attraction_score"
                    ],
                reverse=True,
            )

            top_label, top_stats = (
                ranking[0]
            )

            recommendations.append({

                "recommendation_type":
                    "high_attraction",

                "title":
                    f"Move {top_label} "
                    "toward a high-visibility area",

                "recommendation":
                    (
                        f"{top_label} has the "
                        f"highest observed attraction "
                        f"score ({top_stats['attraction_score']:.0f}). "
                        "Consider placing it near the "
                        "entrance or another high-traffic "
                        "display position."
                    ),

                "priority":
                    "high",

                "evidence":
                    top_stats,

            })


        # ----------------------------------------------------
        # High inspection / low purchase
        # ----------------------------------------------------

        for label, stats in scores.items():

            inspections = (
                stats["inspects"]
            )

            picks = (
                stats["picks"]
            )

            if (
                inspections >= 5
                and
                picks < inspections * 0.30
            ):

                recommendations.append({

                    "recommendation_type":
                        "low_conversion",

                    "title":
                        f"Improve presentation of {label}",

                    "recommendation":
                        (
                            f"Customers are inspecting "
                            f"{label} frequently but "
                            "relatively few interactions "
                            "result in a pickup. "
                            "Try a clearer price label, "
                            "better placement, or related "
                            "products nearby."
                        ),

                    "priority":
                        "medium",

                    "evidence":
                        stats,

                })


        # ----------------------------------------------------
        # High returns
        # ----------------------------------------------------

        for label, stats in scores.items():

            if (
                stats["picks"] >= 3
                and
                stats["returns"]
                >=
                stats["picks"] * 0.50
            ):

                recommendations.append({

                    "recommendation_type":
                        "high_return",

                    "title":
                        f"Review display of {label}",

                    "recommendation":
                        (
                            f"{label} is frequently "
                            "picked up and returned. "
                            "Consider improving its "
                            "visibility, pricing information, "
                            "or product description."
                        ),

                    "priority":
                        "medium",

                    "evidence":
                        stats,

                })


        return recommendations


    # ========================================================
    # DASHBOARD SUMMARY
    # ========================================================

    def summary(
        self,
    ) -> Dict[str, Any]:

        recommendations = (
            self.generate_recommendations()
        )

        return {

            "visitor_count":
                self.visitor_count(),

            "attraction_scores":
                self.attraction_scores(),

            "recommendations":
                recommendations,

        }


__all__ = [
    "RetailAssistant",
]