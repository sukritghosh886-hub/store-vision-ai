"""
Store Vision AI
AI Retail Store Assistant

Anonymous retail intelligence engine.

Pipeline:
    Store visits
        -> product attraction events
        -> entry preference
        -> interaction preference
        -> exit preference
        -> purchase conversion
        -> layout recommendations

This module does not perform facial recognition and does not identify
customers personally.

Recommendations are data-driven suggestions. They do not guarantee
increased sales.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional


class RetailAssistant:
    """
    Retail intelligence engine.

    The constructor accepts already-loaded Supabase rows so the class
    can be used by Streamlit, camera_pipeline.py, tests, or an API.
    """

    def __init__(
        self,
        products: Optional[List[Dict[str, Any]]] = None,
        sales: Optional[List[Dict[str, Any]]] = None,
        visits: Optional[List[Dict[str, Any]]] = None,
        attraction_events: Optional[List[Dict[str, Any]]] = None,
        recommendations: Optional[List[Dict[str, Any]]] = None,
        billing_items: Optional[List[Dict[str, Any]]] = None,
    ):
        self.products = products or []
        self.sales = sales or []
        self.visits = visits or []
        self.events = attraction_events or []
        self.recommendations = recommendations or []
        self.billing_items = billing_items or []

        self.product_map = {
            str(p.get("id")): p
            for p in self.products
            if p.get("id") is not None
        }

    # ------------------------------------------------------------------
    # BASIC HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _number(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _parse_time(value: Any) -> Optional[datetime]:
        if not value:
            return None

        if isinstance(value, datetime):
            return value

        text = str(value).strip()

        try:
            return datetime.fromisoformat(
                text.replace("Z", "+00:00")
            )
        except ValueError:
            return None

    @staticmethod
    def _product_key(event: Dict[str, Any]) -> str:
        product_id = event.get("product_id")

        if product_id:
            return str(product_id)

        label = (
            event.get("detected_label")
            or event.get("product_name")
            or event.get("label")
            or "unknown"
        )

        return str(label)

    def _product_name(self, key: str) -> str:
        product = self.product_map.get(str(key))

        if product:
            return (
                product.get("name")
                or product.get("product_name")
                or product.get("sku")
                or str(key)
            )

        return str(key)

    # ------------------------------------------------------------------
    # ATTRACTION ANALYTICS
    # ------------------------------------------------------------------

    def attraction_scores(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate product attention.

        Stronger signals receive more weight:

        approach  -> low weight
        inspect   -> medium weight
        pick      -> strong weight
        carry     -> strong weight
        return    -> negative signal
        """

        stats: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {
                "approaches": 0,
                "inspects": 0,
                "picks": 0,
                "returns": 0,
                "carries": 0,
                "visitors": 0,
                "carry_seconds": 0.0,
            }
        )

        visitor_sets: Dict[str, set] = defaultdict(set)

        for event in self.events:
            key = self._product_key(event)

            event_type = str(
                event.get("event_type")
                or event.get("interaction_type")
                or ""
            ).lower()

            if event_type not in {
                "approach",
                "inspect",
                "pick",
                "return",
                "carry",
            }:
                continue

            stats[key][f"{event_type}s"] += 1

            visit_id = event.get("visit_id")
            person_id = (
                event.get("person_track_id")
                or event.get("person_id")
            )

            visitor_key = (
                visit_id
                or person_id
            )

            if visitor_key:
                visitor_sets[key].add(
                    str(visitor_key)
                )

            duration = self._number(
                event.get("duration_seconds")
                or event.get("carry_seconds"),
                0,
            )

            if event_type == "carry":
                stats[key]["carry_seconds"] += duration

        for key in stats:
            stats[key]["visitors"] = len(
                visitor_sets[key]
            )

            carry_score = min(
                stats[key]["carry_seconds"] / 10.0,
                100.0,
            )

            stats[key]["attraction_score"] = (
                stats[key]["approaches"] * 1.0
                + stats[key]["inspects"] * 2.0
                + stats[key]["picks"] * 4.0
                + stats[key]["carries"] * 5.0
                + carry_score
                - stats[key]["returns"] * 1.0
            )

        return dict(stats)

    # ------------------------------------------------------------------
    # ENTRY / EXIT PREFERENCE
    # ------------------------------------------------------------------

    def entry_exit_preferences(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Determine what anonymous visitors interact with near entry
        and near exit.

        Instead of assuming that the first database event is an
        entry preference, events are classified using the elapsed
        position within the visit.

        First twenty percent  -> entry preference
        Last twenty percent   -> exit preference
        """

        result = {
            "entry": [],
            "exit": [],
        }

        visit_map = {
            str(v.get("id")): v
            for v in self.visits
            if v.get("id") is not None
        }

        for event in self.events:
            visit_id = event.get("visit_id")

            if not visit_id:
                continue

            visit = visit_map.get(
                str(visit_id)
            )

            if not visit:
                continue

            entered = self._parse_time(
                visit.get("entered_at")
                or visit.get("entry_time")
                or visit.get("created_at")
            )

            exited = self._parse_time(
                visit.get("exited_at")
                or visit.get("exit_time")
            )

            event_time = self._parse_time(
                event.get("started_at")
                or event.get("created_at")
                or event.get("event_time")
                or event.get("timestamp")
            )

            if not entered or not event_time:
                continue

            if exited and exited > entered:
                total_seconds = (
                    exited - entered
                ).total_seconds()

                elapsed_seconds = (
                    event_time - entered
                ).total_seconds()

                if total_seconds <= 0:
                    continue

                position = (
                    elapsed_seconds
                    / total_seconds
                )

                if position <= 0.20:
                    result["entry"].append(event)

                elif position >= 0.80:
                    result["exit"].append(event)

            else:
                # If a visit is still active, only classify early
                # events. Exit preference will be calculated after exit.
                elapsed_seconds = (
                    event_time - entered
                ).total_seconds()

                if elapsed_seconds <= 60:
                    result["entry"].append(event)

        return {
            "entry": self._rank_events(
                result["entry"]
            ),
            "exit": self._rank_events(
                result["exit"]
            ),
        }

    def _rank_events(
        self,
        events: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        stats = defaultdict(
            lambda: {
                "events": 0,
                "visitors": set(),
                "picks": 0,
                "inspects": 0,
                "approaches": 0,
                "carries": 0,
            }
        )

        for event in events:
            key = self._product_key(event)

            event_type = str(
                event.get("event_type")
                or event.get("interaction_type")
                or ""
            ).lower()

            stats[key]["events"] += 1

            visit_id = event.get("visit_id")

            if visit_id:
                stats[key]["visitors"].add(
                    str(visit_id)
                )

            if event_type in stats[key]:
                stats[key][event_type + "s"] += 1

        ranking = []

        for key, values in stats.items():
            ranking.append(
                {
                    "product_id": key,
                    "product_name": self._product_name(key),
                    "events": values["events"],
                    "visitors": len(values["visitors"]),
                    "picks": values["picks"],
                    "inspects": values["inspects"],
                    "approaches": values["approaches"],
                    "carries": values["carries"],
                }
            )

        ranking.sort(
            key=lambda x: (
                x["picks"] * 4
                + x["inspects"] * 2
                + x["approaches"]
                + x["carries"] * 5
            ),
            reverse=True,
        )

        return ranking

    # ------------------------------------------------------------------
    # SALES
    # ------------------------------------------------------------------

    def sales_by_product(self) -> Dict[str, int]:
        result = defaultdict(int)

        for sale in self.sales:
            product_id = (
                sale.get("product_id")
                or sale.get("observed_product_id")
            )

            if not product_id:
                continue

            quantity = int(
                self._number(
                    sale.get("quantity"),
                    0,
                )
            )

            result[str(product_id)] += quantity

        return dict(result)

    def conversion_by_product(self) -> List[Dict[str, Any]]:
        attraction = self.attraction_scores()
        sales = self.sales_by_product()

        rows = []

        for key, stats in attraction.items():
            interactions = (
                stats["picks"]
                + stats["inspects"]
                + stats["approaches"]
            )

            units_sold = sales.get(
                key,
                0,
            )

            conversion = 0.0

            if interactions > 0:
                conversion = (
                    units_sold
                    / interactions
                    * 100.0
                )

            rows.append(
                {
                    "product_id": key,
                    "product_name": self._product_name(key),
                    "interactions": interactions,
                    "units_sold": units_sold,
                    "conversion_percent": round(
                        conversion,
                        2,
                    ),
                }
            )

        rows.sort(
            key=lambda x: x["conversion_percent"],
            reverse=True,
        )

        return rows

    # ------------------------------------------------------------------
    # VISITORS
    # ------------------------------------------------------------------

    def visitor_count(self) -> int:
        return len(self.visits)

    def visitor_summary(self) -> Dict[str, Any]:
        completed = 0
        durations = []

        for visit in self.visits:
            entered = self._parse_time(
                visit.get("entered_at")
            )

            exited = self._parse_time(
                visit.get("exited_at")
            )

            if entered and exited and exited > entered:
                completed += 1

                durations.append(
                    (
                        exited - entered
                    ).total_seconds()
                )

        average_duration = 0.0

        if durations:
            average_duration = (
                sum(durations)
                / len(durations)
            )

        return {
            "visitor_count": len(self.visits),
            "completed_visits": completed,
            "average_visit_seconds": round(
                average_duration,
                2,
            ),
        }

    # ------------------------------------------------------------------
    # LAYOUT RECOMMENDATIONS
    # ------------------------------------------------------------------

    def generate_recommendations(
        self,
    ) -> List[Dict[str, Any]]:

        recommendations = []

        scores = self.attraction_scores()

        if not scores:
            return recommendations

        ranking = sorted(
            scores.items(),
            key=lambda x: x[1]["attraction_score"],
            reverse=True,
        )

        # Highest attention product.
        top_key, top_stats = ranking[0]
        top_name = self._product_name(top_key)

        recommendations.append(
            {
                "recommendation_type": "high_attraction",
                "title": (
                    f"Place {top_name} in a high-visibility area"
                ),
                "recommendation": (
                    f"{top_name} currently has the strongest "
                    f"observed attraction score "
                    f"({top_stats['attraction_score']:.1f}). "
                    "Consider testing it near the entrance, "
                    "a main aisle, or another high-traffic display."
                ),
                "priority": "high",
                "evidence": {
                    "product": top_name,
                    **top_stats,
                },
            }
        )

        # Strong inspection but weak pickup.
        for key, stats in ranking:
            inspections = stats["inspects"]
            picks = stats["picks"]

            if (
                inspections >= 5
                and picks < inspections * 0.30
            ):
                name = self._product_name(key)

                recommendations.append(
                    {
                        "recommendation_type": "low_pickup",
                        "title": (
                            f"Improve presentation of {name}"
                        ),
                        "recommendation": (
                            f"Customers inspect {name} often, "
                            "but pickup activity is relatively low. "
                            "Test clearer pricing, better signage, "
                            "stronger packaging visibility, or "
                            "related products nearby."
                        ),
                        "priority": "medium",
                        "evidence": {
                            "product": name,
                            **stats,
                        },
                    }
                )

        # Frequently picked and returned.
        for key, stats in ranking:
            picks = stats["picks"]
            returns = stats["returns"]

            if (
                picks >= 3
                and returns >= picks * 0.50
            ):
                name = self._product_name(key)

                recommendations.append(
                    {
                        "recommendation_type": "high_return",
                        "title": (
                            f"Review display information for {name}"
                        ),
                        "recommendation": (
                            f"{name} is frequently picked up "
                            "and returned. Test clearer pricing, "
                            "product information, or a different "
                            "display position."
                        ),
                        "priority": "medium",
                        "evidence": {
                            "product": name,
                            **stats,
                        },
                    }
                )

        # Entry preference.
        preferences = self.entry_exit_preferences()

        if preferences["entry"]:
            top_entry = preferences["entry"][0]

            recommendations.append(
                {
                    "recommendation_type": "entry_preference",
                    "title": (
                        f"Feature {top_entry['product_name']} near entry"
                    ),
                    "recommendation": (
                        f"Anonymous visitors most often interact "
                        f"with {top_entry['product_name']} early in "
                        "their visits. Consider making this category "
                        "visible from the entrance."
                    ),
                    "priority": "high",
                    "evidence": top_entry,
                }
            )

        # Exit preference.
        if preferences["exit"]:
            top_exit = preferences["exit"][0]

            recommendations.append(
                {
                    "recommendation_type": "exit_preference",
                    "title": (
                        f"Test {top_exit['product_name']} near exit"
                    ),
                    "recommendation": (
                        f"{top_exit['product_name']} receives strong "
                        "late-visit attention. Consider a secondary "
                        "display, reminder, or related-product "
                        "placement closer to the checkout/exit area."
                    ),
                    "priority": "medium",
                    "evidence": top_exit,
                }
            )

        return recommendations

    # ------------------------------------------------------------------
    # DASHBOARD SUMMARY
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        preferences = self.entry_exit_preferences()

        return {
            "visitor_count": self.visitor_count(),
            "visitor_summary": self.visitor_summary(),
            "attraction_scores": self.attraction_scores(),
            "entry_preferences": preferences["entry"],
            "exit_preferences": preferences["exit"],
            "conversion": self.conversion_by_product(),
            "recommendations": self.generate_recommendations(),
        }

    # ------------------------------------------------------------------
    # NATURAL-LANGUAGE ASSISTANT
    # ------------------------------------------------------------------

    def answer(self, question: str) -> str:
        """
        Answer common store-owner questions without requiring an
        external paid LLM.
        """

        text = str(question or "").lower().strip()

        summary = self.summary()

        if any(
            word in text
            for word in (
                "attract",
                "attention",
                "popular",
                "prefer",
            )
        ):
            ranking = sorted(
                summary["attraction_scores"].items(),
                key=lambda x: x[1]["attraction_score"],
                reverse=True,
            )

            if not ranking:
                return (
                    "I do not have enough product-interaction "
                    "data yet."
                )

            key, stats = ranking[0]

            return (
                f"The most attractive product currently appears "
                f"to be {self._product_name(key)} with an attraction "
                f"score of {stats['attraction_score']:.1f}."
            )

        if any(
            word in text
            for word in (
                "entry",
                "enter",
                "entrance",
                "first",
            )
        ):
            entries = summary["entry_preferences"]

            if not entries:
                return (
                    "I do not have enough early-visit data yet "
                    "to determine entry preferences."
                )

            top = entries[0]

            return (
                f"Near entry, visitors most often interacted with "
                f"{top['product_name']}."
            )

        if any(
            word in text
            for word in (
                "exit",
                "leave",
                "leaving",
                "checkout",
            )
        ):
            exits = summary["exit_preferences"]

            if not exits:
                return (
                    "I do not have enough late-visit data yet "
                    "to determine exit preferences."
                )

            top = exits[0]

            return (
                f"Near exit, visitors most often interacted with "
                f"{top['product_name']}."
            )

        if any(
            word in text
            for word in (
                "layout",
                "arrange",
                "placement",
                "place",
                "display",
                "store arrangement",
            )
        ):
            recommendations = summary["recommendations"]

            if not recommendations:
                return (
                    "I need more customer-interaction data before "
                    "making a reliable layout recommendation."
                )

            top = recommendations[0]

            return (
                f"My first layout experiment would be: "
                f"{top['recommendation']}"
            )

        if any(
            word in text
            for word in (
                "sale",
                "sales",
                "purchase",
                "conversion",
            )
        ):
            conversion = summary["conversion"]

            if not conversion:
                return (
                    "There is not enough interaction and sales "
                    "data to calculate product conversion yet."
                )

            top = conversion[0]

            return (
                f"The strongest observed conversion is for "
                f"{top['product_name']}, at "
                f"{top['conversion_percent']:.2f}% based on the "
                "available interaction and sales records."
            )

        if any(
            word in text
            for word in (
                "recommend",
                "suggestion",
                "suggest",
                "advice",
            )
        ):
            recommendations = summary["recommendations"]

            if not recommendations:
                return (
                    "I need more visitor and product-interaction "
                    "data before generating strong recommendations."
                )

            return recommendations[0]["recommendation"]

        return (
            "I can analyse customer attraction, entry preferences, "
            "exit preferences, product interactions, conversion, "
            "and store-layout recommendations."
        )


__all__ = [
    "RetailAssistant",
]