from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List


class RetailAssistant:
    """
    Data-grounded retail intelligence assistant.

    The assistant never invents inventory or sales numbers.
    All business answers are calculated from Supabase records.
    """

    def __init__(self, supabase):
        self.supabase = supabase

    # -----------------------------------------------------
    # Database helpers
    # -----------------------------------------------------

    def _get_products(self) -> List[Dict[str, Any]]:
        response = (
            self.supabase
            .table("products")
            .select("*")
            .order("name")
            .execute()
        )

        return response.data or []

    def _get_sales(self) -> List[Dict[str, Any]]:
        response = (
            self.supabase
            .table("sales")
            .select("*")
            .order("created_at", desc=True)
            .limit(1000)
            .execute()
        )

        return response.data or []

    def _get_alerts(self) -> List[Dict[str, Any]]:
        response = (
            self.supabase
            .table("alerts")
            .select("*")
            .order("created_at", desc=True)
            .limit(100)
            .execute()
        )

        return response.data or []

    def _get_shelf_scans(self) -> List[Dict[str, Any]]:
        response = (
            self.supabase
            .table("shelf_scans")
            .select("*")
            .order("created_at", desc=True)
            .limit(100)
            .execute()
        )

        return response.data or []

    def _get_visits(self) -> List[Dict[str, Any]]:
        response = (
            self.supabase
            .table("visits")
            .select("*")
            .order("entered_at", desc=True)
            .limit(1000)
            .execute()
        )

        return response.data or []

    # -----------------------------------------------------
    # Inventory intelligence
    # -----------------------------------------------------

    def low_stock_products(self):
        products = self._get_products()

        low_stock = []

        for product in products:

            stock = int(
                product.get(
                    "stock_quantity",
                    0,
                )
                or 0
            )

            minimum = int(
                product.get(
                    "minimum_stock",
                    0,
                )
                or 0
            )

            if stock <= minimum:

                low_stock.append(
                    {
                        "name": product.get(
                            "name",
                            "Unknown",
                        ),
                        "sku": product.get(
                            "sku",
                            "N/A",
                        ),
                        "stock": stock,
                        "minimum": minimum,
                        "shortage": max(
                            minimum - stock,
                            0,
                        ),
                    }
                )

        return low_stock

    # -----------------------------------------------------
    # Sales intelligence
    # -----------------------------------------------------

    def sales_summary(self):

        sales = self._get_sales()

        revenue = 0.0
        units = 0

        for sale in sales:

            quantity = int(
                sale.get(
                    "quantity",
                    0,
                )
                or 0
            )

            amount = float(
                sale.get(
                    "total_amount",
                    0,
                )
                or 0
            )

            units += quantity
            revenue += amount

        return {
            "transactions": len(sales),
            "units": units,
            "revenue": revenue,
        }

    # -----------------------------------------------------
    # Top products
    # -----------------------------------------------------

    def top_products(self, limit=5):

        sales = self._get_sales()

        quantities = defaultdict(int)
        revenue = defaultdict(float)

        for sale in sales:

            product_id = sale.get(
                "product_id"
            )

            if not product_id:
                continue

            quantities[product_id] += int(
                sale.get(
                    "quantity",
                    0,
                )
                or 0
            )

            revenue[product_id] += float(
                sale.get(
                    "total_amount",
                    0,
                )
                or 0
            )

        products = self._get_products()

        names = {
            p.get("id"): p
            for p in products
        }

        ranking = []

        for product_id, quantity in quantities.items():

            product = names.get(
                product_id,
                {},
            )

            ranking.append(
                {
                    "name": product.get(
                        "name",
                        "Unknown product",
                    ),
                    "sku": product.get(
                        "sku",
                        "N/A",
                    ),
                    "units": quantity,
                    "revenue": revenue[
                        product_id
                    ],
                }
            )

        ranking.sort(
            key=lambda x: x["units"],
            reverse=True,
        )

        return ranking[:limit]

    # -----------------------------------------------------
    # Alert intelligence
    # -----------------------------------------------------

    def open_alerts(self):

        alerts = self._get_alerts()

        return [
            alert
            for alert in alerts
            if str(
                alert.get(
                    "status",
                    "",
                )
            ).lower()
            == "open"
        ]

    # -----------------------------------------------------
    # Shelf intelligence
    # -----------------------------------------------------

    def shelf_summary(self):

        scans = self._get_shelf_scans()

        if not scans:

            return {
                "scans": 0,
                "empty": 0,
                "occupied": 0,
                "status": "No shelf scans available.",
            }

        latest = scans[0]

        empty = int(
            latest.get(
                "empty_slots",
                0,
            )
            or 0
        )

        occupied = int(
            latest.get(
                "occupied_slots",
                0,
            )
            or 0
        )

        return {
            "scans": len(scans),
            "empty": empty,
            "occupied": occupied,
            "status": latest.get(
                "stock_status",
                "unknown",
            ),
        }

    # -----------------------------------------------------
    # Visitor intelligence
    # -----------------------------------------------------

    def visitor_summary(self):

        visits = self._get_visits()

        entries = len(visits)

        completed = sum(
            1
            for visit in visits
            if visit.get("exited_at")
        )

        active = entries - completed

        return {
            "visits": entries,
            "completed": completed,
            "active": max(
                active,
                0,
            ),
        }

    # -----------------------------------------------------
    # Reorder recommendations
    # -----------------------------------------------------

    def reorder_recommendations(
        self,
        limit=5,
    ):

        products = self._get_products()

        recommendations = []

        for product in products:

            stock = int(
                product.get(
                    "stock_quantity",
                    0,
                )
                or 0
            )

            minimum = int(
                product.get(
                    "minimum_stock",
                    0,
                )
                or 0
            )

            if stock <= minimum:

                shortage = max(
                    minimum - stock,
                    0,
                )

                recommendations.append(
                    {
                        "name": product.get(
                            "name",
                            "Unknown",
                        ),
                        "sku": product.get(
                            "sku",
                            "N/A",
                        ),
                        "stock": stock,
                        "minimum": minimum,
                        "priority": (
                            "CRITICAL"
                            if stock == 0
                            else "HIGH"
                            if shortage >= 5
                            else "MEDIUM"
                        ),
                    }
                )

        priority_order = {
            "CRITICAL": 0,
            "HIGH": 1,
            "MEDIUM": 2,
        }

        recommendations.sort(
            key=lambda x: (
                priority_order.get(
                    x["priority"],
                    99,
                ),
                -x["minimum"],
            )
        )

        return recommendations[:limit]

    # -----------------------------------------------------
    # Natural language interface
    # -----------------------------------------------------

    def answer(self, question: str):

        q = question.lower().strip()

        if not q:

            return (
                "Please ask a retail question."
            )

        # Low stock
        if (
            "low stock" in q
            or "out of stock" in q
            or "stock low" in q
        ):

            products = (
                self.low_stock_products()
            )

            if not products:

                return (
                    "Good news — no products "
                    "are currently below their "
                    "minimum-stock threshold."
                )

            lines = [
                f"- {p['name']} "
                f"({p['sku']}): "
                f"{p['stock']} units remaining "
                f"(minimum {p['minimum']})"
                for p in products
            ]

            return (
                f"I found {len(products)} "
                "low-stock product(s):\n\n"
                + "\n".join(lines)
            )

        # Reorder
        if (
            "reorder" in q
            or "restock" in q
            or "what should i buy" in q
            or "what should i order" in q
        ):

            recommendations = (
                self.reorder_recommendations()
            )

            if not recommendations:

                return (
                    "No immediate reorder "
                    "recommendations."
                )

            lines = [
                f"- **{r['name']}** "
                f"({r['sku']}) — "
                f"{r['priority']} priority, "
                f"{r['stock']} units available "
                f"vs minimum {r['minimum']}"
                for r in recommendations
            ]

            return (
                "Recommended reorder priorities:\n\n"
                + "\n".join(lines)
            )

        # Sales
        if (
            "sales" in q
            or "revenue" in q
            or "sold" in q
            or "earning" in q
        ):

            summary = (
                self.sales_summary()
            )

            return (
                f"Based on the sales records currently "
                f"available, there are "
                f"**{summary['transactions']} transactions**, "
                f"**{summary['units']} units sold**, "
                f"and **₹{summary['revenue']:,.2f} revenue**."
            )

        # Top products
        if (
            "best selling" in q
            or "top product" in q
            or "top products" in q
            or "popular product" in q
        ):

            products = (
                self.top_products()
            )

            if not products:

                return (
                    "There is not enough sales "
                    "data to rank products."
                )

            lines = [
                f"{i}. **{p['name']}** "
                f"({p['sku']}) — "
                f"{p['units']} units, "
                f"₹{p['revenue']:,.2f} revenue"
                for i, p in enumerate(
                    products,
                    start=1,
                )
            ]

            return (
                "Top-selling products:\n\n"
                + "\n".join(lines)
            )

        # Alerts
        if (
            "alert" in q
            or "security" in q
            or "warning" in q
        ):

            alerts = self.open_alerts()

            if not alerts:

                return (
                    "There are currently "
                    "**no open alerts**."
                )

            return (
                f"There are **{len(alerts)} "
                "open alert(s)** requiring review."
            )

        # Shelf
        if (
            "shelf" in q
            or "empty" in q
            or "stock-out" in q
            or "stockout" in q
        ):

            summary = (
                self.shelf_summary()
            )

            if summary["scans"] == 0:

                return (
                    "No shelf scans are currently "
                    "available."
                )

            return (
                f"The latest shelf analysis reports "
                f"**{summary['empty']} empty zone(s)** "
                f"and **{summary['occupied']} occupied "
                f"zone(s)**. "
                f"Overall status: "
                f"**{summary['status']}**."
            )

        # Visitors
        if (
            "visitor" in q
            or "customer" in q
            or "people" in q
            or "visits" in q
        ):

            summary = (
                self.visitor_summary()
            )

            return (
                f"Visitor records show "
                f"**{summary['visits']} visits**, "
                f"with **{summary['completed']} completed** "
                f"and approximately **{summary['active']} "
                "currently active** visits."
            )

        # Overview
        if (
            "overview" in q
            or "summary" in q
            or "dashboard" in q
            or "what is happening" in q
            or "business" in q
        ):

            products = self._get_products()

            low = self.low_stock_products()

            sales = self.sales_summary()

            alerts = self.open_alerts()

            shelf = self.shelf_summary()

            return (
                "### Retail Intelligence Overview\n\n"
                f"- **Products:** {len(products)}\n"
                f"- **Low-stock products:** {len(low)}\n"
                f"- **Sales transactions:** "
                f"{sales['transactions']}\n"
                f"- **Units sold:** {sales['units']}\n"
                f"- **Revenue:** ₹{sales['revenue']:,.2f}\n"
                f"- **Open alerts:** {len(alerts)}\n"
                f"- **Latest empty shelf zones:** "
                f"{shelf['empty']}\n"
            )

        # Help
        return (
            "I can answer questions about your retail data.\n\n"
            "Try:\n"
            "- Which products are low in stock?\n"
            "- What should I reorder?\n"
            "- What are my sales and revenue?\n"
            "- Which products are best selling?\n"
            "- Do I have any open alerts?\n"
            "- Are there empty shelves?\n"
            "- How many visitors have we tracked?\n"
            "- Give me a business overview."
        )