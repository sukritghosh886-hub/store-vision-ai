from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ProductInterest:

    product: str

    views: int = 0

    interactions: int = 0

    purchases: int = 0


class RetailAssistant:

    """
    Retail intelligence engine.

    It calculates:

    - product views
    - interactions
    - purchases
    - interest score
    - conversion rate
    - display recommendations
    """

    def __init__(self):

        self.products: Dict[
            str,
            ProductInterest
        ] = {}

        self.visitor_count = 0

    # --------------------------------------------------
    # Visitor
    # --------------------------------------------------

    def register_visitor(self):

        self.visitor_count += 1

    # --------------------------------------------------
    # Get/create product
    # --------------------------------------------------

    def get_product(
        self,
        product_name: str
    ) -> ProductInterest:

        if product_name not in self.products:

            self.products[
                product_name
            ] = ProductInterest(
                product=product_name
            )

        return self.products[
            product_name
        ]

    # --------------------------------------------------
    # Product viewed
    # --------------------------------------------------

    def observe_product(
        self,
        product_name: str
    ):

        product = self.get_product(
            product_name
        )

        product.views += 1

    # --------------------------------------------------
    # Product interaction
    # --------------------------------------------------

    def register_interaction(
        self,
        product_name: str
    ):

        product = self.get_product(
            product_name
        )

        product.interactions += 1

    # --------------------------------------------------
    # Purchase
    # --------------------------------------------------

    def register_purchase(
        self,
        product_name: str
    ):

        product = self.get_product(
            product_name
        )

        product.purchases += 1

    # --------------------------------------------------
    # Interest score
    # --------------------------------------------------

    @staticmethod
    def calculate_interest_score(
        product: ProductInterest
    ) -> float:

        return (
            product.views * 0.2
            +
            product.interactions * 1.0
            +
            product.purchases * 2.0
        )

    # --------------------------------------------------
    # Conversion rate
    # --------------------------------------------------

    @staticmethod
    def conversion_rate(
        product: ProductInterest
    ) -> float:

        if product.interactions == 0:

            return 0.0

        return (
            product.purchases
            /
            product.interactions
        )

    # --------------------------------------------------
    # Ranking
    # --------------------------------------------------

    def get_rankings(
        self,
        limit: int = 10
    ) -> List[dict]:

        rankings = []

        for product in self.products.values():

            score = (
                self.calculate_interest_score(
                    product
                )
            )

            conversion = (
                self.conversion_rate(
                    product
                )
            )

            rankings.append({

                "product":
                    product.product,

                "views":
                    product.views,

                "interactions":
                    product.interactions,

                "purchases":
                    product.purchases,

                "interest_score":
                    round(
                        score,
                        2
                    ),

                "conversion_rate":
                    round(
                        conversion,
                        4
                    ),
            })

        rankings.sort(
            key=lambda x:
                x["interest_score"],
            reverse=True,
        )

        return rankings[:limit]

    # --------------------------------------------------
    # Generate recommendations
    # --------------------------------------------------

    def generate_recommendations(
        self
    ) -> List[str]:

        rankings = self.get_rankings(
            limit=5
        )

        if not rankings:

            return [
                "Not enough customer data yet."
            ]

        recommendations = []

        # ------------------------------------------
        # Most interesting product
        # ------------------------------------------

        top_product = rankings[0]

        recommendations.append(
            (
                f"Highlight "
                f"'{top_product['product']}' "
                f"because it currently has the "
                f"highest customer-interest score."
            )
        )

        # ------------------------------------------
        # High interaction / low purchase
        # ------------------------------------------

        for product in rankings:

            if (
                product["interactions"] >= 5
                and
                product["purchases"] == 0
            ):

                recommendations.append(
                    (
                        f"'{product['product']}' "
                        "gets customer interaction "
                        "but no recorded purchases. "
                        "Test its price, placement "
                        "or promotion."
                    )
                )

        # ------------------------------------------
        # Product pairing
        # ------------------------------------------

        if len(rankings) >= 2:

            recommendations.append(
                (
                    f"Test placing "
                    f"'{rankings[0]['product']}' "
                    f"near "
                    f"'{rankings[1]['product']}'."
                )
            )

        return recommendations

    # --------------------------------------------------
    # Dashboard summary
    # --------------------------------------------------

    def summary(self) -> dict:

        return {

            "visitors":
                self.visitor_count,

            "top_products":
                self.get_rankings(),

            "recommendations":
                self.generate_recommendations(),
        }