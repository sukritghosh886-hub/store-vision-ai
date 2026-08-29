from __future__ import annotations

import io
import requests

import cv2
import numpy as np


def load_sku_references(
    supabase,
    user_id: str,
):

    products_response = (
        supabase
        .table("products")
        .select(
            "id,name,sku,category"
        )
        .eq(
            "user_id",
            user_id,
        )
        .execute()
    )

    products = (
        products_response.data
        or []
    )

    references_response = (
        supabase
        .table("sku_references")
        .select(
            "id,product_id,image_path"
        )
        .eq(
            "user_id",
            user_id,
        )
        .execute()
    )

    references = (
        references_response.data
        or []
    )

    product_map = {
        product["id"]: product
        for product in products
    }

    result = []

    for reference in references:

        product = product_map.get(
            reference["product_id"]
        )

        if not product:
            continue

        image_url = reference.get(
            "image_path"
        )

        if not image_url:
            continue

        try:

            response = requests.get(
                image_url,
                timeout=10,
            )

            response.raise_for_status()

            array = np.frombuffer(
                response.content,
                dtype=np.uint8,
            )

            image = cv2.imdecode(
                array,
                cv2.IMREAD_COLOR,
            )

            if image is None:
                continue

            result.append(
                {
                    "reference_id":
                        reference["id"],

                    "product_id":
                        product["id"],

                    "name":
                        product["name"],

                    "sku":
                        product["sku"],

                    "category":
                        product.get(
                            "category"
                        ),

                    "image":
                        image,
                }
            )

        except Exception:
            continue

    return result