from __future__ import annotations

import cv2
import numpy as np


def image_embedding(image):

    resized = cv2.resize(
        image,
        (64, 64),
    )

    gray = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2GRAY,
    )

    vector = gray.astype(
        np.float32
    )

    vector /= 255.0

    return vector.flatten()


def compare_images(
    query,
    reference,
):

    a = image_embedding(query)

    b = image_embedding(reference)

    distance = np.linalg.norm(
        a - b
    )

    maximum = np.sqrt(
        len(a)
    )

    return max(
        0.0,
        1.0 - (
            distance / maximum
        ),
    )


def recognize_sku(
    query_image,
    references,
    threshold=0.70,
):

    best = None

    best_score = 0.0

    for reference in references:

        score = compare_images(
            query_image,
            reference["image"],
        )

        if score > best_score:

            best_score = score

            best = reference

    if best is None:
        return None

    if best_score < threshold:
        return None

    return {
        "product_id":
            best["product_id"],

        "sku":
            best["sku"],

        "name":
            best["name"],

        "confidence":
            round(
                best_score,
                3,
            ),
    }