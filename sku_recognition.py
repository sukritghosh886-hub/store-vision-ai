from __future__ import annotations

import cv2
import numpy as np


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Convert an image into a normalized feature representation.
    """

    if image is None:
        raise ValueError("Image is None")

    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

    gray = cv2.resize(
        gray,
        (128, 128),
        interpolation=cv2.INTER_AREA,
    )

    # Histogram captures visual distribution.
    histogram = cv2.calcHist(
        [gray],
        [0],
        None,
        [64],
        [0, 256],
    )

    histogram = cv2.normalize(
        histogram,
        histogram,
    ).flatten()

    # Edge information captures shape/packaging structure.
    edges = cv2.Canny(
        gray,
        50,
        150,
    )

    edges = cv2.resize(
        edges,
        (32, 32),
    )

    edges = edges.astype(
        np.float32
    ) / 255.0

    return np.concatenate(
        [
            histogram,
            edges.flatten(),
        ]
    )


def compare_images(
    query: np.ndarray,
    reference: np.ndarray,
) -> float:

    query_features = preprocess_image(
        query
    )

    reference_features = preprocess_image(
        reference
    )

    # Cosine similarity
    denominator = (
        np.linalg.norm(query_features)
        * np.linalg.norm(reference_features)
    )

    if denominator == 0:
        return 0.0

    similarity = (
        np.dot(
            query_features,
            reference_features,
        )
        / denominator
    )

    return float(
        np.clip(
            similarity,
            0.0,
            1.0,
        )
    )


def recognize_sku(
    query_image: np.ndarray,
    references: list,
    threshold: float = 0.70,
):

    if not references:
        return {
            "matched": False,
            "confidence": 0.0,
            "message": "No SKU references available.",
        }

    best_reference = None
    best_score = 0.0

    for reference in references:

        reference_image = reference.get(
            "image"
        )

        if reference_image is None:
            continue

        score = compare_images(
            query_image,
            reference_image,
        )

        if score > best_score:
            best_score = score
            best_reference = reference

    if best_reference is None:
        return {
            "matched": False,
            "confidence": 0.0,
            "message": "No usable reference images.",
        }

    matched = (
        best_score >= threshold
    )

    return {
        "matched": matched,
        "confidence": round(
            best_score,
            4,
        ),
        "product_id":
            best_reference.get(
                "product_id"
            ),
        "sku":
            best_reference.get(
                "sku"
            ),
        "name":
            best_reference.get(
                "name"
            ),
        "category":
            best_reference.get(
                "category"
            ),
        "message":
            (
                "SKU match found."
                if matched
                else
                "No confident SKU match."
            ),
    }