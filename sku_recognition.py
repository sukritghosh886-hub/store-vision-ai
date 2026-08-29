from __future__ import annotations

import hashlib
import io
from typing import Optional

import cv2
import numpy as np
from PIL import Image


BUCKET = "sku-references"


def _image_to_gray(image_bytes: bytes):

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    rgb = np.array(image)

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY,
    )

    return gray


def _feature_vector(image_bytes: bytes):

    gray = _image_to_gray(
        image_bytes
    )

    gray = cv2.resize(
        gray,
        (64, 64),
    )

    vector = gray.astype(
        np.float32
    )

    vector /= 255.0

    return vector.flatten()


def image_similarity(
    image_a: bytes,
    image_b: bytes,
) -> float:

    a = _feature_vector(
        image_a
    )

    b = _feature_vector(
        image_b
    )

    distance = np.linalg.norm(
        a - b
    )

    max_distance = np.sqrt(
        len(a)
    )

    score = 1.0 - (
        distance /
        max_distance
    )

    return float(
        max(
            0.0,
            min(
                1.0,
                score,
            ),
        )
    )


def reference_path(product_id: str):

    return (
        f"{product_id}/"
        f"reference.jpg"
    )


def save_reference(
    supabase,
    product_id: str,
    image_bytes: bytes,
):

    path = reference_path(
        product_id
    )

    supabase.storage.from_(
        BUCKET
    ).upload(
        path,
        image_bytes,
        {
            "content-type": "image/jpeg",
            "upsert": "true",
        },
    )

    return path


def get_reference_url(
    supabase,
    product_id: str,
) -> Optional[str]:

    path = reference_path(
        product_id
    )

    try:

        return (
            supabase.storage
            .from_(BUCKET)
            .get_public_url(path)
        )

    except Exception:

        return None


def recognize(
    query_image: bytes,
    references: list[dict],
    threshold: float = 0.75,
):

    best = None

    best_score = 0.0

    for reference in references:

        reference_bytes = reference.get(
            "image_bytes"
        )

        if not reference_bytes:
            continue

        score = image_similarity(
            query_image,
            reference_bytes,
        )

        if score > best_score:

            best_score = score

            best = reference

    if (
        best is None
        or best_score < threshold
    ):
        return None

    return {
        "product_id": best.get(
            "product_id"
        ),
        "sku": best.get(
            "sku"
        ),
        "name": best.get(
            "name"
        ),
        "confidence": round(
            best_score,
            3,
        ),
    }