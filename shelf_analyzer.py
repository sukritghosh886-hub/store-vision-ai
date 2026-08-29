from __future__ import annotations

import cv2
import numpy as np


def analyze_shelf(
    image,
    rows: int = 2,
    columns: int = 6,
):

    height, width = image.shape[:2]

    slot_width = width / columns

    slot_height = height / rows

    slots = []

    output = image.copy()

    for row in range(rows):

        for column in range(columns):

            x1 = int(
                column * slot_width
            )

            x2 = int(
                (column + 1)
                * slot_width
            )

            y1 = int(
                row * slot_height
            )

            y2 = int(
                (row + 1)
                * slot_height
            )

            crop = image[
                y1:y2,
                x1:x2,
            ]

            gray = cv2.cvtColor(
                crop,
                cv2.COLOR_BGR2GRAY,
            )

            brightness = float(
                np.mean(gray)
            )

            texture = float(
                np.var(gray)
            )

            occupied = (
                texture > 350
                and brightness < 245
            )

            status = (
                "occupied"
                if occupied
                else "empty"
            )

            slots.append(
                {
                    "row": row + 1,
                    "column": column + 1,
                    "status": status,
                }
            )

            color = (
                (0, 200, 0)
                if occupied
                else (0, 0, 255)
            )

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

            cv2.putText(
                output,
                status,
                (x1 + 5, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

    occupied = sum(
        slot["status"] == "occupied"
        for slot in slots
    )

    empty = sum(
        slot["status"] == "empty"
        for slot in slots
    )

    if empty > 0:
        stock_status = "attention"
    else:
        stock_status = "healthy"

    return {
        "image": output,
        "slots": slots,
        "total_slots": len(slots),
        "occupied_slots": occupied,
        "empty_slots": empty,
        "stock_status": stock_status,
    }