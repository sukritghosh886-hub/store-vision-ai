from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass
class ShelfZone:

    row: int
    column: int

    occupancy: float

    status: str

    bbox: tuple


class ShelfAnalyzer:

    def __init__(
        self,
        rows: int = 4,
        columns: int = 6,
        empty_threshold: float = 0.15,
        low_threshold: float = 0.35,
    ):

        self.rows = rows
        self.columns = columns

        self.empty_threshold = (
            empty_threshold
        )

        self.low_threshold = (
            low_threshold
        )


    def analyse(
        self,
        image: np.ndarray,
    ) -> List[ShelfZone]:

        if image is None:
            return []

        if len(image.shape) == 2:

            gray = image

        else:

            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY,
            )

        height, width = gray.shape

        cell_height = (
            height / self.rows
        )

        cell_width = (
            width / self.columns
        )

        zones = []

        for row in range(
            self.rows
        ):

            for column in range(
                self.columns
            ):

                x1 = int(
                    column * cell_width
                )

                x2 = int(
                    (column + 1)
                    * cell_width
                )

                y1 = int(
                    row * cell_height
                )

                y2 = int(
                    (row + 1)
                    * cell_height
                )

                cell = gray[
                    y1:y2,
                    x1:x2,
                ]

                if cell.size == 0:
                    continue

                edges = cv2.Canny(
                    cell,
                    50,
                    150,
                )

                edge_density = float(
                    np.mean(
                        edges > 0
                    )
                )

                variation = float(
                    np.std(cell)
                )

                occupancy = (
                    edge_density * 0.7
                    +
                    min(
                        variation / 100,
                        1.0,
                    ) * 0.3
                )

                occupancy = float(
                    np.clip(
                        occupancy,
                        0,
                        1,
                    )
                )

                if (
                    occupancy
                    < self.empty_threshold
                ):

                    status = "Empty"

                elif (
                    occupancy
                    < self.low_threshold
                ):

                    status = "Low"

                elif occupancy < 0.65:

                    status = "Medium"

                else:

                    status = "High"

                zones.append(
                    ShelfZone(
                        row=row + 1,
                        column=column + 1,
                        occupancy=occupancy,
                        status=status,
                        bbox=(
                            x1,
                            y1,
                            x2,
                            y2,
                        ),
                    )
                )

        return zones


    def summary(
        self,
        zones: List[ShelfZone],
    ):

        if not zones:

            return {
                "total_zones": 0,
                "average_occupancy": 0,
                "empty_zones": 0,
                "low_zones": 0,
                "medium_zones": 0,
                "high_zones": 0,
            }

        return {

            "total_zones":
                len(zones),

            "average_occupancy":
                sum(
                    z.occupancy
                    for z in zones
                ) / len(zones),

            "empty_zones":
                sum(
                    z.status == "Empty"
                    for z in zones
                ),

            "low_zones":
                sum(
                    z.status == "Low"
                    for z in zones
                ),

            "medium_zones":
                sum(
                    z.status == "Medium"
                    for z in zones
                ),

            "high_zones":
                sum(
                    z.status == "High"
                    for z in zones
                ),
        }


    def draw(
        self,
        image: np.ndarray,
        zones: List[ShelfZone],
    ):

        output = image.copy()

        for zone in zones:

            x1, y1, x2, y2 = zone.bbox

            if zone.status == "Empty":

                color = (
                    0,
                    0,
                    255,
                )

            elif zone.status == "Low":

                color = (
                    0,
                    165,
                    255,
                )

            else:

                color = (
                    0,
                    200,
                    0,
                )

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

            label = (
                f"R{zone.row} "
                f"C{zone.column} "
                f"{zone.status} "
                f"{zone.occupancy:.0%}"
            )

            cv2.putText(
                output,
                label,
                (x1 + 4, y1 + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

        return output