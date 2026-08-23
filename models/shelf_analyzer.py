from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass
class ShelfZone:
    """
    Represents one shelf zone.
    """

    row: int
    column: int
    occupancy: float
    status: str


class ShelfAnalyzer:
    """
    Analyses shelf occupancy using image brightness and edges.

    This is a baseline heuristic system.
    A trained segmentation/detection model can replace it later.
    """

    def __init__(
        self,
        rows: int = 4,
        columns: int = 6,
    ):
        self.rows = rows
        self.columns = columns

    def analyse(self, image: np.ndarray) -> List[ShelfZone]:
        """
        Divide the image into a grid and estimate occupancy
        for each zone.
        """

        if image is None:
            return []

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        height, width = gray.shape

        cell_height = height // self.rows
        cell_width = width // self.columns

        zones = []

        for row in range(self.rows):
            for column in range(self.columns):

                y1 = row * cell_height
                y2 = (
                    height
                    if row == self.rows - 1
                    else (row + 1) * cell_height
                )

                x1 = column * cell_width
                x2 = (
                    width
                    if column == self.columns - 1
                    else (column + 1) * cell_width
                )

                cell = gray[y1:y2, x1:x2]

                if cell.size == 0:
                    continue

                # Edge density
                edges = cv2.Canny(
                    cell,
                    50,
                    150,
                )

                edge_density = np.mean(edges > 0)

                # Standard deviation measures visual variation.
                variation = float(np.std(cell))

                # Combine signals.
                occupancy = (
                    edge_density * 0.7
                    + min(variation / 100.0, 1.0) * 0.3
                )

                occupancy = float(
                    np.clip(occupancy, 0.0, 1.0)
                )

                if occupancy < 0.15:
                    status = "Empty"
                elif occupancy < 0.35:
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
                    )
                )

        return zones

    def summary(self, zones: List[ShelfZone]) -> dict:
        """
        Generate high-level shelf statistics.
        """

        if not zones:
            return {
                "average_occupancy": 0.0,
                "empty_zones": 0,
                "low_zones": 0,
                "medium_zones": 0,
                "high_zones": 0,
            }

        average = sum(
            zone.occupancy for zone in zones
        ) / len(zones)

        return {
            "average_occupancy": average,
            "empty_zones": sum(
                zone.status == "Empty"
                for zone in zones
            ),
            "low_zones": sum(
                zone.status == "Low"
                for zone in zones
            ),
            "medium_zones": sum(
                zone.status == "Medium"
                for zone in zones
            ),
            "high_zones": sum(
                zone.status == "High"
                for zone in zones
            ),
        }

    def create_heatmap(
        self,
        zones: List[ShelfZone],
    ) -> np.ndarray:
        """
        Convert zone occupancy into a 2D heatmap array.
        """

        heatmap = np.zeros(
            (self.rows, self.columns),
            dtype=np.float32,
        )

        for zone in zones:
            heatmap[
                zone.row - 1,
                zone.column - 1,
            ] = zone.occupancy

        return heatmap