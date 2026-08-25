from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from models.person_tracker import PersonTrack


Point = Tuple[int, int]


@dataclass
class CrossingEvent:
    track_id: int
    direction: str
    point: Point


class EntryExitLineCounter:
    """
    Detects when a tracked person crosses a virtual entrance line.

    The line is represented by two points.

    direction values:
        "entry"
        "exit"

    The algorithm uses the person's center point and compares
    which side of the line the person was on before and after
    the current frame.
    """

    def __init__(
        self,
        line_start: Point,
        line_end: Point,
        cooldown_frames: int = 20,
    ):
        self.line_start = line_start
        self.line_end = line_end
        self.cooldown_frames = cooldown_frames

        self.previous_side: Dict[int, int] = {}
        self.cooldown: Dict[int, int] = {}

    @staticmethod
    def _side(
        point: Point,
        line_start: Point,
        line_end: Point,
    ) -> int:

        px, py = point
        x1, y1 = line_start
        x2, y2 = line_end

        value = (
            (x2 - x1) * (py - y1)
            - (y2 - y1) * (px - x1)
        )

        if value > 0:
            return 1

        if value < 0:
            return -1

        return 0

    def reset(self) -> None:
        self.previous_side.clear()
        self.cooldown.clear()

    def update(
        self,
        tracks: Dict[int, PersonTrack],
    ) -> Optional[CrossingEvent]:

        for track_id in list(self.cooldown):
            self.cooldown[track_id] -= 1

            if self.cooldown[track_id] <= 0:
                del self.cooldown[track_id]

        for track_id, track in tracks.items():

            point = track.center

            current_side = self._side(
                point,
                self.line_start,
                self.line_end,
            )

            if current_side == 0:
                continue

            previous = self.previous_side.get(
                track_id
            )

            self.previous_side[track_id] = current_side

            if previous is None:
                continue

            if previous == current_side:
                continue

            if track_id in self.cooldown:
                continue

            direction = self._get_direction(
                previous,
                current_side,
            )

            self.cooldown[track_id] = (
                self.cooldown_frames
            )

            return CrossingEvent(
                track_id=track_id,
                direction=direction,
                point=point,
            )

        return None

    @staticmethod
    def _get_direction(
        previous_side: int,
        current_side: int,
    ) -> str:

        # Crossing from negative to positive is treated
        # as entry. Reverse it if your camera orientation
        # requires the opposite interpretation.

        if previous_side < current_side:
            return "entry"

        return "exit"