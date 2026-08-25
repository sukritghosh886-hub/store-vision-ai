from dataclasses import dataclass
from typing import Dict, List, Tuple

from models.person_tracker import PersonTrack


Point = Tuple[int, int]


@dataclass
class CrossingEvent:
    track_id: int
    direction: str
    point: Point


class EntryExitLineCounter:
    """
    Detects multiple people crossing a virtual entrance/exit line.

    The counter keeps the previous side of the line for every
    tracked person and generates an event whenever that person
    changes sides.

    Direction:
        negative -> positive = entry
        positive -> negative = exit

    If your camera is mounted in the opposite orientation,
    use reverse_direction=True.
    """

    def __init__(
        self,
        line_start: Point,
        line_end: Point,
        cooldown_frames: int = 20,
        reverse_direction: bool = False,
    ):
        self.line_start = line_start
        self.line_end = line_end
        self.cooldown_frames = max(0, cooldown_frames)
        self.reverse_direction = reverse_direction

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

    def remove_track(self, track_id: int) -> None:
        """
        Remove tracking state for a person who has disappeared.
        """
        self.previous_side.pop(track_id, None)
        self.cooldown.pop(track_id, None)

    def update(
        self,
        tracks: Dict[int, PersonTrack],
    ) -> List[CrossingEvent]:
        """
        Process all tracks in the current frame.

        Returns every crossing detected in this frame.
        """
        events: List[CrossingEvent] = []

        # Decrease cooldown counters.
        for track_id in list(self.cooldown.keys()):
            self.cooldown[track_id] -= 1

            if self.cooldown[track_id] <= 0:
                del self.cooldown[track_id]

        active_ids = set(tracks.keys())

        # Remove stale tracking state.
        stale_ids = (
            set(self.previous_side.keys())
            - active_ids
        )

        for track_id in stale_ids:
            self.remove_track(track_id)

        # Process EVERY person instead of returning after
        # the first crossing.
        for track_id, track in tracks.items():

            point = track.center

            current_side = self._side(
                point,
                self.line_start,
                self.line_end,
            )

            # Ignore points exactly on the line.
            if current_side == 0:
                continue

            previous_side = self.previous_side.get(
                track_id
            )

            self.previous_side[track_id] = current_side

            # First observation of this person.
            if previous_side is None:
                continue

            # Person did not cross the line.
            if previous_side == current_side:
                continue

            # Ignore repeated crossing during cooldown.
            if track_id in self.cooldown:
                continue

            direction = self._get_direction(
                previous_side,
                current_side,
            )

            if self.reverse_direction:
                direction = (
                    "exit"
                    if direction == "entry"
                    else "entry"
                )

            self.cooldown[track_id] = (
                self.cooldown_frames
            )

            events.append(
                CrossingEvent(
                    track_id=track_id,
                    direction=direction,
                    point=point,
                )
            )

        return events

    @staticmethod
    def _get_direction(
        previous_side: int,
        current_side: int,
    ) -> str:

        if previous_side < current_side:
            return "entry"

        return "exit"