from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math


BBox = Tuple[int, int, int, int]


@dataclass
class PersonDetection:
    """
    One detected person.

    bbox:
        (x, y, width, height)

    confidence:
        Detector confidence between 0 and 1.
    """

    bbox: BBox
    confidence: float = 1.0


@dataclass
class PersonTrack:
    """
    Persistent track for one person.
    """

    track_id: int
    bbox: BBox
    confidence: float
    missed_frames: int = 0

    @property
    def center(self) -> Tuple[int, int]:
        x, y, w, h = self.bbox
        return (
            x + w // 2,
            y + h // 2,
        )


def _center(bbox: BBox) -> Tuple[int, int]:
    x, y, w, h = bbox
    return (
        x + w // 2,
        y + h // 2,
    )


def _distance(a: BBox, b: BBox) -> float:
    ax, ay = _center(a)
    bx, by = _center(b)

    return math.hypot(
        ax - bx,
        ay - by,
    )


class CentroidPersonTracker:
    """
    Lightweight multi-person tracker.

    This class intentionally separates tracking from detection.

    A production detector such as YOLO can later provide
    PersonDetection objects without changing the tracking
    and entry/exit logic.
    """

    def __init__(
        self,
        max_distance: float = 100.0,
        max_missed_frames: int = 15,
    ):
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames

        self._next_id = 1
        self._tracks: Dict[int, PersonTrack] = {}

    @property
    def tracks(self) -> Dict[int, PersonTrack]:
        return dict(self._tracks)

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def _new_track(
        self,
        detection: PersonDetection,
    ) -> PersonTrack:

        track = PersonTrack(
            track_id=self._next_id,
            bbox=detection.bbox,
            confidence=detection.confidence,
        )

        self._tracks[track.track_id] = track
        self._next_id += 1

        return track

    def update(
        self,
        detections: List[PersonDetection],
    ) -> List[PersonTrack]:

        if not detections:
            expired = []

            for track_id, track in self._tracks.items():
                track.missed_frames += 1

                if track.missed_frames > self.max_missed_frames:
                    expired.append(track_id)

            for track_id in expired:
                del self._tracks[track_id]

            return list(self._tracks.values())

        existing_ids = set(self._tracks.keys())
        unmatched_detections = set(range(len(detections)))
        matched_tracks = set()

        candidates = []

        for track_id in existing_ids:
            track = self._tracks[track_id]

            for index, detection in enumerate(detections):
                distance = _distance(
                    track.bbox,
                    detection.bbox,
                )

                if distance <= self.max_distance:
                    candidates.append(
                        (
                            distance,
                            track_id,
                            index,
                        )
                    )

        candidates.sort(key=lambda item: item[0])

        for _, track_id, detection_index in candidates:

            if track_id in matched_tracks:
                continue

            if detection_index not in unmatched_detections:
                continue

            detection = detections[detection_index]
            track = self._tracks[track_id]

            track.bbox = detection.bbox
            track.confidence = detection.confidence
            track.missed_frames = 0

            matched_tracks.add(track_id)
            unmatched_detections.remove(detection_index)

        for track_id in existing_ids - matched_tracks:
            track = self._tracks[track_id]
            track.missed_frames += 1

        expired = [
            track_id
            for track_id, track in self._tracks.items()
            if track.missed_frames > self.max_missed_frames
        ]

        for track_id in expired:
            del self._tracks[track_id]

        for detection_index in unmatched_detections:
            self._new_track(
                detections[detection_index]
            )

        return list(self._tracks.values())