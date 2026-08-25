from models.person_tracker import (
    CentroidPersonTracker,
    PersonDetection,
)


def test_person_receives_track_id():

    tracker = CentroidPersonTracker()

    detections = [
        PersonDetection(
            bbox=(100, 100, 50, 100),
            confidence=0.95,
        )
    ]

    tracks = tracker.update(detections)

    assert len(tracks) == 1
    assert tracks[0].track_id == 1


def test_same_person_keeps_track_id():

    tracker = CentroidPersonTracker(
        max_distance=100
    )

    first = tracker.update(
        [
            PersonDetection(
                bbox=(100, 100, 50, 100)
            )
        ]
    )

    second = tracker.update(
        [
            PersonDetection(
                bbox=(105, 105, 50, 100)
            )
        ]
    )

    assert first[0].track_id == second[0].track_id


def test_different_people_receive_different_ids():

    tracker = CentroidPersonTracker()

    tracks = tracker.update(
        [
            PersonDetection(
                bbox=(100, 100, 50, 100)
            ),
            PersonDetection(
                bbox=(500, 100, 50, 100)
            ),
        ]
    )

    ids = {
        track.track_id
        for track in tracks
    }

    assert len(ids) == 2