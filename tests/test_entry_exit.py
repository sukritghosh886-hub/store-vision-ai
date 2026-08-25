from models.person_tracker import PersonTrack
from vision.line_counter import EntryExitLineCounter


def make_track(track_id, x, y):

    return PersonTrack(
        track_id=track_id,
        bbox=(x, y, 50, 100),
        confidence=0.95,
    )


def test_entry_crossing():

    counter = EntryExitLineCounter(
        line_start=(0, 200),
        line_end=(500, 200),
    )

    # First position above the line.
    counter.update(
        {
            1: make_track(
                1,
                100,
                100,
            )
        }
    )

    # Move below the line.
    event = counter.update(
        {
            1: make_track(
                1,
                100,
                250,
            )
        }
    )

    assert event is not None
    assert event.track_id == 1
    assert event.direction == "entry"


def test_no_event_without_crossing():

    counter = EntryExitLineCounter(
        line_start=(0, 200),
        line_end=(500, 200),
    )

    counter.update(
        {
            1: make_track(
                1,
                100,
                100,
            )
        }
    )

    event = counter.update(
        {
            1: make_track(
                1,
                100,
                120,
            )
        }
    )

    assert event is None