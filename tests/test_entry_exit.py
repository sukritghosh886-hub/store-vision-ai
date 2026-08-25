from models.person_tracker import PersonTrack
from vision.line_counter import EntryExitLineCounter


def make_track(
    track_id: int,
    x: int,
    y: int,
) -> PersonTrack:

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

    counter.update(
        {
            1: make_track(
                1,
                100,
                50,
            )
        }
    )

    events = counter.update(
        {
            1: make_track(
                1,
                100,
                250,
            )
        }
    )

    assert len(events) == 1
    assert events[0].track_id == 1
    assert events[0].direction == "entry"


def test_exit_crossing():

    counter = EntryExitLineCounter(
        line_start=(0, 200),
        line_end=(500, 200),
    )

    counter.update(
        {
            1: make_track(
                1,
                100,
                250,
            )
        }
    )

    events = counter.update(
        {
            1: make_track(
                1,
                100,
                50,
            )
        }
    )

    assert len(events) == 1
    assert events[0].track_id == 1
    assert events[0].direction == "exit"


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
                50,
            )
        }
    )

    events = counter.update(
        {
            1: make_track(
                1,
                100,
                100,
            )
        }
    )

    assert events == []


def test_multiple_people_crossing_same_frame():

    counter = EntryExitLineCounter(
        line_start=(0, 200),
        line_end=(500, 200),
    )

    # Two people start on the same side.
    counter.update(
        {
            1: make_track(1, 100, 50),
            2: make_track(2, 300, 50),
        }
    )

    # Both cross the line in the same frame.
    events = counter.update(
        {
            1: make_track(1, 100, 250),
            2: make_track(2, 300, 250),
        }
    )

    assert len(events) == 2

    track_ids = {
        event.track_id
        for event in events
    }

    assert track_ids == {1, 2}


def test_cooldown_prevents_duplicate_event():

    counter = EntryExitLineCounter(
        line_start=(0, 200),
        line_end=(500, 200),
        cooldown_frames=5,
    )

    counter.update(
        {
            1: make_track(1, 100, 50)
        }
    )

    first_events = counter.update(
        {
            1: make_track(1, 100, 250)
        }
    )

    assert len(first_events) == 1

    # Move back across immediately.
    second_events = counter.update(
        {
            1: make_track(1, 100, 50)
        }
    )

    assert len(second_events) == 0


def test_reverse_direction():

    counter = EntryExitLineCounter(
        line_start=(0, 200),
        line_end=(500, 200),
        reverse_direction=True,
    )

    counter.update(
        {
            1: make_track(1, 100, 50)
        }
    )

    events = counter.update(
        {
            1: make_track(1, 100, 250)
        }
    )

    assert len(events) == 1
    assert events[0].direction == "exit"