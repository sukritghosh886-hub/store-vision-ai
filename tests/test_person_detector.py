from models.person_detector import PersonDetector


def test_person_detector_class_exists():

    detector = PersonDetector()

    assert detector is not None


def test_person_class_id():

    assert PersonDetector.PERSON_CLASS_ID == 0