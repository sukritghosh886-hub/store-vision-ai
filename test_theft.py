from models.theft_detector import TheftDetector


detector = TheftDetector(

    # Fake exit zone
    exit_zone=(
        900,
        0,
        1280,
        720
    ),

    min_carry_frames=3
)


person = {

    "track_id": 17,

    "bbox": [
        900,
        100,
        1100,
        600
    ],

    "confidence": 0.95
}


book = {

    "instance_id": 1,

    "class_name": "book",

    "confidence": 0.92,

    "bbox": [
        950,
        300,
        1000,
        400
    ]
}


people = [person]

products = [book]


# No payment.
paid_items = {}


for frame_number in range(5):

    alerts = detector.update(

        people=people,

        products=products,

        paid_items=paid_items
    )

    if alerts:

        for alert in alerts:

            print(
                "SUSPECTED THEFT"
            )

            print(
                "Person:",
                alert.track_id
            )

            print(
                "Item:",
                alert.item_name
            )

            print(
                "Confidence:",
                alert.confidence
            )