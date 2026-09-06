from ultralytics import YOLO
from pathlib import Path


MODEL_PATH = "models/store_products_seg.pt"
TEST_DIR = "dataset/images/test"

model = YOLO(MODEL_PATH)

total_images = 0
total_detections = 0


for image_path in Path(TEST_DIR).glob("*"):

    if image_path.suffix.lower() not in {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    }:
        continue

    results = model.predict(
        source=str(image_path),
        conf=0.25,
        iou=0.50,
        imgsz=960,
        max_det=1000,
        device="cpu",
        verbose=False
    )

    result = results[0]

    total_images += 1

    if result.boxes is None:
        count = 0
    else:
        count = len(result.boxes)

    total_detections += count

    class_counts = {}

    if result.boxes is not None:

        for cls in result.boxes.cls.tolist():

            class_id = int(cls)

            name = model.names[class_id]

            class_counts[name] = (
                class_counts.get(name, 0) + 1
            )

    print("\nImage:", image_path.name)
    print("Total objects:", count)
    print("Class counts:", class_counts)


print("\n==============================")
print("TEST COMPLETE")
print("==============================")
print("Images tested:", total_images)
print("Total detections:", total_detections)