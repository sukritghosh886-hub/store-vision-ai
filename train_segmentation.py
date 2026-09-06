from ultralytics import YOLO
import os


BASE_MODEL = os.getenv(
    "STORE_VISION_BASE_MODEL",
    "yolov8n-seg.pt"
)

DATASET = "dataset/data.yaml"

model = YOLO(BASE_MODEL)

results = model.train(
    data=DATASET,

    epochs=100,

    imgsz=960,

    batch=4,

    workers=2,

    device="cpu",

    project="runs/store_vision",

    name="product_segmentation",

    patience=20,

    pretrained=True,

    cache=False,

    verbose=True
)

print("\nTraining completed.")
print("Best model should be in:")
print("runs/store_vision/product_segmentation/weights/best.pt")