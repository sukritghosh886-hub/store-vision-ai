
"""Store Vision AI — ML Pipeline.

High-level utilities for model preparation and prediction.
"""

from pathlib import Path

from .generate_dataset import generate_dataset
from .predict import predict_risk
from .train_model import train_model


BASE_DIR = Path(__file__).resolve().parent


def prepare_synthetic_model():
    """Prepare the existing prototype model."""
    generate_dataset()
    train_model()

    return {
        "status": "success",
        "training_data": "synthetic",
        "model_file": str(
            BASE_DIR / "store_vision_ml_model.joblib"
        ),
    }


def predict_visit(features):
    """Run the existing risk predictor."""
    return predict_risk(features)


if __name__ == "__main__":
    print(prepare_synthetic_model())
