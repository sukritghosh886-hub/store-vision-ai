from pathlib import Path

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Main directories
ASSETS_DIR = BASE_DIR / "assets"
DATA_DIR = BASE_DIR / "sample_data"
MODELS_DIR = BASE_DIR / "models"

# Supported image formats
SUPPORTED_IMAGE_TYPES = [
    "jpg",
    "jpeg",
    "png",
    "webp",
]

# Application information
APP_NAME = "Store Vision AI"
APP_VERSION = "1.0.0"

# Analysis defaults
DEFAULT_SHELF_ROWS = 4
DEFAULT_SHELF_COLUMNS = 6

# Detection thresholds
DEFAULT_BRIGHTNESS_THRESHOLD = 220
DEFAULT_EDGE_THRESHOLD = 50

# Create directories if they don't exist
ASSETS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)