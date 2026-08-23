from io import BytesIO
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


def uploaded_file_to_pil(uploaded_file) -> Image.Image:
    """
    Convert a Streamlit UploadedFile to a PIL image.
    """

    if uploaded_file is None:
        raise ValueError("No image was uploaded.")

    data = uploaded_file.getvalue()

    if not data:
        raise ValueError("Uploaded file is empty.")

    image = Image.open(
        BytesIO(data)
    )

    return image.convert("RGB")


def pil_to_opencv(
    image: Image.Image,
) -> np.ndarray:
    """
    Convert PIL RGB image to OpenCV BGR format.
    """

    rgb = np.array(image)

    bgr = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR,
    )

    return bgr


def opencv_to_pil(
    image: np.ndarray,
) -> Image.Image:
    """
    Convert OpenCV BGR image to PIL RGB image.
    """

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    return Image.fromarray(rgb)


def resize_image(
    image: Image.Image,
    max_width: int = 1200,
) -> Image.Image:
    """
    Resize an image while preserving its aspect ratio.
    """

    width, height = image.size

    if width <= max_width:
        return image

    ratio = max_width / width

    new_size: Tuple[int, int] = (
        max_width,
        int(height * ratio),
    )

    return image.resize(
        new_size,
        Image.Resampling.LANCZOS,
    )


def image_statistics(
    image: Image.Image,
) -> dict:
    """
    Calculate basic image statistics.
    """

    array = np.array(image)

    brightness = float(
        np.mean(array)
    )

    contrast = float(
        np.std(array)
    )

    return {
        "width": image.width,
        "height": image.height,
        "brightness": brightness,
        "contrast": contrast,
    }