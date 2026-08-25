from typing import Generator, Union

import cv2
import numpy as np


CameraSource = Union[int, str]


class Camera:
    """
    OpenCV camera/video source.

    Supports:
        0                  -> local webcam
        video.mp4          -> video file
        rtsp://...         -> IP camera
    """

    def __init__(
        self,
        source: CameraSource = 0,
        width: int = 1280,
        height: int = 720,
    ):
        self.source = source
        self.width = width
        self.height = height

        self.capture = None

    def open(self) -> bool:
        if self.capture is not None:
            self.release()

        self.capture = cv2.VideoCapture(self.source)

        if not self.capture.isOpened():
            self.capture = None
            return False

        self.capture.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            self.width,
        )

        self.capture.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            self.height,
        )

        return True

    def read(self):
        if self.capture is None:
            if not self.open():
                return False, None

        return self.capture.read()

    def frames(self) -> Generator[np.ndarray, None, None]:
        while True:
            success, frame = self.read()

            if not success:
                break

            yield frame

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()