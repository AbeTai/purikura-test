from __future__ import annotations

import threading
import time
from typing import Protocol

import cv2
import numpy as np

from purikura_test.api_models import CameraInfo


class CameraSource(Protocol):
    camera_id: int

    def start(self) -> None: ...

    def read(self) -> np.ndarray | None: ...

    def stop(self) -> None: ...


class OpenCVCameraSource:
    def __init__(self, camera_id: int = 0, width: int = 1280, height: int = 720) -> None:
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self._capture: cv2.VideoCapture | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._capture is not None and self._capture.isOpened():
                return
            capture = cv2.VideoCapture(self.camera_id)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if not capture.isOpened():
                capture.release()
                raise RuntimeError(f"Camera {self.camera_id} could not be opened")
            self._capture = capture

    def read(self) -> np.ndarray | None:
        with self._lock:
            if self._capture is None or not self._capture.isOpened():
                return None
            ok, frame = self._capture.read()
            if not ok:
                return None
            return frame

    def stop(self) -> None:
        with self._lock:
            if self._capture is not None:
                self._capture.release()
                self._capture = None


MacBuiltinCameraSource = OpenCVCameraSource


def discover_cameras(max_index: int = 5, probe_seconds: float = 0.2) -> list[CameraInfo]:
    cameras: list[CameraInfo] = []
    for camera_id in range(max_index):
        capture = cv2.VideoCapture(camera_id)
        try:
            time.sleep(probe_seconds)
            available = capture.isOpened()
            if available:
                ok, _ = capture.read()
                available = bool(ok)
            cameras.append(
                CameraInfo(
                    id=camera_id,
                    name=("Built-in camera" if camera_id == 0 else f"Camera {camera_id}"),
                    available=available,
                )
            )
        finally:
            capture.release()
    return cameras
