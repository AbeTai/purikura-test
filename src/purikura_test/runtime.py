from __future__ import annotations

import io
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from purikura_test.api_models import EffectSettings, PerformanceSummary
from purikura_test.camera import CameraSource, OpenCVCameraSource
from purikura_test.effects import EffectPipeline, FrameAsset
from purikura_test.repository import CaptureRepository


@dataclass
class EncodedFrame:
    blob: bytes
    mime: str
    width: int
    height: int


class PurikuraRuntime:
    def __init__(
        self,
        repository: CaptureRepository,
        *,
        camera: CameraSource | None = None,
        pipeline: EffectPipeline | None = None,
    ) -> None:
        self.repository = repository
        self.camera: CameraSource = camera or OpenCVCameraSource(0)
        self._pipeline = pipeline
        self.settings = EffectSettings()
        self.current_frame_id: int | None = None
        self._frame_asset: FrameAsset | None = None
        self._pending_frame: np.ndarray | None = None
        self._latest_processed: np.ndarray | None = None
        self._latest_jpeg: bytes | None = None
        self._performance = PerformanceSummary()
        self._lock = threading.RLock()
        self._pipeline_lock = threading.Lock()
        self._stop = threading.Event()
        self._camera_thread: threading.Thread | None = None
        self._processor_thread: threading.Thread | None = None

    @property
    def camera_id(self) -> int:
        return self.camera.camera_id

    def start(self) -> None:
        if self._camera_thread is not None and self._camera_thread.is_alive():
            return
        self._ensure_pipeline()
        self._stop.clear()
        self.camera.start()
        self._camera_thread = threading.Thread(target=self._read_loop, name="purikura-camera", daemon=True)
        self._processor_thread = threading.Thread(target=self._process_loop, name="purikura-processor", daemon=True)
        self._camera_thread.start()
        self._processor_thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._camera_thread is not None:
            self._camera_thread.join(timeout=2.0)
            self._camera_thread = None
        if self._processor_thread is not None:
            self._processor_thread.join(timeout=2.0)
            self._processor_thread = None
        self.camera.stop()

    def switch_camera(self, camera_id: int) -> None:
        with self._lock:
            was_running = self._camera_thread is not None and self._camera_thread.is_alive()
        if was_running:
            self.stop()
        self.camera = OpenCVCameraSource(camera_id)
        if was_running:
            self.start()

    def update_settings(self, settings: EffectSettings) -> EffectSettings:
        with self._lock:
            self.settings = settings
            return self.settings

    def performance(self) -> PerformanceSummary:
        with self._lock:
            return self._performance.model_copy()

    def set_frame(self, frame_id: int | None) -> None:
        with self._lock:
            self.current_frame_id = frame_id
            if frame_id is None:
                self._frame_asset = None
                return

        frame_blob = self.repository.get_frame_blob(frame_id)
        if frame_blob is None:
            raise KeyError(f"Frame {frame_id} does not exist")

        image_blob, _ = frame_blob
        image_bgra = decode_png_to_bgra(image_blob)
        with self._lock:
            self._frame_asset = FrameAsset(id=frame_id, name=f"Frame {frame_id}", image_bgra=image_bgra)

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            if self._latest_jpeg is not None:
                return self._latest_jpeg
            frame = None if self._latest_processed is None else self._latest_processed.copy()
        if frame is None:
            return None
        started = time.perf_counter()
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            return None
        blob = encoded.tobytes()
        with self._lock:
            self._latest_jpeg = blob
            self._performance.encode_ms = (time.perf_counter() - started) * 1000
        return blob

    def capture_current(self) -> EncodedFrame | None:
        with self._lock:
            frame = None if self._latest_processed is None else self._latest_processed.copy()
        if frame is None:
            return None

        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            return None
        height, width = frame.shape[:2]
        return EncodedFrame(blob=encoded.tobytes(), mime="image/jpeg", width=width, height=height)

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            raw = self.camera.read()
            if raw is None:
                time.sleep(0.05)
                continue
            with self._lock:
                if self._pending_frame is not None:
                    self._performance.dropped_frames += 1
                self._pending_frame = raw
            time.sleep(0.001)

    def _process_loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                raw = self._pending_frame
                self._pending_frame = None
                settings = self.settings
                frame_asset = self._frame_asset
            if raw is None:
                time.sleep(0.005)
                continue

            started = time.perf_counter()
            processed = self._ensure_pipeline().apply(raw, settings, frame_asset)
            processed_at = time.perf_counter()
            ok, encoded = cv2.imencode(".jpg", processed, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            encoded_at = time.perf_counter()
            processing_ms = (processed_at - started) * 1000
            encode_ms = (encoded_at - processed_at) * 1000
            with self._lock:
                self._latest_processed = processed
                self._latest_jpeg = encoded.tobytes() if ok else None
                self._performance.processing_ms = processing_ms
                self._performance.encode_ms = encode_ms
                self._performance.effective_fps = 1000 / processing_ms if processing_ms > 0 else 0.0
                self._performance.profile = settings.processing_profile

    def _ensure_pipeline(self) -> EffectPipeline:
        if self._pipeline is not None:
            return self._pipeline
        with self._pipeline_lock:
            if self._pipeline is None:
                self._pipeline = EffectPipeline()
            return self._pipeline


def decode_png_to_bgra(image_blob: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(image_blob)) as image:
        image = image.convert("RGBA")
        rgba = np.array(image)
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)


def validate_png(image_blob: bytes) -> bytes:
    try:
        with Image.open(io.BytesIO(image_blob)) as image:
            if image.format != "PNG":
                raise ValueError("Only PNG frames are supported")
            normalized = image.convert("RGBA")
            output = io.BytesIO()
            normalized.save(output, format="PNG")
            return output.getvalue()
    except UnidentifiedImageError as exc:
        raise ValueError("Only PNG frames are supported") from exc
