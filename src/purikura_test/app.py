from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from purikura_test.api_models import (
    CameraInfo,
    CameraSelection,
    CaptureCreated,
    CaptureSummary,
    CurrentFrameSelection,
    EffectSettings,
    FrameSummary,
)
from purikura_test.camera import discover_cameras
from purikura_test.repository import CaptureRepository
from purikura_test.runtime import PurikuraRuntime, validate_png


def create_app(
    *,
    runtime: PurikuraRuntime | None = None,
    start_camera: bool | None = None,
) -> FastAPI:
    repository = runtime.repository if runtime is not None else CaptureRepository()
    app_runtime = runtime or PurikuraRuntime(repository)
    should_start_camera = (
        start_camera
        if start_camera is not None
        else os.getenv("PURIKURA_DISABLE_CAMERA", "0") not in {"1", "true", "yes"}
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        repository.init_schema()
        if should_start_camera:
            try:
                app_runtime.start()
            except RuntimeError as exc:
                # Keep the API/UI available so users can switch cameras or inspect errors.
                print(f"Camera startup failed: {exc}")
        try:
            yield
        finally:
            app_runtime.stop()

    app = FastAPI(title="Purikura Test", lifespan=lifespan)
    app.state.runtime = app_runtime
    static_dir = Path(__file__).with_name("static")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (static_dir / "index.html").read_text(encoding="utf-8")

    @app.get("/api/cameras", response_model=list[CameraInfo])
    def cameras() -> list[CameraInfo]:
        return discover_cameras()

    @app.put("/api/camera", response_model=CameraInfo)
    def select_camera(selection: CameraSelection) -> CameraInfo:
        try:
            app_runtime.switch_camera(selection.camera_id)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return CameraInfo(id=selection.camera_id, name=f"Camera {selection.camera_id}", available=True)

    @app.get("/api/preview.mjpeg")
    def preview() -> StreamingResponse:
        boundary = "frame"

        def stream() -> AsyncIterator[bytes]:
            while True:
                image = app_runtime.latest_jpeg()
                if image is None:
                    time.sleep(0.1)
                    continue
                yield (
                    f"--{boundary}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(image)}\r\n\r\n".encode()
                    + image
                    + b"\r\n"
                )
                time.sleep(0.033)

        return StreamingResponse(stream(), media_type=f"multipart/x-mixed-replace; boundary={boundary}")

    @app.get("/api/effects", response_model=EffectSettings)
    def get_effects() -> EffectSettings:
        return app_runtime.settings

    @app.put("/api/effects", response_model=EffectSettings)
    def put_effects(settings: EffectSettings) -> EffectSettings:
        return app_runtime.update_settings(settings)

    @app.post("/api/frames", response_model=FrameSummary)
    async def upload_frame(file: UploadFile = File(...)) -> FrameSummary:
        blob = await file.read()
        try:
            normalized = validate_png(blob)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return repository.add_frame(name=file.filename or "frame.png", image_blob=normalized)

    @app.get("/api/frames", response_model=list[FrameSummary])
    def frames() -> list[FrameSummary]:
        return repository.list_frames()

    @app.put("/api/frame/current", response_model=CurrentFrameSelection)
    def set_current_frame(selection: CurrentFrameSelection) -> CurrentFrameSelection:
        try:
            app_runtime.set_frame(selection.frame_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return selection

    @app.post("/api/captures", response_model=CaptureCreated)
    def create_capture() -> CaptureCreated:
        encoded = app_runtime.capture_current()
        if encoded is None:
            raise HTTPException(status_code=409, detail="No processed camera frame is available yet")
        record = repository.add_capture(
            camera_id=app_runtime.camera_id,
            settings=app_runtime.settings,
            frame_id=app_runtime.current_frame_id,
            image_blob=encoded.blob,
            image_mime=encoded.mime,
            width=encoded.width,
            height=encoded.height,
        )
        return CaptureCreated(id=record.id)

    @app.get("/api/captures", response_model=list[CaptureSummary])
    def captures() -> list[CaptureSummary]:
        return repository.list_captures()

    @app.get("/api/captures/{capture_id}/image")
    def capture_image(capture_id: int) -> Response:
        image = repository.get_capture_image(capture_id)
        if image is None:
            raise HTTPException(status_code=404, detail="Capture not found")
        blob, mime = image
        return Response(content=blob, media_type=mime)

    return app


app = create_app()
