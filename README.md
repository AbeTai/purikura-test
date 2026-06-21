# Purikura Test

FastAPI + OpenCV based prototype for realtime purikura-style camera effects.

## Features

- Realtime MJPEG browser preview.
- Camera abstraction for Mac built-in cameras now and USB cameras later.
- MediaPipe Face Landmarker + SelfieMulticlass based purikura effects.
- Skin smoothing, whitening, eye enlargement, face slim, makeup, tone, and debug overlay controls.
- Quality/Fast processing profiles for comparing fidelity and realtime latency.
- Transparent PNG frame upload and alpha compositing.
- Capture persistence in SQLite through a repository interface.
- Browser UI that can be shown on the Mac display or a TV connected over HDMI.

## Run

```bash
uv sync --all-extras
uv run python scripts/download_models.py
uv run uvicorn purikura_test.app:app --reload
```

Open <http://127.0.0.1:8000>.

On macOS, allow Terminal or the app launching Uvicorn to access the camera when prompted.

## Test

```bash
uv run pytest
```

## Benchmark

```bash
uv run python scripts/benchmark_pipeline.py --profile both
```

## Notes

- The initial camera source uses OpenCV camera indexes. Built-in cameras are usually index `0`; USB cameras typically appear as additional indexes.
- Captures are stored as JPEG blobs in `data/purikura.sqlite3` by default.
- Uploaded frames are stored as PNG blobs and resized to the active preview frame before compositing.
- `scripts/download_models.py` downloads `models/face_landmarker.task` and `models/selfie_multiclass_256x256.tflite`; both are ignored by git.
- Debug overlay modes are `Off`, `Landmarks`, `Masks`, `Parts`, and `All`.
- `GET /api/performance` returns recent processing latency, preview encoding latency, effective FPS, dropped frame count, and active profile.
