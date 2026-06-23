from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from time import perf_counter

import cv2
import numpy as np

from purikura_test.api_models import EffectSettings
from purikura_test.effects import EffectPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark purikura processing profiles.")
    parser.add_argument("--profile", choices=["quality", "fast", "both"], default="both")
    parser.add_argument("--image", type=Path, default=None, help="Optional input image path.")
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--fast-process-width",
        type=int,
        choices=[448, 512, 640],
        default=640,
        help="Internal width used by the Fast profile.",
    )
    return parser.parse_args()


def load_frame(path: Path | None, width: int, height: int) -> np.ndarray:
    if path is not None:
        frame = cv2.imread(str(path))
        if frame is None:
            raise SystemExit(f"Could not read image: {path}")
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    x = np.linspace(64, 220, width, dtype=np.uint8)
    y = np.linspace(72, 190, height, dtype=np.uint8)[:, None]
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = 96
    frame[:, :, 1] = y
    frame[:, :, 2] = x
    return frame


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))
    return sorted(values)[index]


def benchmark_profile(pipeline: EffectPipeline, frame: np.ndarray, profile: str, iterations: int, warmup: int) -> None:
    settings = EffectSettings(processing_profile=profile)
    for _ in range(warmup):
        pipeline.apply(frame, settings)

    samples = []
    for _ in range(iterations):
        started = perf_counter()
        output = pipeline.apply(frame, settings)
        elapsed_ms = (perf_counter() - started) * 1000
        if output.shape != frame.shape:
            raise SystemExit(f"{profile} returned unexpected shape: {output.shape} != {frame.shape}")
        samples.append(elapsed_ms)

    avg = mean(samples)
    print(
        f"{profile}: avg={avg:.1f}ms p95={percentile(samples, 0.95):.1f}ms "
        f"min={min(samples):.1f}ms max={max(samples):.1f}ms fps={1000 / avg:.1f}"
    )


def main() -> None:
    args = parse_args()
    frame = load_frame(args.image, args.width, args.height)
    profiles = ["quality", "fast"] if args.profile == "both" else [args.profile]
    pipeline = EffectPipeline(fast_process_width=args.fast_process_width)
    for profile in profiles:
        benchmark_profile(pipeline, frame, profile, args.iterations, args.warmup)


if __name__ == "__main__":
    main()
