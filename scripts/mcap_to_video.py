# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Render a 2x2 camera grid mp4 from a composable MCAP episode."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import tyro

from dexdata import Reader

# Fixed 2x2 grid layout, keyed by the composable camera topic strings
# (matches the canonical vega_1u_gripper spec).
GRID_CAMERAS = [
    ["/camera/head_left/rgb/video", "/camera/head_right/rgb/video"],
    ["/camera/left_wrist/rgb/video", "/camera/right_wrist/rgb/video"],
]


def main(
    path: Path,
    output: Path | None = None,
    fps: float | None = None,
) -> None:
    """Render a 2x2 camera grid mp4 for one composable episode.

    Args:
        path: Path to the composable episode directory.
        output: Output mp4 path (default: ``<path>/concatenated.mp4``).
        fps: Frames per second (default: ``record_hz`` from
            ``metadata.json`` if present, else 20).
    """
    path = path.expanduser()
    output = output.expanduser() if output else path / "concatenated.mp4"

    print(f"Loading composable episode from {path}")
    reader = Reader(path)
    episode = reader.read_episode()

    if fps is None:
        if reader.metadata is not None and reader.metadata.collection.record_hz:
            fps = float(reader.metadata.collection.record_hz)
            print(f"  Using fps={fps} from metadata.json (collection.record_hz)")
        else:
            fps = 20.0
            print(f"  No metadata.json record_hz; defaulting to fps={fps}")

    cameras: dict[str, np.ndarray] = {}
    for row_cams in GRID_CAMERAS:
        for topic in row_cams:
            if topic in episode.signals:
                cameras[topic] = episode.signals[topic]
                print(
                    f"  Found camera: {topic} with shape {episode.signals[topic].shape}"
                )

    if not cameras:
        print("No grid cameras found in episode")
        return

    sample = next(iter(cameras.values()))
    num_frames, h, w, c = sample.shape
    grid_h, grid_w = 2 * h, 2 * w
    black_frame = np.zeros((h, w, c), dtype=np.uint8)

    print(f"Streaming {num_frames} frames at {fps} fps to {output}")
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        # Composable handler decodes to RGB uint8.
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{grid_w}x{grid_h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    for t in range(num_frames):
        rows = []
        for row_cams in GRID_CAMERAS:
            row_frames = []
            for topic in row_cams:
                frame = cameras[topic][t] if topic in cameras else black_frame
                row_frames.append(frame)
            rows.append(np.concatenate(row_frames, axis=1))
        grid = np.concatenate(rows, axis=0)
        proc.stdin.write(grid.tobytes())

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        print(f"ffmpeg error: {proc.stderr.read().decode()}")
    else:
        print(f"Done. Output: {output}")


if __name__ == "__main__":
    tyro.cli(main)
