# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Simple video reader that loads all frames as numpy arrays."""

import logging
from pathlib import Path

import av
import numpy as np

logger = logging.getLogger(__name__)


MAX_FRAMES = 10_000


def read_video_frames(video_path: str | Path) -> np.ndarray:
    """
    Load all frames from a video file.

    Args:
        video_path: Path to the video file.

    Returns:
        np.ndarray: Array of shape (N, H, W, C) with dtype uint8,
                    where N is number of frames, C is 3 (RGB).

    Raises:
        FileNotFoundError: If video file doesn't exist.
        RuntimeError: If video has no frames or decoding fails.
        ValueError: If video has more than MAX_FRAMES frames.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Check frame count before loading
    meta = get_video_metadata(video_path)
    if meta["num_frames"] is not None and meta["num_frames"] > MAX_FRAMES:
        raise ValueError(
            f"Video has {meta['num_frames']} frames, exceeds limit of {MAX_FRAMES}: {video_path}"
        )

    frames = []
    with av.open(str(video_path), mode="r") as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"  # Enable multithreaded decoding

        for frame in container.decode(stream):
            # Convert to RGB numpy array (H, W, C)
            img = frame.to_ndarray(format="rgb24")
            frames.append(img)

    if len(frames) == 0:
        raise RuntimeError(f"No frames decoded from video: {video_path}")

    if len(frames) > MAX_FRAMES:
        raise ValueError(
            f"Video has {len(frames)} frames, exceeds limit of {MAX_FRAMES}: {video_path}"
        )

    return np.stack(frames, axis=0)


def get_video_metadata(video_path: str | Path) -> dict:
    """
    Get metadata from a video file.

    Args:
        video_path: Path to the video file.

    Returns:
        dict with keys: height, width, fps, num_frames, duration_s, codec
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    with av.open(str(video_path), mode="r") as container:
        stream = container.streams.video[0]

        # Calculate fps from base_rate
        fps = (
            float(stream.base_rate) if stream.base_rate else float(stream.average_rate)
        )

        # Get duration
        if stream.duration is not None and stream.time_base is not None:
            duration_s = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration_s = float(container.duration / av.time_base)
        else:
            duration_s = None

        # Estimate frame count
        num_frames = stream.frames if stream.frames > 0 else None
        if num_frames is None and duration_s is not None and fps:
            num_frames = int(duration_s * fps)

        return {
            "height": stream.height,
            "width": stream.width,
            "fps": fps,
            "num_frames": num_frames,
            "duration_s": duration_s,
            "codec": stream.codec.name,
        }
