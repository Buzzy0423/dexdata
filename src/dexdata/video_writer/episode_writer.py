# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .config import VideoWriterConfig
from .video_writer import VideoWriter


class EpisodeVideoWriter:
    """
    Thin wrapper around the dataset VideoWriter/SerialVideoWriter for per-episode dumps.
    Accepts BGR or RGB frames and ensures a single output file per episode.
    """

    def __init__(
        self,
        video_dir: str | Path,
        writer_config: VideoWriterConfig | None = None,
        fps: int | None = None,
        input_format: str = "bgr",
    ) -> None:
        self.video_dir = Path(video_dir)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.input_format = input_format.lower()

        self.writer_config = writer_config or VideoWriterConfig()
        if fps is not None:
            self.writer_config.fps = fps

        if self.writer_config.writer_type == "serial":
            from .serial_video_writer import SerialVideoWriter

            self._writer = SerialVideoWriter(
                self.writer_config, dataset_root=self.video_dir
            )
        else:
            self._writer = VideoWriter(self.writer_config, dataset_root=self.video_dir)

        self._closed = False

    def _prepare_frames(
        self, frames: list[np.ndarray] | np.ndarray | torch.Tensor
    ) -> torch.Tensor:
        """Normalize incoming frames to uint8 RGB torch tensor with shape (T, H, W, C)."""
        if torch.is_tensor(frames):
            frames_np = frames.detach().cpu().numpy()
        else:
            frames_np = np.asarray(frames)

        if frames_np.ndim != 4:
            raise ValueError(
                f"Expected frames with 4 dims, got shape {frames_np.shape}"
            )

        # Convert from (T, C, H, W) to (T, H, W, C) if needed
        if frames_np.shape[1] in (1, 3, 4) and frames_np.shape[-1] not in (1, 3, 4):
            frames_np = np.transpose(frames_np, (0, 2, 3, 1))

        # Ensure uint8 - handle float [0,1] vs uint8 [0,255] formats
        if frames_np.dtype != np.uint8:
            # Check if values are in float [0,1] range (LeRobot format)
            if frames_np.max() <= 1.0:
                frames_np = (frames_np * 255).clip(0, 255).astype(np.uint8)
            else:
                frames_np = np.clip(frames_np, 0, 255).astype(np.uint8)

        # Convert BGR -> RGB if requested
        if self.input_format == "bgr" and frames_np.shape[-1] >= 3:
            if frames_np.shape[-1] == 3:
                frames_np = frames_np[..., [2, 1, 0]]
            else:  # assume BGRA
                frames_np = frames_np[..., [2, 1, 0, 3]]

        frames_np = np.ascontiguousarray(frames_np)
        return torch.from_numpy(frames_np)

    def write_episode(
        self, frames: list[np.ndarray] | np.ndarray | torch.Tensor, filename: str | Path
    ) -> Path:
        """
        Encode a single episode to a video file. The destination is closed immediately.
        """
        if self._closed:
            raise RuntimeError("EpisodeVideoWriter is closed")

        dest_path = Path(filename)
        if not dest_path.is_absolute():
            dest_path = self.video_dir / dest_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        frames_tensor = self._prepare_frames(frames)
        self._writer.add_episode(dest_path, frames_tensor)
        self._writer.close_destination(dest_path)
        return dest_path

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._writer.close_writer()

    def __enter__(self) -> EpisodeVideoWriter:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def write_episode_video(
    frames: list[np.ndarray] | np.ndarray | torch.Tensor,
    filename: str | Path,
    fps: int | None = None,
    input_format: str = "bgr",
    writer_config: VideoWriterConfig | None = None,
) -> Path:
    """
    Convenience helper to encode a single episode video using EpisodeVideoWriter.
    Uses serial writer by default (VideoWriterConfig default).
    """
    with EpisodeVideoWriter(
        video_dir=Path(filename).parent,
        writer_config=writer_config,
        fps=fps,
        input_format=input_format,
    ) as writer:
        return writer.write_episode(frames, filename)
