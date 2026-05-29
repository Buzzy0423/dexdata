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

import subprocess
from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import torch

DEFAULT_FPS = 20
CAMERA_RESOLUTION: tuple[int, int] = (600, 960)  # (height, width)


@dataclass
class VideoWriterConfig:
    fps: int = DEFAULT_FPS
    source_resolution: tuple[int, int] = CAMERA_RESOLUTION  # (height, width)
    policy_resolution: tuple[int, int] = (400, 640)  # (height, width)
    writer_type: str = "parallel"  # "serial" or "parallel"
    video_backend: str = "pyav"
    codec: str = "libx264"
    pix_fmt: str = "yuv420p"
    crf: int = 23
    gop: int | None = 2

    # Encoding configuration
    batch_buffer_size: int = 500
    frames_per_file: int = 10_000
    num_workers: int = 4
    max_retries: int = 3
    max_queue_size: int = 10


class JobStatus(Enum):
    SCHEDULED = 1
    DISPATCHED = 2
    COMPLETED = 3
    FAILED = 4


class DestinationStatus(Enum):
    ONGOING = 1
    CLOSED = 2
    REMUXED = 3


@dataclass
class EncodeJob:
    chunk_path: Path  # path to the chunk file
    chunk_idx: int  # index of the chunk
    num_frames: int = 0
    destination: VideoDestination | None = None  # destination to write the chunk to
    video_config: VideoWriterConfig = field(
        default_factory=VideoWriterConfig
    )  # video configuration
    status: JobStatus = JobStatus.SCHEDULED  # status of the job
    future: Future | None = None  # future for the encode job (set after dispatch)


@dataclass
class VideoDestination:
    destination_path: Path | None = None
    num_frames: int = 0
    encode_jobs: list[EncodeJob] = field(default_factory=list)
    status: DestinationStatus = DestinationStatus.ONGOING
    # Per-destination buffer for frames before dispatch
    buffer: list[torch.Tensor] = field(default_factory=list)
    buffer_frame_count: int = 0

    @property
    def num_chunks(self) -> int:
        return len(self.encode_jobs)

    @property
    def chunks_completed(self) -> int:
        return sum(1 for job in self.encode_jobs if job.status == JobStatus.COMPLETED)

    @property
    def chunks_pending(self) -> int:
        return len(self.encode_jobs) - self.chunks_completed

    @property
    def chunks_failed(self) -> int:
        return sum(1 for job in self.encode_jobs if job.status == JobStatus.FAILED)

    @property
    def is_ready_to_remux(self) -> bool:
        """Ready to remux if: closed, no failures, all jobs completed."""
        return (
            self.status == DestinationStatus.CLOSED
            and self.chunks_failed == 0
            and self.chunks_pending == 0
        )

    def generate_filelist_path(self) -> Path:
        if self.destination_path is None:
            raise RuntimeError("Destination path is not set")
        filelist_path = self.destination_path.with_suffix(".txt")
        with open(filelist_path, "w") as f:
            for job in self.encode_jobs:
                f.write(f"file '{job.chunk_path.absolute()}'\n")
        return filelist_path

    def remux(self) -> None:
        """
        Remux the chunks into a single video file.
        """
        if not self.is_ready_to_remux:
            raise RuntimeError(
                "Cannot remux destination. Destination not closed or jobs not completed."
            )

        if self.destination_path is None:
            raise RuntimeError("Destination path is not set")

        # Handle empty destination (no chunks to remux)
        if not self.encode_jobs:
            self.status = DestinationStatus.REMUXED
            return

        self.destination_path.parent.mkdir(parents=True, exist_ok=True)
        filelist_path = self.generate_filelist_path()

        # Lossless concatenation with FFmpeg
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",  # Overwrite output
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(filelist_path),
                "-c",
                "copy",  # Lossless copy
                str(self.destination_path),
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg concat failed: {result.stderr}")

        # Clean up chunk files after successful concat
        for job in self.encode_jobs:
            job.chunk_path.unlink(missing_ok=True)
        filelist_path.unlink(missing_ok=True)

        self.status = DestinationStatus.REMUXED

    def close(self) -> None:
        """Mark this destination as closed (no more episodes will be added)."""
        self.status = DestinationStatus.CLOSED

    def collect_completed(self) -> tuple[int, int]:
        """
        Collect completed futures and update job statuses.

        Returns:
            Tuple of (completed_count, failed_count) for this collection pass.
        """
        completed = 0
        failed = 0

        for job in self.encode_jobs:
            if job.status != JobStatus.DISPATCHED:
                continue
            if job.future is None:
                continue
            if not job.future.done():
                continue

            try:
                # Get result (chunk path) - this will raise if encoding failed
                job.future.result(timeout=0)
                job.status = JobStatus.COMPLETED
                completed += 1
            except Exception:
                job.status = JobStatus.FAILED
                failed += 1
            finally:
                # Clear future reference
                job.future = None

        return completed, failed

    def add_episode_frames(self, frames: torch.Tensor) -> None:
        """Add frames to the destination's buffer."""
        self.buffer.append(frames)
        self.buffer_frame_count += frames.shape[0]
        self.num_frames += frames.shape[0]

    def clear_buffer(self) -> list[torch.Tensor]:
        """Clear and return the buffer contents."""
        buffer_copy = self.buffer.copy()
        self.buffer.clear()
        self.buffer_frame_count = 0
        return buffer_copy
