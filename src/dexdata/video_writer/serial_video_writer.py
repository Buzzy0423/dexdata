# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""
SerialVideoWriter class for synchronous batched video encoding.

A simplified alternative to VideoWriter that performs encoding serially
in the main process. Useful for debugging, smaller workloads, or when
multiprocessing overhead is undesirable.

Inherits from VideoWriter and overrides dispatch logic to encode synchronously.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import torch
from loguru import logger

from .config import (
    EncodeJob,
    JobStatus,
    VideoDestination,
    VideoWriterConfig,
)
from .video_writer import VideoWriter
from .worker_encode import VideoEncoderRegistry


class SerialVideoWriter(VideoWriter):
    """
    Serial batched video writer using TorchCodec/PyAV.

    Inherits from VideoWriter but performs all encoding synchronously
    in the main process instead of using a ProcessPoolExecutor.

    Useful for:
    - Debugging encoding issues
    - Small workloads where multiprocessing overhead isn't worthwhile
    - Environments where spawning processes is problematic

    Example:
        writer = SerialVideoWriter(writer_config)

        # Add episodes to different destinations
        writer.add_episode("cam1/video.mp4", episode1_frames)
        writer.add_episode("cam2/video.mp4", episode2_frames)

        # Close writer (encodes remaining, remuxes all destinations)
        writer.close_writer()
    """

    def __init__(
        self,
        writer_config: VideoWriterConfig,
        dataset_root: Path | None = None,
    ) -> None:
        """
        Initialize the SerialVideoWriter.

        Args:
            writer_config: Configuration for video encoding (fps, codec, crf, etc.)
        """
        self._writer_config = writer_config

        # Multiple destinations tracked by path
        self._destinations: dict[Path, VideoDestination] = {}

        # Global chunk counter for unique temp file names
        self._chunk_idx: int = 0

        # Temp directory for chunk files
        if dataset_root is None:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="serial_videowriter_chunks_"))
        else:
            self._temp_dir = Path(dataset_root).joinpath("serial_videowriter_chunks_")
            self._temp_dir.mkdir(parents=True, exist_ok=True)

        # Get encoder function (no process pool needed)
        self._encoder_fn = VideoEncoderRegistry.get(self._writer_config.video_backend)

        logger.info("SerialVideoWriter initialized (serial mode)")
        logger.info(
            f"SerialVideoWriter using {self._writer_config.video_backend} backend"
        )
        self._closed = False

    def _opportunistic_update(self) -> None:
        """No-op in serial mode - all jobs complete synchronously."""
        pass

    def _dispatch_buffer(self, destination: VideoDestination) -> None:
        """Encode current buffer synchronously (overrides async dispatch)."""
        if not destination.buffer:
            return

        # Concatenate all buffered tensors along time dimension. Pass the
        # natural (T, H, W, C) layout straight to the encoder — PyAV's
        # rgb24 frames are (H, W, C), so a TCHW round-trip here would
        # just be a wasted ~2 GB memcpy.
        buffer_frames = destination.clear_buffer()
        frames_thwc = torch.cat(buffer_frames, dim=0).contiguous()
        del buffer_frames
        num_frames = frames_thwc.shape[0]

        # Prepare chunk path and job
        chunk_path = self._temp_dir / f"chunk_{self._chunk_idx:06d}.mp4"

        # Create encode job
        job = EncodeJob(
            chunk_path=chunk_path,
            chunk_idx=self._chunk_idx,
            num_frames=num_frames,
            destination=destination,
            video_config=self._writer_config,
            status=JobStatus.DISPATCHED,
        )
        destination.encode_jobs.append(job)

        # Prepare encoder args - pass tensor directly (T, H, W, C)
        args = {
            "chunk_idx": self._chunk_idx,
            "frames": frames_thwc,
            "fps": self._writer_config.fps,
            "codec": self._writer_config.codec,
            "pix_fmt": self._writer_config.pix_fmt,
            "crf": self._writer_config.crf,
            "gop": self._writer_config.gop,
            "chunk_path": str(chunk_path),
        }

        # Encode synchronously
        try:
            self._encoder_fn(args)
            job.status = JobStatus.COMPLETED
            logger.debug(
                f"Encoded chunk {self._chunk_idx} ({num_frames} frames) for {destination.destination_path}"
            )
        except Exception as e:
            job.status = JobStatus.FAILED
            raise RuntimeError(f"Chunk {self._chunk_idx} encoding failed: {e}") from e

        # Increment global chunk counter
        self._chunk_idx += 1

    def _wait_for_all_jobs(self, timeout_per_job: float = 300.0) -> None:
        """No-op in serial mode - all jobs already complete."""
        pass

    def close_writer(self) -> dict[Path, int]:
        """
        Finalize and close the SerialVideoWriter.

        Encodes all remaining buffers, remuxes all destinations,
        and cleans up resources.

        Returns:
            Dictionary mapping destination paths to their total frame counts.
        """
        if self._closed:
            return {
                dest.destination_path: dest.num_frames
                for dest in self._destinations.values()
                if dest.destination_path is not None
            }

        self._closed = True

        try:
            # Close all ongoing destinations (flush buffers)
            from .config import DestinationStatus

            for dest in self._destinations.values():
                if dest.status == DestinationStatus.ONGOING:
                    if dest.buffer:
                        self._dispatch_buffer(dest)
                    dest.close()

                # Remux if ready (all jobs complete in serial mode)
                if dest.status == DestinationStatus.CLOSED:
                    dest.remux()

        finally:
            # Cleanup temp directory (no pool to shutdown)
            shutil.rmtree(self._temp_dir, ignore_errors=True)

        return {
            dest.destination_path: dest.num_frames
            for dest in self._destinations.values()
            if dest.destination_path is not None
        }

    @property
    def pending_jobs(self) -> int:
        """Total number of jobs currently being encoded (always 0 in serial mode)."""
        return 0
