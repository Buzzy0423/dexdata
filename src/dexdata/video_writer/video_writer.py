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
VideoWriter class for async batched video encoding with TorchCodec.

Uses ProcessPoolExecutor with tenacity retry for fault-tolerant parallel encoding.
Episodes are buffered until threshold is reached, then dispatched to workers.
Supports multiple simultaneous destinations with lazy remuxing.
"""

from __future__ import annotations

import multiprocessing as mp
import shutil
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from .config import (
    DestinationStatus,
    EncodeJob,
    JobStatus,
    VideoDestination,
    VideoWriterConfig,
)
from .worker_encode import VideoEncoderRegistry


def _save_tensor_for_mmap(tensor: torch.Tensor, path: Path) -> dict:
    """Write a contiguous tensor's raw bytes to a file for mmap-based IPC.

    Returns metadata dict (path, dtype, shape) that the worker uses to
    reconstruct the tensor via numpy.memmap. Only the metadata (~100 bytes)
    goes through the pipe — the actual frame data stays on the filesystem
    and is served from the OS page cache.
    """
    t = tensor.contiguous()
    meta = {
        "path": str(path),
        "dtype": str(t.numpy().dtype),
        "shape": list(t.shape),
    }
    with open(path, "wb") as f:
        f.write(t.numpy().tobytes())
    return meta


def _load_tensor_from_mmap(meta: dict) -> torch.Tensor:
    """Reconstruct a tensor by memory-mapping the raw file written by _save_tensor_for_mmap."""
    mm = np.memmap(
        meta["path"], dtype=meta["dtype"], mode="r", shape=tuple(meta["shape"])
    )
    return torch.from_numpy(np.array(mm))


class VideoWriter:
    """
    Async batched video writer using TorchCodec with ProcessPoolExecutor.

    Supports multiple simultaneous destinations. Buffers episode tensors per
    destination and dispatches encoding workers when buffer exceeds threshold.
    Lazy remuxing occurs opportunistically when destinations are closed and
    all their jobs complete.

    Example:
        writer = VideoWriter(writer_config)

        # Add episodes to different destinations
        writer.add_episode("cam1/video.mp4", episode1_frames)
        writer.add_episode("cam2/video.mp4", episode2_frames)
        writer.add_episode("cam1/video.mp4", episode3_frames)

        # Close a destination (no more episodes)
        writer.close_destination("cam1/video.mp4")

        # Add more to remaining destinations
        writer.add_episode("cam2/video.mp4", episode4_frames)

        # Close writer (waits for all jobs, remuxes all destinations)
        writer.close_writer()
    """

    def __init__(
        self,
        writer_config: VideoWriterConfig,
        dataset_root: str | None,
    ) -> None:
        """
        Initialize the VideoWriter.

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
            self._temp_dir = Path(tempfile.mkdtemp(prefix="videowriter_chunks_"))
        else:
            self._temp_dir = Path(dataset_root).joinpath("videowriter_chunks_")
            self._temp_dir.mkdir(parents=True, exist_ok=True)

        # ProcessPoolExecutor with spawn context for CUDA safety
        self._ctx = mp.get_context("spawn")
        self._pool = ProcessPoolExecutor(
            max_workers=self._writer_config.num_workers,
            mp_context=self._ctx,
        )
        logger.info(
            f"VideoWriter initialized with {self._writer_config.num_workers} workers"
        )
        logger.info(f"VideoWriter using {self._writer_config.video_backend} backend")
        self._closed = False

        # Reentrant lock guarding ``_destinations``, ``_chunk_idx``,
        # ``_closed`` and every per-destination ``buffer`` /
        # ``encode_jobs`` / ``status`` mutation. Public entry points
        # (``add_episode``, ``close_destination``, ``close_writer``)
        # acquire this lock; private helpers assume the caller already
        # holds it. We release it temporarily during ``wait()`` on
        # encoder futures so other threads can keep feeding the pool.
        self._state_lock = threading.RLock()

        # Optional background queue-depth monitor. ``enable_queue_logging``
        # spawns a daemon thread that samples destination state at the
        # given interval; the thread exits cleanly when ``close_writer``
        # clears ``_qlog_stop``.
        self._qlog_stop: threading.Event | None = None
        self._qlog_thread: threading.Thread | None = None

    def _opportunistic_update(self) -> None:
        """
        Opportunistic manager hook called on every API entry point.

        1. Collect completed futures from all destinations
        2. Check for failures, update job statuses
        3. Remux any destinations where: closed AND all_jobs_complete AND not_remuxed
        """
        for dest in self._destinations.values():
            # Skip already remuxed destinations
            if dest.status == DestinationStatus.REMUXED:
                continue

            # Collect completed futures for this destination
            dest.collect_completed()

            # Remux if ready
            if dest.is_ready_to_remux:
                dest.remux()
                logger.info(f"Destination {dest.destination_path} remuxed")

    def _get_or_create_destination(self, dest_path: Path) -> VideoDestination:
        """Get existing destination or create a new one."""
        if dest_path not in self._destinations:
            self._destinations[dest_path] = VideoDestination(
                destination_path=dest_path,
            )
        return self._destinations[dest_path]

    def add_episode(self, dest_path: str | Path, frames: torch.Tensor) -> None:
        """
        Add an episode to a destination's buffer. Non-blocking.

        When the buffer exceeds the threshold, a worker is dispatched
        to encode the buffered frames. Thread-safe via
        :attr:`_state_lock` — multiple producer threads may call this
        concurrently to keep the encoder pool fed across cameras.

        Args:
            dest_path: Output video file path for this episode.
            frames: Episode tensor of shape (T, H, W, C) where T is time/frames,
                   H is height, W is width, C is channels (RGB).
        """
        # Validate shape outside the lock — input is caller-private.
        if frames.ndim != 4:
            raise ValueError(f"Expected 4D tensor (T, H, W, C), got {frames.ndim}D")
        t, h, w, c = frames.shape
        if c not in (1, 3, 4):
            raise ValueError(f"Expected 1, 3, or 4 channels, got {c}")

        dest_path = Path(dest_path)
        with self._state_lock:
            if self._closed:
                raise RuntimeError("VideoWriter is closed")

            self._opportunistic_update()
            destination = self._get_or_create_destination(dest_path)
            if destination.status != DestinationStatus.ONGOING:
                raise RuntimeError(
                    f"Destination {dest_path} is closed, cannot add episodes"
                )
            destination.add_episode_frames(frames)
            if destination.buffer_frame_count >= self._writer_config.batch_buffer_size:
                self._dispatch_buffer(destination)

    def _wait_for_queue_space(self) -> None:
        """
        Block until queue size is below max_queue_size.

        This prevents unbounded memory growth by waiting for pending jobs
        to complete before submitting new ones. Uses FIRST_COMPLETED to wake
        as soon as any job finishes. Releases :attr:`_state_lock` around
        the blocking ``wait()`` so peer producer threads can keep
        feeding the buffer while we sleep.
        """
        while self.pending_jobs >= self._writer_config.max_queue_size:
            self._opportunistic_update()

            if self.pending_jobs >= self._writer_config.max_queue_size:
                pending_futures = self._get_pending_futures()
                if pending_futures:
                    # Drop the lock for the blocking wait so peer
                    # threads can still add_episode/close_destination.
                    self._state_lock.release()
                    try:
                        done, _ = wait(pending_futures, return_when=FIRST_COMPLETED)
                    finally:
                        self._state_lock.acquire()
                    self._process_completed_futures(done)

    def _get_pending_futures(self) -> set:
        """Get all pending futures across all destinations."""
        futures = set()
        for dest in self._destinations.values():
            for job in dest.encode_jobs:
                if job.status == JobStatus.DISPATCHED and job.future is not None:
                    futures.add(job.future)
        return futures

    def _process_completed_futures(self, done_futures: set) -> None:
        """Process completed futures and update job statuses."""
        for dest in self._destinations.values():
            for job in dest.encode_jobs:
                if job.future in done_futures:
                    try:
                        job.future.result()  # Raises if failed
                        job.status = JobStatus.COMPLETED
                        logger.info(f"Chunk {job.chunk_idx} encoding completed")
                    except Exception as e:
                        job.status = JobStatus.FAILED
                        raise RuntimeError(
                            f"Chunk {job.chunk_idx} encoding failed: {e}"
                        ) from e
                    finally:
                        job.future = None

    def _dispatch_buffer(self, destination: VideoDestination) -> None:
        """Dispatch current buffer to a worker for encoding."""
        if not destination.buffer:
            return

        # Wait until queue has space before dispatching
        self._wait_for_queue_space()

        # Concatenate all buffered tensors along time dimension. Keep the
        # natural (T, H, W, C) layout — the encoder consumes THWC
        # directly so a TCHW round-trip is wasted memcpy.
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

        # Write frames to a mmap-able file instead of pickling through pipes.
        # The worker memory-maps the raw bytes back — avoids pipe buffer limits
        # and works in Docker (no /dev/shm dependency).
        frames_path = self._temp_dir / f"frames_{self._chunk_idx:06d}.raw"
        frames_meta = _save_tensor_for_mmap(frames_thwc, frames_path)
        del frames_thwc

        args = {
            "chunk_idx": self._chunk_idx,
            "frames_meta": frames_meta,
            "fps": self._writer_config.fps,
            "codec": self._writer_config.codec,
            "pix_fmt": self._writer_config.pix_fmt,
            "crf": self._writer_config.crf,
            "gop": self._writer_config.gop,
            "chunk_path": str(chunk_path),
        }

        # Submit to pool and store future on job
        future = self._pool.submit(
            VideoEncoderRegistry.get(self._writer_config.video_backend), args
        )
        job.future = future

        # Increment global chunk counter
        self._chunk_idx += 1

    def close_destination(self, dest_path: str | Path) -> None:
        """
        Mark a destination as closed (no more episodes will be added).

        This flushes any remaining buffer for the destination and marks it
        closed. The actual remux happens opportunistically (FFmpeg
        invocation is done outside the lock so it doesn't block peers).

        Args:
            dest_path: Path to the destination to close.
        """
        dest_path = Path(dest_path)
        to_remux: VideoDestination | None = None
        with self._state_lock:
            if self._closed:
                raise RuntimeError("VideoWriter is closed")

            self._opportunistic_update()

            if dest_path not in self._destinations:
                raise ValueError(f"Destination {dest_path} does not exist")
            destination = self._destinations[dest_path]
            if destination.status != DestinationStatus.ONGOING:
                return

            if destination.buffer:
                self._dispatch_buffer(destination)
            destination.close()
            logger.info(f"Destination {dest_path} closed")

            # Refresh state; the dest may now be remux-ready if all its
            # chunks landed during the close. We snapshot the dest for
            # remux below — outside the lock, because FFmpeg is slow.
            for d in self._destinations.values():
                d.collect_completed()
            if destination.is_ready_to_remux:
                to_remux = destination

        if to_remux is not None:
            to_remux.remux()
            logger.info(f"Destination {to_remux.destination_path} remuxed")

    def close_writer(self) -> dict[Path, int]:
        """
        Finalize and close the VideoWriter.

        Waits for all encoding jobs to complete, remuxes all remaining
        destinations, and cleans up resources. Pool wait + FFmpeg remux
        happen outside :attr:`_state_lock`.
        """
        with self._state_lock:
            if self._closed:
                return {
                    dest.destination_path: dest.num_frames
                    for dest in self._destinations.values()
                    if dest.destination_path is not None
                }
            self._closed = True

            # Flush every still-buffered destination — under the lock,
            # since _dispatch_buffer mutates per-dest state and the
            # global _chunk_idx.
            for dest in self._destinations.values():
                if dest.status == DestinationStatus.ONGOING:
                    if dest.buffer:
                        self._dispatch_buffer(dest)
                    dest.close()
            destinations_snapshot = list(self._destinations.values())

        # Stop the queue logger (if any) before we start blocking on the
        # encoder pool — its samples are no longer useful past here.
        if self._qlog_stop is not None:
            self._qlog_stop.set()
            if self._qlog_thread is not None:
                self._qlog_thread.join(timeout=2.0)

        try:
            # Wait for the encoder pool to drain (slow; no lock held).
            self._wait_for_all_jobs()

            # Remux closed destinations (FFmpeg subprocess; slow). No
            # other thread can be calling close_destination because
            # ``_closed`` is set, so we're the only remuxer.
            for dest in destinations_snapshot:
                if dest.status == DestinationStatus.CLOSED:
                    dest.remux()
        finally:
            self._pool.shutdown(wait=True)
            shutil.rmtree(self._temp_dir, ignore_errors=True)

        return {
            dest.destination_path: dest.num_frames
            for dest in destinations_snapshot
            if dest.destination_path is not None
        }

    def enable_queue_logging(self, interval_s: float = 2.0) -> None:
        """Start a daemon thread that periodically logs encoder-queue depth.

        Off by default — opt-in for diagnostics when profiling
        throughput. Sampled fields per tick (each acquired under
        :attr:`_state_lock` so the snapshot is consistent):

        * ``buffered`` — total frames sitting in per-destination buffers
          (haven't been dispatched yet)
        * ``pending``  — chunks submitted to the pool but not finished
        * ``done``     — chunks the pool has completed
        * ``dests``    — number of destinations currently ``ONGOING``

        The thread exits cleanly when :meth:`close_writer` sets the
        stop event.
        """
        if self._qlog_thread is not None and self._qlog_thread.is_alive():
            return  # already running
        self._qlog_stop = threading.Event()

        def _run(stop: threading.Event, interval: float) -> None:
            t0 = time.monotonic()
            while not stop.is_set():
                with self._state_lock:
                    buffered = sum(
                        d.buffer_frame_count for d in self._destinations.values()
                    )
                    pending = self.pending_jobs
                    done = sum(
                        sum(1 for j in d.encode_jobs if j.status == JobStatus.COMPLETED)
                        for d in self._destinations.values()
                    )
                    ongoing = sum(
                        1
                        for d in self._destinations.values()
                        if d.status == DestinationStatus.ONGOING
                    )
                logger.info(
                    f"queue@{time.monotonic() - t0:5.1f}s  "
                    f"buffered={buffered}  pending={pending}  "
                    f"done={done}  ongoing_dests={ongoing}"
                )
                stop.wait(interval)

        self._qlog_thread = threading.Thread(
            target=_run,
            args=(self._qlog_stop, interval_s),
            daemon=True,
            name="VideoWriterQueueLogger",
        )
        self._qlog_thread.start()

    def _wait_for_all_jobs(self, timeout_per_job: float = 300.0) -> None:
        """
        Wait for all pending encoding jobs across all destinations.

        Args:
            timeout_per_job: Timeout in seconds per job (default 5 minutes).
        """
        for dest in self._destinations.values():
            for job in dest.encode_jobs:
                if job.status != JobStatus.DISPATCHED:
                    continue
                if job.future is None:
                    continue

                try:
                    job.future.result(timeout=timeout_per_job)
                    job.status = JobStatus.COMPLETED
                except Exception as e:
                    job.status = JobStatus.FAILED
                    raise RuntimeError(
                        f"Chunk {job.chunk_idx} encoding failed: {e}"
                    ) from e
                finally:
                    job.future = None

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close_writer()

    @property
    def destinations(self) -> dict[Path, VideoDestination]:
        """All tracked destinations."""
        return self._destinations

    @property
    def pending_jobs(self) -> int:
        """Total number of jobs currently being encoded across all destinations."""
        return sum(
            1
            for dest in self._destinations.values()
            for job in dest.encode_jobs
            if job.status == JobStatus.DISPATCHED
        )

    @property
    def total_buffered_frames(self) -> int:
        """Total number of frames currently in buffers across all destinations."""
        return sum(dest.buffer_frame_count for dest in self._destinations.values())

    @property
    def chunks_dispatched(self) -> int:
        """Total number of chunks dispatched for encoding (lifetime)."""
        return self._chunk_idx
