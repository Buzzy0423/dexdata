# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Memory-bounded composable → LeRobot v2.1 export.

This is the streaming counterpart to ``composable_to_lerobot.py``. The
output dataset is byte-identical (modulo running-stats vs batch-stats
numerical noise) but peak RSS no longer scales with episode length:

* State/action stay fully materialized (KB scale).
* Cameras and depth are decoded one source frame at a time and pushed
  into the underlying :class:`VideoWriter` in 256-frame batches.
* Image stats are accumulated incrementally per chunk; no full ``(T, H,
  W, C)`` array ever exists in main-process memory.

The standard path (:mod:`dexdata.exporters.lerobot` ``write()``) is
unchanged. Reach for this script when episode length pushes the
standard path past your RAM budget.

Usage::

    python scripts/composable_to_lerobot_streaming.py \\
        --in-dir /path/to/composable_episodes \\
        --out-dir /path/to/lerobot_v21_dataset \\
        --task "pick up the cube"

The default chunk size (256 frames) is tuned for 600×960 BGR camera
inputs; bring it down with ``--image-batch-size`` if peak RSS is still
too high on extreme resolutions.
"""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import jsonlines
import numpy as np
import pyarrow.parquet as pq
import torch
import tqdm
import tyro

import dexdata
from dexdata import align_episode, discover_episodes
from dexdata.exporters import ExporterConfig
from dexdata.exporters.canonical import DepthConfig, RGBConfig
from dexdata.exporters.lerobot import (
    DEFAULT_CHUNK_SIZE,
    EPISODES_STATS_PATH,
    INFO_PATH,
    LEGACY_DATA_PATH_TEMPLATE,
    LEGACY_EPISODES_PATH,
    LEGACY_TASKS_PATH,
    LEGACY_VIDEO_PATH_TEMPLATE,
    V21,
    _column_stats,
    _image_column_stats_hardcoded,
    _parquet_table,
    _video_feature_spec,
)
from dexdata.handlers.containers import CompressedVideoSpec, DepthImageSpec
from dexdata.mcap_utils.streaming_reader import (
    StreamingReader,
    nearest_ref_to_src,
)
from dexdata.video_writer.config import VideoWriterConfig
from dexdata.video_writer.video_writer import VideoWriter

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = (
    Path(dexdata.__file__).parent / "exporters" / "config" / "vega_1u_gripper.yaml"
)
ROBOT_STATE_PREFIX = "/robot/state/"


# ---------------------------------------------------------------------------
# Per-chunk transforms — single-frame versions of canonical/lerobot helpers
# ---------------------------------------------------------------------------


def _transform_rgb_frame(frame_rgb: np.ndarray, cfg: RGBConfig) -> np.ndarray:
    """Source RGB ``(H, W, 3)`` uint8 → on-disk RGB ``(out_h, out_w, 3)``.

    Single-frame version of ``_resize_rgb_for_video``. The composable
    ``CompressedVideo`` handler decodes to RGB directly, so this only
    handles the optional resize — no channel swap.
    """
    if cfg.out_hw is None or (frame_rgb.shape[0], frame_rgb.shape[1]) == cfg.out_hw:
        return frame_rgb
    out_h, out_w = cfg.out_hw
    interp = (
        cv2.INTER_AREA
        if (out_h * out_w < frame_rgb.shape[0] * frame_rgb.shape[1])
        else cv2.INTER_LINEAR
    )
    return cv2.resize(frame_rgb, (out_w, out_h), interpolation=interp)


def _transform_depth_frame(depth_m: np.ndarray, cfg: DepthConfig) -> np.ndarray:
    """Source float32 ``(H, W)`` m → on-disk uint8 ``(out_h, out_w, 3)``.

    Mirrors ``_quantize_depth_for_video`` per-frame: clip to ``cfg.clip_m``,
    linear quantise to 0..255, replicate 3 channels, nearest-neighbor
    resize so depth discontinuities don't get smoothed across.
    """
    out_h, out_w = cfg.out_hw
    if depth_m.shape != (out_h, out_w):
        depth_m = cv2.resize(
            depth_m.astype(np.float32, copy=False),
            (out_w, out_h),
            interpolation=cv2.INTER_NEAREST,
        )
    min_m, max_m = cfg.clip_m
    span = max(max_m - min_m, 1e-6)
    clipped = np.clip(depth_m, min_m, max_m)
    u8 = ((clipped - min_m) / span * 255.0).astype(np.uint8)
    u8[depth_m <= 0] = 0
    return np.repeat(u8[:, :, None], 3, axis=2)


# ---------------------------------------------------------------------------
# Streaming stats accumulator
# ---------------------------------------------------------------------------


def _hardcoded_image_stats(channels: int, n_frames: int) -> dict[str, Any]:
    """Hardcoded CLIP-normalization stats — single source of truth via lerobot.

    Delegates shape construction to
    :func:`lerobot._image_column_stats_hardcoded` so streaming and
    standard paths always emit the same values. We pass a zero-byte
    shape probe to dodge the actual pixel data — the helper only
    reads ``shape``.
    """
    probe = np.empty((n_frames, 1, 1, channels), dtype=np.uint8)
    return _image_column_stats_hardcoded(probe)


# ---------------------------------------------------------------------------
# Per-episode driver
# ---------------------------------------------------------------------------


def _pick_reference_topic(spec_topics: list[str]) -> str:
    for t in spec_topics:
        if t.startswith(ROBOT_STATE_PREFIX):
            return t
    raise ValueError(
        f"no {ROBOT_STATE_PREFIX}* topic in episode spec; "
        f"cannot pick an alignment reference (got {spec_topics})"
    )


def _build_streaming_features(
    non_image_canonical: dict[str, np.ndarray],
    image_keys_with_dims: dict[str, tuple[int, int, int]],
    image_keys_is_depth: dict[str, bool],
    config: ExporterConfig,
    fps: float,
) -> dict[str, dict[str, Any]]:
    """Like ``lerobot._build_features`` but doesn't need a materialised dict.

    ``image_keys_with_dims`` is ``{canonical_key: (H, W, C)}`` for the
    *on-disk* resolution (post-resize / post-quantise). The per-stream
    encode config is read from ``config.rgb`` or ``config.depth`` based
    on ``image_keys_is_depth``.
    """
    features: dict[str, dict[str, Any]] = {}
    for key, arr in non_image_canonical.items():
        if arr.ndim == 1:
            shape = [1]
        else:
            shape = list(arr.shape[1:])
        dtype = "float32" if np.issubdtype(arr.dtype, np.floating) else "int64"
        features[key] = {"dtype": dtype, "shape": shape, "names": None}

    for key, (H, W, C) in image_keys_with_dims.items():
        is_depth = image_keys_is_depth[key]
        stream_cfg = config.depth if is_depth else config.rgb
        features[key] = _video_feature_spec(
            H, W, C, fps, stream_cfg.codec, stream_cfg.pix_fmt, is_depth_map=False
        )
    features["timestamp"] = {"dtype": "float32", "shape": [1], "names": None}
    features["frame_index"] = {"dtype": "int64", "shape": [1], "names": None}
    features["episode_index"] = {"dtype": "int64", "shape": [1], "names": None}
    features["index"] = {"dtype": "int64", "shape": [1], "names": None}
    features["task_index"] = {"dtype": "int64", "shape": [1], "names": None}
    return features


def _resolve_on_disk_dims(
    raw_hw: tuple[int, int], is_depth: bool, config: ExporterConfig
) -> tuple[int, int, int]:
    """Apply the same resize policy as the per-frame transform to a (H, W).

    Returns ``(H, W, 3)`` for both RGB and depth (depth is replicated to
    3 channels for h264 yuv420p compatibility).
    """
    if is_depth:
        h, w = config.depth.out_hw
        return h, w, 3
    if config.rgb.out_hw is not None:
        h, w = config.rgb.out_hw
        return h, w, 3
    return raw_hw[0], raw_hw[1], 3


def _iter_batches(
    frame_iter: Iterator[np.ndarray],
    transform,
    batch_size: int,
) -> Iterator[np.ndarray]:
    """Yield stacked, transformed batches of ``batch_size`` frames.

    Holds at most ``batch_size`` transformed frames in a list. The last
    yielded batch may be smaller.
    """
    buf: list[np.ndarray] = []
    for frame in frame_iter:
        buf.append(transform(frame))
        if len(buf) >= batch_size:
            chunk = np.stack(buf)
            buf.clear()
            yield chunk
    if buf:
        yield np.stack(buf)


def _stream_one_camera(
    reader: StreamingReader,
    image_topic: str,
    canonical_key: str,
    ref_to_src: np.ndarray,
    is_depth: bool,
    config: ExporterConfig,
    writer: VideoWriter,
    dest_path: Path,
    batch_size: int,
) -> tuple[dict[str, Any], tuple[int, int, int]]:
    """Stream one camera onto the video writer; return (stats, on_disk_HWC).

    Frames are decoded one at a time, transformed (resize for RGB,
    quantise for depth), accumulated into ``batch_size`` chunks, and
    pushed to ``writer.add_episode``. We emit shape-compliant zero
    image stats — matches the standard path's ``skip_image_stats``
    default (no downstream consumer reads the values).
    """
    transform = (
        (lambda f: _transform_depth_frame(f, config.depth))
        if is_depth
        else (lambda f: _transform_rgb_frame(f, config.rgb))
    )
    src_iter = reader.iter_frames_in_ref_order(image_topic, ref_to_src)
    on_disk_hwc: tuple[int, int, int] | None = None
    n_frames = 0

    for chunk in _iter_batches(src_iter, transform, batch_size):
        if on_disk_hwc is None:
            on_disk_hwc = (chunk.shape[1], chunk.shape[2], chunk.shape[3])
        n_frames += chunk.shape[0]
        # torch.from_numpy is a zero-copy typecast over the numpy slab;
        # VideoWriter then mmaps it for the worker pool. The chunk is
        # released when this iteration ends.
        writer.add_episode(dest_path, torch.from_numpy(chunk))

    writer.close_destination(dest_path)
    if on_disk_hwc is None:
        raise RuntimeError(f"{image_topic}: no frames streamed")
    return _hardcoded_image_stats(on_disk_hwc[2], n_frames), on_disk_hwc


def _topic_to_signal(spec_signals, topic: str):
    for s in spec_signals:
        if s.topic == topic:
            return s
    raise KeyError(topic)


def _convert_one_episode(
    ep_dir: Path,
    ep_idx: int,
    out_root: Path,
    config: ExporterConfig,
    resolved_fps: float,
    task_index: int,
    frame_offset: int,
    max_gap_ns: int | None,
    image_batch_size: int,
    queue_log_interval_s: float = 0.0,
) -> tuple[
    int, dict[str, dict[str, Any]], dict[str, tuple[int, int, int]], dict[str, bool]
]:
    """Convert one composable episode into one v2.1 episode on disk.

    Returns:
        ``(length, stats, image_dims, image_is_depth)`` for use by the
        caller's feature-spec / parquet bookkeeping.
    """
    reader = StreamingReader(ep_dir)
    meta = reader.metadata
    if meta is None:
        raise RuntimeError(f"{ep_dir}: missing metadata.json sidecar")
    spec_topics = [s.topic for s in reader.spec.signals]
    reference = _pick_reference_topic(spec_topics)

    # Non-image bulk read + align — both small (KB scale).
    non_image_ep = reader.read_non_image_episode()
    if reference not in non_image_ep.signals:
        raise RuntimeError(f"{ep_dir}: reference {reference!r} not in non-image topics")
    aligned = align_episode(
        non_image_ep, reference, mode="nearest", max_gap_ns=max_gap_ns
    )
    del non_image_ep

    # Rename non-image to canonical keys; only carry features in config.
    non_image_canonical: dict[str, np.ndarray] = {}
    for topic, ckey in config.features.items():
        if reader.is_image_topic(topic):
            continue
        if topic not in aligned.signals:
            raise RuntimeError(
                f"{ep_dir}: config references {topic!r} not in aligned episode"
            )
        non_image_canonical[ckey] = aligned.signals[topic]

    ref_recv_ts = aligned.recv_timestamps[reference]
    n_ref_frames = int(ref_recv_ts.size)
    del aligned

    # Parquet — numeric columns only; image topics are referenced by mp4 path.
    episode_chunk = ep_idx // DEFAULT_CHUNK_SIZE
    parquet_path = out_root / LEGACY_DATA_PATH_TEMPLATE.format(
        episode_chunk=episode_chunk, episode_index=ep_idx
    )
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    table, length = _parquet_table(
        non_image_canonical, ep_idx, frame_offset, task_index, resolved_fps
    )
    if length != n_ref_frames:
        raise RuntimeError(
            f"{ep_dir}: parquet length {length} != ref frame count {n_ref_frames}"
        )
    pq.write_table(table, parquet_path)

    # Numeric stats.
    episode_stats: dict[str, dict[str, Any]] = {
        k: _column_stats(arr) for k, arr in non_image_canonical.items()
    }
    del non_image_canonical

    # Image topics: stream one camera at a time. Single shared
    # VideoWriter pool so all four cameras' chunks can encode
    # concurrently (the pool keeps `num_workers=4` busy).
    image_topic_to_canonical: dict[str, str] = {
        topic: ckey
        for topic, ckey in config.features.items()
        if reader.is_image_topic(topic)
    }
    image_dims: dict[str, tuple[int, int, int]] = {}
    image_is_depth: dict[str, bool] = {}

    # Split RGB/depth — VideoWriter is locked to one codec config per
    # instance, so RGB and depth need their own writer pools.
    rgb_topics: list[tuple[str, str]] = []
    depth_topics: list[tuple[str, str]] = []
    for topic, ckey in image_topic_to_canonical.items():
        signal = _topic_to_signal(reader.spec.signals, topic)
        is_depth = isinstance(signal.container, DepthImageSpec)
        if is_depth:
            depth_topics.append((topic, ckey))
        else:
            if not isinstance(signal.container, CompressedVideoSpec):
                raise RuntimeError(
                    f"{topic}: unsupported image container "
                    f"{type(signal.container).__name__}"
                )
            rgb_topics.append((topic, ckey))
        image_is_depth[ckey] = is_depth

    fps_i = int(round(resolved_fps))

    def _make_cfg(c: RGBConfig | DepthConfig) -> VideoWriterConfig:
        return VideoWriterConfig(
            writer_type="parallel",
            video_backend="pyav",
            fps=fps_i,
            codec=c.codec,
            pix_fmt=c.pix_fmt,
            crf=c.crf,
            gop=c.gop,
        )

    def _drive_set(items: list[tuple[str, str]], stream_cfg) -> None:
        if not items:
            return
        is_depth_set = stream_cfg is config.depth
        writer = VideoWriter(_make_cfg(stream_cfg), dataset_root=out_root)
        if queue_log_interval_s and queue_log_interval_s > 0:
            writer.enable_queue_logging(interval_s=queue_log_interval_s)

        def _process_one(topic: str, ckey: str) -> None:
            src_ts = reader.topic_timestamps(topic).recv_ns
            ref_to_src = nearest_ref_to_src(ref_recv_ts, src_ts)
            if ref_to_src.size == 0:
                raise RuntimeError(f"{ep_dir}: {topic!r} has no source frames")
            dest_path = out_root / LEGACY_VIDEO_PATH_TEMPLATE.format(
                episode_chunk=episode_chunk,
                video_key=ckey,
                episode_index=ep_idx,
            )
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            stats, hwc = _stream_one_camera(
                reader=reader,
                image_topic=topic,
                canonical_key=ckey,
                ref_to_src=ref_to_src,
                is_depth=is_depth_set,
                config=config,
                writer=writer,
                dest_path=dest_path,
                batch_size=image_batch_size,
            )
            episode_stats[ckey] = stats
            image_dims[ckey] = hwc

        try:
            # One thread per topic so decode+transform can overlap across
            # cameras. PyAV decode and cv2.resize both release the GIL,
            # so threads scale across cores. The shared VideoWriter is
            # thread-safe (RLock around add_episode / close_destination)
            # and dispatches all chunks into the same N-worker
            # ProcessPoolExecutor.
            with ThreadPoolExecutor(
                max_workers=len(items), thread_name_prefix="cam"
            ) as ex:
                futures = [ex.submit(_process_one, t, c) for t, c in items]
                # Re-raise any worker exception by reading results.
                for f in futures:
                    f.result()
        finally:
            writer.close_writer()

    _drive_set(rgb_topics, config.rgb)
    _drive_set(depth_topics, config.depth)

    return length, episode_stats, image_dims, image_is_depth


def main(
    in_dir: Path,
    out_dir: Path,
    task: str,
    config_path: Path = DEFAULT_CONFIG_PATH,
    fps: float | None = None,
    robot_type: str | None = None,
    max_gap_ns: int | None = None,
    image_batch_size: int = 256,
    queue_log_interval_s: float = 0.0,
) -> None:
    """Streaming export — memory-bounded counterpart to ``composable_to_lerobot``.

    Args:
        in_dir: Root holding composable episodes (recursive discovery).
        out_dir: Destination dataset root. Created if missing.
        task: Single task string for the whole batch (LeRobot v2.1
            tasks.jsonl entry).
        config_path: YAML config (defaults to ``vega_1u_gripper.yaml``).
        fps: Override dataset frame rate; else read from each episode's
            ``collection.record_hz``.
        robot_type: Override ``info.json.robot_type``; else first
            episode's ``robot.embodiment``.
        max_gap_ns: Hard cap on the nearest-neighbor gap during state/
            action alignment. Image topics inherit the same nearest-
            neighbor mapping but don't enforce a gap cap (they're
            displayed at policy cadence, not used as truth signals).
        image_batch_size: Frames per add_episode call to the underlying
            video writer. 256 is balanced — smaller drops peak RSS at
            the cost of more dispatch overhead.
        queue_log_interval_s: If >0, the video writer logs encoder-queue
            depth (buffered/pending/done) every N seconds. Useful for
            verifying the encoder pool stays saturated; off by default.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    episode_dirs = discover_episodes(in_dir)
    if not episode_dirs:
        raise SystemExit(f"no composable episodes under {in_dir}")
    logger.info(
        "streaming export: %d episode(s) %s -> %s (batch=%d)",
        len(episode_dirs),
        in_dir,
        out_dir,
        image_batch_size,
    )

    config = ExporterConfig.from_yaml(config_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_index = 0
    (out_dir / LEGACY_TASKS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(out_dir / LEGACY_TASKS_PATH, mode="w") as w:
        w.write({"task_index": task_index, "task": task})

    feat_spec: dict[str, dict[str, Any]] | None = None
    resolved_fps: float | None = fps
    resolved_robot_type: str | None = robot_type
    frame_offset = 0
    total_frames = 0
    total_episodes = 0

    (out_dir / LEGACY_EPISODES_PATH).parent.mkdir(parents=True, exist_ok=True)
    (out_dir / EPISODES_STATS_PATH).parent.mkdir(parents=True, exist_ok=True)
    progress = tqdm.tqdm(episode_dirs, desc="episodes")
    with contextlib.ExitStack() as stack:
        episodes_writer = stack.enter_context(
            jsonlines.open(out_dir / LEGACY_EPISODES_PATH, mode="w")
        )
        stats_writer = stack.enter_context(
            jsonlines.open(out_dir / EPISODES_STATS_PATH, mode="w")
        )

        for ep_idx, ep_dir in enumerate(progress):
            # Resolve fps / robot_type lazily from the first episode's metadata.
            if resolved_fps is None or resolved_robot_type is None:
                from dexdata.metadata import load_metadata

                meta = load_metadata(ep_dir / "metadata.json")
                if resolved_fps is None:
                    rate = meta.collection.record_hz
                    if not rate:
                        raise RuntimeError(
                            "fps not provided and metadata.collection.record_hz "
                            "is missing/zero — pass --fps explicitly"
                        )
                    resolved_fps = float(rate)
                if resolved_robot_type is None:
                    resolved_robot_type = meta.robot.embodiment or "unknown"

            (
                length,
                episode_stats,
                image_dims,
                image_is_depth,
            ) = _convert_one_episode(
                ep_dir=ep_dir,
                ep_idx=ep_idx,
                out_root=out_dir,
                config=config,
                resolved_fps=resolved_fps,
                task_index=task_index,
                frame_offset=frame_offset,
                max_gap_ns=max_gap_ns,
                image_batch_size=image_batch_size,
                queue_log_interval_s=queue_log_interval_s,
            )

            if feat_spec is None:
                # Rebuild non-image arrays' shapes from one tiny re-read
                # for feat_spec. Cheaper to re-read state than carry it
                # back from the convert function.
                first_non_image = {
                    k: v for k, v in episode_stats.items() if k not in image_dims
                }
                # Use a fake array matching the recorded shape for
                # _build_streaming_features dispatch. We only need ndim
                # and shape[1:], which the stats output's "mean" tells us.
                synth_non_image = {}
                for k, s in first_non_image.items():
                    mean = s["mean"]
                    if isinstance(mean, list) and (
                        len(mean) == 1 and not isinstance(mean[0], list)
                    ):
                        synth_non_image[k] = np.empty((1,), dtype=np.float32)
                    else:
                        synth_non_image[k] = np.empty((1, len(mean)), dtype=np.float32)
                feat_spec = _build_streaming_features(
                    synth_non_image,
                    image_dims,
                    image_is_depth,
                    config,
                    resolved_fps,
                )

            stats_writer.write({"episode_index": ep_idx, "stats": episode_stats})
            stats_writer._fp.flush()
            episodes_writer.write(
                {"episode_index": ep_idx, "tasks": [task], "length": int(length)}
            )
            episodes_writer._fp.flush()

            frame_offset += length
            total_frames += length
            total_episodes += 1

    if feat_spec is None or total_episodes == 0:
        raise RuntimeError("No episodes converted")

    video_keys = [k for k, ft in feat_spec.items() if ft.get("dtype") == "video"]
    total_chunks = (
        (total_episodes + DEFAULT_CHUNK_SIZE - 1) // DEFAULT_CHUNK_SIZE
        if total_episodes > 0
        else 0
    )
    info = {
        "codebase_version": V21,
        "robot_type": resolved_robot_type or "unknown",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": total_episodes * len(video_keys),
        "total_chunks": total_chunks,
        "chunks_size": DEFAULT_CHUNK_SIZE,
        "fps": float(resolved_fps),
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": LEGACY_DATA_PATH_TEMPLATE,
        "video_path": LEGACY_VIDEO_PATH_TEMPLATE if video_keys else None,
        "features": feat_spec,
    }
    (out_dir / INFO_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(out_dir / INFO_PATH, "w") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)

    logger.info(
        "Wrote v2.1 dataset (streaming): %d episodes (%d frames) to %s",
        total_episodes,
        total_frames,
        out_dir,
    )


if __name__ == "__main__":
    tyro.cli(main)
