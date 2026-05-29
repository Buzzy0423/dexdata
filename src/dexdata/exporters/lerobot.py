# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""LeRobot **v2.1** dataset writer for composable episodes.

This module is a port of the patched ``dexdata.exporters.lerobot``
(which operates on the older ``EpisodeFeatures`` IR). Three spot edits
relative to the original:

* Input is :class:`CanonicalBundle` (composable's flat IR + composable
  ``EpisodeMetadata``).
* ``meta.record_rate_hz`` → ``meta.collection.record_hz``.
* ``meta.robot_type`` → ``meta.robot.embodiment``.

Everything else — parquet builder, depth quantisation, mp4 encoder
hookup, per-episode stats, combined-variance aggregation, info.json /
episodes.jsonl / tasks.jsonl streaming, the v2.1↔v2.0 downgrade, and
multi-source intersection-combine — is preserved from the patched file
because it operates on the canonical IR and is data-source-agnostic.

Public entry points:

* :func:`write` — :class:`Iterable[CanonicalBundle]` → v2.1 dataset.
* :func:`combine` — merge multiple v2.1 datasets via feature
  intersection.
* :func:`downgrade_v21_to_v20` — flatten per-episode stats into a
  single global ``stats.json``.

Per-camera mp4 encoding goes through the legacy
:class:`dexdata.video_writer.episode_writer.EpisodeVideoWriter` — it's
a clean leaf with uint8-frame input and a video config dataclass. The
composable layer will eventually replace it, but the writer is good as
it is.

Output layout (v2.1)::

    <out_root>/
        meta/info.json
        meta/episodes.jsonl
        meta/tasks.jsonl
        meta/episodes_stats.jsonl
        data/chunk-{NNN}/episode_{NNNNNN}.parquet
        videos/chunk-{NNN}/<observation.images.*>/episode_{NNNNNN}.mp4
"""

from __future__ import annotations

import contextlib
import json
import logging
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import jsonlines
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm

from .canonical import CanonicalBundle, DepthConfig, ExporterConfig, RGBConfig

# ---- v2.1 schema constants ------------------------------------------------
INFO_PATH = "meta/info.json"
STATS_PATH = "meta/stats.json"
EPISODES_STATS_PATH = "meta/episodes_stats.jsonl"
LEGACY_EPISODES_PATH = "meta/episodes.jsonl"
LEGACY_TASKS_PATH = "meta/tasks.jsonl"
LEGACY_DATA_PATH_TEMPLATE = (
    "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
)
LEGACY_VIDEO_PATH_TEMPLATE = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
)
DEFAULT_CHUNK_SIZE = 1000
V20 = "v2.0"
V21 = "v2.1"

IMAGE_PREFIX = "observation.images."
DEPTH_SUFFIX = ".depth"
RGB_SUFFIX = ".rgb"

logger = logging.getLogger(__name__)


# ---- Feature spec helpers -------------------------------------------------


def _feature_spec(arr: np.ndarray) -> dict[str, Any]:
    """info.json features entry for a non-video numpy column."""
    if arr.ndim == 1:
        shape = [1]
    else:
        shape = list(arr.shape[1:])
    dtype = "float32" if np.issubdtype(arr.dtype, np.floating) else "int64"
    return {"dtype": dtype, "shape": shape, "names": None}


def _video_feature_spec(
    H: int,
    W: int,
    C: int,
    fps: float,
    codec: str,
    pix_fmt: str,
    *,
    is_depth_map: bool,
) -> dict[str, Any]:
    return {
        "dtype": "video",
        "shape": [int(H), int(W), int(C)],
        "names": ["height", "width", "channels"],
        "info": {
            "video.fps": float(fps),
            "video.height": int(H),
            "video.width": int(W),
            "video.channels": int(C),
            "video.codec": codec,
            "video.pix_fmt": pix_fmt,
            "video.is_depth_map": bool(is_depth_map),
            "has_audio": False,
        },
    }


def _build_features(
    sample: dict[str, np.ndarray],
    fps: float,
    config: ExporterConfig,
) -> dict[str, dict[str, Any]]:
    """Build info.json ``features`` from a *materialised* sample.

    ``sample`` should be the output of :func:`_materialize_video_arrays` —
    image arrays already have the on-disk ``(T, H, W, 3)`` shape so we
    just read it off rather than recomputing from config.
    """
    features: dict[str, dict[str, Any]] = {}
    for key, arr in sample.items():
        if key.startswith(IMAGE_PREFIX):
            _, H, W, C = arr.shape
            is_depth = key.endswith(DEPTH_SUFFIX)
            stream_cfg = config.depth if is_depth else config.rgb
            features[key] = _video_feature_spec(
                H,
                W,
                C,
                fps,
                stream_cfg.codec,
                stream_cfg.pix_fmt,
                is_depth_map=False,
            )
        else:
            features[key] = _feature_spec(arr)

    features["timestamp"] = {"dtype": "float32", "shape": [1], "names": None}
    features["frame_index"] = {"dtype": "int64", "shape": [1], "names": None}
    features["episode_index"] = {"dtype": "int64", "shape": [1], "names": None}
    features["index"] = {"dtype": "int64", "shape": [1], "names": None}
    features["task_index"] = {"dtype": "int64", "shape": [1], "names": None}
    return features


# ---- Parquet table builder ------------------------------------------------


def _parquet_table(
    numpy_dict: dict[str, np.ndarray],
    episode_index: int,
    frame_offset: int,
    task_index: int,
    fps: float,
) -> tuple[pa.Table, int]:
    T = next(
        (v.shape[0] for k, v in numpy_dict.items() if not k.startswith(IMAGE_PREFIX)),
        None,
    )
    if T is None:
        raise ValueError("Episode has no non-video features to determine length")

    cols: dict[str, Any] = {}
    for key, arr in numpy_dict.items():
        if key.startswith(IMAGE_PREFIX):
            continue
        if arr.ndim == 1:
            cols[key] = arr.tolist()
        else:
            cols[key] = [row.tolist() for row in arr]

    cols["timestamp"] = [float(i) / float(fps) for i in range(T)]
    cols["frame_index"] = list(range(T))
    cols["episode_index"] = [int(episode_index)] * T
    cols["index"] = [frame_offset + i for i in range(T)]
    cols["task_index"] = [int(task_index)] * T
    return pa.table(cols), T


# ---- Depth quantisation ---------------------------------------------------


def _resize_rgb_for_video(arr: np.ndarray, cfg: RGBConfig) -> np.ndarray:
    """Optionally resize ``(T, H, W, 3)`` uint8 RGB to ``cfg.out_hw``.

    No-op when ``cfg.out_hw`` is ``None`` or already matches the source
    resolution. Uses ``cv2.INTER_AREA`` for downscale, ``INTER_LINEAR``
    for upscale.
    """
    if cfg.out_hw is None:
        return arr
    out_h, out_w = cfg.out_hw
    T, H, W, C = arr.shape
    if (H, W) == (out_h, out_w):
        return arr
    import cv2

    interp = cv2.INTER_AREA if (out_h * out_w < H * W) else cv2.INTER_LINEAR
    out = np.empty((T, out_h, out_w, C), dtype=np.uint8)
    for t in range(T):
        out[t] = cv2.resize(arr[t], (out_w, out_h), interpolation=interp)
    return out


def _quantize_depth_for_video(depth: np.ndarray, cfg: DepthConfig) -> np.ndarray:
    """``(T, H, W)`` float32 m → ``(T, out_h, out_w, 3)`` uint8.

    Linear quantise over ``cfg.clip_m`` → 0..255, replicate to 3
    channels (h264 pix_fmt yuv420p needs 3-channel input). Zero or
    invalid inputs map to 0. Resizes via ``cv2.INTER_NEAREST`` to
    avoid smoothing across depth discontinuities.
    """
    import cv2

    out_h, out_w = cfg.out_hw
    min_m, max_m = cfg.clip_m
    T = depth.shape[0]
    out = np.empty((T, out_h, out_w, 3), dtype=np.uint8)
    span = max(max_m - min_m, 1e-6)
    for t in range(T):
        d = depth[t]
        if d.shape != (out_h, out_w):
            d = cv2.resize(
                d.astype(np.float32),
                (out_w, out_h),
                interpolation=cv2.INTER_NEAREST,
            )
        clipped = np.clip(d, min_m, max_m)
        u8 = ((clipped - min_m) / span * 255.0).astype(np.uint8)
        u8[d <= 0] = 0
        out[t] = np.repeat(u8[:, :, None], 3, axis=2)
    return out


# ---- Per-episode stats ----------------------------------------------------


def _column_stats(col: np.ndarray) -> dict[str, Any]:
    """Per-column mean/std/min/max/count over the time axis (1D output)."""
    col = col.astype(np.float64)
    count = int(col.shape[0])
    if col.ndim == 1:
        return {
            "mean": [float(col.mean())],
            "std": [float(col.std())],
            "min": [float(col.min())],
            "max": [float(col.max())],
            "count": [count],
        }
    return {
        "mean": col.mean(axis=0).tolist(),
        "std": col.std(axis=0).tolist(),
        "min": col.min(axis=0).tolist(),
        "max": col.max(axis=0).tolist(),
        "count": [count],
    }


def _image_column_stats(arr: np.ndarray) -> dict[str, Any]:
    """Per-channel stats over a ``(T, H, W, C)`` uint8 image array.

    Reshaped to ``(C, 1, 1)`` to match LeRobot's training-time loader
    which insists on broadcast-compatible shapes for ``(C, H, W)``
    image tensors. Flat ``(C,)`` arrays are rejected by
    ``compute_stats._assert_type_and_shape``.

    Dispatches to a CUDA path when torch + a CUDA device are available
    (~20-70x faster on real teleop arrays); falls back to numpy otherwise.
    Both paths return numerically equivalent values to float32 precision.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return _image_column_stats_cuda(arr)
    except Exception as e:  # noqa: BLE001
        logger.debug("CUDA image stats unavailable, falling back to numpy: %s", e)
    return _image_column_stats_numpy(arr)


def _image_column_stats_numpy(arr: np.ndarray) -> dict[str, Any]:
    """CPU fallback: numpy reductions with float64 accumulators."""
    axes = (0, 1, 2)
    return {
        "mean": arr.mean(axis=axes, dtype=np.float64).reshape(-1, 1, 1).tolist(),
        "std": arr.std(axis=axes, dtype=np.float64).reshape(-1, 1, 1).tolist(),
        "min": arr.min(axis=axes).astype(np.float64).reshape(-1, 1, 1).tolist(),
        "max": arr.max(axis=axes).astype(np.float64).reshape(-1, 1, 1).tolist(),
        "count": [int(arr.shape[0])],
    }


def _image_column_stats_cuda(arr: np.ndarray) -> dict[str, Any]:
    """CUDA path: chunked sum + sum-of-squares + min/max per channel.

    Streams 64-frame chunks (~110 MB uint8) through GPU memory so peak
    VRAM stays bounded regardless of episode length. uint8 → float32
    on-device avoids the 8x float64 materialisation that dominated CPU
    time; float32 has more than enough range for 255²·H·W·T sums.
    """
    import torch

    T, H, W, C = arr.shape
    n = T * H * W
    device = torch.device("cuda")

    s1 = torch.zeros(C, dtype=torch.float64, device=device)
    s2 = torch.zeros(C, dtype=torch.float64, device=device)
    mn = torch.full((C,), 255, dtype=torch.uint8, device=device)
    mx = torch.zeros((C,), dtype=torch.uint8, device=device)

    chunk_t = 64
    for t0 in range(0, T, chunk_t):
        chunk_u8 = torch.from_numpy(arr[t0 : t0 + chunk_t]).to(
            device, non_blocking=True
        )
        chunk_f = chunk_u8.to(torch.float32)
        s1 += chunk_f.sum(dim=(0, 1, 2), dtype=torch.float64)
        s2 += (chunk_f * chunk_f).sum(dim=(0, 1, 2), dtype=torch.float64)
        mn = torch.minimum(mn, chunk_u8.amin(dim=(0, 1, 2)))
        mx = torch.maximum(mx, chunk_u8.amax(dim=(0, 1, 2)))
        del chunk_u8, chunk_f

    mean = (s1 / n).cpu().numpy()
    var = np.clip((s2 / n).cpu().numpy() - mean * mean, 0, None)
    std = np.sqrt(var)
    mn_np = mn.to(torch.float64).cpu().numpy()
    mx_np = mx.to(torch.float64).cpu().numpy()

    return {
        "mean": mean.reshape(-1, 1, 1).tolist(),
        "std": std.reshape(-1, 1, 1).tolist(),
        "min": mn_np.reshape(-1, 1, 1).tolist(),
        "max": mx_np.reshape(-1, 1, 1).tolist(),
        "count": [int(arr.shape[0])],
    }


def _materialize_video_arrays(
    numpy_dict: dict[str, np.ndarray], config: ExporterConfig
) -> dict[str, np.ndarray]:
    """Apply per-stream transforms to image arrays once.

    RGB: optional resize per ``config.rgb.out_hw``.
    Depth: quantise to uint8 3-channel per ``config.depth``.
    Non-image arrays pass through unchanged. Returned dict is suitable
    as input to both :func:`_encode_episode_videos` and
    :func:`_episode_stats` — neither does any more transforming.
    """
    out: dict[str, np.ndarray] = {}
    for key, arr in numpy_dict.items():
        if key.startswith(IMAGE_PREFIX):
            if key.endswith(DEPTH_SUFFIX):
                out[key] = _quantize_depth_for_video(arr, config.depth)
            else:
                out[key] = _resize_rgb_for_video(arr, config.rgb)
        else:
            out[key] = arr
    return out


# CLIP image normalization stats — values are in the [0, 1] domain that
# LeRobot's loader uses after dividing uint8 frames by 255. We hardcode
# these for every image feature so episode stats are essentially free
# and uniform across the dataset: no downstream training workflow
# actually consumes per-episode camera stats (LeRobot ACT/Diffusion
# overwrites with ImageNet stats when ``use_imagenet_stats=True`` — the
# default; gr00t's Eagle3_VLProcessor uses fixed ``[0.5, 0.5, 0.5]``).
# Real reductions saved ~no training quality and ~10-30% wall time per
# episode.
_HARDCODED_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
_HARDCODED_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


def _image_column_stats_hardcoded(arr: np.ndarray) -> dict[str, Any]:
    """Shape-compliant stats from a fixed CLIP normalization recipe.

    Returns ``(C, 1, 1)`` mean/std and scalar ``[0, 1]`` min/max — the
    shape LeRobot's training-time loader requires, with values in the
    normalized [0, 1] domain it consumes. The function only reads
    ``arr.shape`` (channel count + frame count); the pixel values
    aren't touched.
    """
    C = arr.shape[-1]
    if C == 3:
        mean = list(_HARDCODED_IMAGE_MEAN)
        std = list(_HARDCODED_IMAGE_STD)
    else:
        # Single-channel or 4-channel edge case: replicate the R-channel
        # number so the shape contract holds without inventing meaning.
        mean = [_HARDCODED_IMAGE_MEAN[0]] * C
        std = [_HARDCODED_IMAGE_STD[0]] * C
    return {
        "mean": [[[m]] for m in mean],
        "std": [[[s]] for s in std],
        "min": [[[0.0]] for _ in range(C)],
        "max": [[[1.0]] for _ in range(C)],
        "count": [int(arr.shape[0])],
    }


def _episode_stats(numpy_dict: dict[str, np.ndarray]) -> dict[str, dict]:
    """Per-feature stats for one episode.

    Numeric arrays use float64 reductions. Image arrays get the
    hardcoded CLIP-normalization stats (see
    :func:`_image_column_stats_hardcoded` for why we don't compute the
    real per-episode reduction).
    """
    stats: dict[str, dict] = {}
    for key, arr in numpy_dict.items():
        if key.startswith(IMAGE_PREFIX):
            stats[key] = _image_column_stats_hardcoded(arr)
        else:
            stats[key] = _column_stats(arr)
    return stats


def _aggregate_stats(episode_stats: list[dict[str, dict]]) -> dict[str, dict]:
    """Combined-variance aggregation across per-episode stats.

    Shape is preserved automatically since every operation is numpy-
    native: ``(D,)`` inputs stay ``(D,)``, ``(C, 1, 1)`` inputs stay
    ``(C, 1, 1)``.
    """
    all_keys = {k for ep in episode_stats for k in ep}
    aggregated: dict[str, dict] = {}
    for key in all_keys:
        counts, means, stds, mins, maxs = [], [], [], [], []
        for ep in episode_stats:
            s = ep.get(key)
            if s is None:
                continue
            counts.append(np.array(s["count"], dtype=np.float64))
            means.append(np.array(s["mean"], dtype=np.float64))
            stds.append(np.array(s["std"], dtype=np.float64))
            mins.append(np.array(s["min"], dtype=np.float64))
            maxs.append(np.array(s["max"], dtype=np.float64))
        if not counts:
            continue
        total = sum(counts)
        combined_mean = sum(c * m for c, m in zip(counts, means, strict=False)) / total
        combined_var = (
            sum(
                c * (s**2 + (m - combined_mean) ** 2)
                for c, s, m in zip(counts, stds, means, strict=False)
            )
            / total
        )
        aggregated[key] = {
            "mean": combined_mean.tolist(),
            "std": np.sqrt(combined_var).tolist(),
            "min": np.minimum.reduce(mins).tolist(),
            "max": np.maximum.reduce(maxs).tolist(),
            "count": total.tolist(),
        }
    return aggregated


# ---- Per-episode mp4 encoding ---------------------------------------------


def _encode_episode_videos(
    numpy_dict: dict[str, np.ndarray],
    episode_index: int,
    fps: float,
    out_root: Path,
    config: ExporterConfig,
) -> None:
    """Encode every ``observation.images.*`` array into per-episode mp4.

    Image arrays must already be in their on-disk form — RGB resized
    and depth quantised to uint8 3-channel; see
    :func:`_materialize_video_arrays`. RGB and depth use separate
    :class:`VideoWriter` pools because a writer is locked to one codec
    config; within a pool, all cameras dispatch to subprocess workers
    and encode concurrently.
    """
    # Lazy import — keeps the composable layer importable without the
    # legacy video_writer module being on the path (it currently is, but
    # this future-proofs against a leaf removal).
    from dexdata.video_writer.config import VideoWriterConfig
    from dexdata.video_writer.episode_writer import EpisodeVideoWriter

    episode_chunk = episode_index // DEFAULT_CHUNK_SIZE
    fps_i = int(round(fps))

    def _build_writer_cfg(c: RGBConfig | DepthConfig) -> VideoWriterConfig:
        return VideoWriterConfig(
            writer_type="parallel",
            video_backend="pyav",
            fps=fps_i,
            codec=c.codec,
            pix_fmt=c.pix_fmt,
            crf=c.crf,
            gop=c.gop,
        )

    # Bucket by stream type so all RGB cameras share one pool and all
    # depth cameras share another. add_episode queues; close_destination
    # marks done without waiting; the writer's __exit__ -> close_writer
    # blocks once at the end for every dispatched chunk to finish.
    rgb_items: list[tuple[str, np.ndarray]] = []
    depth_items: list[tuple[str, np.ndarray]] = []
    for key, arr in numpy_dict.items():
        if not key.startswith(IMAGE_PREFIX):
            continue
        if key.endswith(DEPTH_SUFFIX):
            depth_items.append((key, arr))
        else:
            rgb_items.append((key, arr))

    def _encode_set(
        items: list[tuple[str, np.ndarray]],
        stream_cfg: RGBConfig | DepthConfig,
    ) -> None:
        if not items:
            return
        with EpisodeVideoWriter(
            video_dir=out_root,
            writer_config=_build_writer_cfg(stream_cfg),
            input_format="rgb",
        ) as writer:
            for key, frames in items:
                dest = out_root / LEGACY_VIDEO_PATH_TEMPLATE.format(
                    episode_chunk=episode_chunk,
                    video_key=key,
                    episode_index=episode_index,
                )
                dest.parent.mkdir(parents=True, exist_ok=True)
                writer.write_episode(frames, dest)

    _encode_set(rgb_items, config.rgb)
    _encode_set(depth_items, config.depth)


# ---- Public: write v2.1 ---------------------------------------------------


def write(
    bundles: Iterable[CanonicalBundle],
    out_root: Path | str,
    *,
    config: ExporterConfig,
    task: str,
    fps: float | None = None,
    robot_type: str | None = None,
) -> None:
    """Write a sequence of :class:`CanonicalBundle` as a LeRobot v2.1 dataset.

    Args:
        bundles: episodes in canonical IR (see :func:`composable_to_canonical`).
        out_root: directory to write the dataset into (created if missing).
        config: conversion config (see :class:`ExporterConfig`).
            ``config.rgb`` / ``config.depth`` drive the per-stream
            mp4 encoder + depth quantisation.
        task: task description string written to ``meta/tasks.jsonl``.
        fps: dataset frame rate. ``None`` → read from each bundle's
            metadata (``collection.record_hz``); they must all agree.
        robot_type: robot label written to ``info.json``. ``None``
            falls back to the first bundle's ``robot.embodiment``.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    feat_spec: dict[str, dict] | None = None
    resolved_fps: float | None = fps
    resolved_robot_type: str | None = robot_type

    frame_offset = 0
    total_frames = 0
    total_episodes = 0
    task_index = 0

    # tasks.jsonl — depends only on inputs, write it up-front so partial
    # progress survives a crash mid-loop.
    (out_root / LEGACY_TASKS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(out_root / LEGACY_TASKS_PATH, mode="w") as w:
        w.write({"task_index": task_index, "task": task})

    # tqdm without total= — bundles may be a lazy generator of unknown length.
    progress = tqdm.tqdm(bundles, desc="episodes")

    (out_root / LEGACY_EPISODES_PATH).parent.mkdir(parents=True, exist_ok=True)
    (out_root / EPISODES_STATS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as stack:
        episodes_writer = stack.enter_context(
            jsonlines.open(out_root / LEGACY_EPISODES_PATH, mode="w")
        )
        stats_writer = stack.enter_context(
            jsonlines.open(out_root / EPISODES_STATS_PATH, mode="w")
        )

        for ep_idx, bundle in enumerate(progress):
            if bundle is None:
                logger.warning("Skipping empty bundle at index %d", ep_idx)
                continue
            numpy_dict = bundle.arrays
            meta = bundle.metadata

            if resolved_fps is None:
                rate = meta.collection.record_hz
                if not rate:
                    raise RuntimeError(
                        "fps not provided and metadata.collection.record_hz "
                        "is missing/zero — pass fps= explicitly to write()"
                    )
                resolved_fps = float(rate)
            if resolved_robot_type is None:
                resolved_robot_type = meta.robot.embodiment or "unknown"

            # Apply per-stream transforms (RGB resize, depth quantise) once
            # here so encode + stats + feature spec all see identical
            # on-disk frames. Non-image arrays pass through unchanged.
            materialised = _materialize_video_arrays(numpy_dict, config)

            if feat_spec is None:
                feat_spec = _build_features(
                    materialised,
                    resolved_fps,
                    config,
                )

            episode_chunk = ep_idx // DEFAULT_CHUNK_SIZE
            parquet_path = out_root / LEGACY_DATA_PATH_TEMPLATE.format(
                episode_chunk=episode_chunk,
                episode_index=ep_idx,
            )
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            table, length = _parquet_table(
                materialised,
                ep_idx,
                frame_offset,
                task_index,
                resolved_fps,
            )
            pq.write_table(table, parquet_path)

            _encode_episode_videos(
                materialised,
                ep_idx,
                resolved_fps,
                out_root,
                config,
            )

            # Stream this episode's metadata to disk now — drop the
            # arrays from memory after. Flush each writer's underlying
            # file object after every record so a kill -9 / OOM-kill
            # mid-conversion can't leave parquet+mp4 on disk for an
            # episode that's missing from episodes.jsonl (LeRobot's
            # reader trusts episodes.jsonl, so such files would become
            # invisible orphans). flush() only — no fsync; this is
            # Python-crash safety, not power-loss safety.
            stats_writer.write(
                {
                    "episode_index": ep_idx,
                    "stats": _episode_stats(materialised),
                }
            )
            stats_writer._fp.flush()
            episodes_writer.write(
                {
                    "episode_index": ep_idx,
                    "tasks": [task],
                    "length": int(length),
                }
            )
            episodes_writer._fp.flush()
            frame_offset += length
            total_frames += length
            total_episodes += 1
            # Release every per-episode array before the for-loop calls
            # next() on the bundle iterator. The for-loop only rebinds
            # ``bundle`` *after* the iterator returns, so without these
            # dels the previous episode's full video stack (~10 GB BGR
            # bundle.arrays + ~4.5 GB resized RGB materialised + pyarrow
            # table) is alive *while* the upstream generator allocates
            # the next episode's read/align/canonical copies. On large
            # episodes that doubles peak RAM unnecessarily.
            del numpy_dict, materialised, table, bundle

    if feat_spec is None or total_episodes == 0:
        raise RuntimeError("No episodes converted")

    # info.json — written last because it needs the final totals.
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
    (out_root / INFO_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(out_root / INFO_PATH, "w") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)

    logger.info(
        "Wrote v2.1 dataset: %d episodes (%d frames) to %s",
        total_episodes,
        total_frames,
        out_root,
    )


# ---- Public: combine multiple v2.1 datasets -------------------------------


# System columns that the parquet builder always emits and that combine()
# must always retain — they are not "features" in the info.json sense but
# are required for the v2.1 reader to function.
_SYSTEM_COLS = frozenset(
    {"timestamp", "frame_index", "episode_index", "index", "task_index"}
)


def _read_jsonl(path: Path) -> list[dict]:
    with jsonlines.open(path, mode="r") as r:
        return list(r)


def combine(
    srcs: list[Path | str],
    out: Path | str,
    *,
    strategy: str = "intersection",
) -> None:
    """Combine multiple LeRobot v2.1 datasets (produced by this writer) into one.

    Episodes are concatenated in the order ``srcs`` is given. Episode indices,
    frame indices, and the global ``index`` column are renumbered. Per-episode
    metadata (``episodes.jsonl``, ``episodes_stats.jsonl``) is streamed to disk
    and flushed inline so a kill-9 mid-combine can't strand parquets that
    aren't listed in the metadata. ``info.json`` is only written on clean
    completion.

    Under ``strategy="intersection"``: only feature keys present in EVERY input
    with matching dtype and shape are retained. Source-specific keys are
    silently dropped (with a ``logger.info``).
    """
    if strategy != "intersection":
        raise NotImplementedError(
            f"combine: strategy={strategy!r} not implemented (only 'intersection')"
        )
    if not srcs:
        raise ValueError("combine: srcs must contain at least one source")

    src_paths = [Path(s) for s in srcs]
    out_root = Path(out)
    out_root.mkdir(parents=True, exist_ok=True)

    # 1. Load each source's info.json + tasks.jsonl + episodes.jsonl.
    src_infos: list[dict] = []
    src_tasks: list[list[dict]] = []
    src_episodes: list[list[dict]] = []
    for sp in src_paths:
        info_p = sp / INFO_PATH
        if not info_p.is_file():
            raise FileNotFoundError(f"{sp}: missing {INFO_PATH}")
        src_infos.append(json.loads(info_p.read_text()))
        src_tasks.append(_read_jsonl(sp / LEGACY_TASKS_PATH))
        src_episodes.append(_read_jsonl(sp / LEGACY_EPISODES_PATH))

    # 2. Validate compatibility.
    for sp, info in zip(src_paths, src_infos, strict=True):
        cv = str(info.get("codebase_version", ""))
        if cv != V21:
            raise ValueError(f"{sp}: codebase_version={cv!r}, expected {V21!r}")

    fps_values = [float(info["fps"]) for info in src_infos]
    if len(set(fps_values)) != 1:
        raise ValueError(f"combine: fps differs across inputs: {fps_values}")
    shared_fps = fps_values[0]

    robot_types = [info.get("robot_type", "unknown") for info in src_infos]
    if len(set(robot_types)) != 1:
        logger.warning(
            "combine: robot_type differs across inputs %s; using first (%s)",
            robot_types,
            robot_types[0],
        )
    shared_robot_type = robot_types[0]

    chunks_size = int(src_infos[0].get("chunks_size", DEFAULT_CHUNK_SIZE))
    for sp, info in zip(src_paths, src_infos, strict=True):
        cs = int(info.get("chunks_size", DEFAULT_CHUNK_SIZE))
        if cs != chunks_size:
            raise ValueError(
                f"{sp}: chunks_size={cs} differs from first source ({chunks_size})"
            )

    data_template = src_infos[0].get("data_path", LEGACY_DATA_PATH_TEMPLATE)
    video_template = src_infos[0].get("video_path", LEGACY_VIDEO_PATH_TEMPLATE)
    for sp, info in zip(src_paths, src_infos, strict=True):
        if info.get("data_path") != data_template:
            raise ValueError(f"{sp}: data_path template differs from first source")
        vt = info.get("video_path")
        if vt is not None and video_template is not None and vt != video_template:
            raise ValueError(f"{sp}: video_path template differs from first source")

    # 3. Compute intersection of features.
    feature_dicts = [info.get("features", {}) for info in src_infos]
    candidate_keys = set(feature_dicts[0].keys())
    for fd in feature_dicts[1:]:
        candidate_keys &= set(fd.keys())

    retained_keys: list[str] = []
    for key in feature_dicts[0]:
        if key not in candidate_keys:
            continue
        if key in _SYSTEM_COLS:
            continue
        spec0 = feature_dicts[0][key]
        dtype0 = spec0.get("dtype")
        shape0 = list(spec0.get("shape", []))
        ok = True
        for fd in feature_dicts[1:]:
            spec = fd[key]
            if spec.get("dtype") != dtype0 or list(spec.get("shape", [])) != shape0:
                logger.warning(
                    "combine: dropping %r — schema mismatch across sources "
                    "(dtype/shape)",
                    key,
                )
                ok = False
                break
        if ok:
            retained_keys.append(key)

    for src_idx, fd in enumerate(feature_dicts):
        dropped = [
            k
            for k in fd
            if k not in retained_keys
            and k not in _SYSTEM_COLS
            and k not in candidate_keys
        ]
        if dropped:
            logger.info(
                "combine: source %d (%s) — dropping %d source-specific keys: %s",
                src_idx,
                src_paths[src_idx],
                len(dropped),
                sorted(dropped),
            )

    if not retained_keys:
        raise ValueError("no shared feature keys after intersection")

    retained_set = set(retained_keys)
    retained_video_keys = [
        k for k in retained_keys if feature_dicts[0][k].get("dtype") == "video"
    ]

    # 4. Merge tasks.jsonl. Preserve first-appearance order.
    unified_tasks: list[str] = []
    seen_tasks: set[str] = set()
    task_remaps: list[dict[int, int]] = []
    for tasks in src_tasks:
        local_remap: dict[int, int] = {}
        for rec in sorted(tasks, key=lambda r: int(r["task_index"])):
            old_idx = int(rec["task_index"])
            t_str = str(rec["task"])
            if t_str not in seen_tasks:
                seen_tasks.add(t_str)
                unified_tasks.append(t_str)
            new_idx = unified_tasks.index(t_str)
            local_remap[old_idx] = new_idx
        task_remaps.append(local_remap)

    (out_root / LEGACY_TASKS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(out_root / LEGACY_TASKS_PATH, mode="w") as w:
        for new_idx, t_str in enumerate(unified_tasks):
            w.write({"task_index": new_idx, "task": t_str})

    (out_root / LEGACY_EPISODES_PATH).parent.mkdir(parents=True, exist_ok=True)
    (out_root / EPISODES_STATS_PATH).parent.mkdir(parents=True, exist_ok=True)

    new_ep_idx = 0
    global_frame_offset = 0
    total_frames = 0

    with contextlib.ExitStack() as stack:
        episodes_writer = stack.enter_context(
            jsonlines.open(out_root / LEGACY_EPISODES_PATH, mode="w")
        )
        stats_writer = stack.enter_context(
            jsonlines.open(out_root / EPISODES_STATS_PATH, mode="w")
        )

        for src_idx, sp in enumerate(src_paths):
            episodes = src_episodes[src_idx]
            remap = task_remaps[src_idx]

            stats_path = sp / EPISODES_STATS_PATH
            if not stats_path.is_file():
                raise FileNotFoundError(f"{sp}: missing {EPISODES_STATS_PATH}")
            stats_by_ep: dict[int, dict] = {
                int(rec["episode_index"]): rec for rec in _read_jsonl(stats_path)
            }

            for ep_record in sorted(episodes, key=lambda r: int(r["episode_index"])):
                src_ep_idx = int(ep_record["episode_index"])
                src_chunk = src_ep_idx // chunks_size
                src_parquet = (
                    sp
                    / "data"
                    / f"chunk-{src_chunk:03d}"
                    / f"episode_{src_ep_idx:06d}.parquet"
                )
                tbl = pq.read_table(src_parquet)

                keep_cols = [
                    c
                    for c in tbl.column_names
                    if c in retained_set or c in _SYSTEM_COLS
                ]
                tbl = tbl.select(keep_cols)
                length = len(tbl)

                ep_idx_arr = pa.array([new_ep_idx] * length, type=pa.int64())
                global_idx_arr = pa.array(
                    [global_frame_offset + i for i in range(length)],
                    type=pa.int64(),
                )
                old_task_idx_col = tbl.column("task_index").to_pylist()
                new_task_idx = [int(remap[int(ti)]) for ti in old_task_idx_col]
                task_idx_arr = pa.array(new_task_idx, type=pa.int64())

                ep_col_idx = tbl.column_names.index("episode_index")
                tbl = tbl.set_column(ep_col_idx, "episode_index", ep_idx_arr)
                idx_col_idx = tbl.column_names.index("index")
                tbl = tbl.set_column(idx_col_idx, "index", global_idx_arr)
                ti_col_idx = tbl.column_names.index("task_index")
                tbl = tbl.set_column(ti_col_idx, "task_index", task_idx_arr)

                new_chunk = new_ep_idx // chunks_size
                dst_parquet = (
                    out_root
                    / "data"
                    / f"chunk-{new_chunk:03d}"
                    / f"episode_{new_ep_idx:06d}.parquet"
                )
                dst_parquet.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(tbl, dst_parquet)

                for vkey in retained_video_keys:
                    src_video = (
                        sp
                        / "videos"
                        / f"chunk-{src_chunk:03d}"
                        / vkey
                        / f"episode_{src_ep_idx:06d}.mp4"
                    )
                    dst_video = (
                        out_root
                        / "videos"
                        / f"chunk-{new_chunk:03d}"
                        / vkey
                        / f"episode_{new_ep_idx:06d}.mp4"
                    )
                    dst_video.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src_video, dst_video)

                episodes_writer.write(
                    {
                        "episode_index": new_ep_idx,
                        "tasks": ep_record.get("tasks", []),
                        "length": int(ep_record["length"]),
                    }
                )
                episodes_writer._fp.flush()

                src_stats_rec = stats_by_ep.get(src_ep_idx)
                if src_stats_rec is None:
                    raise RuntimeError(
                        f"{sp}: episodes_stats.jsonl missing entry for "
                        f"episode_index={src_ep_idx}"
                    )
                src_stats = src_stats_rec.get("stats", {})
                filtered_stats = {
                    k: v for k, v in src_stats.items() if k in retained_set
                }
                stats_writer.write(
                    {
                        "episode_index": new_ep_idx,
                        "stats": filtered_stats,
                    }
                )
                stats_writer._fp.flush()

                ep_len = int(ep_record["length"])
                total_frames += ep_len
                global_frame_offset += ep_len
                new_ep_idx += 1

    total_chunks = (
        (new_ep_idx + chunks_size - 1) // chunks_size if new_ep_idx > 0 else 0
    )
    out_features: dict[str, dict] = {}
    for key in retained_keys:
        out_features[key] = feature_dicts[0][key]
    out_features["timestamp"] = {"dtype": "float32", "shape": [1], "names": None}
    out_features["frame_index"] = {"dtype": "int64", "shape": [1], "names": None}
    out_features["episode_index"] = {"dtype": "int64", "shape": [1], "names": None}
    out_features["index"] = {"dtype": "int64", "shape": [1], "names": None}
    out_features["task_index"] = {"dtype": "int64", "shape": [1], "names": None}

    info = {
        "codebase_version": V21,
        "robot_type": shared_robot_type or "unknown",
        "total_episodes": new_ep_idx,
        "total_frames": total_frames,
        "total_tasks": len(unified_tasks),
        "total_videos": new_ep_idx * len(retained_video_keys),
        "total_chunks": total_chunks,
        "chunks_size": chunks_size,
        "fps": float(shared_fps),
        "splits": {"train": f"0:{new_ep_idx}"},
        "data_path": data_template,
        "video_path": video_template if retained_video_keys else None,
        "features": out_features,
    }
    (out_root / INFO_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(out_root / INFO_PATH, "w") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)

    logger.info(
        "Combined %d sources → %d episodes (%d frames) at %s",
        len(src_paths),
        new_ep_idx,
        total_frames,
        out_root,
    )


# ---- Public: v2.1 → v2.0 downgrade ----------------------------------------


def downgrade_v21_to_v20(
    src: Path | str,
    out: Path | str | None = None,
) -> Path:
    """Downgrade a LeRobot v2.1 dataset to v2.0 layout.

    Aggregates ``meta/episodes_stats.jsonl`` (per-episode) into a single
    global ``meta/stats.json``, then removes ``episodes_stats.jsonl`` and
    flips ``info.codebase_version`` to ``v2.0``.
    """
    src = Path(src)
    if out is None:
        work = src
    else:
        out = Path(out)
        if out.resolve() != src.resolve():
            if out.exists():
                shutil.rmtree(out)
            shutil.copytree(src, out)
        work = out

    info_path = work / INFO_PATH
    eps_stats_path = work / EPISODES_STATS_PATH

    if not info_path.is_file():
        raise FileNotFoundError(f"{work}: missing meta/info.json")
    info = json.loads(info_path.read_text())
    if str(info.get("codebase_version", "")) == V20:
        logger.info("%s: already v2.0, nothing to do", work)
        return work
    if not eps_stats_path.is_file():
        raise FileNotFoundError(
            f"{work}: cannot downgrade — meta/episodes_stats.jsonl is missing"
        )

    per_episode: list[dict] = []
    with jsonlines.open(eps_stats_path, mode="r") as r:
        for rec in r:
            per_episode.append(rec["stats"])

    # v2.0's stats.json is purely numeric — image features have no
    # meaningful global aggregate semantic in the flat v2.0 layout.
    numeric_only = [
        {k: v for k, v in s.items() if not k.startswith(IMAGE_PREFIX)}
        for s in per_episode
    ]
    aggregated = _aggregate_stats(numeric_only)

    stats_path = work / STATS_PATH
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(aggregated, f, indent=2)

    eps_stats_path.unlink()
    info["codebase_version"] = V20
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)

    logger.info("Downgraded %s to %s", work, V20)
    return work
