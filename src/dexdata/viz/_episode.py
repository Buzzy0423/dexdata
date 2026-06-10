# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Rerun-based visualization for composable :class:`Episode` objects.

The legacy :mod:`dexdata.viz` package is wired against ``RobotState`` and
``RobotAction`` proto-shaped data; this module is its composable peer,
driven entirely off the topic-path identity of the spec. There are no
hardcoded field names and no embodiment knowledge — every signal is
dispatched off its container type:

* ``NumericArray``  → one Rerun ``Scalars`` series per element.
* ``foxglove.CompressedVideo`` → one ``Image`` view; frames logged per timestep.
* ``DepthImage``    → one ``Image`` view, colorized via matplotlib viridis,
  invalid (0.0) pixels rendered black.
* ``foxglove.Pose`` → seven ``Scalars`` series under ``{topic}/pos/{x,y,z}``
  and ``{topic}/quat/{x,y,z,w}`` (xyzw order, matching :mod:`handlers.pose`).

The ``timestamp`` Rerun timeline uses each signal's own
``recv_timestamps`` (MCAP ``log_time``) directly — no implicit
fps-based clock — so signals at different rates show up at their
actual times, in nanoseconds. ``log_time`` is the only timestamp
domain coherent across multi-publisher recordings (each publisher has
its own clock; the recorder stamps ``log_time`` uniformly), so it's
the right choice for cross-stream alignment in the viewer. A
synthetic ``frame_index`` timeline is also published using the
longest signal as the reference, so callers can scrub by frame even
when the episode is multi-rate.

Two entry points:

* :func:`visualize_episode` — log an in-memory :class:`Episode` to Rerun.
* :func:`visualize_episode_dir` — convenience that opens a composable
  episode directory via :class:`Reader` and forwards.
"""

from __future__ import annotations

import gc
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np

from ..mcap_utils.episode import Episode
from ..mcap_utils.reader import Reader

logger = logging.getLogger(__name__)


def visualize_episode(
    episode: Episode,
    *,
    app_name: str = "dexdata_composable",
    mode: str = "local",
    save: bool = False,
    output_path: Path | str | None = None,
) -> Path | None:
    """Log an :class:`Episode` to Rerun and either spawn the viewer or save a ``.rrd``.

    Args:
        episode: Composable episode (per-topic stacked arrays + timestamps).
        app_name: Rerun application name.
        mode: ``"local"`` spawns the desktop viewer; ``"distant"`` runs a
            web server (legacy-style). Ignored when ``save=True``.
        save: If True, write a ``.rrd`` file to ``output_path`` instead.
        output_path: Required when ``save=True``.

    Returns:
        The saved ``.rrd`` path when ``save=True``, otherwise ``None``.
    """
    if save and output_path is None:
        raise ValueError("output_path required when save=True")

    import rerun as rr
    import rerun.blueprint as rrb

    rr.init(app_name, spawn=False)
    gc.collect()

    web_viewer_url: str | None = None
    if not save and mode == "distant":
        recording = rr.get_global_data_recording()
        if recording is None:
            raise RuntimeError("Rerun recording was not initialized")

        grpc_port = int(os.environ.get("DEXDATA_RERUN_GRPC_PORT", "9876"))
        web_port = int(os.environ.get("DEXDATA_RERUN_WEB_PORT", "9090"))
        server_memory_limit = os.environ.get(
            "DEXDATA_RERUN_SERVER_MEMORY_LIMIT", "4GiB"
        )
        server_uri = recording.serve_grpc(
            grpc_port=grpc_port,
            server_memory_limit=server_memory_limit,
        )
        rr.serve_web_viewer(
            web_port=web_port,
            open_browser=False,
            connect_to=server_uri,
        )
        web_viewer_url = (
            f"http://localhost:{web_port}/?url={quote(server_uri, safe='')}"
        )
        logger.info(f"Rerun web viewer: {web_viewer_url}")

    spec = episode.spec
    blueprint = _build_blueprint(spec, rrb)
    rr.send_blueprint(blueprint)

    # Pick the longest signal as the reference for the synthetic frame_index
    # timeline. Multi-rate viz is still meaningful: shorter signals just stop
    # advancing once they end.
    n_frames_per_topic = {topic: arr.shape[0] for topic, arr in episode.signals.items()}
    if not any(n_frames_per_topic.values()):
        logger.warning("Episode has no frames on any topic; nothing to log.")
    else:
        max_topic = max(n_frames_per_topic, key=lambda t: n_frames_per_topic[t])
        logger.info(
            f"Reference for frame_index timeline: {max_topic} "
            f"({n_frames_per_topic[max_topic]} frames)"
        )

    import time as _time

    n_signals = sum(1 for s in spec.signals if episode.signals[s.topic].shape[0] > 0)
    logger.info(f"Logging {n_signals} signals to Rerun…")
    t_log_start = _time.perf_counter()
    for i, signal in enumerate(spec.signals, start=1):
        topic = signal.topic
        arr = episode.signals[topic]
        ts = episode.recv_timestamps[topic]
        if arr.shape[0] == 0:
            continue
        ctype = signal.container.type_name
        t0 = _time.perf_counter()
        if ctype == "NumericArray":
            _log_numeric_array(rr, topic, arr, ts)
        elif ctype == "foxglove.CompressedVideo":
            _log_video(rr, topic, arr, ts)
        elif ctype == "DepthImage":
            _log_depth(rr, topic, arr, ts)
        elif ctype == "foxglove.Pose":
            _log_pose(rr, topic, arr, ts)
        else:
            logger.warning(f"{topic}: no viz handler for container {ctype}")
            continue
        dt = _time.perf_counter() - t0
        logger.info(
            f"  [{i:2d}/{len(spec.signals)}] {ctype:<25} "
            f"{topic} ({arr.shape[0]} frames) — {dt:.2f}s"
        )
    logger.info(f"Logged in {_time.perf_counter() - t_log_start:.2f}s")

    if save:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Writing .rrd to {out}…")
        t0 = _time.perf_counter()
        rr.save(str(out))
        logger.info(
            f"Saved Rerun recording to {out} ({_time.perf_counter() - t0:.2f}s)"
        )
        return out

    if mode == "local":
        return _spawn_viewer_with_temp_recording(rr)
    if mode == "distant":
        logger.info(
            f"Serving Rerun web viewer at {web_viewer_url} (Ctrl+C to exit)."
        )
        try:
            import time

            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    return None


def visualize_episode_dir(
    out_dir: Path | str,
    *,
    app_name: str | None = None,
    mode: str = "local",
    save: bool = False,
    output_path: Path | str | None = None,
) -> Path | None:
    """Open a composable episode directory and visualize."""
    import time as _time

    out_dir = Path(out_dir)
    logger.info(f"Reading episode from {out_dir}…")
    t0 = _time.perf_counter()
    reader = Reader(out_dir)
    episode = reader.read_episode()
    logger.info(
        f"Read episode ({len(episode.signals)} topics, "
        f"{_time.perf_counter() - t0:.2f}s)"
    )
    return visualize_episode(
        episode,
        app_name=app_name or out_dir.name,
        mode=mode,
        save=save,
        output_path=output_path,
    )


# ---- per-container loggers ------------------------------------------------


def _log_numeric_array(rr: Any, topic: str, arr: np.ndarray, ts: np.ndarray) -> None:
    """Log each element of a (T, *shape) array as a separate scalar series.

    Uses Rerun's columnar :func:`send_columns` API — one call per
    element-dimension covering the full timeline — instead of a
    per-frame ``rr.log`` loop. Drops FFI overhead from
    ``n_frames * (2 + n_elems)`` calls to just ``n_elems`` per topic.
    """
    flat = arr.reshape(arr.shape[0], -1)
    n_frames, n_elems = flat.shape
    if n_frames == 0:
        return
    indexes = _time_columns(rr, ts, n_frames)
    for d in range(n_elems):
        rr.send_columns(
            entity_path=f"{topic}/{d}",
            indexes=indexes,
            columns=rr.Scalars.columns(scalars=flat[:, d].astype(np.float64)),
        )


# Keep image chunks small enough for live gRPC/Web transport. Large recordings
# can otherwise become a single several-hundred-MiB message that never reaches
# the viewer, even though saving the same recording to an RRD file succeeds.
_IMAGE_CHUNK_BYTES = 64 * 1024 * 1024


def _image_chunk_size(H: int, W: int, C: int) -> int:
    """Number of frames per send_columns batch given the Arrow i32 cap."""
    per_frame = H * W * C
    return max(1, _IMAGE_CHUNK_BYTES // per_frame)


def _send_image_columns(
    rr: Any,
    topic: str,
    frames_thwc: np.ndarray,
    ts: np.ndarray,
    color_model: str,
) -> None:
    """Send a (T, H, W, C) uint8 stack to Rerun in i32-safe chunks.

    Per-chunk size is computed so the concatenated buffer stays under
    ``_IMAGE_CHUNK_BYTES`` — Rerun's Arrow backend uses ``Binary`` (i32
    offsets) for image buffers and silently drops any column whose
    total bytes exceed the i32 max (~2.1 GB), which is why we have
    to chunk for any camera that exceeds it.
    """
    T, H, W, C = frames_thwc.shape
    if T == 0:
        return
    fmt = rr.components.ImageFormat(
        width=W, height=H, color_model=color_model, channel_datatype="U8"
    )
    frames_thwc = np.ascontiguousarray(frames_thwc)
    chunk = _image_chunk_size(H, W, C)
    for t0 in range(0, T, chunk):
        t1 = min(t0 + chunk, T)
        n = t1 - t0
        buffers = [frames_thwc[t].tobytes() for t in range(t0, t1)]
        rr.send_columns(
            entity_path=topic,
            indexes=[
                rr.TimeColumn(
                    "timestamp", timestamp=ts[t0:t1].astype(np.float64) * 1e-9
                ),
                rr.TimeColumn(
                    "frame_index", sequence=np.arange(t0, t1, dtype=np.int64)
                ),
            ],
            columns=rr.Image.columns(buffer=buffers, format=[fmt] * n),
        )


def _log_video(rr: Any, topic: str, arr: np.ndarray, ts: np.ndarray) -> None:
    """Log a (T, H, W, 3) RGB uint8 stack as JPEG-encoded image columns.

    The composable ``CompressedVideo`` handler decodes to RGB directly,
    but sending raw RGB makes even short recordings several GiB in the
    web viewer. JPEG encoding keeps remote visualization practical without
    changing the source MCAP recording.
    """
    import cv2

    quality = int(os.environ.get("DEXDATA_RERUN_JPEG_QUALITY", "90"))
    color_order = os.environ.get(
        "DEXDATA_RERUN_VIDEO_COLOR_ORDER", "RGB"
    ).upper()
    if color_order not in {"RGB", "BGR"}:
        raise ValueError(
            "DEXDATA_RERUN_VIDEO_COLOR_ORDER must be RGB or BGR"
        )
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    batch_size = 64

    for t0 in range(0, arr.shape[0], batch_size):
        t1 = min(t0 + batch_size, arr.shape[0])
        blobs: list[bytes] = []
        for frame in arr[t0:t1]:
            bgr = (
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if color_order == "RGB"
                else frame
            )
            ok, encoded = cv2.imencode(".jpg", bgr, encode_params)
            if not ok:
                raise RuntimeError(f"Failed to JPEG-encode frame for {topic}")
            blobs.append(encoded.tobytes())

        rr.send_columns(
            entity_path=topic,
            indexes=[
                rr.TimeColumn(
                    "timestamp", timestamp=ts[t0:t1].astype(np.float64) * 1e-9
                ),
                rr.TimeColumn(
                    "frame_index", sequence=np.arange(t0, t1, dtype=np.int64)
                ),
            ],
            columns=rr.EncodedImage.columns(
                blob=blobs,
                media_type=["image/jpeg"] * len(blobs),
            ),
        )


def _log_depth(rr: Any, topic: str, arr: np.ndarray, ts: np.ndarray) -> None:
    """Log a (T, H, W) float32 depth stack as colorized images, columnarly."""
    if arr.shape[0] == 0:
        return
    colored = _colorize_depth_stack(arr)  # (T, H, W, 3) uint8 RGB
    _send_image_columns(rr, topic, colored, ts, color_model="RGB")


def _log_pose(rr: Any, topic: str, arr: np.ndarray, ts: np.ndarray) -> None:
    """Log a (T, 7) [px, py, pz, qx, qy, qz, qw] pose stack as 7 scalars.

    Columnar via :func:`send_columns` — same reason as
    :func:`_log_numeric_array`.
    """
    labels = ("pos/x", "pos/y", "pos/z", "quat/x", "quat/y", "quat/z", "quat/w")
    n_frames = arr.shape[0]
    if n_frames == 0:
        return
    indexes = _time_columns(rr, ts, n_frames)
    for i, label in enumerate(labels):
        rr.send_columns(
            entity_path=f"{topic}/{label}",
            indexes=indexes,
            columns=rr.Scalars.columns(scalars=arr[:, i].astype(np.float64)),
        )


def _time_columns(rr: Any, ts: np.ndarray, n_frames: int) -> list:
    """Shared ``timestamp`` + ``frame_index`` TimeColumn pair."""
    return [
        rr.TimeColumn("timestamp", timestamp=ts.astype(np.float64) * 1e-9),
        rr.TimeColumn("frame_index", sequence=np.arange(n_frames, dtype=np.int64)),
    ]


# ---- blueprint ------------------------------------------------------------


def _build_blueprint(spec: Any, rrb: Any) -> Any:
    """Two-pane layout: visual signals on the left, scalar groups on the right."""
    visual_views: list[Any] = []
    scalar_views: list[Any] = []
    seen_origins: set[str] = set()

    for signal in spec.signals:
        topic = signal.topic
        ctype = signal.container.type_name
        if ctype in ("foxglove.CompressedVideo", "DepthImage"):
            # Images are logged on the origin entity itself. Rerun 0.33's
            # default "$origin/**" query only includes descendants.
            visual_views.append(
                rrb.Spatial2DView(name=topic, origin=topic, contents=topic)
            )
        elif ctype in ("NumericArray", "foxglove.Pose"):
            # Group scalar series by topic — one TimeSeriesView per signal.
            if topic not in seen_origins:
                seen_origins.add(topic)
                scalar_views.append(rrb.TimeSeriesView(name=topic, origin=topic))

    if visual_views and scalar_views:
        n_cols = min(2, len(visual_views))
        return rrb.Blueprint(
            rrb.Horizontal(
                rrb.Grid(*visual_views, grid_columns=n_cols),
                rrb.Vertical(*scalar_views),
                column_shares=[2, 1],
            ),
            collapse_panels=False,
        )
    if visual_views:
        n_cols = min(2, len(visual_views))
        return rrb.Blueprint(
            rrb.Grid(*visual_views, grid_columns=n_cols),
            collapse_panels=False,
        )
    if scalar_views:
        return rrb.Blueprint(rrb.Vertical(*scalar_views), collapse_panels=False)
    return rrb.Blueprint(collapse_panels=False)


# ---- helpers --------------------------------------------------------------


def _colorize_depth_stack(depth: np.ndarray) -> np.ndarray:
    """Colorize a (T, H, W) float32 depth stack to (T, H, W, 3) uint8.

    Uses per-stream global min/max over valid pixels for stable colors
    across frames; invalid (0.0) pixels — composable's invalid sentinel,
    not the legacy NaN — are rendered black.
    """
    from matplotlib import cm

    valid_mask = depth > 0
    if valid_mask.any():
        valid = depth[valid_mask]
        d_min, d_max = float(valid.min()), float(valid.max())
    else:
        d_min, d_max = 0.0, 1.0
    span = max(d_max - d_min, 1e-6)

    norm = np.clip((depth - d_min) / span, 0.0, 1.0)
    rgba = cm.viridis(norm)
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    rgb[~valid_mask] = 0
    return rgb


def _spawn_viewer_with_temp_recording(rr: Any) -> Path:
    """Save the active recording to a temp .rrd and spawn the rerun viewer.

    Spawning via a saved file rather than rr.spawn() / gRPC streaming
    avoids the script-exit-before-flush race and a few other transport
    fragilities; the extra disk-serialise cost is the price for those
    guarantees.
    """
    import time as _time

    tf = tempfile.NamedTemporaryFile(suffix=".rrd", delete=False)
    path = Path(tf.name)
    tf.close()
    logger.info(f"Writing temp .rrd to {path}…")
    t0 = _time.perf_counter()
    rr.save(str(path))
    logger.info(f"Wrote temp .rrd ({_time.perf_counter() - t0:.2f}s)")
    logger.info(f"Opening Rerun viewer on {path}")
    subprocess.Popen(
        ["rerun", str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return path
