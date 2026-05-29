# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Camera-pipeline latency visualization for composable episodes.

Replaces the legacy ``visualize_camera_lag`` (which read raw envelope
timestamps via ``TeleopMcapReader.load_raw``). Composable :class:`Reader`
exposes both ``publish_time`` and ``log_time`` per message via
:meth:`iter_messages`; this module collects them per camera and logs
the three standard lag metrics:

* **log_lag** — inter-frame ``log_time`` delta (ms).
* **publish_lag** — inter-frame ``publish_time`` delta (ms).
* **lp_delta** — per-frame ``log_time - publish_time`` (ms), the
  subscriber-pipeline latency for the frame.

The output blueprint mirrors the legacy one: cameras tiled on the left,
lag plots stacked on the right.
"""

from __future__ import annotations

import gc
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..mcap_utils.reader import Reader

logger = logging.getLogger(__name__)

AlignTimeSource = Literal["log", "publish"]


def visualize_camera_lag(
    episode_dir: Path | str,
    *,
    app_name: str | None = None,
    mode: str = "local",
    save: bool = False,
    output_path: Path | str | None = None,
    align: AlignTimeSource = "log",
) -> Path | None:
    """Visualize per-camera frame-to-frame latency for a composable episode.

    Args:
        episode_dir: Composable episode directory.
        app_name: Rerun app name (defaults to ``episode_dir.name``).
        mode: ``"local"`` spawns the viewer, ``"distant"`` runs a web
            server. Ignored when ``save=True``.
        save: If True, write a ``.rrd`` instead of opening a viewer.
        output_path: Required when ``save=True``.
        align: Which timestamp to use for the Rerun timeline:
            ``"log"`` (subscriber receive time) or ``"publish"``
            (sensor capture time, decision 2.13).
    """
    if save and output_path is None:
        raise ValueError("output_path required when save=True")

    import rerun as rr
    import rerun.blueprint as rrb

    episode_dir = Path(episode_dir)
    reader = Reader(episode_dir)

    # Pick out camera topics by container type.
    camera_topics = [
        s.topic
        for s in reader.spec.signals
        if s.container.type_name == "foxglove.CompressedVideo"
    ]
    if not camera_topics:
        raise ValueError(f"{episode_dir}: no CompressedVideo topics in spec")

    # Per-topic timestamps + frames. iter_messages preserves both
    # publish and log timestamps in source order.
    per_cam: dict[str, dict[str, np.ndarray]] = {}
    for topic in camera_topics:
        frames: list[np.ndarray] = []
        pub_ns: list[int] = []
        log_ns: list[int] = []
        for value, pt, rt in reader.iter_messages(topic):
            frames.append(value)
            pub_ns.append(pt)
            log_ns.append(rt)
        per_cam[topic] = {
            "frames": np.stack(frames, axis=0) if frames else np.empty((0,)),
            "publish_ns": np.asarray(pub_ns, dtype=np.int64),
            "log_ns": np.asarray(log_ns, dtype=np.int64),
        }

    rr.init(app_name or episode_dir.name, spawn=False)
    gc.collect()

    blueprint = _build_lag_blueprint(camera_topics, rrb)
    rr.send_blueprint(blueprint)

    # Pick the timeline origin t0 from the requested envelope across all
    # cameras so timestamps align across panels.
    ts_key = "log_ns" if align == "log" else "publish_ns"
    t0 = min(int(d[ts_key][0]) for d in per_cam.values() if d[ts_key].size > 0)

    for topic, d in per_cam.items():
        ts = d[ts_key]
        if ts.size == 0:
            continue
        log_lag_ms = np.diff(d["log_ns"]) / 1e6
        pub_lag_ms = np.diff(d["publish_ns"]) / 1e6
        lp_delta_ms = (d["log_ns"] - d["publish_ns"]) / 1e6

        for i in range(ts.shape[0]):
            elapsed_s = (int(ts[i]) - t0) / 1e9
            rr.set_time("timestamp", duration=elapsed_s)
            rr.set_time("frame_index", sequence=i)
            # Composable handler decodes to RGB; pass straight to Rerun.
            rr.log(topic, rr.Image(d["frames"][i]))
            rr.log(
                f"camera_lag/lp_delta{topic}",
                rr.Scalars(float(lp_delta_ms[i])),
            )
            if i > 0:
                rr.log(
                    f"camera_lag/log_lag{topic}",
                    rr.Scalars(float(log_lag_ms[i - 1])),
                )
                rr.log(
                    f"camera_lag/publish_lag{topic}",
                    rr.Scalars(float(pub_lag_ms[i - 1])),
                )

    if save:
        out = Path(output_path)  # type: ignore[arg-type]
        out.parent.mkdir(parents=True, exist_ok=True)
        rr.save(str(out))
        logger.info(f"Saved Rerun lag recording to {out}")
        return out

    if mode == "local":
        return _spawn_viewer(rr)
    if mode == "distant":
        rr.serve_web(open_browser=False)
        logger.info("Serving Rerun web viewer (Ctrl+C to exit).")
        try:
            import time

            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    return None


def _build_lag_blueprint(camera_topics: list[str], rrb: Any) -> Any:
    """Cameras on the left (grid), three lag plots stacked on the right."""
    visual_views = [
        rrb.Spatial2DView(name=topic, origin=topic) for topic in camera_topics
    ]
    lag_views = [
        rrb.TimeSeriesView(name="log_lag (ms)", origin="camera_lag/log_lag"),
        rrb.TimeSeriesView(name="publish_lag (ms)", origin="camera_lag/publish_lag"),
        rrb.TimeSeriesView(name="log−publish (ms)", origin="camera_lag/lp_delta"),
    ]
    n_cols = min(2, len(visual_views))
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Grid(*visual_views, grid_columns=n_cols),
            rrb.Vertical(*lag_views),
            column_shares=[2, 1],
        ),
        collapse_panels=False,
    )


def _spawn_viewer(rr: Any) -> Path:
    """Save to a temp .rrd and spawn the viewer.

    Same pattern as :mod:`_episode` — spawning via a saved file avoids
    IPC timeouts on large recordings.
    """
    tf = tempfile.NamedTemporaryFile(suffix=".rrd", delete=False)
    path = Path(tf.name)
    tf.close()
    rr.save(str(path))
    logger.info(f"Opening Rerun viewer on {path}")
    subprocess.Popen(
        ["rerun", str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return path
