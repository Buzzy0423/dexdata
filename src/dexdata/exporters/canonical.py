# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Composable → canonical LeRobot IR adapter + exporter config schema.

The canonical IR is a flat ``{lerobot_key: ndarray}`` dict where every
array has the same first-axis length T (the episode length in frames).
LeRobot's v2.1/v2.0 dataset format expects this shape; the IR is also
useful for any other "one row per timestep" downstream (HDF5, zarr,
plain parquet).

Composable :class:`Episode` data needs two transformations before it
fits this contract:

1. **Topic-string → canonical key rename.** Composable's topic paths
   (``/robot/state/left_arm/qpos``) become flat dotted LeRobot keys
   (``observation.state.left_arm.abs_qpos``). The map is driven by an
   :class:`ExporterConfig` loaded from per-conversion YAML.

2. **Per-container boundary conversions.** Composable's
   ``CompressedVideo`` handler now decodes to RGB uint8 directly, so
   no channel swap is needed at this boundary. Depth is already
   float32 metres with 0.0=invalid on both sides — no conversion
   needed.

The container type (RGB vs depth vs numeric) comes from each topic's
``Spec`` entry, not from the channel name — the spec is authoritative
and the YAML doesn't repeat it. The encode/clip configs themselves
live in :class:`RGBConfig` / :class:`DepthConfig` on the
:class:`ExporterConfig` and are applied at mp4-encode time downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

if TYPE_CHECKING:
    from ..mcap_utils.episode import Episode
    from ..metadata import EpisodeMetadata


# ---- encode configs (per stream type) ------------------------------------


@dataclass(frozen=True)
class RGBConfig:
    """mp4 encode settings shared by every RGB camera in a conversion.

    ``out_hw`` (optional): if set, frames are resized to ``(height, width)``
    before encoding (and stats are computed over the resized frames so
    they match what's in the mp4). ``None`` keeps source resolution.
    Resize uses ``cv2.INTER_AREA`` for downscale / ``cv2.INTER_LINEAR``
    for upscale.
    """

    codec: str = "libx264"
    pix_fmt: str = "yuv420p"
    crf: int = 23
    gop: int = 2
    out_hw: tuple[int, int] | None = None


@dataclass(frozen=True)
class DepthConfig:
    """Depth quantisation + mp4 encode settings.

    ``clip_m`` defines the linear quantisation range in metres; values
    outside (or 0/NaN) become the uint8 invalid sentinel 0. ``out_hw``
    is the resampled (height, width) used for the mp4 — depth is
    nearest-neighbor resized to this before quantisation.
    """

    codec: str = "libx264"
    pix_fmt: str = "yuv420p"
    crf: int = 23
    gop: int = 2
    clip_m: tuple[float, float] = (0.2, 1.2)
    out_hw: tuple[int, int] = (400, 640)


# ---- exporter config -----------------------------------------------------


@dataclass(frozen=True)
class ExporterConfig:
    """One conversion job: per-stream encode settings + a flat feature map.

    The YAML organises features by group (``state``, ``action``,
    ``camera``, etc.) with a ``prefix`` per group, purely as
    organisational sugar for the human editor. On load we flatten to a
    single ``{composable_path: canonical_key}`` map — the converter
    sees only that flat map and dispatches per-feature on the
    composable :class:`Spec`'s container type.
    """

    rgb: RGBConfig
    depth: DepthConfig
    features: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExporterConfig:
        """Load a conversion config from a YAML file.

        Raises:
            ValueError: a group is missing ``prefix`` or ``features``,
                or two groups produce the same composable_path.
        """
        path = Path(path)
        raw: dict[str, Any] = yaml.safe_load(path.read_text())
        rgb_raw = dict(raw.get("rgb", {}))
        if "out_hw" in rgb_raw and rgb_raw["out_hw"] is not None:
            rgb_raw["out_hw"] = tuple(rgb_raw["out_hw"])
        rgb = RGBConfig(**rgb_raw)
        depth_raw = dict(raw.get("depth", {}))
        if "clip_m" in depth_raw:
            depth_raw["clip_m"] = tuple(depth_raw["clip_m"])
        if "out_hw" in depth_raw:
            depth_raw["out_hw"] = tuple(depth_raw["out_hw"])
        depth = DepthConfig(**depth_raw)

        features: dict[str, str] = {}
        for group_name, group in raw.items():
            if group_name in ("rgb", "depth"):
                continue
            if not isinstance(group, dict):
                raise ValueError(f"{path}: group {group_name!r} must be a mapping")
            prefix = group.get("prefix")
            if not prefix:
                raise ValueError(f"{path}: group {group_name!r} missing 'prefix'")
            group_features = group.get("features") or {}
            for composable_path, suffix in group_features.items():
                if composable_path in features:
                    raise ValueError(
                        f"{path}: duplicate composable path {composable_path!r} "
                        f"in group {group_name!r}"
                    )
                features[composable_path] = f"{prefix}.{suffix}"

        return cls(rgb=rgb, depth=depth, features=features)


@dataclass(frozen=True)
class CanonicalBundle:
    """One episode in canonical IR, ready for any dataset-format exporter.

    ``arrays`` values all share the same first-axis length ``num_frames``.
    Image arrays are RGB uint8 ``(T, H, W, 3)``. Depth arrays are float32
    metres ``(T, H, W)`` with ``0.0`` as the invalid sentinel.
    ``metadata`` is the source episode's composable metadata — exporters
    pull ``record_hz``, ``embodiment`` etc. directly off it.
    """

    arrays: dict[str, np.ndarray]
    metadata: EpisodeMetadata
    num_frames: int


def composable_to_canonical(
    episode: Episode,
    metadata: EpisodeMetadata,
    config: ExporterConfig,
) -> CanonicalBundle:
    """Build a :class:`CanonicalBundle` from a composable episode.

    Args:
        episode: Materialised composable episode (from
            :meth:`Reader.read_episode`). Every composable path in
            ``config.features`` must be present in ``episode.signals``.
        metadata: The composable episode's sidecar metadata. Passed
            through to the bundle so format exporters can read
            ``record_hz`` / ``embodiment`` / ``task`` etc.
        config: Conversion config (see :class:`ExporterConfig`). Its
            flat ``features`` map drives the rename; the per-stream
            ``rgb`` / ``depth`` settings are passed to downstream
            writers via the bundle's consumer, not stored on it.

    Returns:
        :class:`CanonicalBundle` with renamed keys, BGR→RGB camera
        conversion applied, and a frame count taken from the first
        (non-image) array's leading axis.

    Raises:
        KeyError: ``config.features`` references a topic not in
            ``episode.signals``.
        ValueError: arrays disagree on the leading-axis length (the
            exporter requires aligned multi-rate input).
    """
    missing = set(config.features) - set(episode.signals)
    if missing:
        raise KeyError(f"config references topics not in episode: {sorted(missing)}")

    arrays: dict[str, np.ndarray] = {}
    for topic, canonical_key in config.features.items():
        # Both composable's CompressedVideo handler and LeRobot's mp4
        # writer use RGB; no per-frame channel swap needed.
        arrays[canonical_key] = episode.signals[topic]

    lengths = {k: int(v.shape[0]) for k, v in arrays.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(
            f"arrays have inconsistent frame counts (caller must align "
            f"multi-rate input first): {lengths}"
        )
    num_frames = next(iter(lengths.values())) if lengths else 0

    return CanonicalBundle(
        arrays=arrays,
        metadata=metadata,
        num_frames=num_frames,
    )
