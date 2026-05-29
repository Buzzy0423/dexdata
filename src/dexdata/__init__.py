# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""dexdata: spec-driven data recording and replay for Dexmate robots.

The package is organized around a single design (see ``DESIGN.md``):

* Top-level holds the format-agnostic abstractions — :class:`Spec`,
  :class:`SignalSpec`, :class:`EpisodeMetadata`, multi-rate alignment.
* :mod:`dexdata.handlers` holds the data-type machinery — container
  specs (numeric array, depth image, compressed video, pose, point
  cloud) and codecs (PNG-16 depth). Anything tied to a specific data
  type lives here.
* :mod:`dexdata.mcap_utils` holds the MCAP-on-disk machinery —
  :class:`Writer`, :class:`Reader`, :class:`Episode`, the handler
  state machines, and the protobuf containers.
* :mod:`dexdata.converters` reads non-composable inputs and emits
  composable episodes (legacy ``dexdata-qa`` episodes today; more later).
* :mod:`dexdata.exporters` writes composable episodes to downstream
  dataset formats (LeRobot v2.1 + v2.0 downgrade today).
* :mod:`dexdata.viz` is the Rerun-based viewer (CLI: ``dexdata-viz``).
* :mod:`dexdata.video_writer` is the legacy mp4 encoder leaf, still
  used by the LeRobot exporter.
"""

__version__ = "0.0.1"

from .align import align_episode
from .handlers import (
    CONTAINER_TYPES,
    CompressedVideoSpec,
    ContainerSpec,
    DepthImageSpec,
    NumericArraySpec,
    PointCloudSpec,
    PoseSpec,
    build_container,
    build_container_from_metadata,
    decode_png16,
    encode_png16,
)
from .mcap_utils import (
    EPISODE_FILE,
    Episode,
    Reader,
    Writer,
    discover_episodes,
)
from .metadata import (
    EpisodeMetadata,
    load_metadata,
    new_episode_id,
    save_metadata,
)
from .spec import SignalSpec, Spec, flatten_channels, load_spec
from .viz import visualize_episode, visualize_episode_dir

__all__ = [
    "CONTAINER_TYPES",
    "CompressedVideoSpec",
    "ContainerSpec",
    "DepthImageSpec",
    "EPISODE_FILE",
    "Episode",
    "EpisodeMetadata",
    "NumericArraySpec",
    "PointCloudSpec",
    "PoseSpec",
    "Reader",
    "SignalSpec",
    "Spec",
    "VEGA_1U_GRIPPER_LEGACY_MAP",
    "Writer",
    "__version__",
    "align_episode",
    "build_container",
    "build_container_from_metadata",
    "convert_legacy_episode",
    "decode_png16",
    "discover_episodes",
    "encode_png16",
    "flatten_channels",
    "legacy_metadata_to_composable",
    "load_metadata",
    "load_spec",
    "new_episode_id",
    "save_metadata",
    "visualize_episode",
    "visualize_episode_dir",
]
