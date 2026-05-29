# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Data-type handling: container specs and codecs.

Format-agnostic, data-specific handling procedures shared across the
writer/reader, viz, and exporters:

* :mod:`containers` — :class:`ContainerSpec` and per-type subclasses
  (numeric array, depth image, compressed video, pose, point cloud)
  plus the registry used by :class:`Spec`.
* :mod:`depth_codec` — PNG-16 encode/decode for depth images.

Anything that operates on a specific data type — codecs, validators,
shape helpers — belongs here, not in :mod:`mcap_utils`.
"""

from .containers import (
    CONTAINER_TYPES,
    CompressedVideoSpec,
    ContainerSpec,
    DepthImageSpec,
    NumericArraySpec,
    PointCloudSpec,
    PoseSpec,
    build_container,
    build_container_from_metadata,
)
from .depth_codec import decode_png16, encode_png16

__all__ = [
    "CONTAINER_TYPES",
    "CompressedVideoSpec",
    "ContainerSpec",
    "DepthImageSpec",
    "NumericArraySpec",
    "PointCloudSpec",
    "PoseSpec",
    "build_container",
    "build_container_from_metadata",
    "decode_png16",
    "encode_png16",
]
