# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""MCAP-format-specific machinery for the spec-driven layer.

The data-type machinery (:class:`ContainerSpec`, codecs) lives in
:mod:`dexdata.handlers`. The high-level model (:class:`Spec`,
:class:`SignalSpec`, :class:`EpisodeMetadata`, multi-rate alignment)
lives at the top of ``dexdata``. This subpackage holds the parts that
are tied to MCAP-on-disk format:

* :class:`Writer` / :class:`Reader` — open and close ``episode.mcap``.
* :class:`Episode` — the materialized read view (one stacked array
  per topic plus envelope timestamps).
* :func:`discover_episodes` — enumerate episode directories on disk.
* :mod:`handlers` — per-container serialize/deserialize state machines.
* :mod:`proto` — owned protobuf containers (``NumericArray``,
  ``DepthImage``); foxglove protos are imported directly inside the
  handlers that use them.
"""

from .discovery import discover_episodes
from .episode import Episode
from .handlers import (
    HANDLER_TYPES,
    CompressedVideoHandler,
    DepthImageHandler,
    Handler,
    NumericArrayHandler,
    PoseHandler,
    make_handler,
    runtime_field_types,
)
from .reader import Reader
from .writer import EPISODE_FILE, Writer

__all__ = [
    "CompressedVideoHandler",
    "DepthImageHandler",
    "EPISODE_FILE",
    "Episode",
    "HANDLER_TYPES",
    "Handler",
    "NumericArrayHandler",
    "PoseHandler",
    "Reader",
    "Writer",
    "discover_episodes",
    "make_handler",
    "runtime_field_types",
]
