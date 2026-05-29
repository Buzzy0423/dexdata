# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Composable-layer exporters.

The exporters here translate composable :class:`Episode` data into
downstream-friendly dataset formats. Each exporter is data-agnostic
beyond its input contract (typically the canonical IR produced by
:func:`composable_to_canonical`) — the composable layer is the
*producer*, the exporter is the *format*. Adding a new exporter
(zarr, hdf5, parquet-flat, …) only requires touching the format
layer; the data adapter is reused.

Public surface:

* :class:`CanonicalBundle` — flat ``{key: ndarray}`` + composable
  ``EpisodeMetadata`` + frame count. The shared input contract for
  every dataset-format exporter.
* :func:`composable_to_canonical` — composable :class:`Episode` +
  :class:`EpisodeMetadata` + :class:`ExporterConfig` →
  :class:`CanonicalBundle`, with topic renames and BGR→RGB camera
  conversion.
* :class:`ExporterConfig` (+ :class:`RGBConfig`, :class:`DepthConfig`)
  — per-conversion settings loaded from YAML (see
  ``exporters/config/*.yaml``).
* :mod:`.lerobot` — LeRobot v2.1 writer (with v2.1↔v2.0 downgrade and
  multi-dataset combine).
"""

from .canonical import (
    CanonicalBundle,
    DepthConfig,
    ExporterConfig,
    RGBConfig,
    composable_to_canonical,
)

__all__ = [
    "CanonicalBundle",
    "DepthConfig",
    "ExporterConfig",
    "RGBConfig",
    "composable_to_canonical",
]
