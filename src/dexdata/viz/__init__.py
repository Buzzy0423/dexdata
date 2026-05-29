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

This package replaces the legacy ``dexdata.viz`` (which spoke to the
``RobotState``/``RobotAction`` proto shape via ``TeleopMcapReader``).
Everything here is topic-path driven off the composable spec — no
embodiment knowledge, dispatch is per container type.

Three entry points:

* :func:`visualize_episode` — in-memory :class:`Episode` → Rerun.
* :func:`visualize_episode_dir` — open a composable episode dir
  via :class:`Reader` and visualize.
* :func:`visualize_camera_lag` — per-camera frame-to-frame latency
  analysis from raw envelope timestamps. Spawns a Rerun layout that
  pairs each camera's image stream with its log/publish lag plots.

CLI entry: ``dexdata-viz`` (registered in ``pyproject.toml``), with
modes mirroring the legacy CLI: ``--save``, ``--load``, ``--lag``,
``--mode {local, distant}``.
"""

from ._episode import visualize_episode, visualize_episode_dir
from ._lag import visualize_camera_lag

__all__ = [
    "visualize_camera_lag",
    "visualize_episode",
    "visualize_episode_dir",
]
